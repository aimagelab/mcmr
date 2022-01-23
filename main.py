import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time

import cv2
import imageio
import lpips
import numpy as np
import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures.meshes import Meshes
from torch.utils.tensorboard import SummaryWriter

from datasets.cub.dataset import CUBDataset
from datasets.pascal3d.dataset import PascalDataset
from datasets.pascal3d.split_train_val_test_VOC import cad_num_per_class
from models.inceptionV3 import InceptionV3
from models.mcmr import MCMRNet
from models.renderer_softras import NeuralRenderer as SOFTRAS_renderer
from utils.geometry import y_rot
from utils.lab_color import rgb_to_lab, lab_to_rgb
from utils.losses import kp_l2_loss, deform_l2reg, camera_loss, quat_reg, GraphLaplacianLoss
from utils.metrics import get_IoU, get_L1, get_SSIM, get_FID, compute_mean_and_cov, get_feat
from utils.transformations import quaternion_matrix, euler_from_matrix
from utils.visualize_results import vis_results


def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class MultiShapePredictor():
    def __init__(self, args, DEBUG, now):
        self.args = args
        self.DEBUG = DEBUG
        self.now = now
        self.set_num_classes()
        if not self.args.qualitative_results:
            self.set_logger()
        self.set_device_and_workers()

    def set_num_classes(self):
        """
        Set number of trained classes/sub-classes

        Returns:
        """
        assert len(self.args.classes) > 0
        if self.args.sub_classes:
            assert len(self.args.classes) == 1
            assert self.args.classes[0] in ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                                            'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
            self.num_classes = cad_num_per_class[self.args.classes[0]]
            self.classes = self.args.classes
        else:
            if len(self.args.classes) == 1 and self.args.classes[0] == 'all':
                if self.args.dataset_name == 'pascal':
                    self.num_classes = 12
                    self.classes = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                                    'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
                elif self.args.dataset_name == 'cub':
                    self.num_classes = 1
            else:
                assert all([pascal_class in ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                                             'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
                            for pascal_class in self.args.classes])
                self.num_classes = len(self.args.classes)
                self.classes = self.args.classes
            if self.args.single_mean_shape:
                self.num_classes = 1
        if self.args.num_learned_shapes is not None:
            self.num_classes = self.args.num_learned_shapes

    def set_logger(self):
        """
        Initialize logger and checkpoints/log directories

        Returns:
        """
        if self.DEBUG:
            self.log_dir = (Path(self.args.log_dir) / self.now)
        else:
            if self.args.is_training:
                self.log_dir = Path(self.args.log_dir)
            else:
                self.log_dir = Path(self.args.log_dir)
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.DEBUG:
            if self.args.is_training:
                self.checkpoint_dir = (Path(self.args.checkpoint_dir) / self.now)
                if not self.checkpoint_dir.is_dir():
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            if self.args.is_training:
                self.checkpoint_dir = Path(self.args.checkpoint_dir)
                if not self.checkpoint_dir.is_dir():
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_writer = SummaryWriter(str(self.log_dir))

    def set_device_and_workers(self):
        """
        Set device type and number of workers

        Returns:
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if self.DEBUG:
            self.num_workers = 0
        else:
            self.num_workers = 4

    def log_arguments(self):
        """
        Log current experiment arguments

        Returns:
        """
        param_str = ''
        for key, value in self.args.__dict__.items():
            param_str += f'{key}: {value}  \n'
        self.log_writer.add_text('Parameters', param_str)

    def define_dataset(self):
        """
        Initialize dataset and dataloader

        Returns:
        """
        assert self.args.dataset_name in ['pascal', 'cub']
        if self.args.dataset_name == 'pascal':
            if self.args.is_training:
                self.dataset_train = PascalDataset(init_mode='train',
                                                   dataset_dir=self.args.dataset_dir, classes=self.classes,
                                                   img_size=self.args.img_size, demo_mode=self.args.demo,
                                                   aug_mode=not self.args.disable_aug, cmr_mode=self.args.cmr_mode)
                self.dataset_train.train()
                self.dataset_eval = PascalDataset(init_mode='eval',
                                                  dataset_dir=self.args.dataset_dir, classes=self.classes,
                                                  img_size=self.args.img_size, demo_mode=self.args.demo,
                                                  aug_mode=False, cmr_mode=self.args.cmr_mode)
                self.dataset_eval.eval()
            else:
                self.dataset_test = PascalDataset(init_mode='test',
                                                  dataset_dir=self.args.dataset_dir, classes=self.classes,
                                                  img_size=self.args.img_size, demo_mode=self.args.demo,
                                                  aug_mode=False, cmr_mode=self.args.cmr_mode)
                self.dataset_test.test()
        if self.args.dataset_name == 'cub':
            if self.args.is_training:
                self.dataset_train = CUBDataset(init_mode='train',
                                                dataset_dir=self.args.dataset_dir, img_size=self.args.img_size,
                                                demo_mode=self.args.demo, aug_mode=not self.args.disable_aug)
                self.dataset_train.train()
                self.dataset_eval = CUBDataset(init_mode='val',
                                               dataset_dir=self.args.dataset_dir, img_size=self.args.img_size,
                                               demo_mode=self.args.demo, aug_mode=False)
                self.dataset_eval.eval()
            else:
                self.dataset_test = CUBDataset(init_mode='test',
                                               dataset_dir=self.args.dataset_dir, img_size=self.args.img_size,
                                               demo_mode=self.args.demo, aug_mode=False)
                self.dataset_test.test()
        if self.args.is_training:
            print(f'Training set images: {len(self.dataset_train)}')
            print(f'Validation set images: {len(self.dataset_eval)}')
            self.dataloader = {'train': torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                                                                    shuffle=True, num_workers=self.num_workers,
                                                                    worker_init_fn=init_worker, drop_last=True),
                               'val': torch.utils.data.DataLoader(self.dataset_eval, batch_size=1, shuffle=False,
                                                                  num_workers=self.num_workers,
                                                                  worker_init_fn=init_worker)}
        else:
            print(f'Test set images: {len(self.dataset_test)}')
            self.dataloader = {}
            self.dataloader['test'] = torch.utils.data.DataLoader(self.dataset_test, batch_size=1, shuffle=False,
                                                                  num_workers=self.num_workers,
                                                                  worker_init_fn=init_worker)

    def define_model(self):
        """
        Initialize network with or without pretrained weights

        Returns:
        """
        self.starting_epoch = 1
        self.best_IoU = 0.
        self.img_size = (self.args.img_size, self.args.img_size)
        self.G_net = MCMRNet(self.img_size, self.args, nz_feat=self.args.nz_feat,
                             num_classes=self.num_classes,
                             texture_type=self.args.texture_type)
        self.G_net = self.G_net.to(self.device)

        if self.args.pretrained_weights.stem != '':
            self.load_checkpoint(self.args.pretrained_weights)

    def define_renderer(self):
        """
        Initialize renderer and useful variables for rendering (faces, default texture for meanshape)

        Returns:
        """
        # SOFTRAS
        self.faces = {}
        if self.args.is_training:
            self.faces['train'] = self.G_net.faces.expand(self.args.batch_size, -1, 3).to(self.device)
            self.faces['val'] = self.G_net.faces.view(1, -1, 3).to(self.device)
        else:
            self.faces['test'] = self.G_net.faces.view(1, -1, 3).to(self.device)

        self.renderer = {
            'train': SOFTRAS_renderer(img_size=self.args.img_size, camera_mode='projection',
                                      orig_size=self.args.img_size, texture_type=self.args.texture_type),
            'val': SOFTRAS_renderer(img_size=self.args.img_size, camera_mode='projection',
                                    orig_size=self.args.img_size, texture_type=self.args.texture_type)
        }
        self.renderer['test'] = self.renderer['val']

        if self.args.save_results:
            self.visual_renderer = SOFTRAS_renderer(img_size=self.args.img_size, camera_mode='projection',
                                                    orig_size=self.args.img_size, background=[1, 1, 1],
                                                    texture_type=self.args.texture_type, anti_aliasing=True,
                                                    clamp_lighting=True)

        # default texture for mean/final shape visualization (if there is no texture prediction)
        self.default_texture = {}
        if self.args.texture_type == 'surface':
            if self.args.is_training:
                self.default_texture['train'] = torch.ones((self.args.batch_size, self.faces['train'].shape[1],
                                                            36, 3)).float().to(self.device)
                self.default_texture['val'] = torch.ones((1, self.faces['val'].shape[1],
                                                          36, 3)).float().to(self.device)
            else:
                self.default_texture['test'] = torch.ones((1, self.faces['test'].shape[1],
                                                           36, 3)).float().to(self.device)
        elif self.args.texture_type == 'vertex':
            if self.args.is_training:
                self.default_texture['train'] = torch.ones((self.args.batch_size, self.G_net.mean_v.shape[1],
                                                            3)).float().to(self.device)
                self.default_texture['val'] = torch.ones((1, self.G_net.mean_v.shape[1],
                                                          3)).float().to(self.device)
            else:
                self.default_texture['test'] = torch.ones((1, self.G_net.mean_v.shape[1],
                                                           3)).float().to(self.device)
        else:
            raise ValueError

        blue = torch.tensor([156, 199, 234.], device=self.device) / 255.
        for k, v in self.default_texture.items():
            self.default_texture[k] = v * blue

    def define_criterion(self):
        """
        Initialize losses and optimizer function

        Returns:
        """

        # shape
        self.projection_loss = kp_l2_loss
        self.mask_loss_fn = torch.nn.MSELoss()
        self.deform_reg_fn = deform_l2reg
        self.graph_laplacian_fn = {}
        if self.args.is_training:
            self.graph_laplacian_fn['train'] = GraphLaplacianLoss(faces=self.faces['train'][0],
                                                                  numV=self.G_net.mean_v.shape[1])
            self.graph_laplacian_fn['val'] = GraphLaplacianLoss(faces=self.faces['val'][0],
                                                                numV=self.G_net.mean_v.shape[1])
        else:
            self.graph_laplacian_fn['test'] = GraphLaplacianLoss(faces=self.faces['test'][0],
                                                                 numV=self.G_net.mean_v.shape[1])
        self.cam_loss_fn = camera_loss
        self.cam_quat_reg = quat_reg

        # texture
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)
        self.color_loss = torch.nn.MSELoss()
        self.pixel_loss = torch.nn.MSELoss()
        self.light_loss_fn = {}

        # learned classification
        if self.args.use_learned_class and self.args.class_loss_wt > 0:
            self.class_loss = torch.nn.CrossEntropyLoss()

    def define_optimizer(self, load_chekpoint=True):
        if self.args.use_sgd:
            self.G_optimizer = torch.optim.SGD(self.G_net.parameters(), lr=self.args.G_learning_rate,
                                               momentum=self.args.beta1)
        else:
            self.G_optimizer = torch.optim.Adam(self.G_net.parameters(), lr=self.args.G_learning_rate,
                                                betas=(self.args.beta1, 0.999))

        if load_chekpoint and self.args.pretrained_weights.stem != '':
            self.G_optimizer.load_state_dict(self.checkpoint['G_optimizer'])
            print('restored G_optimizer')

    def set_losses_dict(self, split):
        self.losses = {}

        self.losses['Mask Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Deformation Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Laplacian Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Deformations Laplacian Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Graph Laplacian Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Camera Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Camera Quat Reg'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

        self.losses['Perceptual Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Color Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Pixel Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

        if self.args.use_learned_class and self.args.class_loss_wt > 0:
            self.losses['Class Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

        self.losses['Total Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['Smoothed Total Loss'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

        self.losses['IoU Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        if split == 'val':
            self.losses['IoU Metric (pred cam)'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        if split == 'test':
            if self.args.dataset_name == 'pascal':
                if self.args.single_mean_shape:
                    num_classes = len(self.classes)
                    self.losses['IoU Metric per class'] = np.empty((len(self.dataloader[split]), num_classes))
                    self.losses['SSIM Metric per class'] = np.empty((len(self.dataloader[split]), num_classes))
                    self.losses['L1 Metric per class'] = np.empty((len(self.dataloader[split]), num_classes))
                    self.losses['FID Metric per class'] = np.empty((len(self.dataloader[split]), num_classes))
                else:
                    self.losses['IoU Metric per class'] = np.empty((len(self.dataloader[split]), self.num_classes))
                    self.losses['SSIM Metric per class'] = np.empty((len(self.dataloader[split]), self.num_classes))
                    self.losses['L1 Metric per class'] = np.empty((len(self.dataloader[split]), self.num_classes))
                    self.losses['FID Metric per class'] = np.empty((len(self.dataloader[split]), self.num_classes))
                self.losses['IoU Metric per class'].fill(np.nan)
                self.losses['SSIM Metric per class'].fill(np.nan)
                self.losses['L1 Metric per class'].fill(np.nan)
                self.losses['FID Metric per class'].fill(np.nan)

            self.losses['Scale Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Cx offset Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Cy offset Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Azimuth error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Elevation error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Roll error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['W quat error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['X quat error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Y quat error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
            self.losses['Z quat error Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

            # Frechet Inception Distance (FID) - appearance score based on InceptionV3 features distance
            self.perception_net = InceptionV3([3])
            self.perception_net = self.perception_net.to(self.device)
            self.perception_net.eval()
            self.losses['FID Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

        self.losses['L1 Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)
        self.losses['SSIM Metric'] = np.zeros((len(self.dataloader[split]), 1), dtype=np.float32)

    def save_checkpoint(self, path):
        """
        Save a PyTorch checkpoint.

        Args:
            path (str): path where the checkpoint will be saved.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint.pth')

        save_dict = {
            'epoch': self.epoch,
            'best_IoU': self.best_IoU,
            'G_net': self.G_net.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
            'args': self.args,
        }

        torch.save(save_dict, path)

    def load_checkpoint(self, path, strict=True):
        """
        Load a PyTorch checkpoint.

        Args:
            path (str): checkpoint path.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint.pth')

        print(f'restoring checkpoint {str(path)}')

        self.checkpoint = torch.load(str(path), map_location=self.device)

        self.starting_epoch = self.checkpoint['epoch'] + 1
        print(f'starting epoch: {self.starting_epoch}')

        if 'best_IoU' in self.checkpoint:
            self.best_IoU = self.checkpoint['best_IoU']
            print(f'best IoU: {self.best_IoU}')

        if self.args.sdf_subdivide_steps != []:
            num_subdivide = (np.asarray(self.args.sdf_subdivide_steps) < self.starting_epoch).sum()
            if num_subdivide > 0:
                for i in range(num_subdivide):
                    _ = self.G_net.subdivide_mesh()
                    if self.args.double_subdivide:
                        _ = self.G_net.subdivide_mesh()
            self.G_net.albedo_predictor.set_uv_sampler(self.G_net.icosa_sphere_v.detach().cpu().numpy(),
                                                       self.G_net.icosa_sphere_f.detach().cpu().numpy(),
                                                       self.args.tex_size)

        ret = self.G_net.load_state_dict(self.checkpoint['G_net'], strict=strict)
        print('restored G_net. Key errors:')
        print(ret)

        self.old_args = self.checkpoint['args']
        print('previous args:')
        print(self.old_args)

    def _convert_texture(self, texture):
        if self.args.color_space == 'rgb':
            if self.args.decoder_name != 'SPADE' or self.args.texture_type == 'vertex':
                texture = (texture + 1) / 2.
        elif self.args.color_space == 'lab':
            if self.args.decoder_name != 'SPADE' or self.args.texture_type == 'vertex':
                # original range is in [-1, 1] -> we need: [[0,1], [-1,1], [-1,1]]
                texture = texture + torch.tensor([1, 0, 0], device=texture.device)
                texture = texture * torch.tensor([0.5, 1, 1], device=texture.device)
            else:
                # original range is in [0, 1] -> we need: [[0,1], [-1,1], [-1,1]]
                texture = texture + torch.tensor([0, -0.5, -0.5], device=texture.device)
                texture = texture * torch.tensor([1, 2, 2], device=texture.device)
            # converting range: [[0,1], [-1,1], [-1,1]] -> [[0,100], [-110,110], [-110,110]]
            texture = texture * torch.tensor([100, 110, 110], device=texture.device)
            if self.args.texture_type == 'surface':
                texture = lab_to_rgb(texture.permute(0, 3, 1, 2)).permute((0, 2, 3, 1))
            elif self.args.texture_type == 'vertex':
                texture = lab_to_rgb(texture.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute((0, 2, 1))
            texture = texture.clamp(0, 1)
        else:
            raise ValueError

        return texture

    def save_results(self, save_dir, idx, verts, faces, texture, default_texture, Rt_pred, K_pred, scale_pred,
                     Rt_gt, K_gt, wms=False):
        self.visual_renderer = SOFTRAS_renderer(img_size=self.args.img_size, camera_mode='projection',
                                                orig_size=self.args.img_size, background=[1, 1, 1],
                                                texture_type=self.args.texture_type, anti_aliasing=True,
                                                clamp_lighting=True)

        if self.args.dataset_name == 'pascal':
            scale = 0.2
            light_a_shape = 0.65
            light_d_shape = 0.38
            R0 = cv2.Rodrigues(np.array([np.pi / 2, 0, 0]))[0]
        else:
            scale = 0.5
            light_a_shape = 0.65
            light_d_shape = 0.38
            R0 = cv2.Rodrigues(np.array([np.pi / 3, 0, 0]))[0]
        light_dir_shape = [0., 1., 1.]
        # pred cam
        self.visual_renderer.set_camera(K_pred, Rt_pred)
        light_dir = (torch.inverse(Rt_pred[0, :3, :3]) @ torch.tensor(light_dir_shape, device=Rt_pred.device).T).T
        self.visual_renderer.set_lighting(1., 0., [0, 1, 0])
        img_pred_texture, _ = self.visual_renderer(verts, faces, texture)
        img_pred_texture = (img_pred_texture.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        self.visual_renderer.set_lighting(light_a_shape, light_d_shape, light_dir / torch.norm(light_dir))
        img_pred_shape, _ = self.visual_renderer(verts, faces, default_texture)
        img_pred_shape = (img_pred_shape.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if wms:
            cv2.imwrite(str(save_dir / 'wms' / 'texture' / f'{idx:04}.png'), img_pred_texture[..., ::-1])
            cv2.imwrite(str(save_dir / 'wms' / 'shape' / f'{idx:04}.png'), img_pred_shape[..., ::-1])
        else:
            cv2.imwrite(str(save_dir / 'images' / 'texture' / f'{idx:04}.png'), img_pred_texture[..., ::-1])
            cv2.imwrite(str(save_dir / 'images' / 'shape' / f'{idx:04}.png'), img_pred_shape[..., ::-1])

        # pred cam flipped
        diag_mat = torch.diag(torch.tensor([-1., 1., 1.], device=Rt_pred.device))
        inv_K = K_pred.clone()
        inv_K[:, 0, 2] = 256 - inv_K[:, 0, 2] - 1.
        inv_Rt = Rt_pred.clone()
        inv_Rt[:, :3, :3] = torch.matmul(diag_mat, torch.matmul(inv_Rt[:, :3, :3], diag_mat))
        inv_Rt[:, 0, 3] = inv_Rt[:, 0, 3] * -1
        self.visual_renderer.set_camera(inv_K, inv_Rt)
        light_dir = (torch.inverse(inv_Rt[0, :3, :3]) @ torch.tensor(light_dir_shape, device=inv_Rt.device).T).T
        self.visual_renderer.set_lighting(1., 0., [0, 1, 0])
        img_pred_texture, _ = self.visual_renderer(verts, faces, texture)
        img_pred_texture = (img_pred_texture.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        self.visual_renderer.set_lighting(light_a_shape, light_d_shape, light_dir / torch.norm(light_dir))
        img_pred_shape, _ = self.visual_renderer(verts, faces, default_texture)
        img_pred_shape = (img_pred_shape.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if wms:
            cv2.imwrite(str(save_dir / 'wms-flipped' / 'texture' / f'{idx:04}.png'), img_pred_texture[..., ::-1])
            cv2.imwrite(str(save_dir / 'wms-flipped' / 'shape' / f'{idx:04}.png'), img_pred_shape[..., ::-1])
        else:
            cv2.imwrite(str(save_dir / 'images-flipped' / 'texture' / f'{idx:04}.png'), img_pred_texture[..., ::-1])
            cv2.imwrite(str(save_dir / 'images-flipped' / 'shape' / f'{idx:04}.png'), img_pred_shape[..., ::-1])

        # canonic matrix
        R1 = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]
        R = R1.dot(R0)
        R = torch.FloatTensor(R).float().to(Rt_gt.device)
        rot_init = R.clone()
        R *= scale
        Rt_canonic = torch.zeros((3, 4)).unsqueeze(0).float().to(Rt_gt.device)
        Rt_canonic[:, :3, :3] = R
        Rt_canonic[:, -1, -1] = 2.732
        K_canonic = K_gt.clone()
        K_canonic[:, 0, 2] = 256. / 2
        K_canonic[:, 1, 2] = 256. / 2

        if not self.args.qualitative_results:
            # canonic view
            self.visual_renderer.set_camera(K_canonic, Rt_canonic)
            light_dir = (torch.inverse(Rt_canonic[0, :3, :3]) @ torch.tensor(light_dir_shape, device=Rt_canonic.device).T).T
            self.visual_renderer.set_lighting(light_a_shape, light_d_shape, light_dir / torch.norm(light_dir))
            img_pred_texture, _ = self.visual_renderer(verts, faces, texture)
            img_pred_texture = (img_pred_texture.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pred_shape, _ = self.visual_renderer(verts, faces, default_texture)
            img_pred_shape = (img_pred_shape.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / 'canonical' / 'texture' / f'{idx:04}.png'), img_pred_texture[..., ::-1])
            cv2.imwrite(str(save_dir / 'canonical' / 'shape' / f'{idx:04}.png'), img_pred_shape[..., ::-1])

            # gif canonic view
            writer_texture = imageio.get_writer(str(save_dir / 'gifs' / 'texture' / f'{idx:04}.gif'), mode='I')
            writer_shape = imageio.get_writer(str(save_dir / 'gifs' / 'shape' / f'{idx:04}.gif'), mode='I')
            for deg in np.arange(0, 360, 10):
                # apply curr_rot to vertices
                curr_rot = y_rot(torch.FloatTensor([np.radians(deg)]), pytorch=True).to(Rt_canonic.device)
                Rt_canonic[:, :3, :3] = (curr_rot @ rot_init).unsqueeze(0).float().to(Rt_canonic.device) * scale
                self.visual_renderer.set_camera(K_canonic, Rt_canonic)
                light_dir = (torch.inverse(Rt_canonic[0, :3, :3]) @ torch.tensor(light_dir_shape, device=Rt_canonic.device).T).T
                self.visual_renderer.set_lighting(1., 0., [0, 1, 0])
                img_pred_texture, _ = self.visual_renderer(verts, faces, texture)
                img_pred_texture = (img_pred_texture.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                self.visual_renderer.set_lighting(light_a_shape, light_d_shape, light_dir / torch.norm(light_dir))
                img_pred_shape, _ = self.visual_renderer(verts, faces, default_texture)
                img_pred_shape = (img_pred_shape.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                writer_texture.append_data(img_pred_texture)
                writer_shape.append_data(img_pred_shape)
                if deg % 30 == 0:
                    cv2.imwrite(str(save_dir / 'canonical-rotations' / 'texture' / f'{idx:04}_{deg:04}.png'),
                                img_pred_texture[..., ::-1])
                    cv2.imwrite(str(save_dir / 'canonical-rotations' / 'shape' / f'{idx:04}_{deg:04}.png'),
                                img_pred_shape[..., ::-1])
            writer_texture.close()
            writer_shape.close()
        else:
            size = 512
            self.visual_renderer = SOFTRAS_renderer(img_size=size, camera_mode='projection',
                                                    orig_size=size, background=[1, 1, 1],
                                                    texture_type=self.args.texture_type, anti_aliasing=True,
                                                    clamp_lighting=True)
            K = K_pred.clone()
            K[:, 0, 2] += 128
            K[:, 1, 2] += 128
            Rt_pred_copy = Rt_pred.clone()
            rot_init = (Rt_pred_copy[0, :3, :3] / scale_pred).clone()
            for deg in np.arange(0, 360, 30):
                # apply curr_rot to vertices
                curr_rot = y_rot(torch.FloatTensor([np.radians(deg)]), pytorch=True).to(Rt_pred_copy.device)
                Rt_pred_copy[:, :3, :3] = (curr_rot @ rot_init).unsqueeze(0).float().to(Rt_pred_copy.device) * scale_pred
                self.visual_renderer.set_camera(K, Rt_pred_copy)
                light_dir = (torch.inverse(Rt_pred_copy[0, :3, :3]) @ torch.tensor(light_dir_shape,
                                                                                 device=Rt_canonic.device).T).T
                self.visual_renderer.set_lighting(1., 0., [0, 1, 0])
                img_pred_texture, _ = self.visual_renderer(verts, faces, texture)
                img_pred_texture = (img_pred_texture.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8)
                self.visual_renderer.set_lighting(light_a_shape, light_d_shape, light_dir / torch.norm(light_dir))
                img_pred_shape, _ = self.visual_renderer(verts, faces, default_texture)
                img_pred_shape = (img_pred_shape.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if wms:
                    cv2.imwrite(str(save_dir / 'wms-rotations' / 'texture' / f'{idx:04}_{deg:04}.png'),
                                img_pred_texture[128:-128, :, ::-1])
                    cv2.imwrite(str(save_dir / 'wms-rotations' / 'shape' / f'{idx:04}_{deg:04}.png'),
                                img_pred_shape[128:-128, :, ::-1])
                else:
                    cv2.imwrite(str(save_dir / 'images-rotations' / 'texture' / f'{idx:04}_{deg:04}.png'),
                                img_pred_texture[128:-128, :, ::-1])
                    cv2.imwrite(str(save_dir / 'images-rotations' / 'shape' / f'{idx:04}_{deg:04}.png'),
                                img_pred_shape[128:-128, :, ::-1])

    def run_epoch(self, phase):
        global_iter_start_time = time()

        if phase != 'train':
            eval_inputs = []
            eval_input_K = []
            eval_input_Rt = []
            eval_mean_shape = []
            eval_pred_v = []
            eval_albedo = []
            eval_shape_light_pred = []
            eval_texture_pred = []

        if phase == 'test':
            meanshape_weights = []
            meanshape_weights_count = []

        smoothed_total_loss = 0
        epoch_iter = 0

        if phase == 'train' and self.epoch in self.args.sdf_subdivide_steps:
            print(f'Applying subdivision of the meanshape @ (before) Epoch {self.epoch}')
            _ = self.G_net.subdivide_mesh()
            if self.args.double_subdivide:
                _ = self.G_net.subdivide_mesh()

            if self.args.sdf_subdivide_update_wt is True:
                # self.args.triangle_loss_wt /= 10.
                self.args.deform_reg_wt /= 10.
                self.args.delta_v_triangle_loss_wt /= 10.

            self.define_renderer()

            if self.args.texture_type == 'surface':
                self.G_net.albedo_predictor.set_uv_sampler(self.G_net.icosa_sphere_v.detach().cpu().numpy(),
                                                           self.G_net.icosa_sphere_f.detach().cpu().numpy(),
                                                           self.args.tex_size)

            self.define_criterion()
            self.define_optimizer(load_chekpoint=False)

        if (np.asarray(self.args.G_lr_steps) < 0).sum() == 0:
            if self.epoch in self.args.G_lr_steps:
                for g in self.G_optimizer.param_groups:
                    g['lr'] = self.args.G_lr_steps_values[self.args.G_lr_steps.index(self.epoch)]

        if phase == 'test' and not self.args.qualitative_results and not self.args.save_results:
            if self.args.dataset_name == 'pascal':
                from scipy.io import savemat
                shape_save_dir = f'./pred_shapes'
                Path(shape_save_dir).mkdir(parents=True, exist_ok=True)
                shape_dict = {}
                for curr_class in self.classes:
                    shape_dict[curr_class] = []

        for i, batch in enumerate(self.dataloader[phase]):
            if self.args.qualitative_results:
                if self.args.dataset_name == 'pascal':
                    samples_idxs = np.arange(len(self.dataloader[phase]))
                else:
                    samples_idxs = np.arange(len(self.dataloader[phase]))
                if i > max(samples_idxs):
                    break
                if i not in samples_idxs:
                    continue
            iter_start_time = time()

            batch_size = batch['image'].shape[0]

            input_tensor_imgs = batch['image_tensor'].float().to(self.device)
            input_imgs = batch['image'].float().to(self.device)
            input_masks = batch['mask'].float().to(self.device)
            input_intrinsic = batch['intrinsic'].float().to(self.device)
            input_R = batch['rot_matr'].float().to(self.device)
            input_tr = batch['tr_vect'].float().to(self.device)[:, None, :]
            input_cams = batch['cam_rottr'].float().to(self.device)
            if self.args.sub_classes:
                input_idxs = batch['cad_idx']
            else:
                input_idxs = batch['class_idx']
            if self.args.single_mean_shape:
                input_idxs = torch.zeros_like(batch['class_idx'])

            # Get meanshape
            mean_shape = self.G_net.get_mean_shape()
            mean_shape = mean_shape.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if not self.args.use_learned_class:
                input_idxs_onehot = torch.zeros(batch_size, self.num_classes).float().to(self.device)
                for n, idx in enumerate(input_idxs):
                    input_idxs_onehot[n, idx.item()] = 1
                input_idxs_onehot = input_idxs_onehot.unsqueeze(-1).unsqueeze(-1)
                mean_shape = (mean_shape * input_idxs_onehot).sum(1)

            # Generator
            delta_v, rot_cam_unnorm, rot_cam, tr_cam, albedo, class_pred, mean_shape = self.G_net(input_tensor_imgs, mean_shape)

            if phase == 'test':
                np_class_pred = class_pred.detach().cpu().numpy()
                meanshape_weights.append(np_class_pred.squeeze(0).copy())
                sel_class = np.argmax(np_class_pred, axis=-1)
                np_class_pred[:] = 0
                np_class_pred[:, sel_class] = 1
                meanshape_weights_count.append(np_class_pred.squeeze(0))

            # add delta to vertices
            pred_v = mean_shape + delta_v

            if phase == 'test' and self.args.dataset_name == 'pascal' and not self.args.qualitative_results and not self.args.save_results:
                shape_dict[self.classes[batch['class_idx'].item()]].append({
                    'vertices': pred_v.detach().cpu().squeeze(0).numpy(),
                    'faces': self.faces[phase].detach().cpu().squeeze(0).numpy() + 1,
                    'gt_index': batch['cad_idx'].item() + 1
                })

            # randomly flip the predicted shape/mask to force the network to predict symmetric shapes and textures
            if phase == 'train' and not self.args.disable_random_flip_predictions:
                flip_mask = (torch.rand((input_imgs.shape[0])) < 0.5).bool()

                img_flip_mask = torch.zeros(input_tensor_imgs.shape, dtype=torch.bool, device=input_tensor_imgs.device)
                img_flip_mask[flip_mask] = 1
                input_tensor_imgs = img_flip_mask.float() * torch.flip(input_tensor_imgs, dims=(-1,)) + (1 - img_flip_mask.float()) * input_tensor_imgs
                input_imgs = img_flip_mask.float() * torch.flip(input_imgs, dims=(-1,)) + (1 - img_flip_mask.float()) * input_imgs

                img_flip_mask = torch.zeros(input_masks.shape, dtype=torch.bool, device=input_masks.device)
                img_flip_mask[flip_mask] = 1
                input_masks = img_flip_mask.float() * torch.flip(input_masks, dims=(-1,)) + (1 - img_flip_mask.float()) * input_masks

                input_intrinsic[flip_mask, 0, 2] = input_imgs.shape[-1] - input_intrinsic[flip_mask, 0, 2] - 1
                diag_mat = torch.diag(torch.tensor([-1., 1., 1.], device=input_R.device))
                input_R[flip_mask] = torch.matmul(diag_mat, torch.matmul(input_R[flip_mask], diag_mat))
                input_tr[flip_mask, :, 0] = input_tr[flip_mask, :, 0] * -1

                input_cams[flip_mask, 1] = -1 * input_cams[flip_mask, 1] - 1 / 128  # -1/128 due to division by 128 instead of 127.5
                input_cams[flip_mask, 5] = -1 * input_cams[flip_mask, 5]
                input_cams[flip_mask, 6] = -1 * input_cams[flip_mask, 6]

                tr_flip_mask = torch.zeros(tr_cam.shape, dtype=torch.bool, device=tr_cam.device)
                tr_flip_mask[:, 1] = flip_mask
                tr_cam = tr_flip_mask.float() * (-1 * tr_cam - 1 / 128) + (1 - tr_flip_mask.float()) * tr_cam

                rot_flip_mask = torch.zeros(rot_cam.shape, dtype=torch.bool, device=rot_cam.device)
                rot_flip_mask[:, 2] = flip_mask
                rot_flip_mask[:, 3] = flip_mask
                rot_cam = rot_flip_mask.float() * (-1 * rot_cam) + (1 - rot_flip_mask.float()) * rot_cam

            cams_pred = torch.cat([tr_cam, rot_cam], 1)
            if phase == 'test':
                rot_mat_pred = quaternion_matrix(rot_cam.detach().cpu().numpy().squeeze(0))[:3, :3]
                el_pred, roll_pred, az_pred = euler_from_matrix(np.linalg.inv(rot_mat_pred))
                el_pred = torch.Tensor([(np.degrees(el_pred) + 90) * -1]).float().to(self.device)
                roll_pred = torch.Tensor([np.degrees(roll_pred)]).float().to(self.device)
                az_pred = torch.Tensor([((np.degrees(az_pred) * 100) % 36000) / 100]).float().to(self.device)

            # set renderers cameras
            if phase != 'train':
                scale_pred, cx_offset_pred, cy_offset_pred = tr_cam.squeeze(0)
                cx_pred = (cx_offset_pred * (self.args.img_size // 2)) + (self.args.img_size // 2)
                cy_pred = (cy_offset_pred * (self.args.img_size // 2)) + (self.args.img_size // 2)
                Rt = torch.from_numpy(quaternion_matrix(rot_cam.detach().cpu().numpy().squeeze(0))).float().unsqueeze(0).to(self.device)
                Rt[:, :3, :3] = (torch.ones_like(Rt[:, :3, :3]) * scale_pred) * Rt[:, :3, :3]
                Rt[:, -2, -1] = 5
                Rt = Rt[:, :-1, :]
                intrinsic = torch.eye(3).unsqueeze(0).to(self.device)
                if self.args.dataset_name == 'pascal':
                    intrinsic[:, 0, 0] = 3000
                    intrinsic[:, 1, 1] = 3000
                elif self.args.dataset_name == 'cub':
                    intrinsic[:, 0, 0] = 1000
                    intrinsic[:, 1, 1] = 1000
                intrinsic[:, 0, 2] = cx_pred
                intrinsic[:, 1, 2] = cy_pred

                if phase == 'test' and self.args.use_gt_camera:
                    intrinsic = input_intrinsic
                    input_Rt = torch.zeros((batch_size, 3, 4)).float().to(self.device)
                    input_Rt[:, :3, :3] = input_R
                    input_Rt[:, :, -1:] = input_tr.transpose(2, 1)
                    Rt = input_Rt

                if phase == 'test' and self.args.save_results:
                    input_Rt = torch.zeros((batch_size, 3, 4)).float().to(self.device)
                    input_Rt[:, :3, :3] = input_R
                    input_Rt[:, :, -1:] = input_tr.transpose(2, 1)
                    if self.args.qualitative_results:
                        self.save_results(self.save_dir, i, pred_v, self.faces[phase],
                                          self._convert_texture(albedo),
                                          self.default_texture[phase],
                                          Rt, intrinsic, scale_pred, input_Rt, input_intrinsic, False)
                        self.save_results(self.save_dir, i, mean_shape, self.faces[phase],
                                          self._convert_texture(albedo),
                                          self.default_texture[phase],
                                          Rt, intrinsic, scale_pred, input_Rt, input_intrinsic, True)
                    else:
                        self.save_results(self.save_dir, i, pred_v, self.faces[phase],
                                          self._convert_texture(albedo),
                                          self.default_texture[phase],
                                          Rt, intrinsic, scale_pred, input_Rt, input_intrinsic, False)

                self.renderer[phase].set_camera(intrinsic, Rt)

                if phase == 'val':
                    input_Rt = torch.zeros((batch_size, 3, 4)).float().to(self.device)
                    input_Rt[:, :3, :3] = input_R
                    input_Rt[:, :, -1:] = input_tr.transpose(2, 1)
            else:
                input_Rt = torch.zeros((batch_size, 3, 4)).float().to(self.device)
                input_Rt[:, :3, :3] = input_R
                input_Rt[:, :, -1:] = input_tr.transpose(2, 1)
                self.renderer[phase].set_camera(input_intrinsic, input_Rt)

            # generate object texture
            albedo = self._convert_texture(albedo)

            self.renderer[phase].set_lighting(1., 0., [0, 1, 0])

            if phase == 'val':
                # compute mask with predicted pose and than visualize result with gt pose for Tensorboard log
                _, mask_pred_cam = self.renderer[phase](pred_v, self.faces[phase], albedo)
                self.renderer[phase].set_camera(input_intrinsic, input_Rt)
                texture_pred, mask_pred = self.renderer[phase](pred_v, self.faces[phase], albedo)
            else:
                texture_pred, mask_pred = self.renderer[phase](pred_v, self.faces[phase], albedo)
            if torch.any(torch.isnan(texture_pred)):
                print(f"Found NaN values in texture in image #{i} - epoch #{self.epoch}")
                texture_pred = torch.zeros_like(texture_pred, dtype=texture_pred.dtype,
                                                device=texture_pred.device)
                mask_pred = torch.any(texture_pred > 0., dim=-3).float()
            if torch.any(torch.isnan(mask_pred)):
                print(f"Found NaN values in mask in image #{i} - epoch #{self.epoch}")
                mask_pred = torch.any(texture_pred > 0., dim=-3).float()

            # Compute losses
            mask_loss = self.mask_loss_fn(mask_pred, input_masks)
            deform_reg = self.deform_reg_fn(delta_v)
            laplacian_loss = mesh_laplacian_smoothing(Meshes(verts=pred_v, faces=self.faces[phase]), method='uniform')
            laplacian_delta_loss = mesh_laplacian_smoothing(Meshes(verts=delta_v, faces=self.faces[phase]), method='uniform')
            graph_laplacian_loss = self.graph_laplacian_fn[phase](pred_v)

            cam_loss = self.cam_loss_fn(cams_pred, input_cams, 0)
            cam_quat_reg_loss = self.cam_quat_reg(rot_cam_unnorm)

            tex_percept_loss = self.perceptual_loss(texture_pred * input_masks.unsqueeze(1),
                                                    input_imgs * input_masks.unsqueeze(1)).mean()

            input_imgs_lab = rgb_to_lab(input_imgs)
            texture_pred_lab = rgb_to_lab(texture_pred)

            gt_L, gt_a, gt_b = torch.chunk(input_imgs_lab, 3, dim=1)
            out_L, out_a, out_b = torch.chunk(texture_pred_lab, 3, dim=1)

            tex_color_loss = self.color_loss(torch.cat((out_a, out_b), 1) * input_masks.unsqueeze(1),
                                             torch.cat((gt_a, gt_b), 1) * input_masks.unsqueeze(1))

            tex_pixel_loss = self.pixel_loss(torch.cat([out_L, out_L, out_L], 1) * input_masks.unsqueeze(1),
                                             torch.cat([gt_L, gt_L, gt_L], 1) * input_masks.unsqueeze(1))
            if torch.any(torch.isnan(tex_pixel_loss)):
                print('this should never happen! :(')
                tex_pixel_loss = 0.

            if self.args.use_learned_class and self.args.class_loss_wt > 0:
                class_loss = self.class_loss(class_pred, input_idxs.to(self.device))

            # Sum up weighted losses and priors
            total_loss = 0.
            total_loss += self.args.mask_loss_wt * mask_loss
            total_loss += self.args.deform_reg_wt * deform_reg
            total_loss += self.args.laplacian_wt * laplacian_loss
            total_loss += self.args.laplacian_delta_wt * laplacian_delta_loss
            total_loss += self.args.graph_laplacian_wt * graph_laplacian_loss
            total_loss += self.args.cam_loss_wt * cam_loss
            total_loss += self.args.cam_reg_wt * cam_quat_reg_loss

            total_loss += self.args.tex_percept_loss_wt * tex_percept_loss
            total_loss += self.args.tex_color_loss_wt * tex_color_loss
            total_loss += self.args.tex_pixel_loss_wt * tex_pixel_loss

            if self.args.use_learned_class and self.args.class_loss_wt > 0:
                total_loss += self.args.class_loss_wt * class_loss

            smoothed_total_loss = smoothed_total_loss * 0.99 + 0.01 * total_loss.item()

            # print(total_loss)

            if phase == 'train':
                self.G_optimizer.zero_grad()
                total_loss.backward()
                self.G_optimizer.step()

            self.losses['Mask Loss'][epoch_iter] = mask_loss.item()
            self.losses['Deformation Loss'][epoch_iter] = deform_reg.item()
            self.losses['Laplacian Loss'][epoch_iter] = laplacian_loss.item()
            self.losses['Deformations Laplacian Loss'][epoch_iter] = laplacian_delta_loss.item()
            self.losses['Graph Laplacian Loss'][epoch_iter] = graph_laplacian_loss.item()
            self.losses['Camera Loss'][epoch_iter] = cam_loss.item()
            self.losses['Camera Quat Reg'][epoch_iter] = cam_quat_reg_loss.item()

            self.losses['Perceptual Loss'][epoch_iter] = tex_percept_loss.item()
            self.losses['Color Loss'][epoch_iter] = tex_color_loss.item()
            self.losses['Pixel Loss'][epoch_iter] = tex_pixel_loss.item()

            if self.args.use_learned_class and self.args.class_loss_wt > 0:
                self.losses['Class Loss'][epoch_iter] = class_loss.item()

            self.losses['Total Loss'][epoch_iter] = total_loss.item()
            self.losses['Smoothed Total Loss'][epoch_iter] = smoothed_total_loss

            self.losses['IoU Metric'][epoch_iter] = get_IoU(input_masks, mask_pred)
            if phase == 'val':
                self.losses['IoU Metric (pred cam)'][epoch_iter] = get_IoU(input_masks, mask_pred_cam)

            if phase == 'test':
                self.losses['Scale Metric'][epoch_iter] = torch.nn.L1Loss()(scale_pred, input_cams[:, 0]).item()
                cx_gt = (input_cams[:, 1] * (self.args.img_size // 2)) + (self.args.img_size // 2)
                self.losses['Cx offset Metric'][epoch_iter] = torch.nn.L1Loss()(cx_pred, cx_gt).item()
                cy_gt = (input_cams[:, 2] * (self.args.img_size // 2)) + (self.args.img_size // 2)
                self.losses['Cy offset Metric'][epoch_iter] = torch.nn.L1Loss()(cy_pred, cy_gt).item()
                rot_mat_gt = quaternion_matrix(input_cams[:, 3:].detach().cpu().numpy().squeeze(0))[:3, :3]
                el_gt, roll_gt, az_gt = euler_from_matrix(np.linalg.inv(rot_mat_gt))
                el_gt = torch.Tensor([(np.degrees(el_gt) + 90) * -1]).float().to(self.device)
                roll_gt = torch.Tensor([np.degrees(roll_gt)]).float().to(self.device)
                az_gt = torch.Tensor([((np.degrees(az_gt) * 100) % 36000) / 100]).float().to(self.device)
                self.losses['Azimuth error Metric'][epoch_iter] = torch.nn.L1Loss()(az_pred, az_gt).item()
                self.losses['Elevation error Metric'][epoch_iter] = torch.nn.L1Loss()(el_pred, el_gt).item()
                self.losses['Roll error Metric'][epoch_iter] = torch.nn.L1Loss()(roll_pred, roll_gt).item()
                quat_pred = cams_pred[:, 3:] * -1 if cams_pred[:, 3] < 0 else cams_pred[:, 3:]
                quat_gt = input_cams[:, 3:] * -1 if input_cams[:, 3] < 0 else input_cams[:, 3:]
                self.losses['W quat error Metric'][epoch_iter] = torch.nn.L1Loss()(quat_pred[:, 0], quat_gt[:, 0]).item()
                self.losses['X quat error Metric'][epoch_iter] = torch.nn.L1Loss()(quat_pred[:, 1], quat_gt[:, 1]).item()
                self.losses['Y quat error Metric'][epoch_iter] = torch.nn.L1Loss()(quat_pred[:, 2], quat_gt[:, 2]).item()
                self.losses['Z quat error Metric'][epoch_iter] = torch.nn.L1Loss()(quat_pred[:, 3], quat_gt[:, 3]).item()

                real_feat = get_feat(self.perception_net, input_imgs * input_masks.unsqueeze(1))
                pred_feat = get_feat(self.perception_net, texture_pred)
                real_m, real_c = compute_mean_and_cov(real_feat[None, ...])
                pred_m, pred_c = compute_mean_and_cov(pred_feat[None, ...])
                self.losses['FID Metric'][epoch_iter] = get_FID(mu1=real_m, sigma1=real_c, mu2=pred_m, sigma2=pred_c)

            self.losses['L1 Metric'][epoch_iter] = get_L1(input_imgs * input_masks.unsqueeze(1),
                                                          texture_pred).item()
            self.losses['SSIM Metric'][epoch_iter] = get_SSIM(input_imgs * input_masks.unsqueeze(1),
                                                              texture_pred).item()

            if phase == 'test':
                if self.args.dataset_name == 'pascal':
                    if self.args.sub_classes:
                        class_id = batch['cad_idx']
                    else:
                        class_id = batch['class_idx']
                    self.losses['IoU Metric per class'][epoch_iter, class_id] = self.losses['IoU Metric'][epoch_iter][0]
                    self.losses['SSIM Metric per class'][epoch_iter, class_id] = self.losses['SSIM Metric'][epoch_iter][0]
                    self.losses['L1 Metric per class'][epoch_iter, class_id] = self.losses['L1 Metric'][epoch_iter][0]
                    self.losses['FID Metric per class'][epoch_iter, class_id] = self.losses['FID Metric'][epoch_iter][0]

            self.total_steps += 1
            epoch_iter += 1

            if phase != 'train' and i in self.eval_idxs:
                eval_inputs.append(input_imgs)
                if phase == 'test':
                    eval_input_K.append(intrinsic)
                else:
                    eval_input_K.append(input_intrinsic)
                if phase == 'test':
                    eval_input_Rt.append(Rt)
                else:
                    eval_input_Rt.append(input_Rt)
                eval_pred_v.append(pred_v)
                eval_mean_shape.append(mean_shape)

                eval_albedo.append(albedo)
                eval_shape_light_pred.append(texture_pred)
                if self.args.texture_type == 'surface':
                    uv_images_pred = self.G_net.albedo_predictor.uvimage_pred.detach()
                    uv_images_pred = self._convert_texture(uv_images_pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    eval_texture_pred.append(uv_images_pred)
                else:
                    eval_texture_pred = None

            iter_end_time = time()
            print(f'{phase.capitalize()} iteration [{i} of {len(self.dataloader[phase])}] '
                  f'{(iter_end_time - iter_start_time):.2g} s')

        if phase == 'test':
            print(f'MEANSHAPE WEIGHTS:\n{list(np.mean(meanshape_weights, axis=0))}')
            print(f'MEANSHAPE WEIGHTS COUNT:\n{list(np.mean(meanshape_weights_count, axis=0))}')

        if not self.args.qualitative_results:
            if phase == 'test' and not self.args.save_results:
                if self.args.dataset_name == 'pascal':
                    for k, v in shape_dict.items():
                        save_dict = {}
                        save_dict[k] = v
                        savemat(f'{shape_save_dir}/{k}.mat', save_dict)

            global_iter_end_time = time()
            print(f'{phase.capitalize()} time/itr '
                  f'{((global_iter_end_time - global_iter_start_time) / (len(self.dataloader[phase]))):.2g} s')

            # Visualize predicted texture, textured meanshape and deformed meanshape
            if (not self.args.disable_display_visuals) and ((self.epoch % self.args.display_freq == 0) or
                                                            (phase == 'test')):
                if phase == 'train':
                    faces = self.faces[phase]
                    default_texture = self.default_texture[phase]
                else:
                    input_imgs = torch.cat(eval_inputs)
                    input_intrinsic = torch.cat(eval_input_K)
                    input_Rt = torch.cat(eval_input_Rt)
                    pred_v = torch.cat(eval_pred_v)
                    mean_shape = torch.cat(eval_mean_shape)

                    albedo = torch.cat(eval_albedo)
                    texture_pred = torch.cat(eval_shape_light_pred)
                    if eval_texture_pred is not None:
                        uv_images_pred = torch.cat(eval_texture_pred)

                    self.renderer[phase].set_camera(input_intrinsic, input_Rt)

                    faces = self.faces[phase].expand(len(self.eval_idxs), -1, -1)

                    if self.args.texture_type == 'surface':
                        default_texture = self.default_texture[phase].expand(len(self.eval_idxs), -1, -1, -1)
                    elif self.args.texture_type == 'vertex':
                        default_texture = self.default_texture[phase].expand(len(self.eval_idxs), -1, -1)

                # Get deformed shape with texture and light effects (if predicted)
                shape_light_pred = texture_pred
                shape_light_frontal = None
                if phase == 'train' and self.args.texture_type == 'surface':
                    uv_images_pred = self.G_net.albedo_predictor.uvimage_pred.detach()
                    uv_images_pred = self._convert_texture(uv_images_pred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

                if shape_light_pred is not None:
                    shape_light_pred = shape_light_pred.detach().cpu()
                    shape_light_pred = torch.clamp(shape_light_pred, 0, 1)
                if shape_light_frontal is not None:
                    shape_light_frontal = shape_light_frontal.detach().cpu()
                    shape_light_frontal = torch.clamp(shape_light_frontal, 0, 1)

                # Get meanshape with light effects (if present)
                self.renderer[phase].set_lighting(light_direction=[0, 0, -1])
                mean_shape_pred, _ = self.renderer[phase](mean_shape, faces, default_texture)
                mean_shape_pred = mean_shape_pred.detach().cpu()
                mean_shape_pred = torch.clamp(mean_shape_pred, 0, 1)

                # Log visual results
                if phase == 'train':
                    num_samples = batch_size
                else:
                    num_samples = len(self.eval_idxs)

                if num_samples > 8:
                    num_res = num_samples // 2
                    if phase == 'train':
                        num_row = num_samples // 4
                    else:
                        num_row = 8
                else:
                    num_res = num_samples
                    num_row = num_samples // 2

                vis_results(f'{phase.capitalize()}', self.log_writer, self.epoch, num_row,
                            input_imgs[:int(num_res)],
                            mean_shape_pred[:int(num_res)],
                            shape_light_frontal[:int(num_res)] if shape_light_frontal is not None else None,
                            shape_light_pred[:int(num_res)] if shape_light_pred is not None else None,
                            uv_images_pred[:int(num_res)]if uv_images_pred is not None else None)

            # Log/Print losses and metrics
            if self.args.is_training:
                for k, v in self.losses.items():
                    self.log_writer.add_scalar(f'{phase.capitalize()}/{k}', v.mean(), self.epoch)
            else:
                metrics_str = ''
                for k, v in self.losses.items():
                    if k in ['IoU Metric per class', 'SSIM Metric per class',
                             'L1 Metric per class', 'FID Metric per class']:
                        print(f'{k}:')
                        metrics_str += f'{k}:\n'
                        for i in range(v.shape[1]):
                            print(f'Class {i}: {np.nanmean(v[:, i])}')
                            metrics_str += f'Class {i}: {np.nanmean(v[:, i])}\n'
                    else:
                        print(f'{k}: {v.mean()}')
                        metrics_str += f'{k}: {v.mean()}\n'
                if self.args.dataset_name == 'pascal':
                    metrics_filename = ''
                    for i, curr_class in enumerate(self.classes):
                        metrics_filename += f'{curr_class}-'
                    metrics_filename += f'{self.num_classes:03}'
                    if self.args.cmr_mode is False:
                        metrics_filename += '-PointRend'
                    if self.args.use_gt_camera:
                        metrics_filename += '-gtcam'
                else:
                    metrics_filename = f'CUB-{self.num_classes:03}'
                    if self.args.use_gt_camera:
                        metrics_filename += '-gtcam'
                with open(f'./metrics_results/{metrics_filename}.txt', 'w') as fd:
                    fd.write(metrics_str)
                    fd.close()

            if phase == 'train':
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(self.epoch, self.total_steps))
                save_filename = 'net_latest.pth'
                save_path = str(self.checkpoint_dir / save_filename)
                self.save_checkpoint(save_path)

                if self.epoch % self.args.save_epoch_freq == 0:
                    save_filename = f'net_{self.epoch:04}.pth'
                    save_path = str(self.checkpoint_dir / save_filename)
                    self.save_checkpoint(save_path)

            if phase == 'val':
                if self.losses['IoU Metric (pred cam)'].mean() > self.best_IoU:
                    self.best_IoU = self.losses['IoU Metric (pred cam)'].mean()
                    save_filename = f'net_best_IoU.pth'
                    save_path = str(self.checkpoint_dir / save_filename)
                    self.save_checkpoint(save_path)

    def init_train(self):
        self.args.sdf_subdivide_steps = [int(x) for x in str(args.sdf_subdivide_steps).split(',')] \
            if args.sdf_subdivide_steps != '' else []

        self.log_arguments()
        self.define_dataset()
        self.define_model()
        self.define_renderer()
        self.define_criterion()
        self.define_optimizer()

        self.total_steps = 0
        for self.epoch in range(self.starting_epoch, self.args.num_epochs + 1):
            print(f'Epoch {self.epoch}')
            self.train()
            self.val()

        self.log_writer.close()

    def init_test(self):
        self.args.sdf_subdivide_steps = [int(x) for x in str(args.sdf_subdivide_steps).split(',')] \
            if args.sdf_subdivide_steps != '' else []

        if not self.args.qualitative_results:
            self.log_arguments()
        self.define_dataset()
        self.define_model()
        self.define_renderer()
        self.define_criterion()

        self.total_steps = 0
        self.epoch = 0
        self.test()

        if not self.args.qualitative_results:
            self.log_writer.close()

    def train(self):
        self.G_net.train()

        self.set_losses_dict('train')

        return self.run_epoch('train')

    def val(self):
        self.G_net.eval()

        self.set_losses_dict('val')

        self.eval_set_idxs = np.arange(0, len(self.dataset_eval))
        self.eval_idxs = np.sort(np.random.choice(self.eval_set_idxs, 8, replace=False))

        with torch.no_grad():
            return self.run_epoch('val')

    def test(self):
        self.G_net.eval()

        self.set_losses_dict('test')

        self.eval_set_idxs = np.arange(0, len(self.dataset_test))
        self.eval_idxs = np.sort(np.random.choice(self.eval_set_idxs, 64, replace=False))
        self.visual_idxs = self.eval_set_idxs[::30]

        if self.args.save_results:
            if self.args.dataset_name == 'cub':
                if self.args.qualitative_results:
                    self.save_dir = self.args.save_dir / 'birds-qualitative'
                else:
                    self.save_dir = self.args.save_dir / 'birds'
            else:
                dir_name = ''
                for i, class_name in enumerate(self.classes):
                    if i == len(self.classes) - 1:
                        dir_name += f'{class_name}'
                    else:
                        dir_name += f'{class_name}-'
                if self.args.qualitative_results:
                    dir_name += '-qualitative'
                self.save_dir = self.args.save_dir / dir_name
            if self.args.qualitative_results:
                sub_dirs = ['images', 'images-flipped', 'images-rotations', 'wms', 'wms-flipped', 'wms-rotations']
            else:
                sub_dirs = ['gifs', 'images', 'images-flipped', 'canonical-rotations']
            for sub_dir in sub_dirs:
                for sub_sub_dir in ['texture', 'shape']:
                    curr_dir = self.save_dir / sub_dir / sub_sub_dir
                    if not curr_dir.is_dir():
                        curr_dir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            return self.run_epoch('test')


if __name__ == '__main__':
    parser = ArgumentParser()

    # Losses
    parser.add_argument('--mask_loss_wt', type=float, default=0., help='mask loss weight')
    parser.add_argument('--cam_loss_wt', type=float, default=0., help='weights to camera loss')
    parser.add_argument('--cam_reg_wt', type=float, default=0., help='weights to camera regularization')
    parser.add_argument('--deform_reg_wt', type=float, default=0., help='regularization to deformation')
    parser.add_argument('--laplacian_wt', type=float, default=0., help='weights to laplacian smoothness prior')
    parser.add_argument('--laplacian_delta_wt', type=float, default=0.,
                        help='weights to deformations laplacian smoothness prior')
    parser.add_argument('--graph_laplacian_wt', type=float, default=0.,
                        help='weights to graph laplacian smoothness prior')
    parser.add_argument('--tex_percept_loss_wt', type=float, default=0., help='weights to tex perceptual loss')
    parser.add_argument('--tex_color_loss_wt', type=float, default=0., help='weights to tex color loss')
    parser.add_argument('--tex_pixel_loss_wt', type=float, default=0., help='weights to tex pixel loss')
    parser.add_argument('--use_learned_class', action='store_true',
                        help='learn-based selection of cad/class meanshape')
    parser.add_argument('--num_learned_shapes', type=int, default=None,
                        help='number of learnable meanshapes (if None use number of classes)')
    parser.add_argument('--class_loss_wt', type=float, default=0., help='weights to class loss')

    # Renderer
    parser.add_argument('--texture_type', type=str, default='surface',
                        help='texture type (`surface` (uv image) or `vertex` (vertex color)')
    parser.add_argument('--color_space', type=str, default='rgb', help='color space (`rgb` or `lab`')
    parser.add_argument('--tex_size', type=int, default=6, help='texture resolution per face')

    # Dataset
    parser.add_argument('--dataset_name', type=str, help='dataset name [pascal, cub]')
    parser.add_argument('--dataset_dir', type=Path, help='dataset directory location')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='which classes are used during training ["all" or list of class names]')
    parser.add_argument('--single_mean_shape', action='store_true', help='force to use a single meanshape')
    parser.add_argument('--sub_classes', action='store_true', help='train on PASCAL3D+ 3D models sub-classes')
    parser.add_argument('--disable_aug', action='store_true', help='disable data augmentation during training')
    parser.add_argument('--cmr_mode', action='store_true',
                        help='enable cmr_mode for PASCAL3D+ fair comparison (maskrcnn and cmr gt masks)')
    parser.add_argument('--demo', action='store_true', help='load 100 samples in DEBUG mode')

    # Model
    parser.add_argument('--img_size', type=int, default=256, help='input image resolution')
    parser.add_argument('--fixed_meanshape', action='store_true', help='if True, meanshape is not updated during training')
    parser.add_argument('--disable_resnet_pretrain', action='store_true', help='if True, disable resnet pretraining')
    parser.add_argument('--enc_flat_type', type=str, default='fc',
                        help='latent features flattening operation [fc, avg, view]')
    parser.add_argument('--nz_feat', type=int, default=256, help='Encoded feature size')
    parser.add_argument('--decoder_name', type=str, default='SPADE', help='decoder network [upsample, deconv, SPADE]')
    parser.add_argument('--upconv_mode', type=str, default='bilinear', help='upsample mode')
    parser.add_argument('--norm_mode', type=str, default='none', help='SPADE norm mode [batch, instance, none]')
    parser.add_argument('--subdivide', type=int, default=4,
                        help='# to subdivide icosahedron, 8=642verts, 16=2562 verts')
    parser.add_argument('--sdf_subdivide_steps', type=str, default='',
                        help='list of epochs when subdivision is applied to meanshapes. '
                             'Default: "" (none). Example: "100,200,300"')
    parser.add_argument('--double_subdivide', action='store_true',
                        help='if true activate double subdivision (e.g. 4 to 16)')
    parser.add_argument('--sdf_subdivide_update_wt', action='store_true',
                        help='activate deformation loss weight decay after mesh subdivision')

    # Training
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='size of minibatches')
    parser.add_argument('--pretrained_weights', type=Path, default='', help='pretrained weights directory')
    parser.add_argument('--G_learning_rate', type=float, default=1e-4, help='generator learning rate')
    parser.add_argument('--G_lr_steps', type=int, nargs='+', default=351, help='learning rate epoch steps')
    parser.add_argument('--G_lr_steps_values', type=float, nargs='+', default=1e-5,
                        help='learning rate epoch steps values')
    parser.add_argument('--use_sgd', action='store_true',
                        help='if true uses sgd instead of adam, beta1 is used as momentum')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--disable_random_flip_predictions', action='store_true',
                        help='disable random flips to predicted shape and corresponding masks and textures')

    # Test
    parser.add_argument('--use_gt_camera', action='store_true', help='test using gt camera rototranslation')

    # Saving checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./snapshots', help='root directory for output files')
    parser.add_argument('--save_epoch_freq', type=int, default=50, help='save model every k epochs')

    # Visualization
    parser.add_argument('--log_dir', type=str, default='./results', help='root directory for output results')
    parser.add_argument('--display_freq', type=int, default=20, help='visuals logging frequency')
    parser.add_argument('--disable_display_visuals', action='store_true', help='disable display images')
    parser.add_argument('--save_results', action='store_true', help='if true save results for each test sample')
    parser.add_argument('--qualitative_results', action='store_true',
                        help='if true save a selection of qualitative results')
    parser.add_argument('--save_dir', type=Path, default='./visual_results')

    parser.add_argument('--is_training', action='store_true', help='enable training mode')

    parser.add_argument('--faster', action='store_true', help='disable deterministic mode')

    args = parser.parse_args()

    DEBUG = (sys.gettrace() is not None)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        if not args.faster:
            print('Using deterministic mode')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print('!! Using benchmark mode !!')
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    now = str(datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
    print(now)
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    multi_shape_predictor = MultiShapePredictor(args, DEBUG, now)
    if args.is_training:
        multi_shape_predictor.init_train()
    else:
        multi_shape_predictor.init_test()
