import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from utils.geometry import quaternion_from_matrix
from utils.image import crop, perturb_bbox, square_bbox


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


class CUBDataset(Dataset):
    def __init__(self, dataset_dir: Path, img_size: int = 224, init_mode: str = 'train',
                 demo_mode: bool = False, aug_mode: bool = False):

        if not dataset_dir.is_dir():
            raise OSError(f'Directory {dataset_dir} does not exist.')

        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.demo_mode = demo_mode
        self.aug_mode_args = aug_mode
        self.aug_mode = aug_mode
        self._mode = init_mode
        self.data = self._load_data()

    def __getitem__(self, idx):
        sample = self.data[self.mode][idx].copy()
        sample['bbox'] = sample['bbox'].copy()
        sample['mask'] = sample['mask'].copy()
        sample['kpoints'] = sample['kpoints'].copy()
        sample['sfm_pose'] = sample['sfm_pose'].copy()

        img = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = sample['mask']

        bbox = sample['bbox']

        # kps = sample['kpoints'].copy()
        # kps[:, :2] = (kps[:, :2] - 127.5) / 127.5

        # convert SfM pose to intrinsic and extrinsic
        fixed_z = 5.
        focal = 1000.
        factor = 6.

        scale = sample['sfm_pose'][0].flatten().astype(np.float32)[0]
        trans = sample['sfm_pose'][1].flatten().astype(np.float32)
        rot = sample['sfm_pose'][2].astype(np.float32)

        # 7.732 is the fixed distance used in U-CMR considering camera eye
        orig_scale = (scale * (7.732 / fixed_z)) / focal
        orig_scale *= factor
        K = np.asarray([[focal, 0, trans[0]],
                        [0, focal, trans[1]],
                        [0, 0, 1]])
        extrinsic = np.zeros((3, 4))
        extrinsic[:3, :3] = rot
        extrinsic[-1, -1] = fixed_z

        # bbox perturbation
        if self.aug_mode is True:
            bbox = perturb_bbox(bbox, 0.05, 0.05)
            bbox = square_bbox(bbox)
        else:
            if self.mode == 'val':
                bbox = perturb_bbox(bbox, 0.05, 0.)
                bbox = square_bbox(bbox)
            else:
                bbox = perturb_bbox(bbox, 0., 0.)
                bbox = square_bbox(bbox)

        # image and mask crop
        try:
            img = crop(img, bbox, bgval=0)
            mask = crop(mask, bbox, bgval=0)
        except:
            print(sample['image_path'])
        # kps[:, 0] -= bbox[0]
        # kps[:, 0] = np.clip(kp[:, 0], 0, bbox[2] - bbox[0])
        # kps[:, 1] -= bbox[1]
        # kps[:, 1] = np.clip(kp[:, 1], 0, bbox[3] - bbox[1])
        K[0, 2] -= bbox[0]
        K[1, 2] -= bbox[1]

        # image and mask rescaling
        h, w, _ = img.shape
        img_scale = self.img_size / float(max(h, w))
        new_size = (np.round(np.array(img.shape[:2]) * img_scale)).astype(int)
        img = cv2.resize(img, (new_size[1], new_size[0]))
        mask = cv2.resize(mask, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        # kp[:, :2] *= img_scale
        # K[0, 0] *= img_scale
        K[0, 2] *= img_scale
        K[1, 2] *= img_scale

        # image and mask horizontal flipping
        if self.aug_mode and random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            # kps[:, 0] *= img.shape[1] - kps[:, 0] - 1
            K[0, 2] = img.shape[1] - K[0, 2] - 1
            extrinsic[:3, :3] = np.diag([-1, 1, 1]).dot(extrinsic[:3, :3].dot(np.diag([-1, 1, 1])))
            extrinsic[0, 3] *= -1

        scale = orig_scale * img_scale

        # rotation and translation processing
        rot = extrinsic[:3, :3]
        tr = extrinsic[:, -1]
        cx = (K[0, 2] - (img.shape[1] // 2)) / (img.shape[1] // 2)  # cx normalized offset from image center
        cy = (K[1, 2] - (img.shape[0] // 2)) / (img.shape[0] // 2)  # cy normalized offset from image center

        rot_homogeneous = np.eye(4)
        rot_homogeneous[:3, :3] = rot.copy()
        quat = quaternion_from_matrix(rot_homogeneous, isprecise=False)
        cam_rottr = np.concatenate([[scale, cx, cy], quat])

        rot = (np.ones_like(rot) * scale) * rot

        return {
            'image': (img.copy().astype(np.float32) / 255.).transpose(2, 0, 1),
            'image_tensor': self._to_tensor(img.copy()),
            'mask': mask.astype(np.float32) / 255.,
            'class_idx': 0,
            # 'kpoints_2d_coords': kps,
            'intrinsic': K,
            'rot_matr': rot,
            'tr_vect': tr,
            'cam_rottr': cam_rottr
        }

    def __len__(self):
        return len(self.data[self.mode])

    def train(self):
        self.mode = 'train'
        if self.aug_mode_args:
            self.aug_mode = True

    def eval(self):
        self.mode = 'val'
        self.aug_mode = False

    def test(self):
        self.mode = 'test'
        self.aug_mode = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'test']
        self._mode = value

    @property
    def name(self):
        return self.__class__.__name__

    def _load_data(self):

        data = defaultdict(list)

        split = self._mode

        image_annots = sio.loadmat(str(self.dataset_dir / 'data' / f'{split}_cub_cleaned.mat'),
                                   struct_as_record=False, squeeze_me=True)['images']
        sfm_annots = sio.loadmat(str(self.dataset_dir / 'sfm' / f'anno_{split}.mat'),
                                struct_as_record=False, squeeze_me=True)['sfm_anno']

        if split == 'val':
            # random selection of almost 10% of validation samples
            val_num_samples = np.sort(np.random.choice(np.arange(0, len(image_annots)), 325, replace=False))
            image_annots = image_annots[val_num_samples]
            sfm_annots = sfm_annots[val_num_samples]

        if self.demo_mode:
            image_annots = image_annots[:100]
            sfm_annots = sfm_annots[:100]

        for image_annot, sfm_annot in zip(image_annots, sfm_annots):
            image_path = str(self.dataset_dir.parents[1] / 'CUB_200_2011' / 'images' / image_annot.rel_path)

            mask = image_annot.mask * 255

            if (mask != 0).sum() == 0:
                continue

            class_idx = image_annot.class_id

            bbox = np.array([image_annot.bbox.x1, image_annot.bbox.y1,
                             image_annot.bbox.x2, image_annot.bbox.y2], dtype=np.float) - 1

            y_mask, x_mask = np.where(mask != 0)
            x_min_mask, y_min_mask = x_mask.min(), y_mask.min()
            x_max_mask, y_max_mask = x_mask.max(), y_mask.max()
            mask_bbox = [x_min_mask, y_min_mask, x_max_mask, y_max_mask]

            if get_iou(mask_bbox, bbox) < 0.5:
                bbox = mask_bbox

            parts = image_annot.parts.T.astype(float)
            kpoints = np.copy(parts)
            vis = kpoints[:, 2] > 0
            kpoints[vis, :2] -= 1

            sfm_pose = [np.copy(sfm_annot.scale), np.copy(sfm_annot.trans), np.copy(sfm_annot.rot)]

            sample = {
                'image_path': image_path,
                'class_idx': class_idx,
                'mask': mask,
                'bbox': bbox,
                'kpoints': kpoints,
                'sfm_pose': sfm_pose
            }

            data[split].append(sample)

        return data

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """
        See: https://pytorch.org/docs/stable/torchvision/models.html

        All pre-trained models expect input images normalized in the same
         way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
         where H and W are expected to be at least 224. The images have to be
         loaded in to a range of [0, 1] and then normalized using mean = [0.485,
         0.456, 0.406] and std = [0.229, 0.224, 0.225]
        """
        image = image.astype(float) / 255.

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        image_tensor = normalize(image_tensor.float(),
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        return image_tensor
