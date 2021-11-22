import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize
from tqdm import tqdm

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


class PascalDataset(Dataset):

    def __init__(self, dataset_dir: Path, classes: list, img_size: int = 224, init_mode: str = 'train',
                 demo_mode: bool = False, aug_mode: bool = False, cmr_mode: bool = False):

        if not dataset_dir.is_dir():
            raise OSError(f'Directory {dataset_dir} does not exist.')

        self.dataset_dir = dataset_dir
        self.classes = classes
        self.img_size = img_size
        self.demo_mode = demo_mode
        self.aug_mode_args = aug_mode
        self.aug_mode = aug_mode
        self.cmr_mode = cmr_mode
        self._mode = init_mode
        self.data = self._load_data()

    def __getitem__(self, idx):
        sample = self.data[self.mode][idx].copy()
        sample['bbox'] = sample['bbox'].copy()
        sample['kpoint_array'] = sample['kpoint_array'].copy()
        sample['intrinsic'] = sample['intrinsic'].copy()
        sample['extrinsic'] = sample['extrinsic'].copy()

        img = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cmr_mode and self.mode == 'test' and self.classes[sample['pascal_class']] == 'car':
            mask = sample['mask_path'].copy()
        else:
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)

        bbox = sample['bbox']

        # kps = sample['kpoint_array'].copy()
        # kps[:, :2] = (kps[:, :2] - 127.5) / 127.5

        # bbox perturbation
        if self.aug_mode is True:
            bbox = perturb_bbox(bbox, 0.05, 0.05)
            bbox = square_bbox(bbox)
        else:
            if self.mode == 'eval':
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
            print(sample['mask_path'])
        # kps[:, 0] -= bbox[0]
        # kps[:, 0] = np.clip(kp[:, 0], 0, bbox[2] - bbox[0])
        # kps[:, 1] -= bbox[1]
        # kps[:, 1] = np.clip(kp[:, 1], 0, bbox[3] - bbox[1])
        sample['intrinsic'][0, 2] -= bbox[0]
        sample['intrinsic'][1, 2] -= bbox[1]

        # image and mask rescaling
        h, w, _ = img.shape
        scale = self.img_size / float(max(h, w))
        new_size = (np.round(np.array(img.shape[:2]) * scale)).astype(int)
        img = cv2.resize(img, (new_size[1], new_size[0]))
        mask = cv2.resize(mask, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        # kp[:, :2] *= scale
        sample['intrinsic'][0, 0] *= scale
        sample['intrinsic'][0, 2] *= scale
        sample['intrinsic'][1, 1] *= scale
        sample['intrinsic'][1, 2] *= scale

        # image and mask horizontal flipping
        if self.aug_mode and random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            # kps[:, 0] *= img.shape[1] - kps[:, 0] - 1
            sample['intrinsic'][0, 2] = img.shape[1] - sample['intrinsic'][0, 2] - 1
            sample['extrinsic'][:3, :3] = np.diag([-1, 1, 1]).dot(sample['extrinsic'][:3, :3].dot(np.diag([-1, 1, 1])))
            sample['extrinsic'][0, 3] *= -1

        # fix focal (3000) and translation offset on Z axis (5m) for all dataset samples
        K = sample['intrinsic']
        extrinsic = sample['extrinsic']

        new_K = K.copy()
        new_K[0, 0] = 3000
        new_K[1, 1] = 3000

        fixed_z = 5.
        scale = K[0, 0] / new_K[0, 0] * fixed_z / extrinsic[-1, -1]

        new_extrinsic = extrinsic.copy()
        new_extrinsic[-1, -1] = fixed_z
        new_extrinsic[-1, :-1] /= K[0, 0] / new_K[0, 0]

        # rotation and translation processing
        rot = new_extrinsic[:3, :3]
        tr = new_extrinsic[:, -1]
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
            'class_idx': sample['pascal_class'],
            'cad_idx': sample['cad_idx'],
            # 'kpoints_2d_coords': kps,
            'intrinsic': new_K,
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
        self.mode = 'eval'
        self.aug_mode = False

    def test(self):
        self.mode = 'test'
        self.aug_mode = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'eval', 'test']
        self._mode = value

    @property
    def name(self):
        return self.__class__.__name__

    def _load_data(self):

        data = defaultdict(list)

        split = self._mode

        for class_sub_dir in sorted(self.dataset_dir.glob('*/')):
            if class_sub_dir.stem in self.classes:
                if self.cmr_mode and split == 'test' and class_sub_dir.stem == 'car':
                    # CMR and U-CMR have the same test set
                    test_UCMR = loadmat(str('./UCMR_P3D_data/p3d/data/car_val.mat'),
                                        struct_as_record=False, squeeze_me=True)['images']
                    test_UCMR_bbox = [[x.bbox.x1, x.bbox.y1, x.bbox.x2, x.bbox.y2]
                                      for x in loadmat(str('./UCMR_P3D_data/p3d/data/car_val.mat'),
                                                       struct_as_record=False, squeeze_me=True)['images']]

                meta_filenames = np.loadtxt(str(class_sub_dir / f'{class_sub_dir.stem}_{split}.txt'), dtype=str)
                if self.demo_mode:
                    meta_filenames = meta_filenames[:100]

                for meta_f in tqdm(meta_filenames, f'[{self.name}] Loading {class_sub_dir.stem} {split} set'):
                    meta_f = Path(meta_f)
                    with meta_f.open('r') as f:
                        img_annotations = yaml.load(f, Loader=yaml.Loader)

                    image_path = str(class_sub_dir / 'images' / img_annotations['image_name'])

                    if self.cmr_mode and split == 'test' and class_sub_dir.stem == 'car':
                        if img_annotations['bbox'] not in test_UCMR_bbox:
                            continue
                        mask_idx = test_UCMR_bbox.index(img_annotations['bbox'])
                        mask = test_UCMR[mask_idx].mask * 255
                    else:
                        mask_paths = [
                            'masks_VOC',
                            'masks_hariharan',
                            'masks_cmr',
                            'masks' if self.cmr_mode is True else 'masks_pointrend',
                        ]

                        if split != 'test':
                            for path in mask_paths:
                                mask_path = str(class_sub_dir / path / f'{meta_f.stem}.png')
                                if Path(mask_path).is_file():
                                    break
                        else:
                            if self.cmr_mode is True:
                                # same test masks as in CMR and U-CMR
                                mask_path = str(class_sub_dir / 'masks_cmr' / f'{meta_f.stem}.png')
                            else:
                                # all gt masks
                                for path in mask_paths[:3]:
                                    mask_path = str(class_sub_dir / path / f'{meta_f.stem}.png')
                                    if Path(mask_path).is_file():
                                        break

                        if not Path(mask_path).is_file():
                            continue

                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if (mask != 0).sum() == 0:
                        continue

                    bbox = np.asarray(img_annotations['bbox'])

                    y_mask, x_mask = np.where(mask != 0)
                    x_min_mask, y_min_mask = x_mask.min(), y_mask.min()
                    x_max_mask, y_max_mask = x_mask.max(), y_mask.max()
                    mask_bbox = [x_min_mask, y_min_mask, x_max_mask, y_max_mask]

                    if get_iou(mask_bbox, bbox) < 0.5:
                        bbox = mask_bbox

                    pascal_class = self.classes.index(img_annotations['pascal_class'])

                    cad_idx = img_annotations['cad_idx']

                    kpoints = np.asarray(img_annotations['kpoints_2d'], dtype=np.float32)

                    # keypoint filtering as done in CMR and U-CMR
                    if (kpoints[:, 2] == 1).sum() <= 3:
                        continue

                    K = np.asarray(img_annotations['intrinsic'], dtype=np.float32)
                    extrinsic = np.asarray(img_annotations['extrinsic'], dtype=np.float32)

                    if extrinsic[-1, -1] == 0:
                        continue

                    sample = {
                        'image_path': image_path,
                        'mask_path': mask if self.cmr_mode and split == 'test' and class_sub_dir.stem == 'car' else mask_path,
                        'bbox': bbox,
                        'pascal_class': pascal_class,
                        'cad_idx': cad_idx,
                        'kpoint_array': kpoints,
                        'intrinsic': K,
                        'extrinsic': extrinsic,
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
