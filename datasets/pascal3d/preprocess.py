from argparse import ArgumentParser
from pathlib import Path

import yaml

from datasets.pascal3d.preprocess_VOC_cmr_masks import process_CMR_masks
from datasets.pascal3d.preprocess_VOC_hariharan_masks import process_Hariharan_masks
from datasets.pascal3d.preprocess_VOC_masks import process_VOC_masks
from datasets.pascal3d.preprocess_annotations import process_annotations
from datasets.pascal3d.preprocess_masks_MaskRCNN import generate_MaskRCNN_masks
from datasets.pascal3d.preprocess_masks_PointRend import generate_PointRend_masks

pascal_classes = [
    'aeroplane',
    'bicycle',
    'boat',
    'bottle',
    'bus',
    'car',
    'chair',
    'diningtable',
    'motorbike',
    'sofa',
    'train',
    'tvmonitor'
]

pascal_classes_id = {
    1: 'aeroplane',
    2: 'bicycle',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    9: 'chair',
    11: 'diningtable',
    14: 'motorbike',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

pascal_classes_to_COCO = {
    'aeroplane': [4],
    'bicycle': [1],
    'boat': [8],
    'bottle': [39],
    'bus': [5],
    'car': [2, 7],
    'chair': [56],
    'diningtable': [60],
    'motorbike': [3],
    'sofa': [57],
    'train': [6],
    'tvmonitor': [62]
}


class CustomDumper(yaml.Dumper):
    # Super neat hack to preserve the mapping key order. See https://stackoverflow.com/a/52621703/1497385
    def represent_dict_preserve_order(self, data):
        return self.represent_dict(data.items())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, default='your-folder/datasets/PASCAL3D+',
                        help='dateset main directory')
    parser.add_argument('--results_dir', type=Path, default='your-folder/datasets/PASCAL_final',
                        help='dateset main directory')
    args = parser.parse_args()

    print("Processing annotations")
    process_annotations(args.dataset_dir, args.results_dir)
    print("Generating MaskRCNN masks")
    generate_MaskRCNN_masks(args.dataset_dir, args.results_dir)
    print("Generating PointRend masks")
    generate_PointRend_masks(args.dataset_dir, args.results_dir)
    print("Process VOC masks")
    process_VOC_masks(args.dataset_dir, args.results_dir)
    print("Process CMR masks")
    process_CMR_masks(args.dataset_dir, args.results_dir)
    print("Process Hariharan masks")
    process_Hariharan_masks(args.dataset_dir, args.results_dir)
    print("PREPROCESSING ENDED SUCCESSFULLY!")
