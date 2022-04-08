from argparse import ArgumentParser
from pathlib import Path

import yaml

from datasets.pascal3d.preprocess_VOC_cmr_masks import process_CMR_masks
from datasets.pascal3d.preprocess_VOC_hariharan_masks import process_Hariharan_masks
from datasets.pascal3d.preprocess_VOC_masks import process_VOC_masks
from datasets.pascal3d.preprocess_annotations import process_annotations
from datasets.pascal3d.preprocess_masks_MaskRCNN import generate_MaskRCNN_masks
from datasets.pascal3d.preprocess_masks_PointRend import generate_PointRend_masks


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
