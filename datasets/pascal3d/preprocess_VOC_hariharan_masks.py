from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

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


def process_Hariharan_masks(dataset_dir, results_dir):
    for pascal_class_id, pascal_class in pascal_classes.items():
        print(pascal_class)

        result_dir = results_dir / f'{pascal_class}'
        for sub_dir in ['masks_hariharan']:
            curr_dir = result_dir / sub_dir
            if not curr_dir.is_dir():
                curr_dir.mkdir(parents=True, exist_ok=True)

        with open(str(dataset_dir / 'annotations' / f'pascal3d_{pascal_class}_keypoints.txt'), 'r') as fp:
            pascal_annotations = fp.readlines()
            for i, line in enumerate(pascal_annotations):
                line = line.replace('\n', '')
                pascal_annotations[i] = line.split(',')

        with open(str(dataset_dir / 'annotations' / f'pascal3d_{pascal_class}_difficulty.txt'), 'r') as fp:
            pascal_difficulty_annotations = fp.readlines()
            for i, line in enumerate(pascal_difficulty_annotations):
                line = line.replace('\n', '')
                pascal_difficulty_annotations[i] = line.split(',')
            pascal_difficulty_annotations = np.asarray(pascal_difficulty_annotations)

        for n, annot in tqdm(enumerate(pascal_annotations)):
            if annot[1] != 'pascal':
                continue

            occluded = int(annot[25])
            truncated = int(annot[26])
            difficult = int(pascal_difficulty_annotations[n, 1])
            if occluded or truncated or difficult:
                continue

            mask_path = dataset_dir / 'masks_hariharan' / str(pascal_class_id)  # type: Path
            masks = sorted(mask_path.glob('%s__???.png' % annot[2][:-4]))
            if len(masks) == 0:
                print(f'Masks not found for image "{annot[2]}"!')
                continue

            masks = [cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in masks]

            img_h = int(annot[3])
            img_w = int(annot[4])

            # object bbox
            bbox = [int(annot[7]), int(annot[8]), int(annot[9]), int(annot[10])]
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] >= img_w:
                bbox[2] = img_w - 1
            if bbox[3] >= img_h:
                bbox[3] = img_h - 1

            # compute and save object mask
            bbox_pixels = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            found_masks = []
            diff_masks = []
            for mask in masks:
                sum_mask = np.sum(mask) / 255
                sum_mask_bbox = np.sum(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]) / 255
                if sum_mask * 0.95 < sum_mask_bbox < sum_mask * 1.05 and sum_mask > bbox_pixels * 0.2:
                    found_masks.append(mask)
                    diff_masks.append(sum_mask - sum_mask_bbox)
            if len(found_masks) > 0:
                assert not (result_dir / 'masks_hariharan' / f'{n:05}.png').is_file()
                cv2.imwrite(str(result_dir / 'masks_hariharan' / f'{n:05}.png'), found_masks[np.argmin(np.abs(diff_masks))])

            if not (result_dir / 'masks_hariharan' / f'{n:05}.png').is_file():
                print('No mask found for image', annot[2], 'with bbox', bbox, 'annotation id', n)
