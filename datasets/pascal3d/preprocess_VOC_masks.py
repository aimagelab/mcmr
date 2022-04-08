import cv2
import numpy as np
from tqdm import tqdm

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


def process_VOC_masks(dataset_dir, results_dir):
    with (dataset_dir / 'VOC_mask_colors_RGB.txt').open('rt') as fd:
        VOC_colors = fd.readlines()
        VOC_colors = np.asarray([x.strip().split()[::-1] for x in VOC_colors], dtype=np.uint8)
        # border_color = np.asarray([192, 224, 224], dtype=np.uint8)

    with (dataset_dir / 'VOC_classes.txt').open('rt') as fd:
        VOC_classes = fd.readlines()
        VOC_classes = np.asarray([x.strip() for x in VOC_classes], dtype=str)

    for pascal_class in pascal_classes:
        print(pascal_class)
        result_dir = results_dir / f'{pascal_class}'
        for sub_dir in ['masks_VOC']:
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

            mask_objects = cv2.imread(str(dataset_dir / 'SegmentationObject' / annot[2].replace('.jpg', '.png')))
            mask_classes = cv2.imread(str(dataset_dir / 'SegmentationClass' / annot[2].replace('.jpg', '.png')))
            if mask_objects is None:
                print(f'Object mask not found for image "{annot[2]}"!')
                continue
            if mask_classes is None:
                print(f'Class mask not found for image "{annot[2]}"!')
                continue

            # Keypoints
            num_kps = (len(annot) - 27) // 3
            kpoint_idxs = np.arange(27, len(annot)).reshape(num_kps, 3)
            kpoints = np.zeros((num_kps, 3), dtype=int)  # x, y, visibility (0=not visible, 1=visible)

            for k, idxs in enumerate(kpoint_idxs):
                kpoints[k, 0] = int(annot[idxs[1]])
                kpoints[k, 1] = int(annot[idxs[2]])
                if int(annot[idxs[0]]) == 1:
                    kpoints[k, 2] = 1
                else:
                    kpoints[k, 2] = 0

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
            for color in VOC_colors:
                masked_img = np.zeros(mask_objects.shape[:2])
                curr_obj = np.all(mask_objects == color, axis=-1)
                curr_obj_dilated = curr_obj.copy().astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
                curr_obj_dilated = cv2.dilate(curr_obj_dilated, kernel, iterations=2)
                curr_obj_dilated = curr_obj_dilated > 0
                if np.all(curr_obj == False):
                    break
                curr_class = mask_classes[curr_obj]
                assert np.any(curr_class != 0)
                assert np.all(curr_class == curr_class[0])
                curr_color = curr_class[0]
                class_name = VOC_classes[np.argwhere(np.all(curr_color == VOC_colors, axis=-1))[0][0]]
                if class_name == pascal_class:
                    sum_mask = np.sum(curr_obj_dilated)
                    sum_mask_bbox = np.sum(curr_obj_dilated[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                    if sum_mask * 0.95 < sum_mask_bbox < sum_mask * 1.05 and sum_mask > bbox_pixels * 0.2:
                        masked_img[curr_obj] = 255
                        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
                        masked_img = cv2.dilate(masked_img, kernel, iterations=1)
                        found_masks.append(masked_img)
                        diff_masks.append(sum_mask - sum_mask_bbox)
            if len(found_masks) > 0:
                assert not (result_dir / 'masks_VOC' / f'{n:05}.png').is_file()
                cv2.imwrite(str(result_dir / 'masks_VOC' / f'{n:05}.png'), found_masks[np.argmin(np.abs(diff_masks))])
