import cv2
import numpy as np
import yaml
from roipoly import RoiPoly
from scipy.io import loadmat

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


def process_CMR_masks(dataset_dir, results_dir):
    for pascal_class_id, pascal_class in pascal_classes.items():
        print(pascal_class)

        result_dir = results_dir / f'{pascal_class}'
        for sub_dir in ['masks_cmr']:
            curr_dir = result_dir / sub_dir
            if not curr_dir.is_dir():
                curr_dir.mkdir(parents=True, exist_ok=True)

        annot_dir = results_dir / pascal_class
        annot_files = sorted((annot_dir / 'annotations').glob('*.yaml'))

        masks_file = dataset_dir / 'masks_cmr' / f'{pascal_class}.mat'
        curr_masks = loadmat(str(masks_file), struct_as_record=False, squeeze_me=True)['segmentations']
        VOC_image_id = np.atleast_2d(curr_masks.voc_image_id).T
        VOC_rec_id = np.atleast_2d(curr_masks.voc_rec_id).T
        poly_X_idxs = curr_masks.poly_x
        poly_Y_idxs = curr_masks.poly_y

        for annot_file in annot_files:
            n = int(annot_file.stem)

            with annot_file.open('r') as fd:
                img_annot = yaml.load(fd, Loader=yaml.Loader)

            img_name = img_annot['image_name'].split('.')[0]
            w, h = img_annot['image_size'][1], img_annot['image_size'][0]

            # object bbox
            bbox = img_annot['bbox']
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] >= w:
                bbox[2] = w - 1
            if bbox[3] >= h:
                bbox[3] = h - 1

            valid_mask_idxs = np.where(VOC_image_id == img_name)[0]
            masks = []
            for poly_X, poly_Y in zip(poly_X_idxs[valid_mask_idxs], poly_Y_idxs[valid_mask_idxs]):
                assert poly_X.shape[0] == poly_Y.shape[0]

                roi_poly = RoiPoly(color='w', show_fig=False)
                roi_poly.x = poly_X
                roi_poly.y = poly_Y

                mask_img = np.zeros((h, w), dtype=np.uint8)
                mask_img = roi_poly.get_mask(mask_img) * 255

                masks.append(mask_img)

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
                assert not (result_dir / 'masks_cmr' / f'{n:05}.png').is_file()
                cv2.imwrite(str(result_dir / 'masks_cmr' / f'{n:05}.png'), found_masks[np.argmin(np.abs(diff_masks))])

            if not (result_dir / 'masks_cmr' / f'{n:05}.png').is_file():
                print('No mask found for image', img_annot['image_name'], 'with bbox', bbox, 'annotation id', n)
