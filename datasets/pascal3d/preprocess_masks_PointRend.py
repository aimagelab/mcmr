from collections import OrderedDict

import cv2
import numpy as np
import yaml
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend

from datasets.pascal3d.preprocess import pascal_classes_to_COCO, CustomDumper
from utils.geometry import pascal_vpoint_to_extrinsics, intrinsic_matrix


def generate_PointRend_masks(dataset_dir, results_dir):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    config_file = "detectron2_repo/projects/PointRend/configs/InstanceSegmentation/" \
                  "pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/" \
                        "pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor_03 = DefaultPredictor(cfg)

    for pascal_class, COCO_ids in pascal_classes_to_COCO.items():
        result_dir = results_dir / f'{pascal_class}'
        for sub_dir in ['images', 'annotations', 'masks_pointrend']:
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

        CustomDumper.add_representer(OrderedDict, CustomDumper.represent_dict_preserve_order)

        for n, annot in enumerate(pascal_annotations):
            img = cv2.imread(str(dataset_dir / 'images' / f'{pascal_class}_{annot[1]}' / annot[2]))

            # object CAD
            cad_id = int(annot[6]) - 1

            # object bbox
            bbox = [int(annot[7]), int(annot[8]), int(annot[9]), int(annot[10])]
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] >= img.shape[1]:
                bbox[2] = img.shape[1] - 1
            if bbox[3] >= img.shape[0]:
                bbox[3] = img.shape[0] - 1

            # object pose
            az = float(annot[11])
            el = float(annot[12])
            # theta = float(annot[13])
            rad = float(annot[14])
            cx, cy = float(annot[19]), float(annot[20])

            # object occlusion
            occluded = int(annot[25])
            truncated = int(annot[26])
            difficult = int(pascal_difficulty_annotations[n, 1])

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

            if len(list((result_dir / 'images').glob(annot[2]))) == 0:
                cv2.imwrite(str(result_dir / 'images' / annot[2]), img)

            # compute correct intrinsic and extrinsic matrices
            K = intrinsic_matrix(fx=3000, fy=3000, cx=cx, cy=cy)
            extrinsic = pascal_vpoint_to_extrinsics(az_deg=az, el_deg=el, radius=rad)

            # create annotation file
            new_annot = OrderedDict()
            new_annot['image_name'] = annot[2]
            new_annot['image_size'] = [img.shape[0], img.shape[1], img.shape[2]]
            new_annot['pascal_class'] = pascal_class
            new_annot['cad_idx'] = cad_id
            new_annot['occluded'] = occluded
            new_annot['truncated'] = truncated
            new_annot['difficult'] = difficult
            new_annot['bbox'] = bbox
            new_annot['kpoints_2d'] = [[int(kp[0]), int(kp[1]), int(kp[2])] for kp in kpoints]
            new_annot['intrinsic'] = [[float(val[i]) for i in range(len(val))] for val in K]
            new_annot['extrinsic'] = [[float(val[i]) for i in range(len(val))] for val in extrinsic]

            with open(str(result_dir / 'annotations' / f'{n:05}.yaml'), 'w') as fp:
                yaml.dump(new_annot, fp, Dumper=CustomDumper)

            # compute and save object mask
            kpoints = new_annot['kpoints_2d']
            outputs = predictor(img)
            masked_img = np.zeros(img.shape[:2])
            masks = []
            for i, obj_id in enumerate(outputs['instances'].pred_classes):
                if obj_id.item() in COCO_ids:
                    n_kp = 0
                    for kpoint in kpoints:
                        if 0 < kpoint[0] < img.shape[1] and 0 < kpoint[1] < img.shape[0]:
                            if outputs['instances'].pred_masks.to('cpu').numpy()[i][kpoint[1], kpoint[0]] == True:
                                n_kp += 1
                    if n_kp >= (num_kps // 2):
                        masks.append(outputs['instances'].pred_masks.to('cpu').numpy()[i])
            for mask in masks:
                masked_img[mask != False] = 255

            cv2.imwrite(str(result_dir / 'masks_pointrend' / f'{n:05}.png'), masked_img)

            if (masked_img != 0).sum() == 0:
                outputs = predictor_03(img)
                masked_img = np.zeros(img.shape[:2])
                masks = []
                for i, obj_id in enumerate(outputs['instances'].pred_classes):
                    if obj_id.item() in COCO_ids:
                        n_kp = 0
                        for kpoint in kpoints:
                            if 0 < kpoint[0] < img.shape[1] and 0 < kpoint[1] < img.shape[0]:
                                if outputs['instances'].pred_masks.to('cpu').numpy()[i][kpoint[1], kpoint[0]] == True:
                                    n_kp += 1
                        if n_kp >= (num_kps // 2):
                            masks.append(outputs['instances'].pred_masks.to('cpu').numpy()[i])
                for mask in masks:
                    masked_img[mask != False] = 255

                cv2.imwrite(str(result_dir / 'masks_pointrend' / f'{n:05}_conf_03.png'), masked_img)
