import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects import point_rend

from datasets.pascal3d.preprocess import pascal_classes_to_COCO


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
        masks_dir = result_dir / 'masks_pointrend'
        if not masks_dir.is_dir():
            masks_dir.mkdir(parents=True, exist_ok=True)

        with open(str(dataset_dir / 'annotations' / f'pascal3d_{pascal_class}_keypoints.txt'), 'r') as fp:
            pascal_annotations = fp.readlines()
            for i, line in enumerate(pascal_annotations):
                line = line.replace('\n', '')
                pascal_annotations[i] = line.split(',')

        for n, annot in enumerate(pascal_annotations):
            img = cv2.imread(str(dataset_dir / 'images' / f'{pascal_class}_{annot[1]}' / annot[2]))

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

            # compute and save object mask
            kpoints = [[int(kp[0]), int(kp[1]), int(kp[2])] for kp in kpoints]
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
