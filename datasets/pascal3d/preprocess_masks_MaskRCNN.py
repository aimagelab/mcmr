import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

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


def generate_MaskRCNN_masks(dataset_dir, results_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor_03 = DefaultPredictor(cfg)

    for pascal_class, COCO_ids in pascal_classes_to_COCO.items():
        result_dir = results_dir / f'{pascal_class}'
        masks_dir = result_dir / 'masks'
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

            cv2.imwrite(str(result_dir / 'masks' / f'{n:05}.png'), masked_img)

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

                cv2.imwrite(str(result_dir / 'masks' / f'{n:05}_conf_03.png'), masked_img)
