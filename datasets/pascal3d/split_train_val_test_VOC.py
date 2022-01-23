import glob
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

cad_num_per_class = {
    "aeroplane": 8,
    "bicycle": 6,
    "boat": 6,
    "bottle": 8,
    "bus": 6,
    "car": 10,
    "chair": 10,
    "diningtable": 6,
    "motorbike": 5,
    "sofa": 6,
    "train": 4,
    "tvmonitor": 4
}


def split_dataset(orig_dataset_dir, new_dataset_dir):
    sub_dirs = sorted(new_dataset_dir.glob('*/'))
    for sub_dir in sub_dirs:
        meta_files = np.asarray(sorted(glob.glob(os.path.join(str(sub_dir / 'annotations'), '*.yaml'))))

        with open(str(orig_dataset_dir / f'{sub_dir.stem}_val.txt'), 'r') as fp:
            pascal_VOC_validation = fp.readlines()
            for i, line in enumerate(pascal_VOC_validation):
                pascal_VOC_validation[i] = line.replace('\n', '')[:-3]

        train = []
        test = []
        cad_id_list = []
        for meta_file in tqdm(meta_files):
            with Path(meta_file).open(mode='r') as fp:
                img_annot = yaml.load(fp, Loader=yaml.Loader)

            visibility = img_annot['truncated'] | img_annot['occluded'] | img_annot['difficult']
            if visibility == 0:
                img_filename = img_annot['image_name']
                if Path(img_filename).stem in pascal_VOC_validation:
                    test.append(meta_file)
                else:
                    cad_id_list.append(img_annot['cad_idx'])
                    train.append(meta_file)
        cad_id_list = np.asarray(cad_id_list)

        sampled_cad_idxs = []
        for cad_id in range(cad_num_per_class[sub_dir.stem]):
            curr_cad_idxs = np.where(cad_id_list == cad_id)[0]
            sampled_cad_idxs.append(np.random.choice(curr_cad_idxs, int(len(curr_cad_idxs) * 0.05) + 1, replace=False))
        sampled_cad_idxs = np.sort(np.concatenate(sampled_cad_idxs))

        train = np.asarray(train)
        eval = train[sampled_cad_idxs]
        train = np.delete(train, sampled_cad_idxs)
        test = np.asarray(test)

        np.savetxt(str(sub_dir / f'{sub_dir.stem}_train.txt'), train, fmt='%s')
        np.savetxt(str(sub_dir / f'{sub_dir.stem}_eval.txt'), eval, fmt='%s')
        np.savetxt(str(sub_dir / f'{sub_dir.stem}_test.txt'), test, fmt='%s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--orig_dataset_dir', type=Path, default='your-folder/datasets/PASCAL3D+',
                        help='dateset main directory')
    parser.add_argument('--new_dataset_dir', type=Path, default='your-folder/datasets/PASCAL_final',
                        help='dateset main directory')
    args = parser.parse_args()

    split_dataset(args.orig_dataset_dir, args.new_dataset_dir)
