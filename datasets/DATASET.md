# Dataset preparation
Create a ```your-folder/datasets``` directory.

## CUB
1) Download [CUB](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) dataset and unzip it in ```your-folder/datasets```.
2) Download annotations from [U-CMR](https://github.com/shubham-goel/ucmr) using this [link](https://drive.google.com/file/d/1kNGuBpBlUFhBeU0ycqGQD1HGwKQvCZ-y/view?usp=sharing) and unzip them in ```your-folder/datasets```.

At the end you should obtain a ```your-folder/datasets``` directory organized as follows:
```
   .
   ├── CUB_200_2011/             # CUB original dataset
   └── UCMR_CUB_data/            # U-CMR annotations
```


## Pascal3D+

1. Follow option A) or B) to have preprocessed data of Pascal3D+.
2. Download annotations of _car_ class from [U-CMR](https://github.com/shubham-goel/ucmr) using this [link](https://drive.google.com/file/d/1fVSu926c-UOM8PUDZJKqHH3ocX3UOJ7i/view?usp=sharing) and unzip them in ```your-folder/datasets```. These data are used when testing on the _car_ class to reproduce results of Table 2 in the paper.

### <ins> A) Download preprocessed data</ins>
To obtain the already preprocessed data, please send an email to [Alessandro Simoni](https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=125) stating:
1) Your name, title and affiliation
2) Your intended use of the data

### <ins>B) Preprocess data yourself</ins>
Install [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) (v0.3) in your environment running this line:
```bash
python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
```

1) Create ```PASCAL3D+``` directory in ```your-folder/datasets```.
2) Create ```PASCAL3D+/images``` directory  in ```your-folder/datasets```.
3) Download and unzip the [Pascal3D+](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) dataset.
4) Copy and paste the content of ```Images``` folder of the unzipped dataset in ```your-folder/datasets/PASCAL3D+/images```.
5) Copy and paste ```PASCAL/VOCdevkit/VOC2012/SegmentationClass``` folder of the unzipped dataset in ```your-folder/datasets/PASCAL3D+```.
6) Copy and paste ```PASCAL/VOCdevkit/VOC2012/SegmentationObject``` folder of the unzipped dataset in ```your-folder/datasets/PASCAL3D+```.
7) Download the [annotations/masks](https://drive.google.com/file/d/1FU5a__9K3cq5aPn1OX8ZOHoJ_nQqOLDZ/view?usp=sharing) folder and unzip it into ```your-folder/datasets/PASCAL3D+```. 

At the end you should obtain a ```your-folder/datasets/PASCAL3D+``` directory organized as follows:
```
   .
   ├── annotations/             # folder with txt annotation files
   ├── images/                  # folder with Imagenet/PascalVOC image folders
   │   ├── aeroplane_imagenet/    
   │   │                  ├── first_image.jpg
   │   │                  ├── ...
   │   │                  └── last_image.jpg
   │   ├── aeroplane_pascal/             
   │   ├── ...             
   │   ├── tvmonitor_imagenet/            
   │   └── tvmonitor_pascal/                
   ├── masks_cmr/               # folder with CMR masks
   │          ├── aeroplane.mat
   │          ├── ...
   │          └── tvmonitor.mat
   ├── masks_hariharan/         # folder with Hariharan masks
   │          ├── 1
   │          │   ├── first_mask.png
   │          │   ├── ...
   │          │   └── last_mask.png
   │          ├── ...
   │          └── 20
   ├── SegmetationClass/        # folder with segmentation class masks for PascalVOC
   │   │             ├── first_image.jpg
   │   │             ├── ...
   │   │             └── last_image.jpg
   ├── SegmentationObject/      # folder with segmentation object masks for PascalVOC
   │   │             ├── first_image.jpg
   │   │             ├── ...
   │   │             └── last_image.jpg
   ├── aeroplane_val.txt
   ├── bicycle_val.txt
   ├── boat_val.txt
   ├── bottle_val.txt
   ├── bus_val.txt
   ├── car_val.txt
   ├── chair_val.txt
   ├── diningtable_val.txt
   ├── motorbike_val.txt
   ├── sofa_val.txt
   ├── train_val.txt
   ├── tvmonitor_val.txt
   ├── VOC_classes.txt
   └── VOC_mask_colors_RGB.txt
```
Run the preprocessing scripts as follows:
```bash
   python datasets/pascal3d/preprocess.py --dataset_dir your-folder/datasets/PASCAL3D+ --results_dir your-folder/datasets/PASCAL_final
   python datasets/pascal3d/split_train_val_test_VOC.py --orig_dataset_dir your-folder/datasets/PASCAL3D+ --new_dataset_dir your-folder/datasets/PASCAL_final
```

At the end you should obtain a ```your-folder/datasets/PASCAL_final``` directory organized as follows:
```
   .
   ├── aeroplane/                       # class folder
   │   ├── annotations/                 # class annotations folder
   │   │             ├── first_annot.yaml
   │   │             ├── ...
   │   │             └── last_annot.yaml
   │   ├── images/   
   │   │        ├── first_image.jpg
   │   │        ├── ...
   │   │        └── last_image.jpg       
   │   ├── masks/   
   │   │        ├── first_mask.png
   │   │        ├── ...
   │   │        └── last_mask.png         
   │   ├── masks_cmr/   
   │   │        ├── first_mask.png
   │   │        ├── ...
   │   │        └── last_mask.png          
   │   ├── masks_hariharan/  
   │   │        ├── first_mask.png
   │   │        ├── ...
   │   │        └── last_mask.png           
   │   ├── masks_pointrend/ 
   │   │        ├── first_mask.png
   │   │        ├── ...
   │   │        └── last_mask.png            
   │   ├── masks_VOC/       
   │   │        ├── first_mask.png
   │   │        ├── ...
   │   │        └── last_mask.png      
   │   ├── aeroplane_eval.txt            
   │   ├── aeroplane_test.txt       
   │   └── aeroplane_train.txt                    
   ├── bicycle/
   ├── boat/
   ├── bottle/
   ├── bus/
   ├── car/
   ├── chair/
   ├── diningtable/
   ├── motorbike/
   ├── sofa/
   ├── train/
   └── tvmonitor/
```