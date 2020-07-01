# Multi-task Face Recognition Framework

## Introduction
This repository is a multi-task face recognition framework, built on top of the [PyTorch implementation](https://github.com/TreB1eN/InsightFace_Pytorch) of ArcFace.  
It is for our IJCNN'20 paper [Neighborhood-Aware Attention Network for Semi-supervised Face Recognition](https://drive.google.com/file/d/1fNarQTLGRcmf06C1Uhytjcbn3U9hknf0/view?usp=sharing). 
You may refer to the repository [NAAN](https://github.com/labyrinth7x/NAAN) for the fully semi-supervised implementation.

## Requirments
- Python >= 3.5
- Pytorch >= 1.0.0
- numpy
- tensorboardX

## Data Preparation
- Download the full MS-Celeb-1M realeased by [ArcFace](https://github.com/deepinsight/insightface) from [baidu](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ) or [dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0), and move them to the folder ```faces_emore```.
- Download the splitted image list produced by [learn-to-cluster](https://github.com/yl-1993/learn-to-cluster) from [GoogleDrive](https://drive.google.com/file/d/1kurPWh6dm3dWQOLqUAeE-fxHrdnjaULB/view?usp=sharing) or [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ET7lHxOXSjtDiMsgqzLK9LgBi_QW0WVzgZdv2UBzE1Bgzg?e=jZ7kCS), and move them to the folder ```lists```.
- Re-arrange the dataset using ```preprocess.py```. The folder structure of ```emore``` is the same as:
  ```
  emore
   ├── trainset
   ├── testset
      ├── split_1
      ├── split_2
      ├── split_3
      ├── split_4
      ├── split_5
      ├── split_6
      ├── split_7
      ├── split_8
      ├── split_9
  ```

## Training
```
sh train_multi.sh
```
Modify the param ```path``` in ```train_multi.sh``` to the directory of the generated pseudo-label file ```split{}_labels.txt```.
