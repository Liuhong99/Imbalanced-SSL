# Imbalanced-SSL
Code Release for ["Self-supervised Learning is More Robust to Dataset Imbalance"](https://openreview.net/forum?id=4AZz9osqrar&noteId=vwXjA-Os4AW)

## Requirement
- pytorch>=1.6.0
- torchvision
- numpy
- tqdm

## Data
The CIFAR data is available in the [google drive link](https://drive.google.com/file/d/1bFnGMVW3d0RxDoV1L5nbqBMnxXqcdWmV/view?usp=sharing). 

The ImageNet textlists is available in the [google drive link]().

## Usage

### Train on Imbalanced CIFAR
Exponential weight
```
python main_sam_exp.py --data_root data/ --arch resnet18 \
       --learning_rate 0.06 --epochs 1600 --weight_decay 5e-4 --momentum 0.9 \
       --batch_size 512 --gpu 0 \
       --exp_dir <exp_dir> \
       --rho 7.0 --phi 1.6
```
KDE weight
```
python main_sam_weight.py --data_root data/ --arch resnet18 \
       --learning_rate 0.06 --epochs 1600 --weight_decay 5e-4 --momentum 0.9 \
       --batch_size 512 --gpu 0 \
       --weight_path <weight_path> \
       --exp_dir <exp_dir> \
```
No weight
```
python main.py --data_root data/ --arch resnet18 \
       --learning_rate 0.06 --epochs 1600 --weight_decay 5e-4 --momentum 0.9 \
       --batch_size 512 --gpu 0 \
       --exp_dir <exp_dir> \
```
### Evaluation on Balanced CIFAR
```
python main_lincls.py --arch resnet18 --num_cls 10 \
       --batch_size 256 --lr 30.0 --weight_decay 0.0 \
       --pretrained <pretraining_checkpoint_path> --gpu 0 <cifar_data_path>
```
### Train on ImageNet
```
python pretrain_moco_sam_rare.py --dataset imagenet --data data/imagenet \
       --epochs 300 --rho 2 --phi 1.005 \
       --root_path <root_path> --dist-url tcp://localhost:10001
```
### Evaluate on Downstream Datasets
```
CUDA_VISIBLE_DEVICES=0 python ft_moco.py <dataset_path> \
       -d <dataset_name> -sr 100 --lr 0.1 -i 2000 \
       --lr-decay-epochs 5 10 15 --epochs 20  \
       --log <output_path> --pretrained <model_checkpoint_path>
```

## Citation
```
@inproceedings{
liu2022selfsupervised,
title={Self-supervised Learning is More Robust to Dataset Imbalance},
author={Hong Liu and Jeff Z. HaoChen and Adrien Gaidon and Tengyu Ma},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=4AZz9osqrar}
}
```
