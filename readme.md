# TRFRNet

## Introduction
This repository contains code for "Task-relevant Feature Replenishment for Cross-centre Polyp Segmentation"

## Requirements
* torch
* torchvision
* numpy
* opencv
* scipy
* tqdm
* skimage
* PIL

## Usage
### Training:
```bash
python3 train.py  --mode train --load_ckpt None  --A2B _source2target  --sdataset source_dataset  --strain_data_dir /path_to_source_data  --tdataset target_dataset  --ttrain_data_dir /path_to_target_train_data  --tvalid_data_dir /path_to_target_valid_data
```

### Inference:
```bash
python3 test.py  --mode test --load_ckpt checkpoint  --A2B _source2target  --tdataset target_dataset  --ttest_data_dir /path_to_target_test_data
```
