# An Interpretable Multimodal Transformer for Prediction of Molecular Subtypes and Grades in Adult-type Diffuse Gliomas According to the 2021 WHO Classification


![alt man](./figures/main_figure2.png)

Official PyTorch codebase of Multimodal Transformer, a method for molecular subtyping and grading in adult-type diffuse gliomas according to the 2021 WHO classification.

Yunsu Byeon*, Yae Won Park*, Soohyun Lee, HyungSeob Shin, Doohyun Park, Sung Soo Ahn, Kyunghwa Han, Jong Hee Chang, Se Hoon Kim, Seung-Koo Lee, Dosik Hwang


Please cite this paper when you use this code.

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Data Preparation
Datasets must be located in the `data` folder. Within each dataset folder, the following structure is expected:

```
./data/Internal_set/
├── subject001
│   ├── subject001_T1.nii.gz
│   ├── subject001_T1C.nii.gz
│   ├── subject001_T2.nii.gz
│   ├── subject001_FLAIR.nii.gz
│
├── subject002
│   ├── subject002_T1.nii.gz
│   ├── subject002_T1C.nii.gz
│   ├── subject002_T2.nii.gz
│   ├── subject002_FLAIR.nii.gz
│
│ ...
```
The label file (`.xlsx`) must be located outside the dataset folder, an example is as follows:
```
./data/
├── Internal_set
├── Internal_set_label.xlsx
├── TCGA
├── TCGA_label.xlsx
├── UCSF
├── UCSF_label.xlsx
│...
```

**configs/paths**
When adding your dataset in `data` folder, you should modify the `.yaml` file in `configs/paths`.

Example for `configs/paths/train.yaml`:
```
defaults:
  - default

#DATA
data_root: /home/user/Multimodal_Transformer_Glioma/data/Internal_set/
label_root: /home/user/Multimodal_Transformer_Glioma/data/Internal_set/Internal_set.xlsx

save_dir: null
job_num: ''

task_name: "val"

```


## Training

**CNN & Visual Transformer Models**
```
# IDH mutation
python train.py loss=bce metric=binary model=resnet50 cls_mode=idh model.num_classes=1

# 1p/19q codeletion
python train.py loss=bce metric=binary model=resnet50 cls_mode=1p_19q model.num_classes=1

# CNS WHO Grade
python train.py loss=ce metric=multiclass model=resnet50 cls_mode=grade model.num_classes=3
```

**Our Multimodal Transformer Models**
```
# IDH mutation
python train_multimodal.py loss=bce metric=binary model=multimodal_swin_small cls_mode=idh model.num_classes=1

# IDH mutation
python train_multimodal.py loss=bce metric=binary model=multimodal_swin_small cls_mode=1p_19q model.num_classes=1

# IDH mutation
python train_multimodal.py loss=ce metric=multiclass model=multimodal_swin_small cls_mode=idh model.num_classes=3
```

Our code's argument modification is based on [Hydra](https://hydra.cc/). To customize each argument to suit the user, modifications can be made in the `configs` folder.



## Inference
**2D**
```
python inference.py --cmd [cmd mode] --model [model] --cls_mode [class mode] --num_slice_per_patient [# slice] --gpus [gpus] --dimension 2d --resume [.pth 파일 경로]
```

**3D**
```
python inference.py --cmd [cmd mode] --model [model] --cls_mode [class mode] --gpus [gpus] --resume [.pth 파일 경로] --dimension 3d 
```

- cmd : Internal validation set을 할 거라면 'val' 선택 ,  External validation (TCGA) dataset 할 거라면 'test' 선택
- cls_mode : 'idh', '1p_19q' 같이 원하는 모드 선택
- num_slice_per_patient : max_roi_slice 중심으로 몇개의 slice 를 train/val 로 사용할지
- model : resnet50 같은 모델명 설정
