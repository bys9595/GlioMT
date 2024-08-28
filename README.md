# An Interpretable Multimodal Transformer for Prediction of Molecular Subtypes and Grades in Adult-type Diffuse Gliomas

![alt man](./figures/main.png)

Official PyTorch codebase of Multimodal Transformer, a method for molecular subtyping and grading in adult-type diffuse gliomas according to the 2021 WHO classification.

Yunsu Byeon*, Yae Won Park*, Soohyun Lee, HyungSeob Shin, Doohyun Park, Sung Soo Ahn, Kyunghwa Han, Jong Hee Chang, Se Hoon Kim, Seung-Koo Lee, Dosik Hwang


Please cite this paper when you use this code.
Code will be updated when the paper is accepted.

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

### Modify the paths of configs
When adding your dataset in `data` folder, you should modify the `.yaml` file in `configs/paths`.

Example for `configs/paths/train.yaml`:
```
defaults:
  - default

#DATA
data_root: /home/user/Multimodal_Transformer_Glioma/data/Internal_set/
label_root: /home/user/Multimodal_Transformer_Glioma/data/Internal_set.xlsx

task_name: "train"

```

### Label Structure
To use the code directly, create and use a label file in the format `./data/label_sample.xlsx`.


## Training

**CNN & Visual Transformer Models**
```
# Usage
# bash ./scripts/run_train.sh {loss} {metric} {model} {cls_mode} {num_classes}


# IDH mutation
bash ./scripts/run_train.sh bce binary resnet50 idh 1

# 1p/19q codeletion
bash ./scripts/run_train.sh bce binary vit_base 1p_19q 1

# CNS WHO Grade
bash ./scripts/run_train.sh ce multiclass resnet50 grade 3
```

**Multimodal Transformer Models**
```
# Usage
# bash ./scripts/run_multimodal_train.sh {loss} {metric} {model} {cls_mode} {num_classes}


# IDH mutation
bash ./scripts/run_train_multimodal.sh bce binary multimodal_vit_base idh 1

# 1p/19q codeletion
bash ./scripts/run_train_multimodal.sh bce binary multimodal_swin_small 1p_19q 1

# CNS WHO Grade
bash ./scripts/run_train_multimodal.sh ce multiclass multimodal_vit_base grade 3
```

Our code's argument modification is based on [Hydra](https://hydra.cc/). To customize each argument to suit the user, modifications can be made in the `configs` folder.



## Inference
**CNN & Visual Transformer Models**
```
# Usage
# bash ./scripts/run_eval.sh {metric} {model} {cls_mode} {num_classes} {checkpoint_path}


# IDH mutation
bash ./scripts/run_eval.sh binary multimodal_vit_base idh 1 ./exp/runs/idh/20240101/090000/
```

**Multimodal Transformer Models**
```
# Usage
# bash ./scripts/run_eval_multimodal.sh {metric} {model} {cls_mode} {num_classes} {checkpoint_path}


# IDH mutation
bash ./scripts/run_eval_multimodal.sh binary multimodal_vit_base idh 1 ./exp/runs/idh/20240101/090000/
```
