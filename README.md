# GlioMT

![alt man](./figures/main.png)


Official implementation of **"[Interpretable Multimodal Transformer for Prediction of Molecular Subtypes and Grades in Adult-type Diffuse Gliomas](https://www.nature.com/articles/s41746-025-01530-4)"**.

Yunsu Byeon*, Yae Won Park*, Soohyun Lee, Doohyun Park, HyungSeob Shin, Kyunghwa Han, Jong Hee Chang, Se Hoon Kim, Seung-Koo Lee, Sung Soo Ahn, Dosik Hwang


## Updates

***27/11/2024***

[explainability.ipynb](./explainability.ipynb) is updated.

***28/08/2024***

Initial commits

## Abstract
Molecular subtyping and grading of adult-type diffuse gliomas are essential for treatment decisions and patient prognosis. Here, we introduce a robust interpretable multimodal transformer (GlioMT), incorporating imaging and clinical data, to predict the molecular subtype and grade of adult-type diffuse gliomas according to the 2021 WHO classification. GlioMT is trained on multiparametric MRI data from an institutional set of 1,053 patients with adult-type diffuse gliomas (144 oligodendrogliomas, 157 IDH-mutant astrocytomas, and 752 IDH-wildtype glioblastomas) to predict the IDH mutation status, 1p/19q codeletion status, and tumor grade. External validation is performed on 200 and 477 patients from the TCGA and UCSF sets, respectively. GlioMT outperforms conventional CNNs and visual transformers across multiple classification tasks with the highest AUCs for prediction of IDH mutation (0.92 on TCGA and 0.98 on UCSF), 1p/19q codeletion (0.85 on TCGA and 0.81 on UCSF), and grade (0.81 on TCGA and 0.91 on UCSF). Interpretability analysis revealed that GlioMT effectively highlighted tumor regions that are considered most discriminative, underscoring its potential to improve the reliability of clinical decision-making. 
 


## Model Weights
Pre-trained models using the institutional set are available below.


|                   | GlioMT (ours)|
|:-----------------:|:------------------------:|
| IDH mutation      |       [GlioMT_ViT-B](https://drive.google.com/file/d/1VtRU3W_Tl_ghrsa2Fjpmr8Y-yJqaV_sn/view?usp=drive_link)    |
| 1p/19q codeletion |      [GlioMT_Swin-S](https://drive.google.com/file/d/1AkXwnEBg_M0f7eaF11TuSyqu6ktDCuPb/view?usp=drive_link)    |
| Tumor Grade       |      [GlioMT_Swin-S](https://drive.google.com/file/d/1lnasGRbYbA5c3cPmZa0kBwJkixCrV9Sj/view?usp=drive_link)    |



## Getting Started
### 1. Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.

### 2. Data Preparation
Datasets must be located in the `data` folder. Within each dataset folder, the following structure is expected:

```
./data/Internal/
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
├── Internal
├── Internal_label.xlsx
├── TCGA
├── TCGA_label.xlsx
├── UCSF
├── UCSF_label.xlsx
│...
```

`./sample/label_sample.xlsx` is an example label file.


### 3. Modify the paths of configs

When adding your dataset in `data` folder, you should modify the `.yaml` file in `configs/paths`.

Training Example for `configs/paths/train.yaml`:
```
# Training DATA
data_root: /home/user/GlioMT/data/Internal/
label_root: /home/user/GlioMT/data/Internal_label.xlsx
json_root: /home/user/GlioMT/data/dataset_Internal.json
task_name: "train"
```

Validation Example for `configs/paths/TCGA.yaml`:
```
# Validation DATA
data_root: /home/user/GlioMT/data/TCGA/
label_root: /home/user/GlioMT/data/TCGA_label.xlsx
json_root: /home/user/GlioMT/data/dataset_TCGA.json
task_name: "val"
```

## Training

```
# Usage
# bash ./scripts/run_train_GlioMT.sh {loss} {metric} {model} {cls_mode} {num_classes} {slice percentile}


# IDH mutation
bash ./scripts/run_train_GlioMT.sh bce binary GlioMT_vit_base idh 1 75

# 1p/19q codeletion
bash ./scripts/run_train_GlioMT.sh bce binary GlioMT_swin_small 1p_19q 1 75

# CNS WHO Grade
bash ./scripts/run_train_GlioMT.sh ce multiclass GlioMT_vit_base grade 3 25
```

Our code's argument modification is based on [Hydra](https://hydra.cc/). To customize each argument to suit the user, modifications can be made in the `configs` folder.



## Inference
```
# Usage
# bash ./scripts/run_eval_GlioMT.sh {metric} {model} {cls_mode} {num_classes} {slice percentile} {checkpoint_path}


# IDH mutation
bash ./scripts/run_eval_GlioMT.sh binary GlioMT_vit_base idh 1 75 ./exp/runs/idh/20240101/090000/
```


## Interpretability Analysis
[explainability.ipynb](./explainability.ipynb) is an example of interpretability analysis.

## Citation ✏️ 📄
If you find this repo useful for your research, please consider citing the paper as follows:

```
@article{byeon2025interpretable,
  title={Interpretable multimodal transformer for prediction of molecular subtypes and grades in adult-type diffuse gliomas},
  author={Byeon, Yunsu and Park, Yae Won and Lee, Soohyun and Park, Doohyun and Shin, HyungSeob and Han, Kyunghwa and Chang, Jong Hee and Kim, Se Hoon and Lee, Seung-Koo and Ahn, Sung Soo and others},
  journal={NPJ Digital Medicine},
  volume={8},
  number={1},
  pages={140},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
