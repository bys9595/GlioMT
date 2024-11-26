import os
import json
from sklearn.model_selection import KFold
import numpy as np

# data folder path setting
name = 'UCSF'
data_folder = '/mnt/BYS/dataset/UCSF/'
out_path = '/mnt/BYS/GlioMT/data'


out_name = os.path.join(out_path, f'dataset_{name}.json')
# get .nii.gz file list 

nii_files = [f for f in os.listdir(data_folder)]
nii_files.sort()

# Create 5-fold split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(nii_files))

# Initialize dictionary for each fold (both training and validation)
training_data = {'fold_' + str(i): {'train': [], 'val': []} for i in range(5)}

# Organize data by folds
for idx, file in enumerate(nii_files):
    # Find which fold this file belongs to
    for fold_num, (train_idx, val_idx) in enumerate(fold_indices):
        if idx in train_idx:
            training_data['fold_' + str(fold_num)]['train'].append({
                "image": file
            })
        elif idx in val_idx:
            training_data['fold_' + str(fold_num)]['val'].append({
                "image": file
            })

data = {
    "description": "SEV",
    "modality": "MRI",
    "name": name,
    "tensorImageSize": "3D",
    "training": training_data,
    "num_folds": 5
}

# create JSON file
with open(out_name, 'w') as f:
    json.dump(data, f, indent=4)

print("JSON file created successfully!")
