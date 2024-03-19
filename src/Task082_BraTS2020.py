#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """
    from tqdm import tqdm 
    task_name = "Task082_BraTS2020"
    nnUNet_raw_data = "/mai_nas/BYS/brain_metastasis/nnunet/nnUNet_raw_data_base/nnUNet_raw_data"
    downloaded_data_dir = "/mai_nas/BYS/brain_metastasis/data/SEV/sev_ver12/" # data path

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTs = join(target_base, "imagesTs")

    maybe_mkdir_p(target_imagesTs)

    patient_names = []
    patient_list = os.listdir(downloaded_data_dir) # SEV_0001, SEV_0002 ...
    patient_list.sort()
    
    done_list = os.listdir('/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/SEV/')
    done_list = [i.split('.')[0] for i in done_list if i.endswith('.pkl')]
    
    every_list = os.listdir('/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/SEV/')
    every_list = [i.split('.')[0] for i in every_list if i.endswith('.nii.gz')]
    
    only_list = list(set(every_list) - set(done_list))
    
    for patient_name in tqdm(only_list):
        cur = join(downloaded_data_dir, patient_name)
        patient_names.append(patient_name)
        
        t1 = join(cur, patient_name + "_T1_bet.nii.gz")
        t1c = join(cur, patient_name + "_T1C_bet.nii.gz")
        t2 = join(cur, patient_name + "_T2_bet.nii.gz")
        flair = join(cur, patient_name + "_FLAIR_bet.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesTs, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTs, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, patient_name + "_0003.nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2020"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2020"
    json_dict['licence'] = "see BraTS2020 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = 0
    json_dict['numTest'] = len(patient_names)
    json_dict['training'] = 0
    json_dict['test'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]

    save_json(json_dict, join(target_base, "dataset.json"))

   

    # test set
    #  nnUNet_ensemble -f nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold -o ensembled_nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold
    # apply_threshold_to_folder('ensembled_nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold/', 'ensemble_PP200/', 200, 2)
    # convert_labels_back_to_BraTS_2018_2019_convention('ensemble_PP200/', 'ensemble_PP200_converted')

    # export for publication of weights
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA4_BN -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 -t 82 -o nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold.zip --disable_strict
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA3_BN_BD -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 -t 82 -o nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip --disable_strict
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 -t 82 -o nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip --disable_strict
