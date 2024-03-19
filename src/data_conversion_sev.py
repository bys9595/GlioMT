import subprocess
import numpy as np
import os
import natsort
import pandas as pd
from operator import itemgetter
import SimpleITK as sitk
from multiprocessing.pool import Pool
import shutil

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != data.min()
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    if len(mask.shape) == 3:
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minyidx = int(np.min(mask_voxel_coords[1]))
        maxyidx = int(np.max(mask_voxel_coords[1])) + 1
        minxidx = int(np.min(mask_voxel_coords[2]))
        maxxidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minyidx, maxyidx], [minxidx, maxxidx]]
    elif len(mask.shape) == 2:
        minyidx = int(np.min(mask_voxel_coords[0]))
        maxyidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        return [[minyidx, maxyidx], [minxidx, maxxidx]]
    

def conversion(patient_name, data_dir, seg_dir, out_dir):
    # Read seg mask
    seg_mask_itk =  sitk.ReadImage(os.path.join(seg_dir, patient_name +'.nii.gz'))
    seg_mask =  sitk.GetArrayFromImage(seg_mask_itk)
    
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_T1_bet.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_T1C_bet.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_T2_bet.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_FLAIR_bet.nii.gz'))
    
    T1 =  sitk.GetArrayFromImage(T1_itk)
    T1C =  sitk.GetArrayFromImage(T1C_itk)
    T2 =  sitk.GetArrayFromImage(T2_itk)
    FLAIR =  sitk.GetArrayFromImage(FLAIR_itk)
    
    
    
    nonzero_mask = create_nonzero_mask(np.expand_dims(T1, 0))
    bbox_idx = get_bbox_from_mask(nonzero_mask)
    
    T1 = T1[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
    T1C = T1C[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
    T2 = T2[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
    FLAIR = FLAIR[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
    seg_mask = seg_mask[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
    
    os.makedirs(os.path.join(out_dir, patient_name.split('_')[0]), exist_ok=True)
    
    new_t1_itk = sitk.GetImageFromArray(T1)
    sitk.WriteImage(new_t1_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_T1.nii.gz"))
    
    new_t1c_itk = sitk.GetImageFromArray(T1C)
    sitk.WriteImage(new_t1c_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_T1C.nii.gz"))
    
    new_t2_itk = sitk.GetImageFromArray(T2)
    sitk.WriteImage(new_t2_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_T2.nii.gz"))

    new_fl_itk = sitk.GetImageFromArray(FLAIR)
    sitk.WriteImage(new_fl_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_FLAIR.nii.gz"))
    
    new_segmask_itk = sitk.GetImageFromArray(seg_mask)
    # new_fl_itk.CopyInformation(FLAIR_itk)
    sitk.WriteImage(new_segmask_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_seg.nii.gz"))
    
    
    # os.rename(os.path.join(data_dir, patient_name, patient_name+'_T1C_regi_bet.nii.gz'), os.path.join(data_dir, patient_name, patient_name+'_T1C_bet.nii.gz'))        
    
    if os.path.isfile(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_ADC_bet.nii.gz')):
        ADC_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name.split('_')[0]+'_ADC_bet.nii.gz'))
        ADC =  sitk.GetArrayFromImage(ADC_itk)
        ADC = ADC[bbox_idx[0][0]:bbox_idx[0][1], bbox_idx[1][0]:bbox_idx[1][1], bbox_idx[2][0]:bbox_idx[2][1]]
        new_adc_itk = sitk.GetImageFromArray(ADC)
        sitk.WriteImage(new_adc_itk, os.path.join(out_dir, patient_name.split('_')[0], patient_name.split('_')[0] + "_ADC.nii.gz"))
    print(patient_name + '  done')
    
    
if __name__ == "__main__":
    data_dir = "/mai_nas/BYS/brain_metastasis/data/SEV/sev_ver12/"
    # seg_dir = "/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/SEV"
    seg_dir = "/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/SEV/"
    out_dir = "/mnt/BYS/dataset/SEV/"
    # --------------------------------------------------------------------------------------

    names = os.listdir(data_dir)
    names = natsort.natsorted(names)
    

    p = Pool(16)

    args = [[patient_name, data_dir, seg_dir, out_dir] for i, patient_name in enumerate(names)]
    p.starmap_async(conversion, args)
    p.close()
    p.join() 
