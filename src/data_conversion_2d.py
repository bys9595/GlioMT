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
    

def conversion(patient_name, slice_percentile, data_dir, seg_dir, out_dir):
    # Read seg mask
    os.makedirs(os.path.join(out_dir, patient_name), exist_ok=True)
    
    seg_mask_itk =  sitk.ReadImage(os.path.join(seg_dir, patient_name+'.nii.gz'))
    seg_mask =  sitk.GetArrayFromImage(seg_mask_itk)
    
    # Read Image
    T1_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1_bet.nii.gz'))
    T1C_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T1C_bet.nii.gz'))
    T2_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_T2_bet.nii.gz'))
    FLAIR_itk =  sitk.ReadImage(os.path.join(data_dir, patient_name, patient_name+'_FLAIR_bet.nii.gz'))
    
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
    seg_mask[seg_mask !=0] = 1
    
    
    total_img = np.stack([T1, T1C, T2, FLAIR], 0)
    z_seg = seg_mask.sum(-1).sum(-1)

    # 상위 %
    glioma_vol_lower_bound = np.percentile(z_seg[z_seg.nonzero()[0]], 100-slice_percentile, axis=0)
    roi_mask = z_seg > glioma_vol_lower_bound
    roi_idx_list = np.where(roi_mask==True)[0].tolist()
    
    for roi_idx in roi_idx_list:
        save_img = total_img[:, roi_idx]
        save_name = patient_name  + '_' + str(roi_idx).zfill(4) + '.npy'
        np.save(os.path.join(out_dir, patient_name, save_name), save_img)
        
    print(patient_name + '  done')
    
    
if __name__ == "__main__":
    data_dir = "/mai_nas/BYS/brain_metastasis/data/SEV/sev_ver12/"
    seg_dir = "/mai_nas/BYS/brain_metastasis/nnunet/nnunet_results/SEV"
    out_dir = "/mai_nas/BYS/brain_metastasis/preprocessed/SEV_2d"
    slice_percentile= 50
    out_dir = out_dir + '_' + str(slice_percentile)
    # --------------------------------------------------------------------------------------

    names = os.listdir(data_dir)
    names = natsort.natsorted(names)

    p = Pool(16)

    args = [[patient_name, slice_percentile, data_dir, seg_dir, out_dir] for i, patient_name in enumerate(names)]
    p.starmap_async(conversion, args)
    p.close()
    p.join() 
