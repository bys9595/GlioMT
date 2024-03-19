import os
import torch
import numpy as np


'''
Classification Mode selection

- idh : binary, mutant vs wildtype
- 1p_19q : binary, mutant+codeletion vs mutant+no-codeletion
- subtype : 3-class, Mutant + 1p/19q codeletion  vs  Mutant + 1p/19q no codeletion  vs  Wildtype
- lgg_hgg : binary,  LGG vs HGG
- grade : 3-class,    Grade 2 vs Grade 3 vs Grade 4

'''


def return_mean_value(metrics):
    for i in range(len(metrics)):
        if isinstance(metrics[i], torch.Tensor):
            metrics[i] = metrics[i].mean().item()
        elif isinstance(metrics[i], np.ndarray):
            metrics[i] = metrics[i].mean().item()
    
    return metrics


def patient_and_class_selection_with_clinical(args):
    if args.cmd == 'test':
        root_path = args.test_label_root
    else:
        root_path = args.label_root
    
    df_from_excel = pd.read_excel(root_path)
    df_from_excel.replace(['M','male'], -1, inplace=True)
    df_from_excel.replace(['F','female'], 1, inplace=True)
    
    if 'adc' in args.seq_comb:
        adc_mask = np.array(df_from_excel['ADC_exist']).astype(np.bool_)    
    
    # ---------------------------------------------------------------------------------------
    
    if args.cls_mode == 'idh':
        name_list = np.array(df_from_excel['Anony_ID'])
        cls_list = np.array(df_from_excel['IDH_mutation'])
        mask = None
    elif args.cls_mode == '1p_19q':
        mutation = np.array(df_from_excel['IDH_mutation']).astype(np.bool_)
        name_list = np.array(df_from_excel['Anony_ID'])
        cls_list = np.array(df_from_excel['1p/19q codeletion'])
        mask = mutation
    elif args.cls_mode == 'subtype':
        name_list = np.array(df_from_excel['Anony_ID'])
        cls_list = np.array(df_from_excel['Mole_Group_no'])
        cls_list = cls_list - 1 # 1, 2, 3 -> 0, 1, 2
        mask = None
    elif args.cls_mode == 'lgg_hgg':
        name_list = np.array(df_from_excel['Anony_ID'])
        cls_list = np.array(df_from_excel['WHO_23_4']) # LGG : 0,  HGG : 1
        mask = None
    elif args.cls_mode == 'grade':
        name_list = np.array(df_from_excel['Anony_ID'])
        cls_list = np.array(df_from_excel['WHO']) # LGG : 0,  HGG : 1
        cls_list = cls_list - 2 # 2, 3, 4 -> 0, 1, 2
        mask = None
    
    if 'adc' in args.seq_comb:
        if mask is None:
            mask = adc_mask
        else:
            mask = mask * adc_mask

    age_mean = 54.922218319478596
    age_std = 14.65117384080856
    
    age_list = np.array(df_from_excel['Age'])
    age_list = (age_list - age_mean) / age_std
    
    sex_list = np.array(df_from_excel['Sex'])
    sex_list[pd.isna(sex_list)] = -1
    sex_list = sex_list.astype(np.int32)
    
    location_list = np.array(df_from_excel['nonfrontal_0_frontal_1'])
    location_list = location_list * 2 -1

    clinical_feats = np.stack([np.squeeze(age_list[mask]), np.squeeze(sex_list[mask]), np.squeeze(location_list[mask])], 0)
    
    name_list = np.squeeze(name_list[mask])
    cls_list = np.squeeze(cls_list[mask])
    
    return name_list, cls_list, mask, clinical_feats




def kfold_class_balancing_with_clinical(img_name_array, cls_array, clinical_feats, k_fold, k):   
    train_name_list = []
    train_cls_list = []
    train_clinical_feat_list = []

    val_name_list = []
    val_cls_list = []
    val_clinical_feat_list = []
    
    unique = np.unique(cls_array)
    
    for cls in unique:
        mask = (cls_array == cls)
        
        temp_name_list = img_name_array[mask]
        clinical_feats_list = clinical_feats[:, mask]
        
        len_train = 0
        for i in range(k_fold):
            if i == k:
                val_name_temp = temp_name_list[k::k_fold]
                val_clinical_feat_temp = clinical_feats_list[:, k::k_fold]
                val_name_list += val_name_temp.tolist()
                val_clinical_feat_list.append(val_clinical_feat_temp)
                
            else:
                train_name_temp = temp_name_list[i::k_fold]
                train_clinical_feat_temp = clinical_feats_list[:, i::k_fold]
                train_name_list += train_name_temp.tolist()
                len_train += len(train_name_temp.tolist())
                train_clinical_feat_list.append(train_clinical_feat_temp)
        
        val_cls_list += [cls] * len(val_name_temp)
        train_cls_list += [cls] * len_train
    
    val_clinical_feat_list = np.concatenate(val_clinical_feat_list, 1).tolist()
    train_clinical_feat_list = np.concatenate(train_clinical_feat_list, 1).tolist()
    
    return train_name_list, train_cls_list, train_clinical_feat_list, val_name_list, val_cls_list, val_clinical_feat_list
        

def normalize_images(t1, t1c, t2, flair, adc):
    t1 = (t1 - t1.mean()) / max(t1.std(), 1e-8)
    t1c = (t1c - t1c.mean()) / max(t1c.std(), 1e-8)
    t2 = (t2 - t2.mean()) / max(t2.std(), 1e-8)
    flair = (flair - flair.mean()) / max(flair.std(), 1e-8)
    
    if adc is None:
        pass
    else:
        adc = (adc - adc.mean()) / max(adc.std(), 1e-8)
    
    return t1, t1c, t2, flair, adc


def normalize_forenorm_images(t1, t1c, t2, flair, adc):
    t1 = (t1 - t1[t1!=0].mean()) / max(t1[t1!=0].std(), 1e-8)
    t1c = (t1c - t1c[t1c!=0].mean()) / max(t1c[t1c!=0].std(), 1e-8)
    t2 = (t2 - t2[t2!=0].mean()) / max(t2[t2!=0].std(), 1e-8)
    flair = (flair - flair[flair!=0].mean()) / max(flair[flair!=0].std(), 1e-8)
    
    if adc is None:
        pass
    else:
        adc = (adc - adc[adc!=0].mean()) / max(adc[adc!=0].std(), 1e-8)
    
    return t1, t1c, t2, flair, adc


def sequence_combination(t1, t1c, t2, flair, adc, seq_comb):
    
    if seq_comb == '4seq':
        img_npy = np.stack([t1, t1c, t2, flair], 0)
    elif seq_comb == '4seq+adc':
        img_npy = np.stack([t1, t1c, t2, flair, adc], 0)
    elif seq_comb == 't2':
        img_npy = np.stack([t2], 0)
    elif seq_comb == 't2+adc':
        img_npy = np.stack([t2, adc], 0)
    
    return img_npy


import pandas as pd
import pickle
def load_radiomics(mask, args):
    if args.cmd == 'test':
        radiomics_dir = args.test_radiomics_root
    else:
        radiomics_dir = args.radiomics_root
    
    
    df_t1 = pd.read_excel(os.path.join(radiomics_dir, 'radi_feat_T1.xlsx'), engine='openpyxl')
    df_t1c = pd.read_excel(os.path.join(radiomics_dir, 'radi_feat_T1C.xlsx'), engine='openpyxl')
    df_t2 = pd.read_excel(os.path.join(radiomics_dir, 'radi_feat_T2.xlsx'), engine='openpyxl')
    df_flair = pd.read_excel(os.path.join(radiomics_dir, 'radi_feat_FLAIR.xlsx'), engine='openpyxl')
    
    T1_feat = np.array(df_t1)[3:, 1:].astype(np.float32) # 1053 x 106(feat)
    T1C_feat = np.array(df_t1c)[3:, 1:].astype(np.float32) # 1053 x 106(feat)
    T2_feat = np.array(df_t2)[3:, 1:].astype(np.float32) # 1053 x 106(feat)
    FLAIR_feat = np.array(df_flair)[3:, 1:].astype(np.float32) # 1053 x 106(feat)
    
    radiomics_patient_names = np.array(df_t1)[:, 0].tolist()[3:]
    feature_names = np.array(df_t1)[1].tolist()[1:] # 106
    
    if mask is not None:
        T1_feat = T1_feat[mask]
        T1C_feat = T1C_feat[mask]
        T2_feat = T2_feat[mask]
        FLAIR_feat = FLAIR_feat[mask]
        
        radiomics_patient_names = np.array(radiomics_patient_names)[mask].tolist()
    
    radiomics_feat = np.stack([T1_feat, T1C_feat, T2_feat, FLAIR_feat], 1) # 1053 x 4 x 106
    
    with open(os.path.join('/mai_nas/BYS/brain_metastasis/preprocessed/sev_analysis/', 'feature_statistics.pkl'), 'rb') as fr:
        feature_statistics = pickle.load(fr)
        
    cls_feature_statistics = feature_statistics[args.cls_mode]
    cls_mean = cls_feature_statistics['mean']
    cls_std = cls_feature_statistics['std']
    
    radiomics_feat = (radiomics_feat - cls_mean) / cls_std
    
    # exclude shape features, remain T1C
    nonshape_feat = np.concatenate((radiomics_feat[:, :, :18], radiomics_feat[:, :, 32:]), 2).reshape(radiomics_feat.shape[0], -1)
    shape_feat = radiomics_feat[:, 1, 18:32]
    radiomics_feat = np.concatenate((nonshape_feat, shape_feat), 1)
    
    return radiomics_feat, radiomics_patient_names, feature_names


import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist

def convert_to_distributed_network(cfg, net, gpu):
    #  Multiprocessing & Distributed Training ----------------------------------------------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()
    
    distributed = cfg.trainer.ddp.world_size > 1 or cfg.trainer.ddp.multiprocessing_distributed
    
    if distributed:
        if cfg.trainer.ddp.multiprocessing_distributed:
            rank = cfg.trainer.ddp.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.trainer.ddp.dist_backend, init_method='tcp://127.0.0.1:' + cfg.trainer.ddp.dist_url, 
                                world_size=torch.cuda.device_count() * cfg.trainer.ddp.world_size, rank=rank)
        print("[!] [Rank {}] Distributed Init Setting Done.".format(rank))
        
    batch_size_per_gpu = cfg.data.batch_size
    workers_per_gpu = cfg.data.workers
    
    if not torch.cuda.is_available():
        print("[Warnning] Using CPU, this will be slow.")
        
    elif distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise, DistributedDataParallel will use all available devices.
        if gpu is not None:
            print("[!] [Rank {}] Distributed DataParallel Setting Start".format(rank))
            
            torch.cuda.set_device(gpu)
            net = net.cuda(gpu)
            
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size ourselves based on the total number of GPUs we have
            batch_size_per_gpu = int(cfg.data.batch_size / ngpus_per_node)
            workers_per_gpu = int((cfg.data.workers + ngpus_per_node - 1) / ngpus_per_node)
            
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        net = net.cuda(gpu)
    else:
        net = torch.nn.DataParallel(net).cuda()
    #  -------------------------------------------------------------------------------------------------------------------------------------------------
    
    return net, batch_size_per_gpu, workers_per_gpu

import math

def load_custom_pretrined(cfg, net):
    checkpoint = torch.load(cfg.custom_pretrained_pth)
    
    state_dict = checkpoint['model']
    
    if cfg.pretrain_which_method == "simmim":
        # SimMIM
        new_state_dict={}
        for k, v in state_dict.items():
            if "encoder." in k:
                name = k[8:]
                new_state_dict[name] = v
        state_dict = new_state_dict
    elif cfg.pretrain_which_method == "cim_ema":
        new_state_dict = {}
        for k, v in state_dict.items():
            if "_c." in k:
                new_state_dict[k.replace('_c', '')] = v

        new_state_dict['cls_token'] = state_dict['cls_token']
        new_state_dict['pos_embed'] = state_dict['pos_embed']
        state_dict = new_state_dict

    

    in_chans = cfg.model.in_chans
    # print(state_dict.keys())
    conv_weight = state_dict['patch_embed.proj.weight']
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
        
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        # NOTE this strategy should be better than random init, but there could be other combinations of
        # the original RGB input layer weights that'd work better for specific cases.
        repeat = int(math.ceil(in_chans / I))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= (I / float(in_chans))
        
    conv_weight = conv_weight.to(conv_type)

    state_dict['patch_embed.proj.weight'] = conv_weight
    
    if 'head.weight' in state_dict.keys():
        state_dict.pop('head' + '.weight', None)
        state_dict.pop('head' + '.bias', None)

        state_dict.pop('head' + '.weight', None)
        state_dict.pop('head' + '.bias', None)
    
    net.load_state_dict(state_dict, strict=False)
    
    # for para in net.parameters():
    #     para.requires_grad = False
    
    # block_list = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3',
    #                 'blocks.4', 'blocks.5', 'blocks.6', 'blocks.7', 
    #                 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11']

    # for name, param in net.named_parameters():
    #     if name[:8] in block_list[args.freeze:]:
    #         param.requires_grad = True
    #     elif name[:4] in ['norm', 'head']:
    #         param.requires_grad = True
    #     elif name[:7] == 'fc_norm':
    #         param.requires_grad = True
    
    return net