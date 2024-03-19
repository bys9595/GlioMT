'''
Classification Mode selection

- idh : binary, mutant vs wildtype
- 1p_19q : binary, mutant+codeletion vs mutant+no-codeletion
- subtype : 3-class, Mutant + 1p/19q codeletion  vs  Mutant + 1p/19q no codeletion  vs  Wildtype
- lgg_hgg : binary,  LGG vs HGG
- grade : 3-class,    Grade 2 vs Grade 3 vs Grade 4

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter

import os
import time
import json
import random
import warnings
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai import transforms
from monai.transforms import Resized, RandAffined, RandFlipd, RandGaussianNoised, RandGaussianSmoothd, RandGaussianSmoothd, RandScaleIntensityd, RandAdjustContrastd, ToTensord

from src.utils import adjust_learning_rate, save_on_master, is_main_process, graph_plot, AverageMeter, get_learning_rate, progress_bar
from src.total_utils import normalize_images, sequence_combination, load_custom_pretrined, convert_to_distributed_network

import hydra
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

# A logger for this file
logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore')



class Brain_Dataset(Dataset):
    def __init__(self, train_val_test, cfg, transform=None):
        self.train_val_test = train_val_test
        self.cfg = cfg
        
        self.data_root = cfg.paths.data_root
        self.label_root = cfg.paths.label_root
        self.slice_percentile = cfg.data.slice_percentile

        self.cls_mode = cfg.cls_mode
        self.seq_comb = cfg.data.seq_comb
        self.k_fold = cfg.data.k_fold  
        self.fold_num = cfg.data.fold_num  
        
        self.transform = transform      
        
        seed_everything(cfg)
        
        img_name_list, cls_list = self.patient_and_class_selection()
        
        # # Train+Val / Test
        # train_name_list, train_cls_list, test_name_list, test_cls_list = self.split_class_balanced_data(img_name_list, cls_list, self.k_fold, 0)
        # # Train / Val
        # train_name_list, train_cls_list, val_name_list, val_cls_list = self.split_class_balanced_data(np.array(train_name_list), np.array(train_cls_list), self.k_fold, self.fold_num)
        
        samples_per_cls = []
        for i in np.unique(cls_list):
            samples_per_cls.append((cls_list == i).sum())
        self.samples_per_cls = torch.tensor(samples_per_cls)
        
        # Train / Val
        train_name_list, train_cls_list, val_name_list, val_cls_list = self.split_class_balanced_data(img_name_list, cls_list, self.k_fold, self.cfg.data.fold_num)     
        
        if self.train_val_test == 'train':
            self.img_name_list = train_name_list
            self.cls_list = train_cls_list
            patient_dict = {'train': train_name_list, 'val': val_name_list}
            # patient_dict = {'train': train_name_list, 'val': val_name_list, 'test': test_name_list}
            with open(os.path.join(cfg.paths.output_dir, 'patients.json'),'w') as f:
                json.dump(patient_dict, f, indent=4)
        elif self.train_val_test == 'val':
            self.img_name_list = val_name_list
            self.cls_list = val_cls_list
        # elif self.train_val_test == 'test':
        #     self.img_name_list = test_name_list
        #     self.cls_list = test_cls_list

        print('{} data, length of dataset: {}'.format(self.train_val_test, len(self.img_name_list)))

    def __len__(self):
        if self.train_val_test == 'train': 
            return len(self.img_name_list) * 10000
        else:
            return len(self.img_name_list)
        
    def return_samples_per_cls(self):
        return self.samples_per_cls

    def cls_bal_selection(self):
        rand_prob = random.uniform(0, 1)
        cls_array = np.array(self.cls_list)
        cls_unique = np.unique(cls_array)
        
        prob_step = 1 / len(cls_unique)
        
        for i in range(len(cls_unique)):
            if rand_prob < (i+1) * prob_step:
                idx_list = np.where(cls_array == i)[0].tolist()
                break
    
        idx = idx_list[random.randint(0, len(idx_list)-1)]
        return idx

    def sequence_combination(self, t1, t1c, t2, flair, adc):
        if self.seq_comb == '4seq':
            img_npy = np.stack([t1, t1c, t2, flair], 0)
        elif self.seq_comb == '4seq+adc':
            img_npy = np.stack([t1, t1c, t2, flair, adc], 0)
        elif self.seq_comb == 't2':
            img_npy = np.stack([t2], 0)
        elif self.seq_comb == 't2+adc':
            img_npy = np.stack([t2, adc], 0)
        
        return img_npy

    def patient_and_class_selection(self):
        
        df_from_excel = pd.read_excel(self.label_root)
        df_from_excel.replace(['M','male'], -1, inplace=True)
        df_from_excel.replace(['F','female'], 1, inplace=True)
        
        # ---------------------------------------------------------------------------------------
        if self.cls_mode == 'idh':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['IDH_mutation'])
            mask = None
        elif self.cls_mode == '1p_19q':
            mutation = np.array(df_from_excel['IDH_mutation']).astype(np.bool_)
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['1p/19q codeletion'])
            mask = mutation
        elif self.cls_mode == 'subtype':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['Mole_Group_no'])
            cls_list = cls_list - 1 # 1, 2, 3 -> 0, 1, 2
            mask = None
        elif self.cls_mode == 'lgg_hgg':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['WHO_23_4']) # LGG : 0,  HGG : 1
            mask = None
        elif self.cls_mode == 'grade':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['WHO']) # LGG : 0,  HGG : 1
            cls_list = cls_list - 2 # 2, 3, 4 -> 0, 1, 2
            mask = None
        
        name_list = np.squeeze(name_list[mask])
        cls_list = np.squeeze(cls_list[mask])
        
        return name_list, cls_list

        
    def split_class_balanced_data(self, img_name_array, cls_array, k_fold, k):   
        '''
        k_fold : total num of fold
        k : fold num
        '''
        train_name_list = []
        train_cls_list = []

        test_name_list = []
        test_cls_list = []
        
        unique = np.unique(cls_array)
        
        for cls in unique:
            mask = (cls_array == cls)
            temp_name_list = img_name_array[mask]
            
            len_train = 0
            for i in range(k_fold):
                if i == k:
                    test_name_temp = temp_name_list[k::k_fold]
                    test_name_list += test_name_temp.tolist()
                else:
                    train_name_temp = temp_name_list[i::k_fold]
                    train_name_list += train_name_temp.tolist()
                    len_train += len(train_name_temp.tolist())
            
            test_cls_list += [cls] * len(test_name_temp)
            train_cls_list += [cls] * len_train
        
        return train_name_list, train_cls_list, test_name_list, test_cls_list  
        
    def __getitem__(self, idx):
        if self.train_val_test == 'train':
            if self.cfg.data.class_balance_load:
                idx = self.cls_bal_selection()
            else:
                idx = idx % len(self.img_name_list)
        else:
            pass
        
        name = self.img_name_list[idx]
                
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1.nii.gz')))        
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1C.nii.gz')))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T2.nii.gz')))
        flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_FLAIR.nii.gz')))
        
        if 'adc' in self.seq_comb:
            adc = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_ADC.nii.gz')))
        else:
            adc = None
        
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name + '_seg.nii.gz')))
        seg[seg !=0] = 1
        
        t1, t1c, t2, flair, adc = normalize_images(t1, t1c, t2, flair, adc)
        img_npy = sequence_combination(t1, t1c, t2, flair, adc, self.seq_comb)
        
        z_seg = seg.sum(-1).sum(-1)

        # 상위 %
        if self.slice_percentile <= 1:
            roi_idx_list = [np.argmax(z_seg)]
        else:
            glioma_vol_lower_bound = np.percentile(z_seg[z_seg.nonzero()[0]], 100-self.slice_percentile, axis=0)
            roi_mask = z_seg > glioma_vol_lower_bound
            roi_idx_list = np.where(roi_mask==True)[0].tolist()
        
        # ---------------------------------------------------------------------------------------------------------------------------------------
        label = np.array(self.cls_list[idx])
        label = torch.from_numpy(label).long()
        
        if self.train_val_test in ['train', 'val']:
            roi_random_idx = random.choice(roi_idx_list)
            img_npy_2d = img_npy[:, roi_random_idx] #CHW
            data_dict = {'image' : img_npy_2d, 'label' : label, 'name' : name}
        else:
            img_npy_2d = img_npy[:, roi_idx_list] # C x S x H x W
            S, A, H, W = img_npy_2d.shape
            img_npy_2d = img_npy_2d.reshape(S*A, H, W)
            
            seg_2d = seg[roi_idx_list]
            data_dict = {'image' : img_npy_2d, 'label' : label, 'name' : name, 'seg' : seg_2d}
        
        if self.transform is not None:
            data_dict = self.transform(data_dict)
                
        return data_dict


def dataloader(cfg, batch_size_per_gpu, workers_per_gpu):    
    if isinstance(cfg.data.training_size, int):
        size = (cfg.data.training_size, cfg.data.training_size)
    elif isinstance(cfg.data.training_size, tuple):
        size = cfg.data.training_size
        
    if cfg.data.aug_type == 'light':
        transform =transforms.Compose([   
                                        Resized(keys=["image"], spatial_size=size, mode="bicubic"),
                                        RandGaussianNoised(keys=['image'], mean=0, std=0.1, prob=0.2),
                                        RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.2),
                                        RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.2),
                                        RandAdjustContrastd(keys=['image'], gamma=(0.75, 1.25), prob=0.2),
                                        ToTensord(keys=["image"])
                                        ])
    elif cfg.data.aug_type == 'heavy':
        transform =transforms.Compose([   
                                        Resized(keys=["image"], spatial_size=size, mode="bicubic"),
                                        RandAffined(keys=["image"], mode="bilinear", prob=0.3, 
                                                    rotate_range=(np.pi/12, np.pi/12), scale_range=(0.15, 0.15), shear_range=(0.15, 0.15), padding_mode="border"),
                                        RandGaussianNoised(keys=['image'], mean=0, std=0.2, prob=0.3),
                                        RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.3),
                                        RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.3),
                                        RandAdjustContrastd(keys=['image'], gamma=(0.75, 1.25), prob=0.3),
                                        ToTensord(keys=["image"])
                                        ])
    
    val_transform = transforms.Compose([
                                        Resized(keys=["image"], spatial_size=size, mode="bicubic"),
                                        ToTensord(keys=["image"]),
                                    ])
    # test_transform = transforms.Compose([
    #                                     Resized(keys=["image", "seg"], spatial_size=size, mode=["bicubic", "nearest"]),
    #                                     ToTensord(keys=["image", "seg"]),
    #                                     ])
    

    transform.set_random_state(seed=cfg.random_seed)
    
    trainset = Brain_Dataset('train', cfg, transform=transform)
    valset = Brain_Dataset('val', cfg, transform=val_transform)
    # testset = Brain_Dataset('test', cfg, transform=test_transform)
    
    if cfg.loss['_target_'].endswith('ClassBalancedLoss'):
        samples_per_cls = trainset.return_samples_per_cls()
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.loss['samples_per_cls'] = samples_per_cls.tolist()
            cfg.loss['no_of_classes'] = cfg.model.num_classes if cfg.model.num_classes!=1 else cfg.model.num_classes+1
    
    if cfg.trainer.ddp.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        

    print("[!] Data Loading Done")
    
    

    train_loader = torch.utils.data.DataLoader(trainset, pin_memory=cfg.data.pin_memory, batch_size=batch_size_per_gpu,
                                                sampler=train_sampler, shuffle=(train_sampler is None), num_workers=workers_per_gpu
                                                )
    
    valid_loader = torch.utils.data.DataLoader(valset, pin_memory=cfg.data.pin_memory, batch_size=batch_size_per_gpu,
                                                sampler=val_sampler, shuffle=True, num_workers=workers_per_gpu
                                                )
    
    # test_loader = torch.utils.data.DataLoader(testset, pin_memory=cfg.data.pin_memory, batch_size=1,
    #                                             sampler=None, shuffle=False, num_workers=workers_per_gpu
    #                                             )

    
    # return train_loader, valid_loader, test_loader, train_sampler
    return train_loader, valid_loader, train_sampler


def seed_everything(cfg: DictConfig)->None:
    os.environ["PL_GLOBAL_SEED"] = str(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(cfg.data.workers)}"
    
    cudnn.benchmark = (not cfg.trainer.deterministic)
    cudnn.deterministic = cfg.trainer.deterministic

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    
    print("[!] Glioma Subtyping and Grading Clasification According to WHO2021")
    print("[!] Created by MAI-LAB, Yunsu Byeon")
    assert cfg.paths.time_dir != None, 'You need to set cfg.paths.time_dir'
    
    # Seed Setting
    seed_everything(cfg)

    # # gpu setting
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    # print("[Info] Finding Empty GPU {}".format(cfg.gpus))
    
    # main
    if cfg.trainer.ddp.multiprocessing_distributed:        
        print("[!] Multi-GPU Training.")
        print('[Info] Save dir:', cfg.paths.output_dir)
        mp.spawn(train_worker, nprocs=torch.cuda.device_count(), cfg=(cfg))
        print('[Info] Save dir:', cfg.paths.output_dir)
    else:
        print("[!] Single-GPU Training")
        print('[Info] Save dir:', cfg.paths.output_dir)
        train_worker(0, cfg)
        print('[Info] Save Model dir:', cfg.paths.output_dir)

    print("RETURN: " + cfg.paths.output_dir) # Do not remove this print line, this is for capturing the path of model

def train_worker(gpu, cfg):   
    
    distributed = cfg.trainer.ddp.world_size > 1 or cfg.trainer.ddp.multiprocessing_distributed
    
    # from timm.models import create_model
    # Define network
    net = hydra.utils.instantiate(cfg.model)
    if cfg.custom_pretrained_pth != None:
        net = load_custom_pretrined(cfg, net)
        logger = logging.getLogger(cfg.custom_pretrained_pth)
        logger.info('Load pretrained model')
    net, batch_size_per_gpu, workers_per_gpu = convert_to_distributed_network(cfg, net, gpu)
    
    # Load Dataset
    train_loader, valid_loader, train_sampler = dataloader(cfg, batch_size_per_gpu, workers_per_gpu)
    
    criterion = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, net.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)    

    writer = SummaryWriter(f"{cfg.paths.output_dir}")
    
    #  load status & Resume Learning
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        
        if distributed:
            net.module.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint['net'])
        
        cfg.start_epoch = checkpoint['epoch'] + 1 
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("[!] Model loaded")

        del checkpoint
    
    train_epoch(criterion, optimizer, scheduler, net, gpu, train_loader, valid_loader, train_sampler, writer, cfg)


def train_epoch(criterion, optimizer, scheduler, net, gpu, train_loader, valid_loader, train_sampler, writer, cfg):
    distributed = cfg.trainer.ddp.world_size > 1 or cfg.trainer.ddp.multiprocessing_distributed
    best_auc = 0
    best_loss = 10000
     
    train_metric_dict={'loss':[], 'acc':[], 'precision': [], 'recall': [], 'f1_score':[], 'auc': []}
    val_metric_dict={'loss':[], 'acc':[], 'precision': [], 'recall': [], 'f1_score':[], 'auc': []}
    
    scaler = torch.cuda.amp.GradScaler() if cfg.trainer.amp else None
    early_stop_count = 0
    
    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.end_epoch):
        start_time = time.time()
        
        lr = adjust_learning_rate(optimizer, epoch+1, cfg)
        
        if distributed:
            train_sampler.set_epoch(epoch)
        #  Train        
        train_loss, train_auc = train(criterion, optimizer, net, epoch,  train_loader, writer, scaler, cfg, distributed)
        
        train_metric_dict['loss'].append(train_loss)
        train_metric_dict['auc'].append(train_auc)

        if distributed:
            dist.barrier()

        #  Validation
        val_loss, val_auc = val(criterion, net, epoch, valid_loader, writer, cfg, distributed)
            
        val_metric_dict['loss'].append(val_loss)
        val_metric_dict['auc'].append(val_auc)
        
        #  Save_dict for saving model
        if distributed:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        
        save_dict = {
                    'net': net_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_auc' : val_auc,
                    }
        
        save_on_master(save_dict, os.path.join(cfg.paths.output_dir, 'checkpoint_latest.pth'))
        
        early_stop_count += 1
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_on_master(save_dict,os.path.join(cfg.paths.output_dir, 'checkpoint_best_loss.pth'))
            if is_main_process():
                print("[!] Save checkpoint with best loss.")
                
        if val_auc > best_auc:
            best_auc = val_auc
            early_stop_count = 0
            save_on_master(save_dict,os.path.join(cfg.paths.output_dir, 'checkpoint_best_auc.pth'))
            if is_main_process():
                print("[!] Save checkpoint with best auc.")
        
        # scheduler.step()
        
        if is_main_process():
            print("Epoch time : {}".format(time.time()-start_time))
        
        
        if early_stop_count > cfg.trainer.early_stop_epooch:
            logger = logging.getLogger('train')
            logger.info('Early Stopping')
            break
            

    # Evaluation Metric Graph
    for key in train_metric_dict.keys():
        graph_plot(train_metric_dict[key], val_metric_dict[key], cfg.paths.output_dir, key)
    
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
        print("[!] [Rank {}] Distroy Distributed process".format(gpu))
    

# Train
def train(criterion, optimizer, net, epoch, train_loader, writer, scaler, cfg, distributed):
    
    train_metrics = hydra.utils.instantiate(cfg.metric)
    train_losses = AverageMeter()
    
    net.train()
    current_LR = get_learning_rate(optimizer)[0]

    iter_num_per_epoch = 0
    for batch_idx, data_dict in enumerate(train_loader):
        inputs = data_dict['image'].cuda(non_blocking=True).float()
        targets = data_dict['label'].cuda(non_blocking=True)
        names = data_dict['name']
        # print(names[0])
                        
        if cfg.model.num_classes ==1 :
            targets = targets.float()
        
        if cfg.trainer.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(1), targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        else:
            outputs = net(inputs)
            
            loss = criterion(outputs.squeeze(1), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
        
        train_metrics.update(outputs, targets)
        train_losses.update(loss.item(), inputs.size(0))
                
        if is_main_process():
            if (batch_idx+1) % cfg.trainer.print_freq == 0:
                progress_bar(epoch, batch_idx, cfg.trainer.iter_per_epoch, cfg.trainer, 'lr: {:.1e} | loss: {:.3f} |'.format(
                current_LR, train_losses.avg))
        
        iter_num_per_epoch += 1
        if iter_num_per_epoch >= cfg.trainer.iter_per_epoch:
            break
    
    auc = train_metrics.compute()
    auc = auc.mean()

    writer.add_scalar('Train/Loss', train_losses.avg, epoch+1)
    writer.add_scalar('Train/auc', auc.mean(), epoch+1)
        
    if distributed:
        dist.barrier()
        
    if is_main_process():
        logger = logging.getLogger('train')
        logger.info('[Epoch {}] [Loss {:.3f}] [lr {:.1e}] AUC {}'.format(
            epoch+1, train_losses.avg, current_LR, 
            str(np.round(auc.item(), decimals=3)),
        ))
    
    return train_losses.avg, auc.item()

def val(criterion, net, epoch, val_loader, writer, cfg, distributed): 
    val_metrics = hydra.utils.instantiate(cfg.metric)
    val_losses = AverageMeter()
  
    net.eval()
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(val_loader):
            inputs = data_dict['image'].cuda(non_blocking=True).float()
            targets = data_dict['label'].cuda(non_blocking=True)

            if cfg.model.num_classes ==1 :
                targets = targets.float()
            
            if cfg.trainer.amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = net(inputs)
                        
                    loss = criterion(outputs.squeeze(1), targets)
            else:
                outputs = net(inputs)
                
                loss = criterion(outputs.squeeze(1), targets)
            
            val_metrics.update(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))
            
    if is_main_process():
        if (batch_idx+1) % cfg.trainer.print_freq == 0:
            progress_bar(epoch,batch_idx, len(val_loader),cfg, ' loss: {:.3f} |'.format(
            val_losses.avg))
    
    auc = val_metrics.compute()
    auc = auc.mean()

    writer.add_scalar('Val/Loss', val_losses.avg, epoch+1)
    writer.add_scalar('Val/auc', auc.mean(), epoch+1)
        

    if distributed:
        dist.barrier()
        
    if is_main_process():
        logger = logging.getLogger('val')
        if cfg.model.num_classes <= 2:
            logger.info('[Epoch {}] [Loss {:.3f}] AUC {}'.format(
                epoch+1, val_losses.avg, 
                str(np.round(auc.item(), decimals=3)),
            ))
        else:
            logger.info('[Epoch {}] [Loss {:.3f}] AUC {} '.format(
                epoch+1, val_losses.avg, 
                str(np.round(auc.item(), decimals=3)),
            ))
    
    return val_losses.avg, auc.item()

        

if __name__ == '__main__': 
    import matplotlib  
    matplotlib.use('Agg')  

    main()