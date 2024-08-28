'''
Classification Mode selection

- idh : binary, mutant vs wildtype
- 1p_19q : binary, mutant+codeletion vs mutant+no-codeletion
- subtype : 3-class, Mutant + 1p/19q codeletion  vs  Mutant + 1p/19q no codeletion  vs  Wildtype
- lgg_hgg : binary,  LGG vs HGG
- grade : 3-class,    Grade 2 vs Grade 3 vs Grade 4

'''
import torch
import torch.backends.cudnn as cudnn
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
from monai.transforms import Resized, RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandGaussianSmoothd, RandScaleIntensityd, RandAdjustContrastd, ToTensord

from src.utils import adjust_learning_rate, save_on_master, graph_plot, AverageMeter, get_learning_rate, progress_bar, normalize_images

import hydra
from omegaconf import DictConfig
from typing import Optional

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
        self.k_fold = cfg.data.k_fold  
        self.fold_num = cfg.data.fold_num  
        
        self.transform = transform      
        
        seed_everything(cfg) # Set Random Seed
    
        img_name_list, cls_list, clinical_feats = self.patient_and_class_selection()
        self.clinical_feats = clinical_feats
        
        # Train / Val
        train_name_list, train_cls_list, val_name_list, val_cls_list = self.split_class_balanced_data(img_name_list, cls_list, self.k_fold, self.cfg.data.fold_num)     
        
        if self.train_val_test == 'train':
            self.img_name_list = train_name_list
            self.cls_list = train_cls_list
            
            fold_dict = {'kfold': self.k_fold, 'fold_num':self.cfg.data.fold_num, 'train': train_name_list, 'val': val_name_list}
            with open(os.path.join(cfg.paths.output_dir, 'fold.json'),'w') as f:
                json.dump(fold_dict, f, indent=4)
        
        elif self.train_val_test == 'val':
            self.img_name_list = val_name_list
            self.cls_list = val_cls_list

        print('{} data, length of dataset: {}'.format(self.train_val_test, len(self.img_name_list)))
        
    def __len__(self):
        if self.train_val_test == 'train': 
            return len(self.img_name_list) * 10000 # for random selection 
        else:
            return len(self.img_name_list)
        
    def patient_and_class_selection(self):
        # Load .xlsx file        
        df_from_excel = pd.read_excel(self.label_root)
        
        df_from_excel.replace(['M', 'm'], 'male', inplace=True)
        df_from_excel.replace(['F', 'f'], 'female', inplace=True)
        
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
        elif self.cls_mode == 'grade':
            name_list = np.array(df_from_excel['Anony_ID'])
            cls_list = np.array(df_from_excel['WHO']) # LGG : 0,  HGG : 1
            cls_list = cls_list - 2 # 2, 3, 4 -> 0, 1, 2
            mask = None
        
        name_list = np.squeeze(name_list[mask])
        cls_list = np.squeeze(cls_list[mask])
        
        age_list = np.array(df_from_excel['Age']).astype(np.int32)
        sex_list = np.array(df_from_excel['Sex'])
        
        clinical_feats={}
        
        clinical_feats['name'] =  name_list
        clinical_feats['age'] =  np.squeeze(age_list[mask])
        clinical_feats['sex'] =  np.squeeze(sex_list[mask])
        
        return name_list, cls_list, clinical_feats
        
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
            idx = idx % len(self.img_name_list)

        name = self.img_name_list[idx]
        
        # Clinical Feats
        clinical_idx = np.where(self.clinical_feats['name']==name)[0][0]
        age = self.clinical_feats['age'][clinical_idx]
        sex = self.clinical_feats['sex'][clinical_idx]
        clinical_feats = [str(age), sex]
                
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1.nii.gz')))        
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T1C.nii.gz')))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_T2.nii.gz')))
        flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name +'_FLAIR.nii.gz')))
        
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, name, name + '_seg.nii.gz')))
        seg[seg !=0] = 1 # integrate all the classes
        
        t1, t1c, t2, flair = normalize_images(t1, t1c, t2, flair)
        img_npy = np.stack([t1, t1c, t2, flair], 0)

        # Setting input axial slice configuration according to slice percentile (%) -------------------------------------------------------------
        z_seg = seg.sum(-1).sum(-1)
        
        if len(z_seg) * self.slice_percentile < 1:
            roi_idx_list = [np.argmax(z_seg)]
        else:
            glioma_vol_lower_bound = np.percentile(z_seg[z_seg.nonzero()[0]], 100-self.slice_percentile, axis=0)
            roi_mask = z_seg > glioma_vol_lower_bound
            roi_idx_list = np.where(roi_mask==True)[0].tolist()
        # ---------------------------------------------------------------------------------------------------------------------------------------
        label = np.array(self.cls_list[idx])
        label = torch.from_numpy(label).long()
        
        if self.train_val_test in ['train']:
            roi_random_idx = random.choice(roi_idx_list)
            img_npy_2d = img_npy[:, roi_random_idx] #CHW
            data_dict = {'image' : img_npy_2d, 'label' : label, 'name' : name, 'clinical_feats': clinical_feats}
        else:
            img_npy_2d = img_npy[:, roi_idx_list] # C x S x H x W
            S, A, H, W = img_npy_2d.shape
            img_npy_2d = img_npy_2d.reshape(S*A, H, W)
            
            data_dict = {'image' : img_npy_2d, 'label' : label, 'name' : name, 'clinical_feats': clinical_feats}
        
        if self.transform is not None:
            data_dict = self.transform(data_dict)
                
        return data_dict



def dataloader(cfg):    
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


    transform.set_random_state(seed=cfg.random_seed)
    
    trainset = Brain_Dataset('train', cfg, transform=transform)
    valset = Brain_Dataset('val', cfg, transform=val_transform)    
    
    
    train_loader = torch.utils.data.DataLoader(trainset, pin_memory=cfg.data.pin_memory, batch_size=cfg.data.batch_size,
                                                shuffle=True, num_workers=cfg.data.workers)
    # batch size of valid_loader should be 1
    valid_loader = torch.utils.data.DataLoader(valset, pin_memory=cfg.data.pin_memory, batch_size=1,
                                                shuffle=True, num_workers=cfg.data.workers)
    
    
    return train_loader, valid_loader


def seed_everything(cfg: DictConfig)->None:
    os.environ["PL_GLOBAL_SEED"] = str(cfg.random_seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(cfg.data.workers)}"
    
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    
    cudnn.benchmark = (not cfg.trainer.deterministic)
    cudnn.deterministic = cfg.trainer.deterministic

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print("[!] Multimodal Transformer for Glioma Subtyping and Grading According to WHO2021")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Seed Setting
    seed_everything(cfg)

    net = hydra.utils.instantiate(cfg.model)
    net = net.cuda()
    
    train_loader, valid_loader = dataloader(cfg)
    print("[!] Data Loading Done")
    
    criterion = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, net.parameters())

    writer = SummaryWriter(f"{cfg.paths.output_dir}")
    
    # Load status & Resume Learning
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        
        net.load_state_dict(checkpoint['net'])
        
        cfg.start_epoch = checkpoint['epoch'] + 1 
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("[!] Model loaded")

        del checkpoint
    
    train_worker(criterion, optimizer, net, train_loader, valid_loader, writer, cfg)
    
    print("RETURN: " + cfg.paths.output_dir) # Do not remove this print line, this is for capturing the path of model


def train_worker(criterion, optimizer, net, train_loader, valid_loader, writer, cfg):
    # Initialize metrics
    best_auc = 0
    best_loss = 10000
    early_stop_count = 0
    
    train_metric_dict={'loss':[], 'acc':[], 'precision': [], 'recall': [], 'f1_score':[], 'auc': []}
    val_metric_dict={'loss':[], 'acc':[], 'precision': [], 'recall': [], 'f1_score':[], 'auc': []}
    
    # Scaler for amp training
    scaler = torch.cuda.amp.GradScaler() if cfg.trainer.amp else None    
    
    # Start Training
    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.end_epoch):
        start_time = time.time()
        
        adjust_learning_rate(optimizer, epoch+1, cfg)
        
        #  Train        
        train_loss, train_auc = train(criterion, optimizer, net, epoch,  train_loader, writer, scaler, cfg)
        
        train_metric_dict['loss'].append(train_loss)
        train_metric_dict['auc'].append(train_auc)

        #  Validation
        val_loss, val_auc = val(criterion, net, epoch, valid_loader, writer, cfg)
            
        val_metric_dict['loss'].append(val_loss)
        val_metric_dict['auc'].append(val_auc)
        
        #  Save_dict for saving model
        net_state_dict = net.state_dict()
        
        save_dict = {
                    'net': net_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_auc' : val_auc,
                    }
        
        save_on_master(save_dict, os.path.join(cfg.paths.output_dir, 'checkpoint_latest.pth'))
        
        early_stop_count += 1
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_on_master(save_dict,os.path.join(cfg.paths.output_dir, 'checkpoint_best_loss.pth'))
            print("[!] Save checkpoint with best loss.")
                
        if val_auc > best_auc:
            best_auc = val_auc
            early_stop_count = 0
            save_on_master(save_dict,os.path.join(cfg.paths.output_dir, 'checkpoint_best_auc.pth'))
            print("[!] Save checkpoint with best auc.")
                
        print("Epoch time : {}".format(time.time()-start_time))
        
        if early_stop_count > cfg.trainer.early_stop_epooch:
            logger = logging.getLogger('train')
            logger.info('Early Stopping')
            break
    
    # Evaluation Metric Graph
    for key in train_metric_dict.keys():
        graph_plot(train_metric_dict[key], val_metric_dict[key], cfg.paths.output_dir, key)
    

# Train
def train(criterion, optimizer, net, epoch, train_loader, writer, scaler, cfg):
    train_metrics = hydra.utils.instantiate(cfg.metric)
    train_losses = AverageMeter()
    
    net.train()
    current_LR = get_learning_rate(optimizer)[0]

    iter_num_per_epoch = 0
    for batch_idx, data_dict in enumerate(train_loader):
        inputs = data_dict['image'].cuda(non_blocking=True).float()
        targets = data_dict['label'].cuda(non_blocking=True)
        clinical_feats = data_dict['clinical_feats']
        names = data_dict['name']
                        
        if cfg.model.num_classes ==1 :
            targets = targets.float()
        
        if cfg.trainer.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = net(inputs, clinical_feats)
                loss = criterion(outputs.squeeze(1), targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        else:
            outputs = net(inputs, clinical_feats)
            
            loss = criterion(outputs.squeeze(1), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
        
        train_metrics.update(outputs, targets)
        train_losses.update(loss.item(), inputs.size(0))
                
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

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [lr {:.1e}] AUC {}'.format(
        epoch+1, train_losses.avg, current_LR, 
        str(np.round(auc.item(), decimals=3)),
    ))

    return train_losses.avg, auc.item()


def slice_ensemble(inputs, clinical_feats, net, criterion, targets, cfg):
    # inputs has the shape (1, SA, H, W)
    _, SA, H, W = inputs.shape
    S, A = cfg.model.in_chans, SA // cfg.model.in_chans
    inputs = inputs[0].view(S, A, H, W).permute(1, 0, 2, 3).contiguous()
    
    batch_size = cfg.data.batch_size
    final_out = None
    total_loss = 0.0
    num_batches = (A + batch_size - 1) // batch_size  # Calculate number of batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, A)

        batch_input = inputs[start_idx:end_idx]  # Shape: (batch_size, S, H, W)
        
        # Forward pass through the network
        clinical_feats_batch = [cli * (end_idx - start_idx) for cli in clinical_feats]
        outputs = net(batch_input, clinical_feats_batch) # Shape: (batch_size, 1, H, W)
        
        # Compute the loss for the batch
        batch_loss = criterion(outputs.squeeze(1), targets.repeat(end_idx - start_idx))
        total_loss += batch_loss.mean() * (end_idx - start_idx)
        
        # Accumulate the outputs
        if final_out is None:
            final_out = outputs.detach().cpu().mean(0).unsqueeze(0) * (end_idx - start_idx)
        else:
            final_out += outputs.detach().cpu().mean(0).unsqueeze(0) * (end_idx - start_idx)
    
    # Average the accumulated outputs
    final_out /= A
    total_loss /= A
    
    return final_out, total_loss


def val(criterion, net, epoch, val_loader, writer, cfg): 
    val_metrics = hydra.utils.instantiate(cfg.metric)
    val_losses = AverageMeter()
  
    net.eval()
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(val_loader):
            inputs = data_dict['image'].cuda(non_blocking=True).float()
            targets = data_dict['label'].cuda(non_blocking=True)
            clinical_feats = data_dict['clinical_feats']
            
            if cfg.model.num_classes ==1 :
                targets = targets.float()
            
            if cfg.trainer.amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs, loss = slice_ensemble(inputs, clinical_feats, net, criterion, targets, cfg)

            else:
                outputs = net(inputs, clinical_feats)
                
                loss = criterion(outputs.squeeze(1), targets)
                
            outputs = outputs.float()
            val_metrics.update(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))
            
    if (batch_idx+1) % cfg.trainer.print_freq == 0:
        progress_bar(epoch,batch_idx, len(val_loader),cfg, ' loss: {:.3f} |'.format(
        val_losses.avg))
    
    auc = val_metrics.compute()
    auc = auc.mean()

    writer.add_scalar('Val/Loss', val_losses.avg, epoch+1)
    writer.add_scalar('Val/auc', auc.mean(), epoch+1)
    
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
    main()