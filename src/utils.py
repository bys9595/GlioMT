import torch
from torch import nn
import torch.distributed as dist
import math
import os, logging
import sys
import time
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

class DirectroyMaker:
    sub_dir_type = ['model','log','config', 'images']
    def __init__(self, root, save_model=True, save_log=True, save_config=True):
        self.root = os.path.expanduser(root)
        self.save_model  = save_model
        self.save_log = save_log
        self.save_config = save_config
        
    def experiments_dir_maker(self,cfg):
        
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            
        now = datetime.now()
        if hasattr(cfg, 'specific_exp_dir'):
            if cfg.specific_exp_dir is None:
                time_idx = '%s_%s_%s' % (now.hour, now.minute,now.second)
            else:
                time_idx = cfg.specific_exp_dir
        else:
            time_idx = '%s_%s_%s' % (now.hour, now.minute,now.second)
        
        # --save_dir
        detail_dir = ''
        if cfg.cmd == 'test':
            detail_dir += '[TEST]'
            detail_dir += str(cfg.resume.split('/')[-3])
            detail_dir += '/' + time_idx + cfg.save_name
        elif cfg.cmd == 'val':
            detail_dir += '[VAL]'
            detail_dir += str(cfg.resume.split('/')[-3])
            detail_dir += '/' + time_idx + cfg.save_name
        elif cfg.cmd == 'ensemble':
            detail_dir += '[ENSEMBLE]'
            detail_dir += '/' + time_idx + cfg.save_name
        else:
            detail_dir += cfg.model
            detail_dir += '__' + time_idx + cfg.save_name
    
        create_dir_list = []
        if self.save_model:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[0])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)
        if self.save_log:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[1])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)    
        if self.save_config:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[2])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)
        
        # images
        add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[3])
        new_path = os.path.join(self.root,add_subdir_to_detail_dir)
        create_dir_list.append(new_path)
        
        
        
        for path in create_dir_list:
            if not os.path.exists(path):
                os.makedirs(path)
                
        return create_dir_list
    
    

def config_save(cfg,PATH):
    import json
    with open(PATH+'/'+ cfg.cmd + '_config.json', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)
    
    
def check_cfg(cfg):
    # --epoch
    try:
        assert cfg.end_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert cfg.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return cfg

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(*cfg, **kwcfg):
    if is_main_process():
        torch.save(*cfg, **kwcfg)

def set_logging_defaults(logdir, cfg, cfg_print=True):
    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, cfg.cmd + '_log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])
    # log cmdline argumetns
    if cfg_print:
        logger = logging.getLogger('main')
        if is_main_process():
            for param in sorted(vars(cfg).keys()):
                logger.info('[cfg] {0} {1}'.format(param, vars(cfg)[param]))
            # logger.info(cfg)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)




term_width = shutil.get_terminal_size().columns
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def progress_bar(epoch,current, total, cfg, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    print("Epoch: [{}]".format(epoch))
    sys.stdout.write("Epoch: [{}/{}]".format(epoch, (cfg.end_epoch - 1)))
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

import math

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    min_lr = 1e-7
    if epoch < cfg.trainer.warmup_epochs:
        lr = cfg.optimizer.lr * epoch / cfg.trainer.warmup_epochs 
    else:
        lr = min_lr + (cfg.optimizer.lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.trainer.warmup_epochs) / (cfg.trainer.end_epoch - cfg.trainer.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

 
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def get_binary_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    correct = (y_true == y_prob).sum().item()
    acc = correct / y_true.size(0)
    return correct, acc


def graph_plot(train_list, val_list, result_root_path, mode):
    save_path = os.path.join(result_root_path, 'images')
    os.makedirs(save_path, exist_ok=True)
    
    plt.plot(np.array(train_list), 'b', label='train')
    plt.plot(np.array(val_list), 'r', label='val')
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.title(mode + ' Graph')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, mode +'.png'))
    plt.close()
    plt.clf()
