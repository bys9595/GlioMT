import torch
from src.models.xai.multimodal_vit_xai import vit_base_patch16_224

# Model
net = vit_base_patch16_224(pretrained=True, pretrained_cfg="augreg_in21k_ft_in1k", pretrained_strict=False, in_chans=4, num_classes=1)        
net = net.cuda()

# Load checkpoint
checkpoint = torch.load('/mai_nas/BYS/glioma/exp/runs/idh/20240220/165814/train/checkpoint_best_auc.pth')
net.load_state_dict(checkpoint['net'])

