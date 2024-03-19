from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformer

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from timm.models._registry import generate_default_cfgs
from timm.models.swin_transformer import checkpoint_filter_fn


from functools import partial
from timm.models._builder import build_model_with_cfg
from timm.layers import trunc_normal_

from transformers import AutoTokenizer, BertModel, AutoModel



class ClinicalSwinT(SwinTransformer):
    """ Clinical SwinT
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1, window_size=7,
                 patch_size=4, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 decoder_depth=2):
        super().__init__(img_size=img_size, in_chans=in_chans, window_size=window_size, 
                         num_classes=num_classes, patch_size=patch_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads)
        
        # self.in_chans = in_chans
        # self.patch_size = patch_size
        # self.depth = depth
        # self.num_heads = num_heads
        
        pretrained_name = "bert-base-uncased"
        # assert pretrained_name is not None, "Only ViT-Base available in Clinical Model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.embed_model = BertModel.from_pretrained(pretrained_name)
        
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 3, 768) * .02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
        self.projector = nn.Linear(self.num_features, 768) if self.num_features != 768 else nn.Identity()
        
        self.decoder_blocks = nn.Sequential(*[
            Block(dim=768,
                num_heads=1,
                qkv_bias=True
                )
            for i in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(768, eps=1e-6)
        
        self.fc = nn.Linear(768, num_classes, bias=True)

    def forward(self, x, clinical_feats):
        age_embed, sex_embed = self.forward_clinical_embed(clinical_feats)
        
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=True)
        x = self.projector(x)
        
        multi_modal_feats = torch.stack([x, age_embed, sex_embed], 1)
        multi_modal_feats = multi_modal_feats + self.decoder_pos_embed
        multi_modal_feats = self.decoder_blocks(multi_modal_feats)
        multi_modal_feats = self.decoder_norm(multi_modal_feats)
    
        cls_token = multi_modal_feats[:, 0]
        out = self.fc(cls_token)

        return out

    
    def forward_clinical_embed(self, clinical_feats):
        age = clinical_feats[0]
        sex = clinical_feats[1]
        
        age = ['old' if int(i)>=55 else 'young' for i in age]
        
        age_token = self.tokenizer(age, return_tensors="pt")
        sex_token = self.tokenizer(sex, return_tensors="pt")

        age_token = {k: v.cuda() for k, v in age_token.items()}
        sex_token = {k: v.cuda() for k, v in sex_token.items()}
        
        self.embed_model.eval()
        with torch.no_grad():                      
            age_embed = self.embed_model(**age_token).last_hidden_state[:, 0]
            sex_embed = self.embed_model(**sex_token).last_hidden_state[:, 0]

        return age_embed, sex_embed


def _create_swin_transformer(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        ClinicalSwinT, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth', ),
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',),
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',),
    'swin_large_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    'swin_tiny_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',),
    'swin_small_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',),
    'swin_base_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',),
    'swin_base_patch4_window12_384.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    # tiny 22k pretrain is worse than 1k, so moved after (untagged priority is based on order)
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',),

    'swin_tiny_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_small_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),
    'swin_large_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_large_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),

    'swin_s3_tiny_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_t-1d53f6a8.pth'),
    'swin_s3_small_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_s-3bb4c69d.pth'),
    'swin_s3_base_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_b-a1e95db4.pth'),
})

        
def create_swint_model(model_name, pretrained=False, **kwargs):    
    return globals()[model_name](pretrained=pretrained, **kwargs)


def swin_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_small_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-B @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer(
        'swin_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_base_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-B @ 384x384
    """
    model_args = dict(patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer(
        'swin_base_patch4_window12_384', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_large_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-L @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer(
        'swin_large_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_large_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-L @ 384x384
    """
    model_args = dict(patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer(
        'swin_large_patch4_window12_384', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_s3_tiny_224(pretrained=False, **kwargs):
    """ Swin-S3-T @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(7, 7, 14, 7), embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_tiny_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_s3_small_224(pretrained=False, **kwargs):
    """ Swin-S3-S @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(14, 14, 14, 7), embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_small_224', pretrained=pretrained, **dict(model_args, **kwargs))


def swin_s3_base_224(pretrained=False, **kwargs):
    """ Swin-S3-B @ 224x224, https://arxiv.org/abs/2111.14725
    """
    model_args = dict(
        patch_size=4, window_size=(7, 7, 14, 7), embed_dim=96, depths=(2, 2, 30, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer('swin_s3_base_224', pretrained=pretrained, **dict(model_args, **kwargs))

