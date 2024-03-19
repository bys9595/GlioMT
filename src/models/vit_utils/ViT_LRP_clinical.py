""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from einops import rearrange
from .layers_ours import *

from .helpers import load_pretrained
from .weight_init import trunc_normal_
from .layer_helpers import to_2tuple

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.models._registry import generate_default_cfgs
from timm.models.vision_transformer import checkpoint_filter_fn
from functools import partial
from timm.models._builder import build_model_with_cfg


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


default_cfgs = generate_default_cfgs({

    # re-finetuned augreg 21k FT on in1k weights
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'vit_base_patch16_384.augreg2_in21k_ft_in1k': _cfg(),
    'vit_base_patch8_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),

    # How to train your ViT (augreg) weights, pretrained on 21k FT on in1k
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    'vit_base_patch16_224.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        hf_hub_id='timm/'),
    'vit_base_patch16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),

    # How to train your ViT (augreg) weights trained on in1k only
    'vit_small_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224.untrained': _cfg(url=''),
    'vit_huge_patch14_224.untrained': _cfg(url=''),
    'vit_giant_patch14_224.untrained': _cfg(url=''),
    'vit_gigantic_patch14_224.untrained': _cfg(url=''),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_large_patch32_224.orig_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        hf_hub_id='timm/',
        num_classes=21843),
    'vit_huge_patch14_224.orig_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # How to train your ViT (augreg) weights, pretrained on in21k
    'vit_tiny_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_large_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz', custom_load=True,
        hf_hub_id='timm/'),
    'vit_base_patch16_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz', custom_load=True,
        hf_hub_id='timm/'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    
    # DINOv2 pretrained - https://arxiv.org/abs/2304.07193 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_in21k_miil-887286df.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221),
    'vit_base_patch16_224_miil.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_1k_miil_84_4-2deb18e3.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear'),

    # Custom timm variants
    'vit_base_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_base_patch16_rpn_224-sw-3b07e89d.pth',
        hf_hub_id='timm/'),
    'vit_medium_patch16_gap_240.sw_in12k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=11821),
    'vit_medium_patch16_gap_256.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95),
    'vit_medium_patch16_gap_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=0.95, crop_mode='squash'),
    'vit_base_patch16_gap_224': _cfg(),

    # CLIP pretrained image tower and related fine-tuned weights
    'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 384, 384)),
    'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 448, 448)),
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in12k_in1k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_base_patch16_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_base_patch16_clip_384.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_384.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),

    'vit_base_patch32_clip_224.laion2b_ft_in12k': _cfg(
        #hf_hub_id='timm/vit_base_patch32_clip_224.laion2b_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=11821),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.openai_ft_in12k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_giant_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_gigantic_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1280),

    'vit_base_patch32_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_large_patch14_clip_224.openai': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_336.openai': _cfg(
        hf_hub_id='timm/', hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), num_classes=768),

    # experimental (may be removed)
    'vit_base_patch32_plus_256.untrained': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),
    'vit_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240), crop_pct=0.95),
    'vit_small_patch16_36x1_224.untrained': _cfg(url=''),
    'vit_small_patch16_18x2_224.untrained': _cfg(url=''),
    'vit_base_patch16_18x2_224.untrained': _cfg(url=''),

    # EVA fine-tuned weights from MAE style MIM - EVA-CLIP target pretrain
    # https://github.com/baaivision/EVA/blob/7ecf2c0a370d97967e86d047d7af9188f78d2df3/eva/README.md#eva-l-learning-better-mim-representations-from-eva-clip
    'eva_large_patch14_196.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_21k_to_1k_ft_88p6.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_21k_to_1k_ft_89p2.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
    'eva_large_patch14_196.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_1k_ft_88p0.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_1k_ft_88p65.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

    'flexivit_small.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.1000ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_1000ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.300ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'flexivit_large.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.patch16_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b16_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.patch30_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b30_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'vit_base_patch16_xp_224.untrained': _cfg(url=''),
    'vit_large_patch14_xp_224.untrained': _cfg(url=''),
    'vit_huge_patch14_xp_224.untrained': _cfg(url=''),

    'vit_base_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_large_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
})



def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0., hook=False):
        super().__init__()
        self.hook = hook
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        if self.hook:
            self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        if self.hook:
            self.save_attn(attn)
            attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., hook=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, hook=hook)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0., hook=False, clinical=False, radiomics=False, clinical_embed_dim=64):
        super().__init__()
        self.hook = hook
        self.num_classes = num_classes
        self.multi_modal = clinical or radiomics
        self.clinical_embed_dim = clinical_embed_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, hook=hook)
            for i in range(depth)])

        
        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            
            if clinical and radiomics:
                self.multi_modal_fc = Mlp(389, clinical_embed_dim, clinical_embed_dim, 0.5)
                # self.mixed_fc = Linear(2*embed_dim, embed_dim)
                self.head = Linear(embed_dim+clinical_embed_dim, num_classes)
            elif clinical:
                self.multi_modal_fc = Mlp(3, clinical_embed_dim, clinical_embed_dim, 0.5)
                # self.mixed_fc = Linear(2*embed_dim, embed_dim)
                self.head = Linear(embed_dim+clinical_embed_dim, num_classes)
            elif radiomics:
                self.multi_modal_fc = Mlp(386, clinical_embed_dim, clinical_embed_dim, 0.5)
                # self.mixed_fc = Linear(embed_dim+clinical_embed_dim, embed_dim)
                self.head = Linear(embed_dim+clinical_embed_dim, num_classes)
            
            # self.head = Linear(embed_dim, num_classes)
            
        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None
        
        self.cat = Cat()

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, clinical_feat=None):
        # clinical_feat : B x 3
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        if self.hook:
            x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1) # B x 768
        
        if self.multi_modal:
            clinical_feat = self.multi_modal_fc(clinical_feat) # B x 3
            mixed_feats = self.cat((x, clinical_feat), dim=1)
            # mixed_feats = self.mixed_fc(mixed_feats)
            # clinical_feat = rearrange(clinical_feat, 'b n d -> b (n d)', n=3, d=self.clinical_embed_dim)
        else:
            clinical_feat = clinical_feat.repeat(1, self.clinical_embed_dim)
            mixed_feats = self.cat((x, clinical_feat), dim=1)
        
        
        mixed_feats = self.head(mixed_feats)
        
        return mixed_feats

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam, clinical = self.cat.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam
        
        elif method == "second_last_layer":
            cam = self.blocks[-2].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-2].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def create_model(model_name, pretrained=False, **kwargs):
    model = globals()[model_name](pretrained=pretrained, **kwargs)
    
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch8_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/8)
    """
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch14_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14)
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic (big-G) model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer(
        'vit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_medium_patch16_gap_240(pretrained=False, **kwargs):
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 240x240
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_medium_patch16_gap_256(pretrained=False, **kwargs):
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 256x256
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_medium_patch16_gap_384(pretrained=False, **kwargs):
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 384x384
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_gap_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/o class token, w/ avg-pool @ 256x256
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_base_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch32_clip_224(pretrained=False, **kwargs):
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch32_clip_384(pretrained=False, **kwargs):
    """ ViT-B/32 CLIP image tower @ 384x384
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch32_clip_448(pretrained=False, **kwargs):
    """ ViT-B/32 CLIP image tower @ 448x448
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_clip_224(pretrained=False, **kwargs):
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_clip_384(pretrained=False, **kwargs):
    """ ViT-B/16 CLIP image tower @ 384x384
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch14_clip_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14) CLIP image tower
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch14_clip_336(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_patch14_clip_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) CLIP image tower.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_patch14_clip_336(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_giant_patch14_clip_224(pretrained=False, **kwargs):
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_giant_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_gigantic_patch14_clip_224(pretrained=False, **kwargs):
    """ ViT-bigG model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

# Experimental models below

def vit_base_patch32_plus_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32+)
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch32_plus_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_plus_240(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16+)
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_rpn_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ residual post-norm
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5,
        class_token=False, block_fn=ResPostBlock, global_pool='avg')
    model = _create_vision_transformer(
        'vit_base_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_36x1_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_small_patch16_36x1_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer(
        'vit_small_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer(
        'vit_base_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def eva_large_patch14_196(pretrained=False, **kwargs):
    """ EVA-large model https://arxiv.org/abs/2211.07636 /via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer(
        'eva_large_patch14_196', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def eva_large_patch14_336(pretrained=False, **kwargs):
    """ EVA-large model https://arxiv.org/abs/2211.07636 via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def flexivit_small(pretrained=False, **kwargs):
    """ FlexiViT-Small
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True)
    model = _create_vision_transformer('flexivit_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def flexivit_base(pretrained=False, **kwargs):
    """ FlexiViT-Base
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True)
    model = _create_vision_transformer('flexivit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def flexivit_large(pretrained=False, **kwargs):
    """ FlexiViT-Large
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True)
    model = _create_vision_transformer('flexivit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch16_xp_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_base_patch16_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch14_xp_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_large_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_huge_patch14_xp_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_huge_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_small_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1.0, img_size=518,
    )
    model = _create_vision_transformer(
        'vit_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1.0, img_size=518,
    )
    model = _create_vision_transformer(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-L/14 for DINOv2
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1.0, img_size=518,
    )
    model = _create_vision_transformer(
        'vit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_giant_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-G/14 for DINOv2
    """

    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192

    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1.0, 
        mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, img_size=518, act_layer=nn.SiLU
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model