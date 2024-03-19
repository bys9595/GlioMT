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

from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class ClinicalSwinT(SwinTransformer):
    """ Clinical SwinT
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1, window_size=7,
                 patch_size=4, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 decoder_depth=2, clini_info_style='bert', embed_trainable=False, clini_embed_token='cls',
                 fusion_style='self-attn', age_cutoff=45, clini=True, mm_num_heads=1):
        super().__init__(img_size=img_size, in_chans=in_chans, window_size=window_size, 
                         num_classes=num_classes, patch_size=patch_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads)
        
        
        self.clini_info_style = clini_info_style
        self.embed_trainable = embed_trainable
        self.clini_embed_token = clini_embed_token
        self.fusion_style = fusion_style
        self.age_cutoff = age_cutoff
        self.clini= clini
        self.mm_num_heads= mm_num_heads
        
        if self.clini_embed_token == 'cls':
            self.clini_embed_token_idx = 0
        elif self.clini_embed_token == 'word':
            self.clini_embed_token_idx = 5
        
        if clini_info_style == 'bert':
            if self.num_features==768:
                key = "bert-base-uncased"
            if self.num_features==1024:
                key = "bert-large-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(key)
            self.embed_model = BertModel.from_pretrained(key)
        elif clini_info_style == 'random':
            self.clini_embedding = nn.Embedding(4, self.num_features)
        
        self.projector = nn.Linear(self.num_features+2, self.num_features)
        # self.predecoder_norm = nn.LayerNorm(768, eps=1e-6)
        
        # self.multimodal_cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))
        # nn.init.normal_(self.multimodal_cls_token, std=1e-6)
        
        if self.clini:
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, 3, self.num_features) * .02)
        else:
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_features) * .02)
        
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
        self.pool_feat = SelectAdaptivePool2d(
                                pool_type='avg',
                                flatten=True,
                                input_fmt='NHWC',
                            )
        
        self.decoder_blocks = nn.Sequential(*[
            Block(dim=self.num_features,
                num_heads=mm_num_heads,
                qkv_bias=True
                )
            for i in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(self.num_features, eps=1e-6)
        

    def forward(self, x, clinical_feats):
        age_embed, sex_embed = self.forward_clinical_embed(clinical_feats)
        
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=True)
        # x = self.predecoder_norm(x)
        # x = self.norm_proj(self.projector(x))
        
        if self.fusion_style == 'self-attn':
            if self.clini:
                multi_modal_feats = torch.stack([x, age_embed, sex_embed], 1)
                # ---
                # cls_token = self.multimodal_cls_token.expand(x.shape[0], -1, -1)
                # multi_modal_feats = torch.cat([cls_token, multi_modal_feats], 1)
                # ---
            else:
                # cls_token = self.multimodal_cls_token.expand(x.shape[0], -1, -1)
                # multi_modal_feats = torch.cat([cls_token, x.unsqueeze(1)], 1)
                multi_modal_feats = x.unsqueeze(1)
                    
                    
            multi_modal_feats = multi_modal_feats + self.decoder_pos_embed
            multi_modal_feats = self.decoder_blocks(multi_modal_feats)
            multi_modal_feats = self.decoder_norm(multi_modal_feats)
        
            cls_token = multi_modal_feats[:, 0]
            out = self.head.flatten(self.head.fc(cls_token))
        
        elif self.fusion_style == 'concat':
            age = torch.tensor([int(i) for i in clinical_feats[0]]).cuda().unsqueeze(1)
            age = (age - 55) / 15
            sex = torch.tensor([1 if i == 'male' else -1 for i in clinical_feats[1]]).cuda().unsqueeze(1)
            
            multi_modal_feats = torch.cat((x, age, sex), 1)
            multi_modal_feats = self.projector(multi_modal_feats)
            out = self.decoder_norm(multi_modal_feats)
            
            out = self.head.flatten(self.head.fc(out))
                

        return out
    

    
    def forward_clinical_embed(self, clinical_feats):
        import math
        age = clinical_feats[0]
        sex = clinical_feats[1]
        
        age = ['an old' if int(i)>=self.age_cutoff else 'a young' for i in age]

        if self.clini_info_style == 'bert':
            age_prompt = ['an mri of ' + age[i] + ' patient' for i in range(len(age))]
            sex_prompt = ['an mri of a ' + sex[i] + ' patient' for i in range(len(sex))]
            
            # # age_list = ['zeros', 'teens', 'twenties', 'thirties', '40s', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties', 'hundreds']
            # age_list = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '100s']

            # age_prompt = ['A patient in their ' + age_list[math.floor(int(age[i])/10)] for i in range(len(age))]
            # sex_prompt = ['A ' + sex[i] + ' patient' for i in range(len(sex))]
                    
            age_token = self.tokenizer(age_prompt, return_tensors="pt", padding=True, truncation=True)
            sex_token = self.tokenizer(sex_prompt, return_tensors="pt")

            age_token = {k: v.cuda() for k, v in age_token.items()}
            sex_token = {k: v.cuda() for k, v in sex_token.items()}
            
            if self.embed_trainable:
                self.embed_model.eval()
                age_embed = self.embed_model(**age_token).last_hidden_state[:, self.clini_embed_token_idx]
                sex_embed = self.embed_model(**sex_token).last_hidden_state[:, self.clini_embed_token_idx]
            else:
                self.embed_model.eval()
                with torch.no_grad():                      
                    age_embed = self.embed_model(**age_token).last_hidden_state[:, self.clini_embed_token_idx]
                    sex_embed = self.embed_model(**sex_token).last_hidden_state[:, self.clini_embed_token_idx]
                
        elif self.clini_info_style == 'random':
            age_idx = [0 if i == 'an old' else 1 for i in age]
            sex_idx = [2 if i == 'male' else 3 for i in sex]
            
            if self.embed_trainable:
                age_embed = self.clini_embedding(torch.Tensor(age_idx).long().cuda())
                sex_embed = self.clini_embedding(torch.Tensor(sex_idx).long().cuda())
            else:
                self.clini_embedding.eval()
                with torch.no_grad():                      
                    age_embed = self.clini_embedding(torch.Tensor(age_idx).long().cuda())
                    sex_embed = self.clini_embedding(torch.Tensor(sex_idx).long().cuda())
        
        # return age_embed
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

