from functools import partial

import torch
import torchvision
from torchvision.models import ViT_B_16_Weights, EfficientNet_V2_S_Weights, EfficientNet_B0_Weights, \
    ConvNeXt_Tiny_Weights, Swin_V2_S_Weights, Swin_T_Weights
from architectures import build_vae_var
import os 
import dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from architectures.vqvae import VQVAE



def get_efficientnet_b0():
    model = torchvision.models.efficientnet_b0(EfficientNet_B0_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_efficientnet_v2_s():
    model = torchvision.models.efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 384
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_convnext_t():
    model = torchvision.models.convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.avgpool(x)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_vit_b_16():
    model = torchvision.models.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        # go through encoder blocks
        for block in self.encoder.layers:
            x = block(x)
            x = yield x, None
        x = self.encoder.ln(x)
        # END OF ENCODER
        # classifier token
        x = x[:, 0]
        x = self.heads(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_swin_v2_s():
    model = torchvision.models.swin_v2_s(Swin_V2_S_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 256
    model.input_channels = 3
    model.number_of_classes = 1000
    return model


def get_swin_t():
    model = torchvision.models.swin_t(Swin_T_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 256
    model.input_channels = 3
    model.number_of_classes = 1000
    return model

# def get_var_d16():
#     model_ckpt = f'var_d16.pth'
#     if not os.path.exists(model_ckpt):
#         os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{model_ckpt}')
    
#     # Params
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
#     # var_ckpt = f'var_d16.pth'

#     # build models    
#     _, model = build_vae_var(
#         V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
#         device=device, patch_nums=patch_nums,
#         num_classes=1000, depth=16, shared_aln=False,
#     )

#     model.num_classes = 1000
#     model.input_size = 256
#     model.input_channels = 3
#     #model.forward_generator = partial(forward_generator, model)

#     model.load_state_dict(torch.load(model_ckpt, map_location=device), strict=True)
#     var_wo_ddp: VAR = compile_model(model, 0)
#     var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    

#     return var, var_wo_ddp

def get_var_d16():
    ################## 1. Download checkpoints and build models
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
    MODEL_DEPTH = 16
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    return var, vae

def compile_model(m, fast):
    if fast == 0:
        return m
    return torch.compile(m, mode={
        1: 'reduce-overhead',
        2: 'max-autotune',
        3: 'default',
    }[fast]) if hasattr(torch, 'compile') else m


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

