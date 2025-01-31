from functools import partial

import torch
import torchvision
from torchvision.models import ViT_B_16_Weights, EfficientNet_V2_S_Weights, EfficientNet_B0_Weights, \
    ConvNeXt_Tiny_Weights, Swin_V2_S_Weights, Swin_T_Weights
from architectures import build_vae_var
import os 
import dist
from torch.nn.parallel import DistributedDataParallel as DDP


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

def get_var_d16():
    model_ckpt = f'var_d16.pth'
    if not os.path.exists(model_ckpt):
        os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{model_ckpt}')
    
    # Params
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    # var_ckpt = f'var_d16.pth'

    # build models    
    _, model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=False,
    )

    def forward_generator(self, inputs):
        """
        A generator-style forward pass:
        - Yields (x, None) after partial computations
        - Finally yields (None, logits) at the end.
        """
        # The VAR forward typically expects two inputs: label_B and x_BLCv_wo_first_l
        # So we unpack them from `inputs`:
        label_B, x_BLCv_wo_first_l = inputs
        B = x_BLCv_wo_first_l.shape[0]

        # Figure out how many tokens we actually process (progressive training or full):
        if self.prog_si >= 0:
            bg, ed = self.begin_ends[self.prog_si]
        else:
            bg, ed = (0, self.L)

        # --------------------------------
        # Step 1: Pre-processing
        # --------------------------------
        with torch.cuda.amp.autocast(enabled=False):
            # Possibly drop the label condition
            label_B = torch.where(
                torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                self.num_classes,
                label_B
            )
            # Class embedding
            sos = cond_BD = self.class_emb(label_B)

            # Expand to first_l and add your pos_start
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            # If progressive scale index is 0, we only keep the sos
            if self.prog_si == 0:
                x_BLC = sos
            else:
                # Otherwise, concatenate the teacher-forced tokens
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            # Add level + position embeddings up to ed
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

        # At this point, you can *yield* your first “intermediate”:
        x_BLC = yield x_BLC, None  # the calling code can optionally modify x_BLC

        # --------------------------------
        # Step 2: Attention bias, type casting
        # --------------------------------
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # Attempt to match main precision
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC           = x_BLC.to(dtype=main_type)
        cond_BD_or_gss  = cond_BD_or_gss.to(dtype=main_type)
        attn_bias       = attn_bias.to(dtype=main_type)

        # --------------------------------
        # Step 3: Go through the backbone blocks
        # --------------------------------
        for i, block in enumerate(self.blocks):
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            x_BLC = yield x_BLC, None  # yield intermediate after each block

        # --------------------------------
        # Step 4: Final projection to logits
        # --------------------------------
        logits = self.get_logits(x_BLC.float(), cond_BD)

        # The final yield: no more intermediate, so (None, logits)
        yield None, logits
        model.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

    model.num_classes = 1000
    model.input_size = 256
    model.input_channels = 3
    model.forward_generator = partial(forward_generator, model)

    model.load_state_dict(torch.load(model_ckpt, map_location=device), strict=True)
    var_wo_ddp: VAR = compile_model(model, 0)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    

    return var, var_wo_ddp

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

