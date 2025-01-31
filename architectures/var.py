import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from architectures.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from architectures.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from architectures.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.number_of_classes = num_classes
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, plotting_PCA=False
    ) -> torch.Tensor:
        """
        Inference method for autoregressive mode, collecting activations per scale.
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if plotting_PCA:
            # Initialize activations dictionary
            activations = {f"scale_{si}": {} for si in range(len(self.patch_nums))}
            def activation_hook(module, input, output, scale_idx):
                layer_name = f"{module.__class__.__name__}_{id(module)}"
                if layer_name not in activations[f"scale_{scale_idx}"]:
                    if "GELU" in layer_name:
                        activations[f"scale_{scale_idx}"][layer_name] = []
                else:
                    print(f"WARNING: Duplicate layer name found: {layer_name}")
                if "GELU" in layer_name: 
                    activations[f"scale_{scale_idx}"][layer_name].append(output.detach().cpu().numpy())

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks:
            b.attn.kv_caching(True)

        # Scale-wise processing
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn

            if plotting_PCA:
                # Dynamically register hooks for the current scale
                for idx, block in enumerate(self.blocks):
                    # Register hooks for `ffn` and its submodules
                    for name, module in block.ffn.named_children():
                        if name == "act":
                            module.register_forward_hook(lambda m, i, o: activation_hook(m, i, o, scale_idx=si))
            # Forward pass for the current scale
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

            if plotting_PCA:
                # Unregister hooks for the current scale to avoid duplicates
                for idx, block in enumerate(self.blocks):
                    for name, module in block.ffn.named_children():
                        if name == "act":
                            module._forward_hooks.clear()

        for b in self.blocks:
            b.attn.kv_caching(False)

        if plotting_PCA:
            # --------------------------------------------------------------------
            # PCA analysis
            # --------------------------------------------------------------------
            import numpy as np
            import pickle
            ##############################################################################
            # SVD-BASED PCA
            ##############################################################################
            def perform_pca_svd(data):
                """
                Perform PCA using Singular Value Decomposition (SVD).
                Returns:
                    explained_variance (ndarray): Variance captured by each component (shape: n_components).
                    eigenvectors (ndarray): Principal component vectors (shape: d x n_components).
                    explained_variance_ratio (ndarray): Fraction of variance explained by each component (shape: n_components).
                    singular_values (ndarray): Singular values of data_centered (shape: n_components).
                """
                # 1) Center the data
                data_mean = np.mean(data, axis=0)
                data_centered = data - data_mean

                # 2) Perform SVD (data_centered is shape (N, d))
                U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

                # 3) Compute explained variance and explained variance ratio
                explained_variance = (S ** 2) / (data.shape[0] - 1)
                total_variance = np.sum(explained_variance)
                explained_variance_ratio = explained_variance / total_variance

                return explained_variance, Vt.T, explained_variance_ratio, S

            ##############################################################################
            # GATHER DATA: # of PCs for 99% variance
            ##############################################################################
            # Suppose 'activations' is a dict of dicts, e.g.:
            # activations = {
            #     'scale1': {
            #         'layer1_GELU': [array_of_activations1, array_of_activations2, ...],
            #         'layer2_GELU': [...],
            #         ...
            #     },
            #     'scale2': { ... },
            #     ...
            # }
            matrix_dict = {}  # NEW dictionary to store the actual matrix_components_patch for each scale
            variance_dict = {}  # NEW dictionary to store the variance of the activations for each scale
            means_dict = {}  # NEW dictionary to store the mean of the activations for each scale
            l1l2_ratios_dict = {}  # NEW dictionary to store the L1/L2 ratios of the activations for each scale
            for scale, layers_dict in activations.items():
                n_components_99 = []
                per_layer_metrics = []
                means = []
                l1l2_ratios_patches = []
                for layer_name, list_of_activations in layers_dict.items():
                    if "GELU" in layer_name:
                        if len(list_of_activations) > 1:
                            raise ValueError("Multiple lists of activations found for a single layer.")
                        
                        list_of_activations = list_of_activations[0].astype(np.float32)  # only one list of activations

                        list_components = []
                        list_patch_metrics = []
                        list_means = []
                        l1l2_ratios = []
                        for patch in range(list_of_activations.shape[1]):
                            patch_activation = list_of_activations[:, patch, :]
                            explained_variance, eigenvectors, explained_variance_ratio, singular_values = perform_pca_svd(patch_activation)
                            # 3) Calculate # of components for 99% variance
                            cumulative_ev = np.cumsum(explained_variance_ratio)
                            n_components = np.searchsorted(cumulative_ev, 0.99) + 1  # integer
                            list_components.append(n_components)

                            # 4) Compute CV metric for each patch
                            variances = np.var(patch_activation, axis=1).mean()  # Shape: (num_features,)
                            mean = np.mean(patch_activation, axis=1).mean()
                            list_patch_metrics.append(variances/mean)

                            # 5) Compute thresholded ratio values close to 0
                            close_to_zero = np.sum(np.abs(patch_activation) < 1e-5) / patch_activation.size
                            list_means.append(close_to_zero)

                            l1_norm = np.sum(np.abs(patch_activation), axis=1)  # Shape (D,)
                            l2_norm = np.sqrt(np.sum(patch_activation**2, axis=1))  # Shape (D,)
                            scalar_metric = l1_norm / l2_norm  # Ratio per feature
                            l1l2_ratios.append(scalar_metric.mean())  # Average across features

                        #dsfsfsdf
                        n_components_99.append(list_components)
                        per_layer_metrics.append(list_patch_metrics)
                        means.append(list_means) 
                        l1l2_ratios_patches.append(l1l2_ratios)

                # If we found something for this scale
                if n_components_99:
                    matrix_components_patch = np.array(n_components_99)
                    matrix_variance_patch = np.array(per_layer_metrics)
                    matrix_means_patch = np.array(means)
                    l1l2_ratios_patches = np.array(l1l2_ratios_patches)
                    # Save the FULL matrix in matrix_dict
                    matrix_dict[scale] = matrix_components_patch
                    variance_dict[scale] = matrix_variance_patch
                    means_dict[scale] = matrix_means_patch
                    l1l2_ratios_dict[scale] = l1l2_ratios_patches
                else:
                    matrix_dict[scale] = None
                    variance_dict[scale] = None
                    means_dict[scale] = None  
                    l1l2_ratios_dict[scale] = None  

            max_ev = len(cumulative_ev) + 1

            # ----------------------------------
            # SAVE matrix_dict & max_ev
            # ----------------------------------

            save_data = {
                "matrix_dict": matrix_dict,
                "variance_dict": variance_dict,
                "non_zero": means_dict,
                "max_ev": max_ev,
                "l1l2_ratios": l1l2_ratios_dict
            }

            with open("data/my_pca_data.pkl", "wb") as f:
                pickle.dump(save_data, f)

            print("PCA data saved to data/my_pca_data.pkl!")
   
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        device = self.class_emb.weight.device  # Get model device

        # Move label_B only if needed
        if label_B.device != device:
            label_B = label_B.to(device)

        device = self.word_embed.weight.device  # Get model device
        # Move x_BLCv_wo_first_l only if needed
        if x_BLCv_wo_first_l.device != device:
            x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device)


        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
