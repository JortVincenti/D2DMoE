from typing import Dict

import torch
from torch import nn
from transformers import apply_chunking_to_forward

from architectures.custom import CustomMultiheadAttention
from architectures.moe.moe_layers import MoELayer
import torch.nn.functional as F
from typing import Optional
from architectures.basic_var import AdaLNSelfAttn, FFN
from torch.profiler import record_function
import time

def moe_attention_forward(self: CustomMultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          _key_padding_mask=None,
                          need_weights=False):
    assert False, "This function should not be called directly" 
    # TODO warning! works only with "simplified" MultiheadAttention
    assert query.size() == key.size() == value.size()
    batch_size, seq_length, embed_dim = query.size()
    gating_data = {}
    if isinstance(self.q_proj, MoELayer):
        query, q_gating_data = self.q_proj(query)
        gating_data.update(q_gating_data)
    else:
        query = self.q_proj(query)
    if isinstance(self.k_proj, MoELayer):
        key, k_gating_data = self.k_proj(key)
        gating_data.update(k_gating_data)
    else:
        key = self.k_proj(key)
    if isinstance(self.v_proj, MoELayer):
        value, v_gating_data = self.v_proj(value)
        gating_data.update(v_gating_data)
    else:
        value = self.v_proj(value)
    # separate the head dimension, and permute dimensions into [Batch, Head, SeqLen, Dims]
    query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    # Determine value outputs
    values, attention = self.scaled_dot_product(query, key, value,
                                                dropout_p=self.dropout_p if self.training else 0.0)
    values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
    values = values.reshape(batch_size, seq_length, embed_dim)
    if isinstance(self.o_proj, MoELayer):
        o, o_routing_data = self.o_proj(values)
        gating_data.update(o_routing_data)
    else:
        o = self.o_proj(values)
    if need_weights:
        return o, attention, gating_data
    else:
        return o, None, gating_data


def moe_var_block_forward(self, x, cond_BD, attn_bias, current_scale):

    if self.scale_switch is not None and current_scale < self.scale_switch:
        if not self.dense_blocks and self.scale_switch: raise ValueError("scale_switch must be none when no dense_ffn is given, use set_dense_module() in VAR!")
        router_stats = x.shape[0] * x.shape[1] * self.ffn.num_experts * self.ffn.expert_dim
        x = self.dense_blocks(x, cond_BD, attn_bias)
        if current_scale == self.scale_switch-1:
            src_attn, dst_attn = self.dense_blocks.attn, self.attn
            dst_attn.cached_k = src_attn.cached_k    # share the same tensor
            dst_attn.cached_v = src_attn.cached_v
    else:
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        x = x + self.drop_path(self.attn(self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
    
        if isinstance(self.ffn, MoELayer):
            ffn_input = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
            y, router_stats = self.ffn(ffn_input) 
            x = x + self.drop_path(y.mul(gamma2))
        else:
            x = x + self.drop_path(self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2))

    return x, router_stats


def moe_var_main_forward(self, class_idx: torch.Tensor, x: torch.Tensor, return_gating_data: bool = False) -> torch.Tensor:
    """
    A reference forward pass that mirrors the original VAR logic,
    but uses the requested signature.

    Args:
        x (torch.Tensor):
            The main input tensor (analogous to x_BLCv_wo_first_l in the original).
            Shape should be [B, L - first_l, Cvae], where Cvae == 32 if
            self.word_embed = Linear(32, 1024, ...).
        class_idx (Optional[torch.Tensor]):
            Class indices (analogous to label_B in the original).
            Shape [B].
        lvl_idx (Optional[torch.Tensor]):
            (Optional) If you want to override or supply a level index,
            though the original code references self.lvl_1L internally.
        return_gating_data (bool):
            If True, returns any MoE gating information from the blocks.

    Returns:
        torch.Tensor: Final logits with shape [B, L, vocab_size], 
                      or ([B, L, vocab_size], gating_data) if return_gating_data=True.
    """
    raise NotImplementedError('Only used for training')
    # -------------------------------------------------------------------------
    # 1) Move inputs onto the correct device(s)
    # -------------------------------------------------------------------------
    device_class = self.class_emb.weight.device  # device of class_emb
    if class_idx is not None and class_idx.device != device_class:
        class_idx = class_idx.to(device_class)

    device_word = self.word_embed.weight.device  # device of word_embed
    if x.device != device_word:
        x = x.to(device_word)

    # -------------------------------------------------------------------------
    # 2) Possibly gather the valid sequence range (bg, ed) and batch size
    #    from your self.begin_ends / self.prog_si logic.
    # -------------------------------------------------------------------------
    bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
    B = x.shape[0]  # batch size

    # -------------------------------------------------------------------------
    # 3) Conditionally drop class indices & build the "sos" embedding
    # -------------------------------------------------------------------------
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # If class_idx is provided, do the cond_drop
        if class_idx is not None:
            class_idx = torch.where(
                torch.rand(B, device=class_idx.device) < self.cond_drop_rate,
                self.num_classes,  # some "no-class" index
                class_idx
            )
            cond_BD = self.class_emb(class_idx)  # [B, 1024]
        else:
            # If no class_idx is given, you might default to a zero or skip
            # But to mirror original code, we require class_idx.
            raise ValueError("class_idx (label_B) is required in this forward pass.")

        # Build the "sos" portion
        # shape of cond_BD: [B, 1024]
        # self.pos_start:   [1, first_l, 1024] (likely a buffer)
        # `sos` becomes [B, first_l, 1024]
        sos = cond_BD.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

        # ---------------------------------------------------------------------
        # 4) If prog_si == 0, just use the SOS tokens.
        #    Otherwise, cat the output of word_embed(x).
        # ---------------------------------------------------------------------
        if self.prog_si == 0:
            x_BLC = sos  # [B, first_l, 1024]
        else:
            # word_embed wants last dim == 32
            # x should be [B, (L - first_l), 32]
            # => output is [B, (L - first_l), 1024]
            embedded_x = self.word_embed(x.float())
            x_BLC = torch.cat((sos, embedded_x), dim=1)  # [B, L, 1024]

        # ---------------------------------------------------------------------
        # 5) Add level embeddings & position embeddings
        #    (Mirroring: x_BLC += self.lvl_embed(...) + self.pos_1LC[:, :ed])
        # ---------------------------------------------------------------------
        # The original code used self.lvl_1L internally. If you want to use
        # lvl_idx from the signature, you could do it here, e.g.:
        #
        #    if lvl_idx is not None:
        #        x_BLC += self.lvl_embed(lvl_idx) + self.pos_1LC[:, :ed]
        #
        # but the original code uses something like:
        x_BLC += (
            self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))  # [B, ed, 1024]
            + self.pos_1LC[:, :ed]                             # [1, ed, 1024] -> broadcast to [B, ed, 1024]
        )

    # -------------------------------------------------------------------------
    # 6) Prepare the attention bias, do any shared_ada_lin transform
    # -------------------------------------------------------------------------
    attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
    cond_BD_or_gss = self.shared_ada_lin(cond_BD)  # [B, 1024]

    # If using mixed precision, unify dtypes
    temp = x_BLC.new_ones(8, 8)
    main_type = torch.matmul(temp, temp).dtype  # e.g. bfloat16 or float16
    x_BLC = x_BLC.to(dtype=main_type)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
    attn_bias = attn_bias.to(dtype=main_type)

    # -------------------------------------------------------------------------
    # 7) Pass through your MoE/Transformer blocks, optionally capturing gating
    # -------------------------------------------------------------------------
    gating_info = []
    for index, block in enumerate(self.blocks):  
        # If your block returns (output, gating_data):
        x_BLC, g_dat = block(x_BLC, cond_BD_or_gss, attn_bias)
        gating_info.append(g_dat)

    # -------------------------------------------------------------------------
    # 8) Final projection (logits), plus the "hack" to keep word_embed params
    #    from being optimized away if prog_si == 0
    # -------------------------------------------------------------------------
    x_BLC = self.get_logits(x_BLC.float(), cond_BD)  # shape [B, L, V]

    if self.prog_si == 0:
        # The original code uses a small trick to keep some grads alive:
        if isinstance(self.word_embed, nn.Linear):
            x_BLC[0, 0, 0] += (
                self.word_embed.weight[0, 0] * 0
                + self.word_embed.bias[0] * 0
            )
        else:
            s = 0
            for p in self.word_embed.parameters():
                if p.requires_grad:
                    s += p.view(-1)[0] * 0
            x_BLC[0, 0, 0] += s

    # -------------------------------------------------------------------------
    # 9) Return final logits (and gating info if requested)
    # -------------------------------------------------------------------------
    if return_gating_data:
        return x_BLC, gating_info
    else:
        return x_BLC


def ffn_filter_condition_var(_model: nn.Module, m: nn.Module):
    if isinstance(m, AdaLNSelfAttn):
        return True