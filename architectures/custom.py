import math

import torch
from torch import nn
from architectures.basic_var import SelfAttention
from utils import find_module_names, get_module_by_name, set_module_by_name, get_module_name, get_parent_module_name
# import utils
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.0, proj_drop=0.0,attn_l2_norm=False, flash_if_available=True):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by the number of heads.'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        self.register_buffer(
            "causal_mask",
            None
        )
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()

    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, attn_bias, need_weights=False):
        """
        x: [B, L, C]
        attn_bias: optional attention mask/bias
        """
        B, L, C = x.shape
        # 1) Compute Q, K, V separately
        q = self.q_proj(x)  # => [B, L, C]
        k = self.k_proj(x)  # => [B, L, C]
        v = self.v_proj(x)  # => [B, L, C]

        # 2) Reshape into [B, L, num_heads, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # Depending on whether we're using flash/xformers, we handle dimension ordering:
        using_flash = self.using_flash and attn_bias is None and q.dtype != torch.float32
        if using_flash or self.using_xform:
            # flash/xformers typically wants shape [B, L, num_heads, head_dim]
            # so we'll keep it as-is, i.e. (B, L, H, D)
            dim_cat = 1
        else:
            # Our "slow_attn" code expects [B, num_heads, L, head_dim]
            # => permute(0, 2, 1, 3)
            q = q.permute(0, 2, 1, 3)  # => [B, H, L, D]
            k = k.permute(0, 2, 1, 3)  # => [B, H, L, D]
            v = v.permute(0, 2, 1, 3)  # => [B, H, L, D]
            dim_cat = 2

        # 3) (Optional) Attn L2 norm logic
        # (We replicate the original code's approach)
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            # If using flash/xformers => shape is [B, L, H, D], we swap head & length dims
            if using_flash or self.using_xform:
                # scale_mul is [1, H, 1, 1], we want it to be broadcast as [B, L, H, D]
                # or equivalently [B, H, L, D] => we do shape transpose
                scale_mul = scale_mul.transpose(1, 2)  # 1,H,1,1 => 1,1,H,1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        # 4) KV caching logic
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                # Append to dimension = dim_cat
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        # 5) Perform the attention using either flash/xformers or slow_attn
        dropout_p = self.attn_drop if self.training else 0.0
        main_type = q.dtype

        if using_flash:
            # shape: [B, L, H, D]
            # flash_attn_func returns [B, L, H, D], flatten to [B, L, C]
            out = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale
            ).view(B, L, C)

        elif self.using_xform:
            # memory_efficient_attention expects same shape as flash => [B, L, H, D]
            # attn_bias needs shape [B, H, L, L], or expanded
            x_bias = None
            if attn_bias is not None:
                x_bias = attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
            out = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=x_bias,
                p=dropout_p,
                scale=self.scale
            ).view(B, L, C)

        else:
            # shape is [B, H, L, D] => slow_attn wants Q, K, V in that shape
            # slow_attn(...) => returns [B, H, L, D], then we .transpose(1,2) => [B, L, H, D] => flatten to [B, L, C]
            out = slow_attn(
                query=q,    # [B, H, L, D]
                key=k,      # [B, H, L, D]
                value=v,    # [B, H, L, D]
                scale=self.scale,
                attn_mask=attn_bias,
                dropout_p=dropout_p
            ).transpose(1, 2).reshape(B, L, C)

        # 6) Output projection
        if need_weights:
            return self.proj_drop(self.o_proj(out)), out
        return self.proj_drop(self.o_proj(out))


def transcribe_mha_params(org_mha, simple_mha):
    if isinstance(org_mha, nn.MultiheadAttention):
        assert org_mha._qkv_same_embed_dim is True
        assert org_mha.embed_dim == simple_mha.embed_dim
        assert org_mha.num_heads == simple_mha.num_heads
        assert org_mha.in_proj_bias is not None and org_mha.bias_v is None and org_mha.bias_k is None
        assert org_mha.out_proj.bias is not None
        assert org_mha.batch_first is True
        embed_dim = org_mha.embed_dim
        simple_mha.q_proj.weight = torch.nn.Parameter(org_mha.in_proj_weight[:embed_dim].clone().detach())
        simple_mha.q_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[:embed_dim].clone().detach())
        simple_mha.k_proj.weight = torch.nn.Parameter(org_mha.in_proj_weight[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.k_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[embed_dim:2 * embed_dim].clone().detach())
        simple_mha.v_proj.weight = torch.nn.Parameter(
            org_mha.in_proj_weight[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.v_proj.bias = torch.nn.Parameter(org_mha.in_proj_bias[2 * embed_dim:3 * embed_dim].clone().detach())
        simple_mha.o_proj.weight = torch.nn.Parameter(org_mha.out_proj.weight.clone().detach())
        simple_mha.o_proj.bias = torch.nn.Parameter(org_mha.out_proj.bias.clone().detach())
    elif isinstance(org_mha, SelfAttention):
        # 1) Extract references to the internal SelfAttention module
        sa = org_mha  # SelfAttention

        # 2) Check shapes and consistency
        embed_dim_org = sa.num_heads * sa.head_dim
        assert embed_dim_org == simple_mha.embed_dim, (
            f"Mismatch in embed_dim: SelfAttention has {embed_dim_org}, "
            f"but simple_mha has {simple_mha.embed_dim}"
        )
        assert sa.mat_qkv.weight.shape[1] == embed_dim_org, (
            f"mat_qkv expected in_features={embed_dim_org}, got {sa.mat_qkv.weight.shape[1]}"
        )

        # 3) Extract QKV weights/bias
        #    mat_qkv.weight: shape = [3*embed_dim, embed_dim]
        #    combined Q-K-V bias: torch.cat((q_bias, zero_k_bias, v_bias)) => shape [3*embed_dim]
        w_qkv = sa.mat_qkv.weight.clone().detach()  # [3*E, E]
        b_qkv = torch.cat((sa.q_bias, sa.zero_k_bias, sa.v_bias), dim=0).clone().detach()  # [3*E]

        # Chunk them into Q, K, V parts
        E = embed_dim_org
        w_q = w_qkv[:E, :]                # [E, E]
        w_k = w_qkv[E:2*E, :]             # [E, E]
        w_v = w_qkv[2*E:3*E, :]           # [E, E]

        b_q = b_qkv[:E]                   # [E]
        b_k = b_qkv[E:2*E]                # [E]
        b_v = b_qkv[2*E:3*E]              # [E]

        # 4) Extract the output projection weights/bias
        w_o = sa.proj.weight.clone().detach()  # [E, E]
        b_o = sa.proj.bias.clone().detach()    # [E]

        # 5) Assign them to the simple_mha parameters
        #    (You must ensure simple_mha has q_proj, k_proj, v_proj, o_proj as nn.Linear modules.)
        simple_mha.q_proj.weight = torch.nn.Parameter(w_q)
        simple_mha.q_proj.bias   = torch.nn.Parameter(b_q)

        simple_mha.k_proj.weight = torch.nn.Parameter(w_k)
        simple_mha.k_proj.bias   = torch.nn.Parameter(b_k)

        simple_mha.v_proj.weight = torch.nn.Parameter(w_v)
        simple_mha.v_proj.bias   = torch.nn.Parameter(b_v)

        simple_mha.o_proj.weight = torch.nn.Parameter(w_o)
        simple_mha.o_proj.bias   = torch.nn.Parameter(b_o)
    else:
        raise NotImplementedError()


def simplify_mha(model):
    mhas_names = find_module_names(model, lambda _model, m: isinstance(m, (nn.MultiheadAttention, SelfAttention)))
    for mha_name in mhas_names:
        org_mha = get_module_by_name(model, mha_name)
        if isinstance(org_mha, nn.MultiheadAttention):
            embed_dim = org_mha.embed_dim
            num_heads = org_mha.num_heads
            dropout = org_mha.dropout
        elif isinstance(org_mha, SelfAttention):
            embed_dim = org_mha.num_heads * org_mha.head_dim
            num_heads = org_mha.num_heads
            dropout = org_mha.attn_drop
        else:
            raise NotImplementedError()

        simple_mha = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout)
        transcribe_mha_params(org_mha, simple_mha)
        set_module_by_name(model, mha_name, simple_mha)


def create_attention_projection_filter_condition(projection_name: str = None):
    def filter_condition(model: nn.Module, m: nn.Module):
        m_name = get_module_name(model, m)
        parent_module = get_module_by_name(model, get_parent_module_name(m_name))
        if isinstance(m, nn.Linear) and ('mat_qkv' in m_name or 'proj' in m_name):
            return True
        return False

    return filter_condition
