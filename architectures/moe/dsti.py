import logging
from copy import deepcopy
from typing import List, Optional, Callable, Union

import torch
from torch import nn
from torchvision.ops import MLP as TorchvisionMLP
from transformers.models.bert import BertLayer
from transformers.activations import PytorchGELUTanh
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.custom import create_attention_projection_filter_condition, CustomMultiheadAttention
from architectures.vit import MLP
from architectures.gpt import MLP as GPTMLP, GPT
from utils import find_module_names, get_module_by_name, set_module_by_name, get_module_name, get_parent_module_name
from architectures.basic_var import SelfAttention, FFN


class SubstitutionMLP(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimpleMLP(torch.nn.Sequential, SubstitutionMLP):
    # Almost the same as MLP from torchvison's ViT
    def __init__(
            self,
            in_size: int,
            hidden_sizes: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation: str = 'gelu',
            bias: bool = True,
            dropout: float = 0.0,
    ):
        from common import ACTIVATION_NAME_MAP
        layers = []
        in_dim = in_size
        for hidden_dim in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_sizes[-1], bias=bias)) # Jort added this to make match output dim of CustomMultiheadAttention
        layers.append(torch.nn.Dropout(dropout))
        torch.nn.Sequential.__init__(self, *layers)


class ResidualMLP(torch.nn.Sequential, SubstitutionMLP):
    def __init__(
            self,
            in_size: int,
            hidden_sizes: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation: str = 'gelu',
            bias: bool = True,
            dropout: float = 0.0,
    ):
        SubstitutionMLP.__init__(self)
        from common import ACTIVATION_NAME_MAP
        layers = []
        in_dim = in_size
        for hidden_dim in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_sizes[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))
        torch.nn.Sequential.__init__(self, *layers)

    def forward(self, x):
        saved_x = x
        for layer in self:
            x = layer(x)
        x = x + saved_x
        return x


MLP_NAME_MAP = {
    'simple': SimpleMLP,
    'residual': ResidualMLP
}


def replace_mha_projections(original_model: nn.Module,
                            flops_factor: float,
                            activation: str = 'gelu',
                            bias: bool = None,
                            dropout: Union[float, str] = 0.0,
                            q_proj: bool = True,
                            k_proj: bool = True,
                            v_proj: bool = True,
                            o_proj: bool = True,
                            mlp_type: str = 'simple',
                            ):
    original_model.eval()
    # print()
    # print(dir(original_model.module))
    # print("Annotations:", original_model.__annotations__)
    # print(f"input_size: {getattr(original_model, 'input_size', None)}")
    # print(f"num_classes: {getattr(original_model, 'num_classes', None)}")
    # print(f"depth: {getattr(original_model, 'depth', None)}")
    # print(f"lvl_1L: {getattr(original_model, 'lvl_1L', None)}")
    # for i, block in enumerate(original_model.module.blocks):
    #     if hasattr(block, 'attn'):
    #         print(f"Block {i} attn module structure: {dir(block.attn)}")
    #         print(f"Block {i} submodules in attn: {list(block.attn.named_children())}")
    # for i, block in enumerate(original_model.module.blocks):
    #     if hasattr(block, 'attn'):
    #         print(f"Block {i} o_proj: {getattr(block.attn, 'o_proj', None)}")
    #         print(f"Block {i} q_proj: {getattr(block.attn, 'q_proj', None)}")
    #         print(f"Block {i} k_proj: {getattr(block.attn, 'k_proj', None)}")
    #         print(f"Block {i} v_proj: {getattr(block.attn, 'v_proj', None)}")
    #print()

    if isinstance(original_model, GPT):
        d = original_model.config.n_embd
    else:
            # d = original_model.hidden_dim Jort: this is not correct
        d = original_model.module.C

    attention_dim = round(flops_factor * (d ** 2 + d) / (2 * d + 1))
    if dropout == 'same':
        dropout = original_model.attention_dropout
    logging.info(f'Determined attention_dim: {attention_dim}'
                 f' (activation: {activation}, bias:{bias}, dropout: {dropout})')
    model = deepcopy(original_model)
    modules_to_replace = []
    # if o_proj:
    #     modules_to_replace += find_module_names(original_model,
    #                                             create_attention_projection_filter_condition('o_proj'))
    # if q_proj:
    #     modules_to_replace += find_module_names(original_model,
    #                                             create_attention_projection_filter_condition('q_proj'))
    # if k_proj:
    #     modules_to_replace += find_module_names(original_model,
    #                                             create_attention_projection_filter_condition('k_proj'))
    # if v_proj:
    #     modules_to_replace += find_module_names(original_model,
    #                                             create_attention_projection_filter_condition('v_proj'))
    # print(o_proj, q_proj, k_proj, v_proj)
    if o_proj:
        modules_to_replace += find_module_names(original_model, create_attention_projection_filter_condition('proj'))
    if q_proj and k_proj and v_proj:
        modules_to_replace += find_module_names(original_model, create_attention_projection_filter_condition('mat_qkv'))
        
    # use hooks to get an example input
    assert len(modules_to_replace) > 0, f'{modules_to_replace=}'
    # replace the selected layers
    for name in modules_to_replace:
        original_module = get_module_by_name(model, name)
        bias = bias if bias is not None else original_module.bias is not None
        
        if 'proj' in name:
            end_d = 1024
        if 'mat_qkv' in name:
            end_d = 3072

        replacement = MLP_NAME_MAP[mlp_type](d,
                                             [attention_dim, end_d],
                                             activation=activation,
                                             bias=bias,
                                             dropout=dropout)
        set_module_by_name(model, name, replacement)
        logging.info(f'Substituting {name}') # with {mlp_type} MLP {replacement}')
    return model, modules_to_replace


def dsti_mlp_filter_condition(_model: nn.Module, m: nn.Module):
    if isinstance(m, (MLP, TorchvisionMLP, SubstitutionMLP, GPTMLP, BertLayer, GemmaMLP, FFN)):
        return True


def eligible_gelu_activation_filter(model: nn.Module, m: nn.Module):
    # TODO handle cases when functional variant is used instead
    m_name = get_module_name(model, m)
    parent_module = get_module_by_name(model, get_parent_module_name(m_name))
    if isinstance(m, (nn.GELU, PytorchGELUTanh)) and isinstance(parent_module, (
    MLP, TorchvisionMLP, SubstitutionMLP, GPTMLP, BertLayer, GemmaMLP, FFN)):
        return True


def mha_gelu_activation_filter(model: nn.Module, m: nn.Module):
    m_name = get_module_name(model, m)
    parent_m_name = get_parent_module_name(m_name)
    grandparent_m_name = get_parent_module_name(parent_m_name)
    parent_module = get_module_by_name(model, parent_m_name)
    grandparent_module = get_module_by_name(model, grandparent_m_name)
    # print("*"*50)
    # print(f"m_name: {m_name}, parent_m_name: {parent_m_name}, grandparent_m_name: {grandparent_m_name}")
    # print(f"m: {m}, parent_module: {parent_module}, grandparent_module: {grandparent_module}")
    # print(isinstance(m, (nn.GELU, PytorchGELUTanh)), isinstance(parent_module, (MLP, TorchvisionMLP, SubstitutionMLP, GPTMLP, BertLayer, GemmaMLP)), isinstance(grandparent_module, (CustomMultiheadAttention, SelfAttention))) 
    # print("*"*50)
    if isinstance(m, (nn.GELU, PytorchGELUTanh)) and isinstance(parent_module, (
    MLP, TorchvisionMLP, SubstitutionMLP, GPTMLP, BertLayer, GemmaMLP)) \
            and isinstance(grandparent_module, (CustomMultiheadAttention, SelfAttention)): # Jort: To make sure that SelfAttention can be added instead of CustomMultiheadAttention
        return True


def find_gelu_activations(model, apply_to):
    print(f"apply_to: {apply_to}")
    if apply_to == 'moe_eligible_only':
        acts_to_sparsify = find_module_names(model, eligible_gelu_activation_filter)
    elif apply_to == 'everywhere':
        acts_to_sparsify = find_module_names(model, lambda _, m: isinstance(m, nn.GELU))
    elif apply_to == 'mha_projections':
        acts_to_sparsify = find_module_names(model, mha_gelu_activation_filter)
    else:
        raise ValueError(f'Invalid apply_to argument value: {apply_to}')
    return acts_to_sparsify


def find_relu_activations(model):
    acts_to_sparsify = find_module_names(model, lambda _, m: isinstance(m, nn.ReLU))
    return acts_to_sparsify


def replace_with_relu(original_model, acts_to_replace):
    model = deepcopy(original_model)
    for act_m_name in acts_to_replace:
        logging.info(f'Replacing {act_m_name} with ReLU')
        set_module_by_name(model, act_m_name, nn.ReLU())
    return model
