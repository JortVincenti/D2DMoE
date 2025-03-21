import logging
from copy import deepcopy
from functools import partial
from typing import List

import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from torchvision.models import VisionTransformer
from transformers import BertPreTrainedModel, BertForSequenceClassification
from transformers.models.bert import BertLayer
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.custom import CustomMultiheadAttention
from architectures.moe.dsti import ResidualMLP
from architectures.moe.moe_layers import MoELayer, MOE_IMPL_MAP, ModuleBatchedExperts, ExecuteAllExperts, \
    CustomKernelExperts
from architectures.moe.moe_models import moe_vit_block_forward, moe_vit_encoder_forward, moe_vit_main_forward, \
    moe_attention_forward, moe_gpt_block_forward, moe_gpt_main_forward, moe_bert_layer_forward, moe_bert_main_forward, \
    moe_gemma_main_forward, moe_gemma_decoder_forward, moe_var_block_forward, moe_var_main_forward
from architectures.nlp import GemmaWrapper
from architectures.vit import VisionTransformer as CustomVisionTransformer
from architectures.gpt import GPT, MLP as GPTMLP
from common import ACTIVATION_NAME_MAP
from utils import find_module_names, get_module_by_name, set_module_by_name
from architectures.pretrained import NullDDP
from architectures.basic_var import FFN
from architectures.var import VAR
import numpy as np


class MoeficationMoE(MoELayer):
    # https://arxiv.org/pdf/2110.01786.pdf
    def __init__(self, name, hidden_dim, num_experts, bias, expert_dim, activation, experts_class='module',
                 add_residual_connection=False, add_intermediate_gating=False):
        super().__init__(name, num_experts)
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.k = num_experts
        self.bias = bias
        # instantiate experts
        self.experts = MOE_IMPL_MAP[experts_class](dim=hidden_dim,
                                                   num_experts=num_experts,
                                                   depth=2,
                                                   expert_dim=self.expert_dim,
                                                   bias=False if not bias else 'without_last',
                                                   activation=activation,
                                                   intermediate_gating=add_intermediate_gating)
        if self.bias:
            self.last_bias = nn.Parameter(torch.zeros(1, hidden_dim))
        self.forward_mode = 'all'
        self.router = None
        self.k = None
        self.tau = None
        self.add_residual_connection = add_residual_connection
        self.routing_mask = None

    def gate(self, x):
        # x is of size (batch_size, sequence_length, dim)
        if self.forward_mode == 'all':
            # sanity check - route to all tensors when router is not present
            routing_tensor = torch.ones(x.size(0), x.size(1), self.num_experts, dtype=x.dtype, device=x.device)
        elif self.forward_mode == 'topk':
            routing_tensor = self.router(x)
            top = torch.topk(routing_tensor, k=self.k, dim=-1)[1]
            routing_tensor = torch.zeros_like(routing_tensor)
            routing_tensor.scatter_(2, top, 1.0)
        elif self.forward_mode == 'dynk':
            predicted_expert_norms = self.router(x)
            # compute expert proportional contributions
            total_norm = predicted_expert_norms.sum(dim=-1, keepdim=True)
            expert_shares = predicted_expert_norms / total_norm
            # we are interested in cumulative contribution of "selected" experts
            # staring with those with the highest contribution
            sorted_expert_shares, expert_share_indices = torch.sort(expert_shares, dim=-1, descending=True)
            cumulative_expert_shares = torch.cumsum(sorted_expert_shares, dim=-1)
            cumulative_mask = cumulative_expert_shares < self.tau
            # one more expert is needed to actually cross the threshold
            # and at least one expert must be executed
            cumulative_mask = cumulative_mask.roll(shifts=1, dims=-1)
            cumulative_mask[..., 0] = True
            # map the selected sorted experts to routing tensor
            routing_tensor = cumulative_mask.gather(dim=-1, index=expert_share_indices.argsort(dim=-1)).to(x.dtype)
        elif self.forward_mode == 'dynk_max':
            predicted_expert_norms = self.router(x)
            max_norms, _ = predicted_expert_norms.max(dim=-1, keepdim=True)
            norm_thresholds = max_norms * (1.0 - self.tau)
            routing_tensor = torch.zeros_like(predicted_expert_norms)
            routing_tensor[predicted_expert_norms >= norm_thresholds] = 1.0
            self.routing_mask = routing_tensor.sum(dim=-1)
        elif self.forward_mode == 'oracle':            
            # Save original shape (ideally [2, 1, 1024])
            orig_size = x.size()  
            # x is currently [2, 1024]. If you intended [2,1,1024], make sure x has that shape.
            #print('x oracle', x.shape)  # Expected: [2, 1024] if orig_size was [2,1024]
            x = x.view(-1, x.size(-1))
            assert x.dim() == 2, f'{x.size()=}'
            x = x.unsqueeze(0)
            
            if isinstance(self.experts, CustomKernelExperts):
                x = self.experts.forward_without_routing(x)
            else:
                for layers in self.experts.layers:
                    x = layers(x)

            # Suppose B = input.size(0), T = input.size(1)
            e = x.size(0)
            B = orig_size[0]
            T = orig_size[1]

            norms = torch.linalg.vector_norm(x, ord=2, dim=-1)
            max_norms = norms.view(e, B, T).permute(1, 2, 0)  # => shape (B, T, e)
            max_norms, _ = norms.max(dim=-1, keepdim=True)
            norm_thresholds = max_norms * (1.0 - self.tau)
            new_routing = torch.zeros_like(norms)
            new_routing[norms >= norm_thresholds] = 1.0
            new_routing = new_routing.transpose(0, 1) 
            routing_tensor = new_routing
            self.routing_mask = new_routing.sum(dim=0)
        else:
            raise ValueError(f'Unsupported forward_mode: {self.forward_mode}')

        # print('routing_tensor', routing_tensor.shape)
        # print('predicted_expert_norms', predicted_expert_norms.shape)
        # print('predicted_expert_norms[0]', predicted_expert_norms[0])
        # print('max_norms', max_norms.shape)
        # print('max_norms[0]', max_norms[0])
        # print('routing_tensor', routing_tensor.shape)
        return routing_tensor
    
    def forward(self, x):
        # x is of size (batch_size, sequence_length, dim)
        routing_tensor = self.gate(x)

        # Check if all tensors are equal to 1
        #assert routing_tensor.eq(1).all(), "Routing tensor is not equal to 1"
        
        orig_size = x.size()
        x = x.view(-1, x.size(-1))
        #print('x', x.shape)
        #print(' routing_tensor.view(-1, routing_tensor.size(-1)',  routing_tensor.view(-1, routing_tensor.size(-1)).shape)
        out = self.experts(x, routing_tensor.view(-1, routing_tensor.size(-1)))
        #print('out true experts', out.shape)

        if self.bias:
            out = out + self.last_bias
        if self.add_residual_connection:
            out = out + x
        out = out.view(orig_size)
        #print('Bias Last', self.last_bias.sum())
        return out, {self.name: (routing_tensor,)}


def replace_layer_with_moe(model, moefied_module_name, num_experts=None, expert_size=None,
                           experts_class='module'):
    original_module = get_module_by_name(model, moefied_module_name)

    if isinstance(original_module, nn.Sequential):
        # ffn is a nn.Sequential
        # with nn.Linear layers at indices 0 and 3
        w1 = original_module[0]
        activation = type(original_module[1])
    elif isinstance(original_module, GPTMLP):
        w1 = original_module.c_fc
        activation = type(original_module.act)
    elif isinstance(original_module, GemmaMLP):
        w1 = original_module.up_proj
        activation = type(original_module.act_fn)
    elif isinstance(original_module, BertLayer):
        w1 = original_module.intermediate.dense
        activation = type(original_module.intermediate.intermediate_act_fn)
        moefied_intermediate_name = f'{moefied_module_name}.intermediate'
        moefied_output_name = f'{moefied_module_name}.output'
        org_module_name = moefied_module_name
        moefied_module_name = f'{moefied_module_name}.mlp'
    elif isinstance(original_module, FFN):
        w1 = original_module.fc1
        activation = type(original_module.act)
    else:
        raise ValueError(f'Unsupported ffn type: {type(original_module)}')
    add_residual = True if isinstance(original_module, ResidualMLP) else False
    add_intermediate_gating = True if isinstance(original_module, GemmaMLP) else False

    hidden_dim = w1.in_features
    d_ff = w1.out_features
    if num_experts is not None:
        assert d_ff % num_experts == 0, f'd_ff has to be divisible by the number of experts'
        expert_size = d_ff // num_experts
    elif expert_size is not None:
        assert d_ff % expert_size == 0, f'd_ff has to be divisible by the expert size'
        num_experts = d_ff // expert_size
    moe_layer = MoeficationMoE(moefied_module_name, hidden_dim, num_experts, w1.bias is not None, expert_size,
                               activation, experts_class, add_residual_connection=add_residual,
                               add_intermediate_gating=add_intermediate_gating)
    logging.info(f'Replacing {moefied_module_name} (FFN hidden size {d_ff}) with {num_experts} experts')
    set_module_by_name(model, moefied_module_name, moe_layer)
    if isinstance(model, GPT):
        # TODO is there a better option to add dropout? MoefiedMLP does not have dropout,
        #  but GPT MLP has it inside the module
        dropout_layer_name = f'{moefied_module_name}.dropout'
        set_module_by_name(model, dropout_layer_name, torch.nn.Dropout())
    if isinstance(model, BertPreTrainedModel):
        # TODO should we delete output layer?
        # logging.info(f'Deleting original FFN modules...')
        set_module_by_name(model, org_module_name + '.dropout', original_module.output.dropout)
        set_module_by_name(model, org_module_name + '.ln', original_module.output.LayerNorm)
        set_module_by_name(model, moefied_intermediate_name, None)
        set_module_by_name(model, moefied_output_name, None)


def replace_with_moes(original_model: nn.Module, num_experts: int = None, expert_size: int = None,
                      experts_class='module', module_filter_contition=None):
    assert num_experts is not None or expert_size is not None, f'either num_experts or expert_size must be passed'
    assert not (num_experts is not None and expert_size is not None), \
        f'num_experts and expert_size cannot be both passed'
    original_model.eval()
    model = deepcopy(original_model)
    modules_to_moeify = find_module_names(original_model, module_filter_contition)
    # calculate size and replace the selected layers with MoE layers
    for name in modules_to_moeify:
        replace_layer_with_moe(model, name, num_experts, expert_size, experts_class)

    # replace forwards so that gating data is also returned
    if isinstance(model, (VisionTransformer, CustomVisionTransformer)):
        for i in range(len(model.encoder.layers)):
            model.encoder.layers[i].forward = partial(moe_vit_block_forward, model.encoder.layers[i])
            if isinstance(model.encoder.layers[i].self_attention, CustomMultiheadAttention):
                model.encoder.layers[i].self_attention.forward = partial(moe_attention_forward,
                                                                         model.encoder.layers[i].self_attention)
        model.encoder.forward = partial(moe_vit_encoder_forward, model.encoder)
        model.forward = partial(moe_vit_main_forward, model)
    elif isinstance(model, GPT):
        for i in range(len(model.transformer.h)):
            model.transformer.h[i].forward = partial(moe_gpt_block_forward, model.transformer.h[i])
            if isinstance(model.transformer.h[i].attn, CustomMultiheadAttention):
                model.transformer.h[i].attn.forward = partial(moe_attention_forward,
                                                               model.transformer.h[i].attn)
        model.forward = partial(moe_gpt_main_forward, model)
    elif isinstance(model, BertPreTrainedModel):
        for i in range(len(model.bert.encoder.layer)):
            model.bert.encoder.layer[i].forward = partial(moe_bert_layer_forward, model.bert.encoder.layer[i])
            if isinstance(model.bert.encoder.layer[i].attention, CustomMultiheadAttention):
                # TODO this will explode for sure
                model.bert.encoder.layer[i].attention.forward = partial(moe_attention_forward,
                                                                        model.bert.encoder.layer[i].attention)
                raise NotImplementedError()
            model.forward = partial(moe_bert_main_forward, model)
    elif isinstance(model, GemmaWrapper):
        for i in range(len(model.gemma.model.layers)):
            model.gemma.model.layers[i].forward = partial(moe_gemma_decoder_forward, model.gemma.model.layers[i])
            if isinstance(model.gemma.model.layers[i].self_attn, CustomMultiheadAttention):
                raise NotImplementedError('Custom attention not supported with Gemma')
        model.gemma.forward = partial(moe_gemma_main_forward, model.gemma)
    elif isinstance(model, NullDDP) or isinstance(getattr(model, 'module', None), NullDDP):
        model = model.module
        for i in range(len(model.blocks)):
            model.blocks[i].forward = partial(moe_var_block_forward, model.blocks[i])
            # if isinstance(model.blocks[i].attn, CustomMultiheadAttention): # TODO
            #     model.blocks[i].attn.forward = partial(moe_attention_forward, model.blocks[i].attn)
        model.forward = partial(moe_var_main_forward, model)
    elif isinstance(model, VAR):
        for i in range(len(model.blocks)):
            model.blocks[i].forward = partial(moe_var_block_forward, model.blocks[i])
        model.forward = partial(moe_var_main_forward, model)
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')

    return model, modules_to_moeify


def param_clustering_split(ffn, moe_layer):
    num_experts = moe_layer.num_experts
    # ffn is a nn.Sequential
    # with nn.Linear layers at indices 0 and 3
    if isinstance(ffn, nn.Sequential):
        w1 = ffn[0]
        w2 = ffn[3]
    elif isinstance(ffn, GPTMLP):
        w1 = ffn.c_fc
        w2 = ffn.c_proj
    elif isinstance(ffn, BertLayer):
        w1 = ffn.intermediate.dense
        w2 = ffn.output.dense
    elif isinstance(ffn, GemmaMLP):
        w1 = ffn.gate_proj # We cluster by gate projection weights because they are actually sparse
        w2 = ffn.down_proj
        w3 = ffn.up_proj
    elif isinstance(ffn, FFN):
        w1 = ffn.fc1
        w2 = ffn.fc2
    else:
        raise ValueError(f'Unsupported ffn type: {type(ffn)}')

    hidden_dim = w1.in_features
    d_ff = w1.out_features
    expert_size = d_ff // num_experts
    w1_normalized = nn.functional.normalize(w1.weight, p=2.0, dim=1)
    #
    labels = KMeansConstrained(n_clusters=num_experts, size_min=expert_size, size_max=expert_size) \
        .fit_predict(w1_normalized.detach().cpu().numpy())

    # split weights into experts by labels
    if isinstance(moe_layer.experts, ModuleBatchedExperts):
        if isinstance(ffn, GemmaMLP):
            raise NotImplementedError()
        assert moe_layer.experts.depth == 2
        # experts is a nn.ModuleList
        # each expert is a nn.Sequential module
        # with nn.Linear layers at indices 0 and 2
        with torch.no_grad():
            filled_neuron_counts = [0 for _ in range(num_experts)]
            for neuron_index, expert_index in enumerate(labels):
                expert_neuron_index = filled_neuron_counts[expert_index]
                moe_layer.experts.e[expert_index][0].weight[expert_neuron_index].copy_(w1.weight[neuron_index])
                if moe_layer.bias:
                    moe_layer.experts.e[expert_index][0].bias[expert_neuron_index].copy_(w1.bias[neuron_index])
                moe_layer.experts.e[expert_index][2].weight[:, expert_neuron_index].copy_(
                    w2.weight[:, neuron_index])
                filled_neuron_counts[expert_index] += 1
            # copy the last layer bias
            if moe_layer.bias:
                moe_layer.last_bias.copy_(w2.bias)
    elif isinstance(moe_layer.experts, ExecuteAllExperts):
        assert moe_layer.experts.depth == 2
        with torch.no_grad():
            filled_neuron_counts = [0 for _ in range(num_experts)]
            for neuron_index, expert_index in enumerate(labels):
                expert_neuron_index = filled_neuron_counts[expert_index]
                moe_layer.experts.layers[0].w[expert_index, :, expert_neuron_index].copy_(w1.weight[neuron_index])
                if moe_layer.bias:
                    moe_layer.experts.layers[0].b[expert_index, :, expert_neuron_index].copy_(w1.bias[neuron_index])
                moe_layer.experts.layers[1].w[expert_index, expert_neuron_index].copy_(w2.weight[:, neuron_index])
               
                filled_neuron_counts[expert_index] += 1
            # copy the last layer bias
            if moe_layer.bias:
                moe_layer.last_bias.copy_(w2.bias)
        
    elif isinstance(moe_layer.experts, CustomKernelExperts):
        assert moe_layer.experts.depth == 2
        with torch.no_grad():
            filled_neuron_counts = [0 for _ in range(num_experts)]
            for neuron_index, expert_index in enumerate(labels):
                expert_neuron_index = filled_neuron_counts[expert_index]
                moe_layer.experts.w1[expert_index, :, expert_neuron_index].copy_(ffn.fc1.weight[neuron_index])
                if moe_layer.bias:
                    moe_layer.experts.b1[expert_index, expert_neuron_index].copy_(ffn.fc1.bias[neuron_index])
                moe_layer.experts.w2[expert_index, expert_neuron_index].copy_(ffn.fc2.weight[:, neuron_index])
                filled_neuron_counts[expert_index] += 1
            # copy the last layer bias
            if moe_layer.bias:
                moe_layer.last_bias.copy_(ffn.fc2.bias)

            # print('Types')
            # print('ffn.fc1.bias', ffn.fc1.bias.dtype)
            # print('ffn.fc1.weight', ffn.fc1.weight.dtype)
            # print('ffn.fc2.bias', ffn.fc2.bias.dtype)
            # print('ffn.fc2.weight', ffn.fc2.weight.dtype)
            # print('-'*100)
            # print('oe_layer.experts.w1[expert_index, :, expert_neuron_index]', moe_layer.experts.w1[expert_index, :, expert_neuron_index].dtype)
            # print('moe_layer.experts.b1[expert_index, expert_neuron_index]', moe_layer.experts.b1[expert_index, expert_neuron_index].dtype)
            # print('moe_layer.experts.w2[expert_index, expert_neuron_index].', moe_layer.experts.w2[expert_index, expert_neuron_index].dtype)
            # print('moe_layer.last_bias', moe_layer.last_bias.dtype)

            
            
    else:
        # TODO
        raise NotImplementedError('Other variants not handled yet')


# def param_clustering_split(ffn, moe_layer):
#     """
#     This version does NOT do any clustering. Instead, it partitions the
#     FFN's hidden dimension contiguously across num_experts. The result:
#     if you route to all experts, the output is exactly the same as the
#     original ffn.

#     ffn is expected to have two layers:
#       w1: first linear  (in_features -> out_features)
#       w2: second linear (out_features -> ???)
#     moe_layer is expected to have:
#       moe_layer.num_experts = number of experts
#       moe_layer.experts = ExecuteAllExperts with depth == 2
#       (like in your snippet)
#     """

#     num_experts = moe_layer.num_experts

#     # 1) Identify the first & second linear layers in `ffn`
#     if isinstance(ffn, nn.Sequential):
#         # Suppose ffn[0] is the first linear, ffn[3] is the second
#         w1 = ffn[0]  # nn.Linear
#         w2 = ffn[3]  # nn.Linear
#     elif hasattr(ffn, "c_fc") and hasattr(ffn, "c_proj"):  # GPT style
#         w1 = ffn.c_fc   # first linear
#         w2 = ffn.c_proj # second linear
#     elif hasattr(ffn, "intermediate") and hasattr(ffn, "output"):  # BERT style
#         w1 = ffn.intermediate.dense
#         w2 = ffn.output.dense
#     elif hasattr(ffn, "fc1") and hasattr(ffn, "fc2"):      # Basic FFN
#         w1 = ffn.fc1
#         w2 = ffn.fc2
#     else:
#         raise ValueError(f'Unsupported ffn type: {type(ffn)}')

#     # 2) Set up dimensions & chunking
#     hidden_dim = w1.in_features
#     d_ff = w1.out_features       # total # of neurons in the first linear
#     expert_size = d_ff // num_experts
#     # We expect (d_ff % num_experts == 0) for a perfect split

#     # Instead of K-Means, we do a simple chunk-based assignment:
#     # e.g. if d_ff=1024, num_experts=16 => each chunk is size 64
#     # chunk 0 => neurons [0..63], chunk 1 => [64..127], etc.
#     labels = np.repeat(np.arange(num_experts), expert_size)  # shape = (d_ff,)

#     if isinstance(moe_layer.experts, ExecuteAllExperts):
#         assert moe_layer.experts.depth == 2, "We only handle 2-layer MoE modules"
#         with torch.no_grad():
#             filled_neuron_counts = [0 for _ in range(num_experts)]

#             # Copy row i of w1 -> row in the assigned expert
#             for neuron_index, expert_index in enumerate(labels):
#                 expert_neuron_index = filled_neuron_counts[expert_index]

#                 # Move w1.weight[i,:] -> first layer of this expert
#                 # shape of moe_layer.experts.layers[0].w is [num_experts, in_features, expert_size]
#                 # so we do   [expert_index, :, expert_neuron_index]
#                 moe_layer.experts.layers[0].w[expert_index, :, expert_neuron_index].copy_(
#                     w1.weight[neuron_index]
#                 )
#                 if moe_layer.bias:
#                     moe_layer.experts.layers[0].b[expert_index, :, expert_neuron_index].copy_(
#                         w1.bias[neuron_index]
#                     )

#                 # Move w2.weight[:, i] -> second layer of this expert
#                 # shape of moe_layer.experts.layers[1].w is [num_experts, expert_size, out_dim]
#                 # but the snippet in your code used shape [num_experts, expert_size], or [num_experts, expert_size, ???]
#                 # We'll match your snippet: layers[1].w[expert_index, expert_neuron_index].copy_(...)
#                 # meaning it expects w2.weight has shape [out_dim, d_ff].
#                 # The original snippet does: w2.weight[:, neuron_index]
#                 # => we place that in [expert_index, expert_neuron_index].
#                 moe_layer.experts.layers[1].w[expert_index, expert_neuron_index].copy_(
#                     w2.weight[:, neuron_index]
#                 )

#                 filled_neuron_counts[expert_index] += 1

#             # copy the last layer bias if present
#             if moe_layer.bias:
#                 moe_layer.last_bias.copy_(w2.bias)
#     else:
#         raise NotImplementedError("Only handle ExecuteAllExperts with depth=2 for now")

#     # Now, if the MoE forward pass sums across ALL experts, the
#     # result is identical to old ffn(x) => w2( w1(x) ) as long as
#     # all sub-layers are used + no gating differences + no additional
#     # normalization or transformations are introduced.



def split_original_parameters(original_model: nn.Module, moe_model: nn.Module, replaced_module_names: List[str]):
    
    if hasattr(original_model, "module"):  # Only access module if it exists
        original_model = original_model.module  # Extract VAR model
        for index, name in enumerate(replaced_module_names):
            new_name =  name.replace("module.", "")
            replaced_module_names[index] = new_name

    original_model.eval()
    assert len(replaced_module_names) > 0
    # calculate size and replace the selected layers with MoE layers
    for i, name in enumerate(replaced_module_names):
        original_module = get_module_by_name(original_model, name)
        moe_module = get_module_by_name(moe_model, name)
        num_experts = moe_module.num_experts
        logging.info(f'Clustering parameters from {name} into {num_experts} experts')
        param_clustering_split(original_module, moe_module)


# class MoeficationRouter(nn.Module):
#     def __init__(self, hidden_dim, num_experts, width=128, depth=2, bias=False, activation='tanh',
#                  output_activation='identity', test_override_outputs_p=None):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         if depth == 1:
#             self.layers.append(nn.Linear(hidden_dim, num_experts, bias=bias))
#         else:
#             self.layers.append(nn.Linear(hidden_dim, width, bias=bias))
#             self.layers.append(ACTIVATION_NAME_MAP[activation]())
#             for i in range(depth - 2):
#                 self.layers.append(nn.Linear(width, width, bias=bias))
#                 self.layers.append(ACTIVATION_NAME_MAP[activation]())
#             self.layers.append(nn.Linear(width, num_experts, bias=bias))
#         self.layers.append(ACTIVATION_NAME_MAP[output_activation]())
#         self.test_override_outputs_p = test_override_outputs_p
#         if test_override_outputs_p is not None:
#             self.response_cache = {}

class MoeficationRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, width=128, depth=2, bias=False, activation='tanh',
                 output_activation='identity', test_override_outputs_p=None):
        super().__init__()
        self.layers = nn.ModuleList()

        if depth == 1:
            self.layers.append(nn.Linear(hidden_dim, num_experts, bias=bias))
        else:
            self.layers.append(nn.Linear(hidden_dim, width, bias=bias))
            self.layers.append(ACTIVATION_NAME_MAP[activation]())

            for i in range(depth - 2):
                self.layers.append(nn.Linear(width, width, bias=bias))
                self.layers.append(ACTIVATION_NAME_MAP[activation]())

            self.layers.append(nn.Linear(width, num_experts, bias=bias))
        
        self.layers.append(ACTIVATION_NAME_MAP[output_activation]())
        self.test_override_outputs_p = test_override_outputs_p
        if test_override_outputs_p is not None:
            self.response_cache = {}

        # Initialize weights properly
        self._initialize_weights(activation)

    def _initialize_weights(self, activation):
        """ Proper weight initialization depending on activation type """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if activation in ['relu', 'leaky_relu', 'gelu']:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif activation in ['tanh', 'sigmoid']:
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')  # Default
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Print initialization info for Linear layers
        # for idx, layer in enumerate(self.layers):
        #     if isinstance(layer, nn.Linear):
        #         print(f"Initialized layer {idx} (Linear):")
        #         print(f"  Weight: min: {layer.weight.data.min().item():.4f}, max: {layer.weight.data.max().item():.4f}, "
        #               f"mean: {layer.weight.data.mean().item():.4f}, std: {layer.weight.data.std().item():.4f}")
        #         if layer.bias is not None:
        #             print(f"  Bias: min: {layer.bias.data.min().item():.4f}, max: {layer.bias.data.max().item():.4f}, "
        #                   f"mean: {layer.bias.data.mean().item():.4f}, std: {layer.bias.data.std().item():.4f}")

        # Print input statistics
        # print("Router input:")
        # print(f"  shape: {x.shape}")
        # print(f"  min: {x.min().item():.4f}, max: {x.max().item():.4f}, "
        #     f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        
        # Iterate through each layer, printing intermediate outputs and (optionally) layer weights
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"After layer {idx} ({layer.__class__.__name__}):")
            # print(f"  shape: {x.shape}")
            # print(f"  min: {x.min().item():.4f}, max: {x.max().item():.4f}, "
            #     f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
            
            # # Optionally, print layer parameters if it is a Linear layer:
            # if isinstance(layer, nn.Linear):
            #     weight = layer.weight.data
            #     bias = layer.bias.data if layer.bias is not None else None
            #     print(f"  Weight: min: {weight.min().item():.4f}, max: {weight.max().item():.4f}, "
            #         f"mean: {weight.mean().item():.4f}, std: {weight.std().item():.4f}")
            #     if bias is not None:
            #         print(f"  Bias: min: {bias.min().item():.4f}, max: {bias.max().item():.4f}, "
            #             f"mean: {bias.mean().item():.4f}, std: {bias.std().item():.4f}")
        
        # Print final output statistics
        # print("Final router output:")
        # print(f"  shape: {x.shape}")
        # print(f"  min: {x.min().item():.4f}, max: {x.max().item():.4f}, "
        #     f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        # num_zeros = (x == 0).sum().item()
        # total = x.numel()
        # print(f"  zeros: {num_zeros}/{total} ({(num_zeros/total)*100:.2f}%)")
        
        
        # for l in self.layers:
        #     x = l(x)

        if self.test_override_outputs_p is not None: # Jort: This is not used
            cache_key = (x.size(0), x.size(1))
            if cache_key in self.response_cache:
                x = self.response_cache[cache_key]
            else:
                with torch.no_grad():
                    x.bernoulli_(self.test_override_outputs_p)
                    self.response_cache[cache_key] = x

        return x


def add_routers(model, router_args):
    moe_module_names = find_module_names(model, lambda _, m: isinstance(m, MoeficationMoE))
    moe_module_dict = {}
    # create gating networks
    for moe_module_name in moe_module_names:
        moe_module = get_module_by_name(model, moe_module_name)
        logging.info(f'Adding router to {moe_module_name}')
        moe_module.router = MoeficationRouter(moe_module.hidden_dim,
                                              moe_module.num_experts,
                                              **router_args)
        moe_module_dict[moe_module_name] = moe_module
    assert len(moe_module_dict) > 0
    return moe_module_dict