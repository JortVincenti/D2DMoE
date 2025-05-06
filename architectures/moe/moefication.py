import logging
from copy import deepcopy
from functools import partial
from typing import List
import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from architectures.custom import CustomMultiheadAttention
from architectures.moe.dsti import ResidualMLP
from architectures.moe.moe_layers import MoELayer, MOE_IMPL_MAP, ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moe_models import moe_var_block_forward, moe_var_main_forward
from common import ACTIVATION_NAME_MAP
from utils import find_module_names, get_module_by_name, set_module_by_name
from architectures.pretrained import NullDDP
from architectures.basic_var import FFN
from architectures.var import VAR
import numpy as np
from torch.profiler import record_function
import torch.nn.functional as F
import time
import warnings

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

        # Compute and store centroids in self.centroids
        all_centroids = []
        #  [num_experts, in_features, neurons_per_expert].)
        for expert_idx in range(self.num_experts):
            # Extract the sub-tensor for this expert: shape [in_features, neurons_per_expert]
            w1_expert = self.experts.w1[expert_idx]
            # Re-normalize each column to unit length (dim=0 means "normalize across rows")
            w1_expert_norm = F.normalize(w1_expert, p=2, dim=0)

            # Compute centroid by taking the mean of columns => shape [in_features]
            centroid = w1_expert_norm.mean(dim=1)
            all_centroids.append(centroid)

        # Stack all expert centroids into [num_experts, in_features]
        self.centroids = torch.stack(all_centroids, dim=0).to('cuda')

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
            max_norms, _ = predicted_expert_norms.max(dim=-1, keepdim=True) #This is for the other 0.
            if isinstance(self.tau, list):
                # Original scale layout, smallest → largest.
                seq_len = x.size(1)                           # e.g. 525 after your split
                full_scale_sizes = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
                assert len(self.tau) == len(full_scale_sizes), "tau list must have 10 entries"

                device, dtype = max_norms.device, max_norms.dtype

                # --- build factors starting from the *last* scale and moving left ----------
                factors_tail = []
                remain       = seq_len

                for tau_val, gsize in zip(reversed(self.tau), reversed(full_scale_sizes)):
                    if remain == 0:
                        break                        # collected enough tokens
                    take = min(gsize, remain)        # gsize is bigger than remain only
                                                    # when seq_len cuts a scale in half
                    factors_tail.append(
                        torch.full((take,), 1.0 - tau_val, device=device, dtype=dtype)
                    )
                    remain -= take

                # factors_tail is built in reverse order – put it back in forward order
                factors_1d = torch.cat(list(reversed(factors_tail)), dim=0)   # length == L
                assert factors_1d.numel() == seq_len, "factor length mismatch!"

                factors = factors_1d.view(1, seq_len, 1) 
                norm_thresholds = max_norms * factors
            else:
                norm_thresholds = max_norms * (1.0 - self.tau)
            routing_tensor = torch.zeros_like(predicted_expert_norms)
            routing_tensor[predicted_expert_norms >= norm_thresholds] = 1.0
            self.routing_mask = routing_tensor.sum(dim=-1)
        elif self.forward_mode == 'oracle':            
            B, T, d = x.size()  # x is (B,T,d)
        
            # Flatten to (B*T, d)
            x_flat = x.view(B*T, d)
            
            # Expand so first dimension is 'e = number_of_experts': (E, B*T, d)
            e = self.num_experts
            x_expanded = x_flat.unsqueeze(0).expand(e, -1, -1)
            
            # Run forward_without_routing
            x_expert_out = self.experts.forward_without_routing(x_expanded)
            
            # norms => shape (E,B*T)
            norms = torch.linalg.vector_norm(x_expert_out, ord=2, dim=-1)
            # Reshape to (E,B,T)
            norms = norms.view(e, B, T)
            # Compute max norms => shape (1,B,T)
            max_norms, _ = norms.max(dim=0, keepdim=True)
            # norm_thresholds => shape (1,B,T)
            if isinstance(self.tau, list):
                # Original scale layout, smallest → largest.
                seq_len = x.size(1)                           # e.g. 525 after your split
                full_scale_sizes = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
                assert len(self.tau) == len(full_scale_sizes), "tau list must have 10 entries"

                device, dtype = max_norms.device, max_norms.dtype

                # --- build factors starting from the *last* scale and moving left ----------
                factors_tail = []
                remain       = seq_len

                for tau_val, gsize in zip(reversed(self.tau), reversed(full_scale_sizes)):
                    if remain == 0:
                        break                        # collected enough tokens
                    take = min(gsize, remain)        # gsize is bigger than remain only
                                                    # when seq_len cuts a scale in half
                    factors_tail.append(
                        torch.full((take,), 1.0 - tau_val, device=device, dtype=dtype)
                    )
                    remain -= take

                # factors_tail is built in reverse order – put it back in forward order
                factors_1d = torch.cat(list(reversed(factors_tail)), dim=0)   # length == L
                assert factors_1d.numel() == seq_len, "factor length mismatch!"

                factors = factors_1d.view(1, 1, seq_len)      # [1, 1, L] for broadcasting
                print(factors.shape)
                print(max_norms.shape)
                norm_thresholds = max_norms * factors
            else:
                norm_thresholds = max_norms * (1.0 - self.tau)

            # new_routing => shape (E,B,T)
            new_routing = torch.zeros_like(norms)
            new_routing[norms >= norm_thresholds] = 1.0

            # Permute => (B,T,E)
            routing_tensor = new_routing.permute(1, 2, 0)
        elif self.forward_mode == 'centroids':
            """
            For each token in x:
            1) Compute dot product similarity with each expert's centroid (cosine or raw).
            2) Take top-k in descending order.
            3) Compute cumulative sum of those k values.
            4) Keep only the subset whose cumsum < ratio * total_sum. 
                (Ensure at least 1 expert is chosen.)
            """
            # x shape: (B, T, D)
            B, T, D = x.size()

            # 1) [Optional] L2 normalize x and centroids (if you want cosine similarity)
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-9)  # (B, T, D)
            centroids_norm = self.centroids / (self.centroids.norm(dim=-1, keepdim=True) + 1e-9)  # (num_experts, D)
            # 2) Compute similarity => (B, T, num_experts)
            similarity = torch.einsum('btd,ed->bte', x_norm, centroids_norm)

            # 2) shift so that the minimum per-token is zero
            sim_min = similarity.min(dim=-1, keepdim=True)[0]      # (B, T, 1)
            similarity = similarity - sim_min                   # now ∨ similarity_pos >= 0

            # 3) top-k retrieval in descending order (here using all experts)
            topk_values, topk_indices = torch.topk(similarity, k=self.num_experts, dim=-1, largest=True, sorted=True)
            #print("[DEBUG] Token[0,0] topk values:", topk_values[0, 0, :].detach().cpu().numpy())

            # 4) Summation and cumulative sum over top-k values
            sum_scores = topk_values.sum(dim=-1, keepdim=True)  # (B, T, 1)
            cumsum_scores = torch.cumsum(topk_values, dim=-1)    # (B, T, k)
            threshold = self.tau * sum_scores                     # (B, T, 1)
            # print("[DEBUG] Token[0,0] total score:", sum_scores[0, 0, 0].item())
            # print("[DEBUG] Token[0,0] cumulative topk scores:", cumsum_scores[0, 0, :].detach().cpu().numpy())
            # print("[DEBUG] Token[0,0] threshold (self.tau * total):", threshold[0, 0, 0].item())

            # 5) Build a boolean mask to decide which experts to keep
            keep_mask = (cumsum_scores <= threshold)
            # Guarantee at least 1 expert remains per token:
            first_expert_mask = torch.zeros_like(keep_mask)
            first_expert_mask[..., 0] = True  # force the top expert to always be kept
            keep_mask = keep_mask | first_expert_mask
            #print("[DEBUG] Token[0,0] keep_mask (boolean):", keep_mask[0, 0, :].detach().cpu().numpy())
            #print("[DEBUG] Token[0,0] keep_mask (as int):", keep_mask[0, 0, :].int().detach().cpu().numpy())
            # Also, print for a few tokens (if desired) the percentage of experts kept:
            keep_fraction = keep_mask.float().mean(dim=-1)
            #print("[DEBUG] Average keep_mask fraction per token (B x T):", keep_fraction.detach().cpu().numpy())

            # 6) Create routing_tensor and scatter 1.0 into the kept positions
            routing_tensor = torch.zeros_like(similarity)  # (B, T, num_experts)
            keep_mask_float = keep_mask.type_as(routing_tensor)  # ensure same dtype
            routing_tensor.scatter_(2, topk_indices, keep_mask_float)
            num_selected_per_token = routing_tensor.sum(dim=-1)
        elif self.forward_mode == 'random':
            # Assume x has shape (B, T, d)
            B, T, d = x.size()  
            e = self.num_experts  # number of experts
            # Instead of computing x_expert_out = self.experts.forward_without_routing(x_expanded)
            # and thresholding the norms, we generate a random routing tensor with 1's with probability tau
            scores = torch.rand(B, T, e, device=x.device)
            # 2) pick the k smallest (or largest) scores per token
            k = max(1, int(e * self.tau))    # ensure at least one expert
            topk = scores.topk(k, dim=-1, largest=False).indices   # (B, T, k)

            # 3) build your binary routing mask
            routing_tensor = torch.zeros_like(scores, dtype=torch.float)
            routing_tensor.scatter_(-1, topk, 1.0)       
        else:
            raise ValueError(f'Unsupported forward_mode: {self.forward_mode}')
        return routing_tensor
    
    def forward(self, x):
        routing_tensor = self.gate(x)
        
        orig_size = x.size()
        x = x.reshape(-1, orig_size[-1])   

        with record_function("FFN_Experts"):
            out = self.experts(x, routing_tensor.reshape(-1, routing_tensor.size(-1)))

        if self.bias:
            out = out + self.last_bias
        if self.add_residual_connection:
            out = out + x
        out = out.view(orig_size)
        return out, self.expert_dim * routing_tensor.sum()


def replace_layer_with_moe(model, moefied_module_name, num_experts=None, expert_size=None,
                           experts_class='module'):
    original_module = get_module_by_name(model, moefied_module_name)

    if isinstance(original_module, nn.Sequential):
        # ffn is a nn.Sequential
        # with nn.Linear layers at indices 0 and 3
        w1 = original_module[0]
        activation = type(original_module[1])
    elif isinstance(original_module, FFN):
        w1 = original_module.fc1
        activation = type(original_module.act)
    else:
        raise ValueError(f'Unsupported ffn type: {type(original_module)}')
    add_residual = True if isinstance(original_module, ResidualMLP) else False
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
                               add_intermediate_gating=False)
    logging.info(f'Replacing {moefied_module_name} (FFN hidden size {d_ff}) with {num_experts} experts')
    set_module_by_name(model, moefied_module_name, moe_layer)


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

    # If instance of VAR or wrapped VAR
    if isinstance(model, NullDDP) or isinstance(getattr(model, 'module', None), NullDDP):
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

    if isinstance(moe_layer.experts, ExecuteAllExperts):
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
    else:
        # TODO
        raise NotImplementedError('Other variants not handled yet')


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
        
        # Iterate through each layer, printing intermediate outputs and (optionally) layer weights
        for idx, layer in enumerate(self.layers):
            x = layer(x)

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