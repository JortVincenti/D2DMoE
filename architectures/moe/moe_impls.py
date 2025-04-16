import torch
import triton
from triton import language as tl

from utils import row_major, column_major, grouped, leaky_relu, relu, tanh, gelu, config_grid

class MoeFirstLayerImplementation(torch.autograd.Function):
    seen_shapes = set()

    # These are the sample_dim values observed and their corresponding best configs.
    BEST_CONFIGS_FROM_LOGS = {
        8 : {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 128, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 128, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        16: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 64, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        32: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 128, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        64: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},

            6*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            6*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            6*576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            6*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            6*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            6*2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            6*4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6*6400:  {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            6*10816: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            6*16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        128: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},

                        
            4*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            4*6400:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*10816: {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*16384: {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        256: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},


            4*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            4*6400:  {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*10816: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },
    }

    def get_best_config(sample_dim: int, expert_dim: int):
        """
        Return the best config from BEST_CONFIGS_FROM_LOGS using a simple threshold lookup.
        If sample_dim is less than or equal to the smallest key, return that config.
        Otherwise, iterate and return the first config where sample_dim is below the threshold.
        If sample_dim is larger than any key, return the last config.
        """
        if expert_dim not in MoeFirstLayerImplementation.BEST_CONFIGS_FROM_LOGS:
            raise ValueError(f"No configs found for expert_dim={expert_dim}")

        expert_configs = MoeFirstLayerImplementation.BEST_CONFIGS_FROM_LOGS[expert_dim]
        if sample_dim not in expert_configs:
            raise ValueError(f"No config found for sample_dim={sample_dim} under expert_dim={expert_dim}")
        
        return expert_configs[sample_dim]

    
    @staticmethod
    def forward(input, weight, bias, sort_indices, expert_bincounts, activation_str='relu'):
        # If weight is 2D, unsqueeze to make it 3D
        if weight.ndim == 2:
            weight = weight.unsqueeze(-1)
        # If sort_indices is 1D, unsqueeze it to make it 2D
        if sort_indices.ndim == 1:
            sort_indices = sort_indices.unsqueeze(1)

        sample_dim = input.size(0)
        num_experts = weight.size(0)
        hidden_dim  = weight.size(1)
        expert_dim  = weight.size(2)

        # Map the activation string to an integer code
        ACTIVATION_MAP = {'relu': 0, 'tanh': 1, 'gelu': 2, 'leaky_relu': 3}
        activation_code = ACTIVATION_MAP.get(activation_str, 0)

        # Get best config from table
        config = MoeFirstLayerImplementation.get_best_config(sample_dim, expert_dim)
        BLOCK_SIZE_BD = config['BLOCK_SIZE_BD']
        BLOCK_SIZE_HD = config['BLOCK_SIZE_HD']
        BLOCK_SIZE_ED = config['BLOCK_SIZE_ED']
        GROUP_SIZE_BD = config['GROUP_SIZE_BD']
        ORDERING_MAP = {'COLUMN_MAJOR': 0, 'ROW_MAJOR': 1, 'GROUPED': 2}
        ordering_code = ORDERING_MAP.get(config['ORDERING'], 0)
        num_stages = config.get('num_stages', 4)
        num_warps  = config.get('num_warps', 4)

        # Output buffer
        out = torch.empty(
            (num_experts, sample_dim, expert_dim),
            device=input.device,
            dtype=input.dtype
        )

        # Compute the grid dimension in Python int arithmetic
        # grid_dim0 = ( (sample_dim + BLOCK_SIZE_BD - 1) // BLOCK_SIZE_BD ) \
        #             * ( (expert_dim + BLOCK_SIZE_ED - 1) // BLOCK_SIZE_ED )
        # grid = (grid_dim0, num_experts)

        grid = lambda META: (triton.cdiv(sample_dim, META['BLOCK_SIZE_BD']) *
                             triton.cdiv(expert_dim, META['BLOCK_SIZE_ED']),
                             num_experts)


        moe_first_kernel[grid](
            # All positional arguments first
            input, input.stride(0), input.stride(1),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            bias, bias.stride(0), bias.stride(1),
            out, out.stride(0), out.stride(1), out.stride(2),
            sort_indices, sort_indices.stride(0), sort_indices.stride(1),
            expert_bincounts,
            sample_dim, hidden_dim, expert_dim,
            num_experts,           # NUM_EXPERTS (tl.constexpr)
            activation_code,       # ACTIVATION  (tl.constexpr)
            BLOCK_SIZE_BD, BLOCK_SIZE_HD, BLOCK_SIZE_ED, GROUP_SIZE_BD,
            ordering_code,         # ORDERING (tl.constexpr)
            # Then any keyword arguments like grid and warps
            #grid=grid,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        # 2) Launch the kernel using .run(...) so we can pass grid as a tuple
        # moe_first_kernel.run(
        #     # All positional arguments first
        #     input, input.stride(0), input.stride(1),
        #     weight, weight.stride(0), weight.stride(1), weight.stride(2),
        #     bias, bias.stride(0), bias.stride(1),
        #     out, out.stride(0), out.stride(1), out.stride(2),
        #     sort_indices, sort_indices.stride(0), sort_indices.stride(1),
        #     expert_bincounts,
        #     sample_dim, hidden_dim, expert_dim,
        #     num_experts,           # NUM_EXPERTS (tl.constexpr)
        #     activation_code,       # ACTIVATION  (tl.constexpr)
        #     BLOCK_SIZE_BD, BLOCK_SIZE_HD, BLOCK_SIZE_ED, GROUP_SIZE_BD,
        #     ordering_code,         # ORDERING (tl.constexpr)
        #     # Then any keyword arguments like grid and warps
        #     grid=grid,
        #     num_stages=num_stages,
        #     num_warps=num_warps,
        #     warmup=False
        # )

                # # Print debug info once
        # shape_key = tuple(input.shape)
        # if shape_key not in MoeFirstLayerImplementation.seen_shapes:
        #     MoeFirstLayerImplementation.seen_shapes.add(shape_key)
            # print("Input shape:", input.shape, 'and expert dim', expert_dim)
            # print("Chosen config:", moe_first_kernel.best_config)
        #     print('-' * 100)

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, bias, sort_indices, expert_bincounts, activation_str) = inputs
        ctx.activation_str = activation_str
        ctx.save_for_backward(input, weight, bias, sort_indices, expert_bincounts)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, bias, sort_indices, expert_bincounts) = ctx.saved_tensors
        activation_str = ctx.activation_str
        raise NotImplementedError('TODO')


@triton.jit
def moe_first_kernel(
    x_ptr, stride_x_bd, stride_x_hd,
    weight_ptr, stride_weight_ned, stride_weight_hd, stride_weight_ed,
    bias_ptr, stride_bias_ned, stride_bias_ed,
    output_ptr, stride_output_ned, stride_output_bd, stride_output_ed,
    sort_indices_ptr, stride_sort_indices_bd, stride_sort_indices_ned,
    expert_bincounts_ptr,
    sample_dim, hidden_dim, expert_dim,
    NUM_EXPERTS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
    BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
    ORDERING: tl.constexpr
):
    # Decide how to map program_id based on ORDERING code
    if ORDERING == 2:
        pid_bd, pid_ed = grouped(
            tl.program_id(axis=0), sample_dim, expert_dim,
            BLOCK_SIZE_BD, BLOCK_SIZE_ED, GROUP_SIZE_BD
        )
    elif ORDERING == 0:
        pid_bd, pid_ed = column_major(
            tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD
        )
    elif ORDERING == 1:
        pid_bd, pid_ed = row_major(
            tl.program_id(axis=0), expert_dim, BLOCK_SIZE_ED
        )

    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty

    # Number of samples for this expert
    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)

    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_ed = (pid_ed * BLOCK_SIZE_ED + tl.arange(0, BLOCK_SIZE_ED)) % expert_dim
        offs_hd = tl.arange(0, BLOCK_SIZE_HD)

        # gather indices for input
        in_data_indices = tl.load(
            sort_indices_ptr + expert_index * stride_sort_indices_ned + offs_bd * stride_sort_indices_bd
        ).to(tl.int64)

        x_ptrs = x_ptr + in_data_indices[:, None] * stride_x_bd + offs_hd[None, :] * stride_x_hd
        w_ptrs = weight_ptr + expert_index * stride_weight_ned + offs_hd[:, None] * stride_weight_hd + offs_ed[None, :] * stride_weight_ed

        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_ED), dtype=tl.float32)

        # Dot-product loop
        for k in range(0, tl.cdiv(hidden_dim, BLOCK_SIZE_HD)):
            x = tl.load(
                x_ptrs,
                mask=offs_hd[None, :] < hidden_dim - k * BLOCK_SIZE_HD,
                other=0.0
            )
            w = tl.load(
                w_ptrs,
                mask=offs_hd[:, None] < hidden_dim - k * BLOCK_SIZE_HD,
                other=0.0
            )
            accumulator += tl.dot(x, w, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_HD * stride_x_hd
            w_ptrs += BLOCK_SIZE_HD * stride_weight_hd

        offs_b_ed = pid_ed * BLOCK_SIZE_ED + tl.arange(0, BLOCK_SIZE_ED)
        b_ptrs = bias_ptr + expert_index * stride_bias_ned + offs_b_ed[None, :] * stride_bias_ed
        accumulator += tl.load(
            b_ptrs,
            mask=offs_b_ed[None, :] < expert_dim,
            other=0.0
        )

        # Activation
        if ACTIVATION == 0:
            accumulator = relu(accumulator)
        elif ACTIVATION == 1:
            accumulator = tanh(accumulator)
        elif ACTIVATION == 2:
            accumulator = gelu(accumulator)
        elif ACTIVATION == 3:
            accumulator = leaky_relu(accumulator)

        out = accumulator.to(x_dtype)
        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        out_ptrs = (
            output_ptr
            + expert_index * stride_output_ned
            + offs_out_bd[:, None] * stride_output_bd
            + offs_b_ed[None, :] * stride_output_ed
        )
        out_mask = (
            (offs_out_bd[:, None] < expert_samples_count)
            & (offs_b_ed[None, :] < expert_dim)
        )
        tl.store(out_ptrs, out, mask=out_mask)


class MoeSecondLayerAtomicImplementation(torch.autograd.Function):
    seen_shapes = set()

    BEST_CONFIGS_SECOND = {
        8: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
        },

        16: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
        },

        32: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 2, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
        },

        64: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            4*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_ED': 32, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*576:   {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*2304:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR','num_stages': 4, 'num_warps': 4},
            4*4096:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'COLUMN_MAJOR',   'num_stages': 4, 'num_warps': 4},
            
            4*6400:  {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*10816: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*16384: {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_HD': 32,  'BLOCK_SIZE_ED': 64, 'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
        },

        128: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},


            4*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*4096:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            4*6400:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
        },

        256: {
            64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4096:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            6400:  {'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 32,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            10816: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            16384: {'BLOCK_SIZE_BD': 32,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            4*64:    {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*256:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR',   'num_stages': 4, 'num_warps': 4},
            4*576:   {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1024:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*1600:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*2304:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*4096:  {'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},

            4*6400:  {'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*10816: {'BLOCK_SIZE_BD': 128,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
            4*16384: {'BLOCK_SIZE_BD': 128,'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,  'GROUP_SIZE_BD': 1, 'ORDERING': 'ROW_MAJOR','num_stages': 4, 'num_warps': 4},
        },
    }

    @staticmethod
    def get_best_config(sample_dim: int, expert_dim: int):
        """
        Return the best config from BEST_CONFIGS_FROM_LOGS using a simple threshold lookup.
        If sample_dim is less than or equal to the smallest key, return that config.
        Otherwise, iterate and return the first config where sample_dim is below the threshold.
        If sample_dim is larger than any key, return the last config.
        """
        if expert_dim not in MoeSecondLayerAtomicImplementation.BEST_CONFIGS_SECOND:
            raise ValueError(f"No configs found for expert_dim={expert_dim}")

        expert_configs = MoeSecondLayerAtomicImplementation.BEST_CONFIGS_SECOND[expert_dim]
        if sample_dim not in expert_configs:
            raise ValueError(f"No config found for sample_dim={sample_dim} under expert_dim={expert_dim}")
        
        return expert_configs[sample_dim]


    @staticmethod
    def forward(input, weight, sort_indices, expert_bincounts):
        # `input` shape is [num_experts, sample_dim, expert_dim=1024]
        num_experts = weight.size(0)
        expert_dim = weight.size(1) 
        hidden_dim = weight.size(2)   
        sample_dim = input.size(1)  

        # Output: shape [sample_dim, hidden_dim]
        out = torch.zeros(
            (sample_dim, hidden_dim),
            device=input.device,
            dtype=input.dtype
        )

        # 1) Lookup best config for this sample_dim
        config = MoeSecondLayerAtomicImplementation.get_best_config(sample_dim, expert_dim)
        BLOCK_SIZE_BD = config['BLOCK_SIZE_BD']
        BLOCK_SIZE_ED = config['BLOCK_SIZE_ED']
        BLOCK_SIZE_HD = config['BLOCK_SIZE_HD']
        GROUP_SIZE_BD = config['GROUP_SIZE_BD']
        ORDERING_str  = config['ORDERING']
        # Convert ordering string to int
        ORDERING_MAP  = {'COLUMN_MAJOR': 'COLUMN_MAJOR', 'ROW_MAJOR': 'ROW_MAJOR', 'GROUPED': 'GROUPED'}
        # In this snippet, we kept ORDERING as a string in the kernel signature (default = 'GROUPED')
        # so we might directly pass ORDERING_str if we want.

        num_stages = config.get('num_stages', 4)
        num_warps  = config.get('num_warps', 4)

        # 2) Compute a grid = (something, num_experts) – or do manual logic
        # Example: same logic as your code
        # We'll do the same function 'grid = lambda META: (...)' approach, but let's do it in pure python
        grid_dim0 = (
            (sample_dim + BLOCK_SIZE_BD - 1) // BLOCK_SIZE_BD
        ) * (
            (hidden_dim + BLOCK_SIZE_HD - 1) // BLOCK_SIZE_HD
        )
        grid = (grid_dim0, num_experts)

        # 3) Launch the kernel with .run(...)
        moe_second_kernel_atomic.run(
            # runtime parameters (positional)
            input, input.stride(0), input.stride(1), input.stride(2),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            out, out.stride(0), out.stride(1),
            sort_indices, sort_indices.stride(0), sort_indices.stride(1),
            expert_bincounts,
            sample_dim,
            expert_dim,
            hidden_dim,
            num_experts,       # NUM_EXPERTS
            BLOCK_SIZE_BD,
            BLOCK_SIZE_HD,
            BLOCK_SIZE_ED,
            GROUP_SIZE_BD,
            ORDERING_str,      # We'll pass e.g. 'ROW_MAJOR' or 'COLUMN_MAJOR'
            # now the kernel config
            grid=grid,
            num_stages=num_stages,
            num_warps=num_warps,
            warmup=False
        )

        # # Debug print first time we see shape
        # shape_key = (num_experts, sample_dim, expert_dim)
        # if shape_key not in MoeSecondLayerAtomicImplementation.seen_shapes:
        #     MoeSecondLayerAtomicImplementation.seen_shapes.add(shape_key)
            # print("Input shape:", input.shape, 'and expert dim', expert_dim)
            # print("Chosen config:", moe_second_kernel_atomic.best_config)
        #     print('-'*100)

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, sort_indices, expert_bincounts) = inputs
        ctx.save_for_backward(input, weight, sort_indices, expert_bincounts)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, sort_indices, expert_bincounts) = ctx.saved_tensors
        raise NotImplementedError('TODO')


@triton.jit
def moe_second_kernel_atomic(
    x_ptr, stride_x_ned, stride_x_bd, stride_x_ed,
    weight_ptr, stride_weight_ned, stride_weight_ed, stride_weight_hd,
    output_ptr, stride_output_bd, stride_output_hd,
    sort_indices_ptr, stride_sort_indices_bd, stride_sort_indices_ned,
    expert_bincounts_ptr,
    sample_dim,
    expert_dim,
    hidden_dim,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
    BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
    ORDERING: tl.constexpr = 'GROUPED'  # We'll override anyway
):
    # same body as your original code except we remove references to autotune
    if ORDERING == 'GROUPED':
        pid_bd, pid_hd = grouped(
            tl.program_id(axis=0),
            sample_dim, hidden_dim,
            BLOCK_SIZE_BD, BLOCK_SIZE_HD, GROUP_SIZE_BD
        )
    elif ORDERING == 'COLUMN_MAJOR':
        pid_bd, pid_hd = column_major(
            tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD
        )
    elif ORDERING == 'ROW_MAJOR':
        pid_bd, pid_hd = row_major(
            tl.program_id(axis=0), hidden_dim, BLOCK_SIZE_HD
        )
    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty
    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_hd = (pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)) % hidden_dim
        offs_ed = tl.arange(0, BLOCK_SIZE_ED)
        x_ptrs = x_ptr + \
                 expert_index * stride_x_ned + \
                 offs_bd[:, None] * stride_x_bd + \
                 offs_ed[None, :] * stride_x_ed
        w_ptrs = weight_ptr + \
                 expert_index * stride_weight_ned + \
                 offs_ed[:, None] * stride_weight_ed + \
                 offs_hd[None, :] * stride_weight_hd
        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
        for k in range(0, tl.cdiv(expert_dim, BLOCK_SIZE_ED)):
            x = tl.load(
                x_ptrs,
                mask=offs_ed[None, :] < expert_dim - k * BLOCK_SIZE_ED,
                other=0.0
            )
            w = tl.load(
                w_ptrs,
                mask=offs_ed[:, None] < expert_dim - k * BLOCK_SIZE_ED,
                other=0.0
            ).to(x_dtype)
            accumulator += tl.dot(x, w, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_ED * stride_x_ed
            w_ptrs += BLOCK_SIZE_ED * stride_weight_ed

        out_data_indices = tl.load(
            sort_indices_ptr + expert_index * stride_sort_indices_ned + offs_bd * stride_sort_indices_bd
        ).to(tl.int64)
        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        offs_out_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        out_ptrs = output_ptr + (
            out_data_indices[:, None] * stride_output_bd
            + offs_out_hd[None, :] * stride_output_hd
        )
        out_mask = (offs_out_bd[:, None] < expert_samples_count) & (offs_out_hd[None, :] < hidden_dim)
        out = accumulator.to(x_dtype)
        tl.atomic_add(out_ptrs, out, mask=out_mask, sem='relaxed', scope='gpu')


class MoeSecondLayerMergingImplementation(torch.autograd.Function):
    seen_shapes = set()
    seen_shapes_merge = set()

    #
    # 1) For moe_second_kernel => your "sample_dim => config" from the logs:
    #
    BEST_CONFIGS_FROM_LOGS = [
        # (sample_dim, config_dict)
        (64,    {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,
            'GROUP_SIZE_BD': 4,  'ORDERING_CODE': 2,   # 2 => GROUPED
            'num_stages': 3, 'num_warps': 8
        }),
        (256,   {
            'BLOCK_SIZE_BD': 32, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 64,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # 1 => ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (576,   {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 128,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (1024,  {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 128,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (1600,  {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 128,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (2304,  {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 128,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (4096,  {
            'BLOCK_SIZE_BD': 64, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 128,
            'GROUP_SIZE_BD': 1,  'ORDERING_CODE': 1,   # ROW_MAJOR
            'num_stages': 4, 'num_warps': 4
        }),
        (6400,  {
            'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 256,
            'GROUP_SIZE_BD': 8,   'ORDERING_CODE': 2,   # GROUPED
            'num_stages': 3, 'num_warps': 8
        }),
        (10816, {
            'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 256,
            'GROUP_SIZE_BD': 16,  'ORDERING_CODE': 2,   # GROUPED
            'num_stages': 3, 'num_warps': 8
        }),
        (16384, {
            'BLOCK_SIZE_BD': 128, 'BLOCK_SIZE_ED': 32, 'BLOCK_SIZE_HD': 256,
            'GROUP_SIZE_BD': 16,  'ORDERING_CODE': 2,   # GROUPED
            'num_stages': 3, 'num_warps': 8
        }),
    ]

    @staticmethod
    def get_best_config(sample_dim: int):
        """
        Return the best second-kernel config from BEST_CONFIGS_FROM_LOGS
        using simple threshold logic. 
        """
        table = MoeSecondLayerMergingImplementation.BEST_CONFIGS_FROM_LOGS
        if sample_dim <= table[0][0]:
            return table[0][1]
        for i in range(len(table) - 1):
            sdim_i, cfg_i = table[i]
            sdim_ip1, cfg_ip1 = table[i+1]
            if sdim_i <= sample_dim < sdim_ip1:
                return cfg_i
        # if bigger than all, return the last
        return table[-1][1]

    #
    # For moe_merge_results_kernel: from your logs, it's always 
    # BLOCK_SIZE_HD=256, num_warps=4, num_stages=4 for hidden_dim=1024.
    # But if you do have multiple shapes, list them all similarly.
    #
    BEST_MERGE_CONFIGS_FROM_LOGS = [
        ((64, 1024),    {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((256, 1024),   {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((576, 1024),   {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((1024, 1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((1600, 1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((2304, 1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((4096, 1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((6400, 1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((10816,1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
        ((16384,1024),  {'BLOCK_SIZE_HD': 256, 'num_stages': 4, 'num_warps': 4}),
    ]

    @staticmethod
    def get_best_merge_config(sample_dim: int, hidden_dim: int):
        """
        Return the best merge-kernel config from BEST_MERGE_CONFIGS_FROM_LOGS.
        """
        table = MoeSecondLayerMergingImplementation.BEST_MERGE_CONFIGS_FROM_LOGS
        # Simple exact match approach, since your logs always used 1024 hidden dim
        for (sdim, hdim), cfg in table:
            if sdim == sample_dim and hdim == hidden_dim:
                return cfg
        # fallback: return the last or some default
        return table[-1][1]
    # ------------------------------------------------------------------------ #
    @staticmethod
    def forward(input, weight, unsort_indices, expert_bincounts, routing_tensor):
        """
          input:  (num_experts, sample_dim, expert_dim)
          weight: (num_experts, expert_dim, hidden_dim)
          unsort_indices: (sample_dim, num_experts)
          expert_bincounts: (num_experts,)
          routing_tensor:   (sample_dim, num_experts)
        """
        device = input.device
        dtype  = input.dtype

        num_experts = weight.size(0)
        expert_dim  = weight.size(1)
        hidden_dim  = weight.size(2)
        sample_dim  = input.size(1)  # as in your original code

        # 1) Create intermediate buffer
        intermediate_out = torch.empty(
            (num_experts, sample_dim, hidden_dim),
            device=device, dtype=dtype
        )

        # 2) Look up best config for moe_second_kernel using sample_dim
        best_cfg = MoeSecondLayerMergingImplementation.get_best_config(sample_dim)
        bbd = best_cfg['BLOCK_SIZE_BD']
        bhd = best_cfg['BLOCK_SIZE_HD']
        bed = best_cfg['BLOCK_SIZE_ED']
        gbd = best_cfg['GROUP_SIZE_BD']
        ord_code = best_cfg['ORDERING_CODE']   # integer: 0=col,1=row,2=grouped
        num_stages = best_cfg['num_stages']
        num_warps  = best_cfg['num_warps']

        # 3) Compute the grid for second kernel
        grid_dim0 = ((sample_dim + bbd - 1) // bbd) * ((hidden_dim + bhd - 1) // bhd)
        grid = (grid_dim0, num_experts)

        # Debug print once
        # shape_key = tuple(input.shape)
        # if shape_key not in MoeSecondLayerMergingImplementation.seen_shapes:
        #     MoeSecondLayerMergingImplementation.seen_shapes.add(shape_key)
        #     print("Input shape (moe_second_kernel):", input.shape)
        #     print("Chosen config for second kernel:", best_cfg)
        #     print('-'*100)

        # 4) Launch second kernel
        moe_second_kernel.run(
            input, input.stride(0), input.stride(1), input.stride(2),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            intermediate_out, intermediate_out.stride(0), intermediate_out.stride(1), intermediate_out.stride(2),
            expert_bincounts,
            sample_dim, expert_dim, hidden_dim,
            num_experts,
            bbd, bhd, bed, gbd,
            ord_code,     # integer ordering
            grid=grid,
            num_stages=num_stages,
            num_warps=num_warps,
            warmup=False
        )

        # 5) Merge partial results -> final out
        out = torch.empty((sample_dim, hidden_dim), device=device, dtype=dtype)

        # pick best config for the merge kernel
        merge_cfg = MoeSecondLayerMergingImplementation.get_best_merge_config(sample_dim, hidden_dim)
        mbhd = merge_cfg['BLOCK_SIZE_HD']
        merge_stages = merge_cfg['num_stages']
        merge_warps  = merge_cfg['num_warps']

        merge_grid = (sample_dim, (hidden_dim + mbhd - 1)//mbhd)

        # Debug print once
        # merge_shape_key = tuple(intermediate_out.shape)
        # if merge_shape_key not in MoeSecondLayerMergingImplementation.seen_shapes_merge:
        #     MoeSecondLayerMergingImplementation.seen_shapes_merge.add(merge_shape_key)
        #     print("Input shape (moe_merge_results_kernel):", intermediate_out.shape)
        #     print("Chosen config for merge kernel:", merge_cfg)
        #     print('-'*100)

        # 5b) Launch merge kernel
        moe_merge_results_kernel.run(
            intermediate_out, intermediate_out.stride(0),
            intermediate_out.stride(1), intermediate_out.stride(2),
            out, out.stride(0), out.stride(1),
            unsort_indices, unsort_indices.stride(0), unsort_indices.stride(1),
            routing_tensor, routing_tensor.stride(0), routing_tensor.stride(1),
            hidden_dim,
            num_experts,
            BLOCK_SIZE_HD=mbhd,
            grid=merge_grid,
            num_stages=merge_stages,
            num_warps=merge_warps,
            warmup=False
        )

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (input, weight, sort_indices, expert_bincounts, routing_tensor) = inputs
        ctx.save_for_backward(input, weight, sort_indices, expert_bincounts, routing_tensor)

    @staticmethod
    def backward(ctx, output_grad):
        (input, weight, sort_indices, expert_bincounts, routing_tensor) = ctx.saved_tensors
        raise NotImplementedError('TODO')

@triton.jit
def moe_second_kernel(
    x_ptr, stride_x_ned, stride_x_bd, stride_x_ed,
    weight_ptr, stride_weight_ned, stride_weight_ed, stride_weight_hd,
    output_ptr, stride_output_ned, stride_output_bd, stride_output_hd,
    # metadata
    expert_bincounts_ptr,
    sample_dim,
    expert_dim,
    hidden_dim,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
    BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
    ORDERING_CODE: tl.constexpr  # an integer: 0=col_major,1=row_major,2=grouped
):
    # Decide how to map based on ORDERING_CODE:
    if ORDERING_CODE == 2:  # GROUPED
        pid_bd, pid_hd = grouped(
            tl.program_id(axis=0), sample_dim, hidden_dim,
            BLOCK_SIZE_BD, BLOCK_SIZE_HD, GROUP_SIZE_BD
        )
    elif ORDERING_CODE == 0:  # COLUMN_MAJOR
        pid_bd, pid_hd = column_major(
            tl.program_id(axis=0), sample_dim, BLOCK_SIZE_BD
        )
    elif ORDERING_CODE == 1:  # ROW_MAJOR
        pid_bd, pid_hd = row_major(
            tl.program_id(axis=0), hidden_dim, BLOCK_SIZE_HD
        )

    expert_index = tl.program_id(axis=1)
    x_dtype = x_ptr.dtype.element_ty

    expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
    bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
    if pid_bd < bd_pids_for_expert:
        offs_bd = (pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)) % expert_samples_count
        offs_hd = (pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)) % hidden_dim
        offs_ed = tl.arange(0, BLOCK_SIZE_ED)

        x_ptrs = (x_ptr
                  + expert_index * stride_x_ned
                  + offs_bd[:, None] * stride_x_bd
                  + offs_ed[None, :] * stride_x_ed)
        w_ptrs = (weight_ptr
                  + expert_index * stride_weight_ned
                  + offs_ed[:, None] * stride_weight_ed
                  + offs_hd[None, :] * stride_weight_hd)

        accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
        # Dot-product loop over the "expert_dim"
        for k in range(0, tl.cdiv(expert_dim, BLOCK_SIZE_ED)):
            valid_k = offs_ed < expert_dim - k*BLOCK_SIZE_ED
            x = tl.load(x_ptrs, mask=valid_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=valid_k[:, None], other=0.0)
            accumulator += tl.dot(x, w, allow_tf32=False)
            # Advance pointers
            x_ptrs += BLOCK_SIZE_ED * stride_x_ed
            w_ptrs += BLOCK_SIZE_ED * stride_weight_ed

        offs_out_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
        offs_out_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)

        out_ptrs = (output_ptr
                    + expert_index * stride_output_ned
                    + offs_out_bd[:, None] * stride_output_bd
                    + offs_out_hd[None, :] * stride_output_hd)

        out_mask = ((offs_out_bd[:, None] < expert_samples_count)
                    & (offs_out_hd[None, :] < hidden_dim))
        out = accumulator.to(x_dtype)
        tl.store(out_ptrs, out, mask=out_mask)



@triton.jit
def moe_merge_results_kernel(
    x_ptr, stride_x_ned, stride_x_bd, stride_x_hd,
    output_ptr, stride_output_bd, stride_output_hd,
    # metadata
    unsort_indices_ptr, stride_unsort_indices_bd, stride_unsort_indices_ned,
    routing_tensor_ptr, stride_routing_tensor_bd, stride_routing_tensor_ned,
    hidden_dim,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_HD: tl.constexpr
):
    pid_bd = tl.program_id(axis=0)
    pid_hd = tl.program_id(axis=1)
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    hd_mask = offs_hd < hidden_dim

    x_dtype = x_ptr.dtype.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_HD,), dtype=tl.float32)
    for i in range(0, NUM_EXPERTS):
        # Check if expert i is active for sample pid_bd
        executed = tl.load(routing_tensor_ptr +
                           pid_bd*stride_routing_tensor_bd +
                           i*stride_routing_tensor_ned)
        if executed > 0:
            sample_index = tl.load(unsort_indices_ptr +
                                   pid_bd*stride_unsort_indices_bd +
                                   i*stride_unsort_indices_ned)
            vals = tl.load(x_ptr
                           + i*stride_x_ned
                           + sample_index*stride_x_bd
                           + offs_hd*stride_x_hd)
            accumulator += vals

    out_ptrs = (output_ptr + pid_bd*stride_output_bd + offs_hd*stride_output_hd)
    tl.store(out_ptrs, accumulator.to(x_dtype), mask=hd_mask)

@triton.autotune(
    configs=config_grid({
        'BLOCK_SIZE_BD': [32, 64, 128],
        'BLOCK_SIZE_HD': [32, 64, 128], }, num_stages=4, num_warps=4),
    # configs=[triton.Config({
    #     'BLOCK_SIZE_BD': 64,
    #     'BLOCK_SIZE_HD': 64}, num_stages=4, num_warps=4)],
    key=['hidden_dim', 'NUM_EXPERTS'],
)
@triton.jit
def moe_merge_results_batched_kernel(x_ptr, stride_x_ned, stride_x_bd, stride_x_hd,
                                     output_ptr, stride_output_bd, stride_output_hd,
                                     # metadata
                                     unsort_indices_ptr, stride_unsort_indices_bd, stride_unsort_indices_ned,
                                     routing_tensor_ptr, stride_routing_tensor_bd, stride_routing_tensor_ned,
                                     sample_dim,
                                     hidden_dim,
                                     NUM_EXPERTS: tl.constexpr,
                                     BLOCK_SIZE_BD: tl.constexpr,
                                     BLOCK_SIZE_HD: tl.constexpr
                                     ):
    pid_bd = tl.program_id(axis=0)
    pid_hd = tl.program_id(axis=1)
    offs_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    # offs_ned = tl.arange(0, NUM_EXPERTS)
    bd_mask = offs_bd < sample_dim
    hd_mask = offs_hd < hidden_dim
    x_dtype = x_ptr.dtype.element_ty
    # TODO vectorize/optimize
    # experts_executed = tl.load(routing_tensor_ptr +
    #                            offs_bd[:, None] * stride_routing_tensor_bd +
    #                            offs_ned[None, :] * stride_routing_tensor_ned)
    accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)
    for i in range(0, NUM_EXPERTS):
        # inefficient uncoalesced load - TODO optimize?
        samples_executed = tl.load(routing_tensor_ptr +
                                   offs_bd * stride_routing_tensor_bd +
                                   i * stride_routing_tensor_ned,
                                   mask=bd_mask)
        samples_executed_mask = (samples_executed > 0) & (bd_mask)
        sample_indices = tl.load(unsort_indices_ptr +
                                 offs_bd * stride_unsort_indices_bd +
                                 i * stride_unsort_indices_ned,
                                 mask=samples_executed_mask)
        load_mask = samples_executed_mask[:, None] & hd_mask[None, :]
        accumulator += tl.load(x_ptr +
                               i * stride_x_ned +
                               sample_indices[:, None] * stride_x_bd +
                               offs_hd[None, :] * stride_x_hd,
                               mask=load_mask)
    out_ptrs = output_ptr + offs_bd[:, None] * stride_output_bd + offs_hd[None, :] * stride_output_hd
    out_mask = bd_mask[:, None] & hd_mask[None, :]
    tl.store(out_ptrs, accumulator.to(x_dtype), mask=out_mask)









# @triton.jit
# def moe_second_kernel_atomic(
#     x_ptr, stride_x_ned, stride_x_bd, stride_x_ed,
#     weight_ptr, stride_weight_ned, stride_weight_ed, stride_weight_hd,
#     output_ptr, stride_output_bd, stride_output_hd,
#     sort_indices_ptr, stride_sort_indices_bd, stride_sort_indices_ned,
#     expert_bincounts_ptr,
#     sample_dim,
#     expert_dim,
#     hidden_dim,
#     NUM_EXPERTS: tl.constexpr,
#     BLOCK_SIZE_BD: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr,
#     BLOCK_SIZE_ED: tl.constexpr, GROUP_SIZE_BD: tl.constexpr,
#     ORDERING: tl.constexpr = 'GROUPED'
# ):
#     # --- Map grid indices based on ordering ---
#     if ORDERING == 'GROUPED':
#         pid_bd, pid_hd = grouped(
#             tl.program_id(axis=0),
#             sample_dim,
#             hidden_dim,
#             BLOCK_SIZE_BD,
#             BLOCK_SIZE_HD,
#             GROUP_SIZE_BD
#         )
#     elif ORDERING == 'COLUMN_MAJOR':
#         pid_bd, pid_hd = column_major(
#             tl.program_id(axis=0),
#             sample_dim,
#             BLOCK_SIZE_BD
#         )
#     elif ORDERING == 'ROW_MAJOR':
#         pid_bd, pid_hd = row_major(
#             tl.program_id(axis=0),
#             hidden_dim,
#             BLOCK_SIZE_HD
#         )
    
#     # Each block processes one expert given by program_id(axis=1)
#     expert_index = tl.program_id(axis=1)
#     x_dtype = x_ptr.dtype.element_ty

#     # Load how many tokens this expert has.
#     expert_samples_count = tl.load(expert_bincounts_ptr + expert_index)
#     bd_pids_for_expert = tl.cdiv(expert_samples_count, BLOCK_SIZE_BD)
#     if pid_bd >= bd_pids_for_expert:
#         return

#     # Compute output offsets for this block.
#     offs_bd = pid_bd * BLOCK_SIZE_BD + tl.arange(0, BLOCK_SIZE_BD)
#     offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
#     bd_mask = offs_bd < expert_samples_count  # Valid sample positions for this expert
#     hd_mask = offs_hd < hidden_dim             # Valid hidden feature positions

#     # For the expert dimension we iterate in blocks of BLOCK_SIZE_ED.
#     offs_ed = tl.arange(0, BLOCK_SIZE_ED)

#     # Initialize an accumulator (fp32) to accumulate the dot-product.
#     accumulator = tl.zeros((BLOCK_SIZE_BD, BLOCK_SIZE_HD), dtype=tl.float32)

#     # Set up pointers for input and weight. The data layout is:
#     #   x:    [num_experts, sample_dim, expert_dim]
#     #   weight: [num_experts, expert_dim, hidden_dim]
#     #
#     # Since the input is grouped by expert (dimension 0 gives expert id),
#     # each expert’s local samples (dimension 1) are processed separately.
#     x_ptrs = x_ptr + expert_index * stride_x_ned \
#                   + offs_bd[:, None] * stride_x_bd \
#                   + offs_ed[None, :] * stride_x_ed
#     w_ptrs = weight_ptr + expert_index * stride_weight_ned \
#                   + offs_ed[:, None] * stride_weight_ed \
#                   + offs_hd[None, :] * stride_weight_hd

#     # Dot-product loop: Process expert_dim in chunks of BLOCK_SIZE_ED.
#     num_steps = (expert_dim + BLOCK_SIZE_ED - 1) // BLOCK_SIZE_ED
#     for k in range(num_steps):
#         # For this step, update a mask to avoid overrunning expert_dim.
#         current_ed_mask = offs_ed < (expert_dim - k * BLOCK_SIZE_ED)
#         # Load a tile from x and weight with proper masking.
#         x_tile = tl.load(x_ptrs, mask=bd_mask[:, None] & current_ed_mask[None, :], other=0.0)
#         w_tile = tl.load(w_ptrs, mask=current_ed_mask[:, None] & hd_mask[None, :], other=0.0).to(tl.float32)
#         accumulator += tl.dot(x_tile, w_tile, allow_tf32=False)
#         # Advance the pointers for the next chunk along the expert dimension.
#         x_ptrs += BLOCK_SIZE_ED * stride_x_ed
#         w_ptrs += BLOCK_SIZE_ED * stride_weight_ed

#     # Use sort_indices to map local sample indices to global output positions.
#     out_data_indices = tl.load(
#         sort_indices_ptr 
#         + expert_index * stride_sort_indices_ned 
#         + offs_bd * stride_sort_indices_bd,
#         mask=bd_mask,
#         other=0
#     ).to(tl.int64)

#     # Compute global output pointers: out is [sample_dim, hidden_dim].
#     out_ptrs = output_ptr \
#                 + out_data_indices[:, None] * stride_output_bd \
#                 + offs_hd[None, :] * stride_output_hd
#     store_mask = bd_mask[:, None] & hd_mask[None, :]
#     out_vals = accumulator.to(x_dtype)

#     # Direct store because the output positions for this expert are unique.
#     tl.store(out_ptrs, out_vals, mask=store_mask)