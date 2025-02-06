import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from architectures.moe.moefication import MoeficationMoE
from data_utils.data import DATASETS_NAME_MAP
from eval import benchmark_moe
from utils import load_model, get_loader
from utils_var import arg_util
from train import setup_data, TrainingContext, setup_accelerator, make_vae
from common import get_default_args as original_get_default_args
import os
import dist

def get_default_args():
    
    default_args = original_get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]
    default_args.runs_dir = Path(os.environ['RUNS_DIR'])
    default_args.dataset = 'TINYIMAGENET_PATH' #'imagenet'
    default_args.dataset_args = {}
    default_args.dataset_args.variant = 'deit3_rrc'# deit3 Jort added this would be for the train class to work for tiny deit3
    default_args.mixup_alpha = None #0.8
    default_args.cutmix_alpha = None #1.0
    default_args.mixup_smoothing = 0.1
    default_args.batch_size = 16 #128
    default_args.loss_type = 'ce'
    default_args.loss_args = {}
    default_args.optimizer_class = 'adam'
    default_args.optimizer_args = {}
    default_args.optimizer_args.lr = 0.001
    default_args.optimizer_args.weight_decay = 0.0
    default_args.scheduler_class = 'cosine' #Jort: This should be 'linear'
    default_args.scheduler_args = {}
    default_args.scheduler_args.eta_min = 1e-6
    default_args.clip_grad_norm = 1.0
    default_args.epochs = 5
    default_args.eval_points = 20
    default_args.use_wandb = False
    default_args.mixed_precision = None
    default_args.runs_dir = Path.cwd() / 'runs'  # Root dir where experiment data was saved.
    default_args.exp_names = []  # Unique experiment names to visualize the results for (excluding exp_id).
    default_args.exp_ids = [0]  # Experiment ids.
    default_args.display_names = None  # Pretty display names that will be used when generating the plot.
    default_args.output_dir = Path.cwd() / 'figures'  # Target directory.
    default_args.data_indices = 0  # Indices of the data samples to display the patches for.
    default_args.num_cheapest = None  # Number of data samples that consumed the least amount of compute.
    default_args.num_costliest = None  # Number of data samples that consumed the most amount of compute.
    default_args.dataset = None  # Dataset to evaluate on. If "None", will use the dataset used by the run.
    default_args.dataset_args = None  # Dataset arguments. If "None", will use same args as in the training run.
    default_args.batch_size = None  # Batch size.
    default_args.use_wandb = False  # Use W&B. Will save and load the models from the W&B cloud.
    default_args.dsti_tau = False  # Tau threshold for selecting experts to execute.
    return default_args


def get_normalization_transform(data):
    if isinstance(data.transforms.transform, transforms.Compose):
        transforms_set = data.transforms.transform.transforms
    else:
        transforms_set = [data.transforms.transform]
    for transform in transforms_set:
        if isinstance(transform, transforms.Normalize):
            return transform


def set_for_eval_with_topk(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = 'topk'
            m.k = 1


def set_for_eval_with_dynk(model, tau, mode='dynk_max'):
    model.eval()
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = mode
            m.tau = tau


def get_moe_costs(accelerator, model, loader, tc):
    unwrapped_model = accelerator.unwrap_model(model)
    set_for_eval_with_topk(unwrapped_model)
    cost_without_experts, token_expert_costs, _ = benchmark_moe(unwrapped_model, loader, tc)
    return cost_without_experts, token_expert_costs


def compute_spatial_load(accelerator, model, loader, gating_data, tc):
    _, token_expert_costs = get_moe_costs(accelerator, model, loader, tc)
    any_moe_gating_data = next(iter(gating_data.values()))
    device = any_moe_gating_data.device
    spatial_load = torch.zeros(any_moe_gating_data.size(0), any_moe_gating_data.size(1), device=device,
                               dtype=torch.double)
    max_spatial_load = torch.zeros_like(spatial_load)
    # gating data entry should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
    for k, v in token_expert_costs.items():
        assert v > 0
        executed_token_expert_pairs = (gating_data[k] > 0.0).to(torch.double).sum(dim=-1)
        spatial_load += executed_token_expert_pairs * v
        max_token_expert_pairs = gating_data[k].size(-1)
        max_spatial_load += max_token_expert_pairs * v
    return spatial_load / max_spatial_load


def merge_by_costs(tensor_dict, costs, x, y, y_pred, gating_data, number_of_samples, descending):
    if 'costs' not in tensor_dict:
        tensor_dict['costs'] = costs.cpu()
        tensor_dict['x'] = x.cpu()
        tensor_dict['y'] = y.cpu()
        tensor_dict['y_pred'] = y_pred.cpu()
        tensor_dict['gating_data'] = {}
        for k in gating_data.keys():
            tensor_dict['gating_data'][k] = gating_data[k].cpu()
    else:
        # concatenate everything
        tensor_dict['costs'] = torch.cat([tensor_dict['costs'], costs.cpu()], dim=0)
        tensor_dict['x'] = torch.cat([tensor_dict['x'], x.cpu()], dim=0)
        tensor_dict['y'] = torch.cat([tensor_dict['y'], y.cpu()], dim=0)
        tensor_dict['y_pred'] = torch.cat([tensor_dict['y_pred'], y_pred.cpu()], dim=0)
        for k in gating_data.keys():
            tensor_dict['gating_data'][k] = torch.cat([tensor_dict['gating_data'][k], gating_data[k].cpu()], dim=0)
    # sort all tensors by compute spent on the sample, and then discard samples
    tensor_dict['costs'], indices = torch.sort(tensor_dict['costs'], 0, descending=descending)
    tensor_dict['costs'] = tensor_dict['costs'][:number_of_samples]
    tensor_dict['x'] = tensor_dict['x'][indices][:number_of_samples]
    tensor_dict['y'] = tensor_dict['y'][indices][:number_of_samples]
    tensor_dict['y_pred'] = tensor_dict['y_pred'][indices][:number_of_samples]
    for k in gating_data.keys():
        tensor_dict['gating_data'][k] = tensor_dict['gating_data'][k][indices][:number_of_samples]


def merge_selected_samples(tensor_dict, costs, x, y, y_pred, gating_data, indices):
    if 'costs' not in tensor_dict:
        tensor_dict['costs'] = costs[indices].cpu()
        tensor_dict['x'] = x[indices].cpu()
        tensor_dict['y'] = y[indices].cpu()
        tensor_dict['y_pred'] = y_pred[indices].cpu()
        tensor_dict['gating_data'] = {}
        for k in gating_data.keys():
            tensor_dict['gating_data'][k] = gating_data[k][indices].cpu()
    else:
        # concatenate everything
        tensor_dict['costs'] = torch.cat([tensor_dict['costs'], costs[indices].cpu()], dim=0)
        tensor_dict['x'] = torch.cat([tensor_dict['x'], x[indices].cpu()], dim=0)
        tensor_dict['y'] = torch.cat([tensor_dict['y'], y[indices].cpu()], dim=0)
        tensor_dict['y_pred'] = torch.cat([tensor_dict['y_pred'], y_pred[indices].cpu()], dim=0)
        for k in gating_data.keys():
            tensor_dict['gating_data'][k] = torch.cat([tensor_dict['gating_data'][k], gating_data[k][indices].cpu()],
                                                      dim=0)


def process_dataset(accelerator, model, data_loader, number_with_least_compute, number_with_most_compute, data_indices,
                    tau, tc):
    assert isinstance(tau, float) and 0.0 < tau <= 1.0, f'{tau=}'
    data_indices = set(data_indices)
    least_compute = {}
    most_compute = {}
    selected_samples = {}
    cost_without_experts, token_expert_costs = get_moe_costs(accelerator, model, data_loader, tc)
    set_for_eval_with_dynk(model, tau)
    current_index = 0
    with torch.no_grad():
        for X, y in data_loader:
            with torch.no_grad():           
                B, V = y.shape[0], tc.model_vae.vocab_size
                X = X.to(dist.get_device(), non_blocking=True)
                label_B = y.to(dist.get_device(), non_blocking=True)
                gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X) # This does not return None
                x_BLCv_wo_first_l = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            print(f'X: {X.shape}, y: {y.shape}, x_BLCv_wo_first_l: {x_BLCv_wo_first_l.shape}')
            y_pred, gating_data = model(y, x_BLCv_wo_first_l, return_gating_data=True)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            merged_gating_data = {}
            for d in gating_data:
                merged_gating_data.update(d)


            gating_data = {k.replace("module.", ""): v[0] for k, v in merged_gating_data.items()}

            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y, gating_data = accelerator.gather_for_metrics((y_pred, y, gating_data))
            # calculate computational cost for each sample
            sample_costs = torch.zeros(X.size(0), device=X.device)
            for moe_name, moe_token_expert_cost in token_expert_costs.items():
                executed_expert_tokens = (gating_data[moe_name] > 0.0).long().sum(dim=(1, 2))
                total_expert_tokens = gating_data[moe_name].size(1) * gating_data[moe_name].size(2)
                sample_costs += executed_expert_tokens / total_expert_tokens * moe_token_expert_cost
            # merge current batch to find samples with least and most compute
            merge_by_costs(least_compute, sample_costs, X, y, y_pred, gating_data, number_with_least_compute, False)
            merge_by_costs(most_compute, sample_costs, X, y, y_pred, gating_data, number_with_most_compute, True)
            # accumulate "selected" images
            indices_current_batch = set(range(current_index, current_index + X.size(0)))
            current_index += X.size(0)
            indices_to_select = data_indices & indices_current_batch
            indices_to_select = [i % X.size(0) for i in indices_to_select]
            if len(indices_to_select) > 0:
                indices_to_select = torch.tensor(indices_to_select, device=X.device)
                merge_selected_samples(selected_samples, sample_costs, X, y, y_pred, gating_data, indices_to_select)
            if number_with_most_compute == 0 and \
                    number_with_least_compute == 0 and \
                    selected_samples['x'].size(0) >= len(data_indices):
                break
    return selected_samples, least_compute, most_compute


def setup_and_process(args, model, run_args, accelerator, tc):
    dataset = args.dataset if args.dataset is not None else run_args.dataset
    dataset_args = args.dataset_args if args.dataset_args is not None else run_args.dataset_args
    # _, _, data = DATASETS_NAME_MAP[dataset](**dataset_args)
    # logging.info(f'Testset size: {len(data)}')
    # dataloader = get_loader(data, run_args.batch_size if args.batch_size is None else args.batch_size, accelerator,
    #                         shuffle=False)
    
    setup_data(args, tc)
    normalization_transform = "normalize_01_into_pm1"

    selected_samples, least_compute, most_compute = process_dataset(accelerator,
                                                                    model,
                                                                    tc.val_loader,
                                                                    args.num_cheapest,
                                                                    args.num_costliest,
                                                                    args.data_indices,
                                                                    args.dsti_tau, tc)
    patch_size = args.patch_size
    return selected_samples, least_compute, most_compute, patch_size, normalization_transform, tc.val_loader


# def denormalize_image(image, normalization_transform):
#     mean = normalization_transform.mean
#     std = normalization_transform.std
#     de_mean = [-m / s for m, s in zip(mean, std)]
#     de_std = [1.0 / s for s in std]
#     denormalized_image = transforms.Normalize(de_mean, de_std)(image)
#     return denormalized_image

def denormalize_image(image, normalization_transform):
    """
    Reverse image normalization.

    - If `transforms.Normalize` was used, apply inverse normalization.
    - If `normalize_01_into_pm1` was used, reverse it with (x + 1) / 2.
    """
    if normalization_transform == 'normalize_01_into_pm1':
        return (image + 1) / 2  # Reverse (x * 2) - 1 → (x + 1) / 2
    elif isinstance(normalization_transform, transforms.Normalize):
        mean = normalization_transform.mean
        std = normalization_transform.std
        de_mean = [-m / s for m, s in zip(mean, std)]
        de_std = [1.0 / s for s in std]
        return transforms.Normalize(de_mean, de_std)(image)
    else:
        return image  # No normalization applied, return as-is

def prepare_image_with_patch_selection(image, g_x, patch_size, normalization_transform):
    # g_x should be of size (seq_len) by now
    assert g_x.dim() == 1, f'{g_x.size()=}'
    assert image.dim() == 3
    assert image.size(0) == 3
    # denormalize the image data
    if normalization_transform is not None:
        image = denormalize_image(image, normalization_transform)
    image = (image.clone() * 255).to(torch.uint8)
    patches_in_row = image.size(-1) // patch_size
    for token_index, token_weight in enumerate(g_x):
        # class token is the first in the sequence
        if token_index > 0:
            token_index -= 1
            patch_x, patch_y = divmod(token_index, patches_in_row)
            from_x, to_x = patch_x * patch_size, (patch_x + 1) * patch_size
            from_y, to_y = patch_y * patch_size, (patch_y + 1) * patch_size
            mask = torch.zeros(1, image.size(-2), image.size(-1), device=image.device, dtype=torch.bool)
            token_value = token_weight.item()
            mask[0, from_x:to_x, from_y:to_y] = True if token_value > 0.0 else False
            image = draw_segmentation_masks(image, mask, alpha=token_value, colors='red')
    return image.permute(1, 2, 0).numpy()



# def prepare_patch_selection_heatmap(image, g_x, patch_size):
#     # g_x should be of size (seq_len) by now
#     assert g_x.dim() == 1, f'{g_x.size()=}'
#     assert image.dim() == 3
#     assert image.size(0) == 3
#     patches_in_row = image.size(-1) // patch_size
#     print(patches_in_row)
#     print(image.size(-1))
#     print(patch_size)
#     print(g_x.size())
#     heatmap = torch.empty(patches_in_row, patches_in_row, device=image.device)
#     print(heatmap.size())
#     for token_index, token_weight in enumerate(g_x):
#         # class token is the first in the sequence
#         if token_index > 0:
#             token_index -= 1
#             print(token_index)
#             patch_x, patch_y = divmod(token_index, patches_in_row)
#             print(patch_x, patch_y)
#             heatmap[patch_x-1, patch_y-1] = token_weight
#     return heatmap.numpy()

def prepare_patch_selection_heatmap(image, g_x, patch_size):
    """
    Prepares patch-selection heatmaps for either a single patch size
    or multiple scales.

    If `patch_size` is an int, this behaves as the original function
    (returns a single heatmap).
    If `patch_size` is a list of scales, it returns a list of heatmaps,
    one for each scale.

    Assumptions:
    1) `g_x` is a 1D gating vector, possibly concatenated across scales.
    2) The first token in each scale segment is a 'class token' and is skipped.
    3) Each scale s uses s^2 patch tokens (plus 1 class token).
    4) The original code subtracts 1 from both patch_x and patch_y. We keep
       that for consistency, but be aware for `scale=1` this may lead to
       indexing at [-1, -1].

    Parameters
    ----------
    image : torch.Tensor
        Image of shape (3, H, W).
    g_x : torch.Tensor
        1D gating vector. If multiple scales are used, it contains concatenated
        segments: each scale has (1 + s^2) tokens (1 class token + s^2 patches).
    patch_size : int or List[int]
        - If int, the old single-scale behavior is used.
        - If list of ints, treat each entry as a scale and slice g_x accordingly.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        - If patch_size is an int, returns a single heatmap as a NumPy array.
        - If patch_size is a list of scales, returns a list of NumPy arrays
          (one per scale).
    """
    assert g_x.dim() == 1, f"{g_x.size()=}"
    assert image.dim() == 3, "image must be [C,H,W]"
    assert image.size(0) == 3, "image[0] must be 3 color channels (RGB)"
    patch_size = [1,2,3,4,5,6,8,10,13,16]
    # --- SINGLE SCALE BEHAVIOR (original code) ---
    if isinstance(patch_size, int):
        patches_in_row = image.size(-1) // patch_size
        print(patches_in_row)
        print(image.size(-1))
        print(patch_size)
        print(g_x.size())

        heatmap = torch.empty(patches_in_row, patches_in_row, device=image.device)
        print(heatmap.size())

        for token_index, token_weight in enumerate(g_x):
            # skip the first token as a "class token"
            if token_index > 0:
                token_index -= 1  # shift index
                print(token_index)
                patch_x, patch_y = divmod(token_index, patches_in_row)
                print(patch_x, patch_y)
                # original code subtracts 1 from patch_x/patch_y:
                heatmap[patch_x - 1, patch_y - 1] = token_weight

        return heatmap.cpu().numpy()

    # --- MULTI-SCALE BEHAVIOR ---
    else:
        # patch_size is a list of scales, e.g. [1,2,3,4,5,6,8,10,13,16]
        scales = patch_size
        heatmaps = []
        
        # We'll slice `g_x` for each scale: each scale has 1 class token + s^2 patch tokens
        offset = 0
        H = image.size(-1)  # assuming square image => width = height

        for s in scales:
            patches_in_row = H // s
            # for debugging/logging:
            print(f"Preparing scale={s} -> patches_in_row={patches_in_row}")

            # Allocate a 2D heatmap on the same device
            heatmap = torch.empty(patches_in_row, patches_in_row, device=image.device)

            # The first token is the class token => skip it
            # Then we have s^2 patch tokens
            # range of tokens for this scale segment:
            scale_token_count = 1 + s*s  # (class token) + (s^2 patch tokens)
            g_x_scale_segment = g_x[offset : offset + scale_token_count]
            offset += scale_token_count

            for token_index, token_weight in enumerate(g_x_scale_segment):
                if token_index == 0:
                    # skip class token
                    continue
                # shift index by 1 to align with patch positions
                real_index = token_index - 1
                patch_x, patch_y = divmod(real_index, patches_in_row)
                # original code subtracts 1 from each dimension
                heatmap[patch_x, patch_y] = token_weight

            heatmaps.append(heatmap.cpu().numpy())

        return heatmaps




def prepare_image(image, normalization_transform):
    image = denormalize_image(image, normalization_transform)
    image = image.permute(1, 2, 0).numpy()
    return image


def generate_spatial_load_figure(x, _pred, _label, spatial_load, patch_size, normalization_transform, mode):
    # TODO add label and prediction info to the image
    # spatial_load should be of size (seq_len)
    assert spatial_load.dim() == 1, f'{spatial_load.size()=}'
    if mode == 'mask':
        image = prepare_image_with_patch_selection(x, spatial_load, patch_size, normalization_transform)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        # fig.suptitle(f'prediction: {pred.argmax()} label: {label}')
        fig.set_tight_layout(True)
        return fig
    elif mode == 'separate':
        # 1) Prepare the original image
        image = prepare_image(x, normalization_transform)

        # 2) Prepare a list of heatmaps (one per scale)
        selection_heatmaps = prepare_patch_selection_heatmap(x, spatial_load, patch_size)
        # ^ This should now return a list of 2D numpy arrays, not a single array.
        import numpy as np
        np.set_printoptions(threshold=100)
        print("Heatmaps returned by `prepare_patch_selection_heatmap`:")

        for i, hm in enumerate(selection_heatmaps, start=1):
            print("********")
            print(f"scale {i}")
            # Print the entire 2D array for this scale
            print(hm)
            print("********")

        # 3) Figure out a good color range for all scales (optional)
        
        all_vals = np.concatenate([hm.ravel() for hm in selection_heatmaps])
        vmin, vmax = all_vals.min(), all_vals.max()

        # 4) Create subplots: one for the image, plus one per heatmap
        n_subplots = 1 + len(selection_heatmaps)
        fig, axes = plt.subplots(1, n_subplots, figsize=(8 * n_subplots, 8))

        # 5) Show the original image in the first subplot
        ax_img = axes[0]
        ax_img.imshow(image)
        ax_img.axis('off')  # turn off tick labels
        ax_img.set_title("Original image")

        # 6) Show each scale's heatmap
        last_imshow = None
        for i, hm in enumerate(selection_heatmaps, start=1):
            ax = axes[i]
            last_imshow = ax.imshow(
                hm, cmap='inferno', interpolation='nearest',
                vmin=vmin, vmax=vmax
            )
            ax.axis('off')
            ax.set_title(f"Scale Heatmap #{i}")

        # 7) Optionally add a colorbar to the right
        #    (this example places the colorbar in free space on the right side)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        fig.colorbar(last_imshow, cax=cbar_ax)

        fig.set_tight_layout(True)
        return fig
    elif mode == 'image_only':
        image = prepare_image(x, normalization_transform)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        # fig.suptitle(f'prediction: {pred.argmax()} label: {label}')
        fig.set_tight_layout(True)
        return fig
import gc

def main(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    args: arg_util.Args = arg_util.init_dist_and_get_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_names = args.display_names if args.display_names is not None else args.exp_names
    accelerator = Accelerator(split_batches=True)
    tc = TrainingContext()
    for exp_name, display_name in zip(args.exp_names, display_names):
        for exp_id in args.exp_ids:
            model, run_args, state, tc.model_var_wo_ddp= load_model(args, exp_name, exp_id)
            if run_args.model_class != 'dsti_router':
                logging.info(f'{exp_name}_{exp_id} is not a dynamic-k MoE model - skipping.')

                # Move to CPU and delete all references
                model.to('cpu')
                del model
                del run_args
                del state

                for obj in list(globals().values()):
                    if isinstance(obj, torch.Tensor) and obj.device.type == "cuda":
                        del obj

                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                print(f'Preparing model for: {exp_name}_{exp_id}')
                
                setup_accelerator(args, tc)
                model = tc.accelerator.prepare(model)
                make_vae(args, tc)
                
                logging.info(f'Generating patch selection plots for: {exp_name}_{exp_id}')
                selected_samples, least_compute, most_compute, patch_size, normalization_transform, loader = \
                    setup_and_process(args, model, run_args, accelerator, tc)
                assert selected_samples['x'].size(0) == len(args.data_indices), f'{selected_samples}'
                named_sets = [('selected', selected_samples), ('cheapest', least_compute), ('costliest', most_compute)]
                for name, tensor_set in named_sets:
                    images = tensor_set['x']
                    preds = tensor_set['y_pred']
                    labels = tensor_set['y']
                    gating_data = tensor_set['gating_data']
                    spatial_load = compute_spatial_load(accelerator, model, loader, gating_data, tc)
                    for j in range(images.size(0)):
                        for mode in ['image_only', 'mask', 'separate']:
                            # for mode in ['image_only', 'mask']:
                            fig = generate_spatial_load_figure(images[j],
                                                               preds[j],
                                                               labels[j],
                                                               spatial_load[j],
                                                               patch_size,
                                                               normalization_transform,
                                                               mode)
                            save_path = args.output_dir / f'{display_name}_spatial_load_{name}_{j}_{mode}.png'
                            fig.savefig(save_path)
                            logging.info(f'Figure saved in {str(save_path)}')
                            plt.close(fig)


if __name__ == "__main__":
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    main(args)
