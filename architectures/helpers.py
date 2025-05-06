import torch
from torch import nn as nn
from torch.nn import functional as F
import math


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    n = abs(num_samples)
    flat_logits = logits_BlV.softmax(dim=-1).view(-1, V)
    samples = torch.multinomial(
        flat_logits,
        num_samples=n,
        replacement=replacement,
        generator=rng
    ).view(B, l, n)

    # ---- Per‑token probabilities ----
    flat_samples      = samples.view(-1, n)                                    # (B*l, n)
    sample_probs_flat = flat_logits.gather(dim=1, index=flat_samples)          # (B*l, n)
    sample_probs      = sample_probs_flat.view(B, l, n)                        # (B, l, n)

    # (optional) detailed probs for the first sample
    probs     = sample_probs[0].flatten().tolist()
    formatted = ", ".join(f"{p*100:.3f}%" for p in probs)                      # "0.257%, 0.431%, …"

    # ---- Batch‑level stats ----
    per_batch         = sample_probs.view(B, -1)                               # (B, l*n)
    mins, maxs, means = (per_batch.min(1).values,
                        per_batch.max(1).values,
                        per_batch.mean(1))
    avg_min, avg_max, avg_mean = (mins.mean().item()*100,
                                maxs.mean().item()*100,
                                means.mean().item()*100)

    # ---- Average (top‑1 – top‑2) log‑prob gap ----
    top2_vals = flat_logits.topk(2, dim=1).values                              # (B*l, 2)
    avg_gap   = (top2_vals[:, 0] - top2_vals[:, 1]).mean().item()*100

    # ---- Entropy per image ----
    p_safe = flat_logits.clamp(min=1e-12)
    H      = -(p_safe * p_safe.log()).sum(dim=-1)                    # (B*l,)
    H_b    = H.view(B, l).mean(dim=1)                                # (B,)

    # ────────── NEW SCALE‑INVARIANT CONFIDENCE METRICS ──────────
    norm_entropy = (H.mean() / math.log(V)).item()                   # ∈[0,1]

    # ---- Greedy‑token class histogram (first image) ----
    greedy          = logits_BlV.argmax(dim=-1)
    counts          = torch.bincount(greedy[0], minlength=V)
    top_class       = counts.argmax().item()
    top_class_count = counts.max().item()


    # ---- adaptive‑threshold confidence ------------------------------
    max_p        = flat_logits.max(dim=-1).values                 # (B*l,)
    uniform_p    = 1.0 / V
    conf_64x      = (max_p > 64  * uniform_p).float().mean().item() * 100
    mean_max_p   = max_p.mean().item() * 100                     # %

    # ────────── ONE‑LINER SUMMARY (add to existing print) ──────────
    #print(
        #f"avg_min={avg_min:.3f}% | avg_max={avg_max:.3f}% | avg_mean={avg_mean:.3f}% | "
        #f"avg_gap={avg_gap:.3f}% |" 
        #f"entropy={H_b.mean():.4f}"
        #f"entropy={H}"
        #f"norm_entropy={norm_entropy:.4f} | "
        #f"mean_max_p={mean_max_p:.2f}% | "
        #f"conf@64xU={conf_64x:.2f}% | "
        #f"top_class={top_class} ({top_class_count}/{l} tokens)"
    #)
    if H.shape[0] == 128*16*16:
        outfile = os.path.join("entropy_values.bin")
        ent_tensor = H.detach()              # torch.Tensor
        # 2.  Move to CPU and convert to float32 numpy
        ent_np = ent_tensor.cpu().numpy().astype(np.float32)

        # 4.  Append to the binary file
        with open(outfile, "ab") as f:
            f.write(ent_np.tobytes())

    return samples
    # replacement = num_samples >= 0
    # num_samples = abs(num_samples)
    # return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'
