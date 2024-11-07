# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func

# try:
    # from triton.ops.flash_attention import attention as attention_triton
from kernels.flash_attention import attention as attention_triton
# except ImportError:
#     attention_triton = None

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False]
headdim_vals = [64]
dim = 2048
dropout_p = 0.0

method = "Triton"

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)

            
            q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                requires_grad=True) for _ in range(3)]
            # Try both values of sequence_parallel and pick the faster one
            try:
                f, b = time_fwd_bwd(
                    attention_triton, q, k, v, causal, headdim**(-0.5),
                    False, repeats=repeats, verbose=False
                )
            except:
                f, b = float('nan'), float('inf')
            try:
                _, b0 = time_fwd_bwd(
                    attention_triton, q, k, v, causal, headdim**(-0.5),
                    True, repeats=repeats, verbose=False
                )
            except:
                b0 = float('inf')
            time_f[config, "Triton"] = f
            time_b[config, "Triton"] = min(b, b0) if min(b, b0) < float('inf') else float('nan')


            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            time_f_b[config, method] = time_f[config, method] + time_b[config, method]
            speed_f[config, method] = efficiency(
                flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                time_f[config, method]
            )
            speed_b[config, method] = efficiency(
                flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                time_b[config, method]
            )
            speed_f_b[config, method] = efficiency(
                flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                time_f_b[config, method]
            )
            print(
                f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
            )
