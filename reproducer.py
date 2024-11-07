import math
import torch


from flash_attn.utils.benchmark import benchmark_fwd_bwd


from kernels.flash_attention import attention as attention_triton

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30 # 1 also fails
device = 'cuda'
dtype = torch.float16
dim = 2048
dropout_p = 0.0

method = "Triton"

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

causal = False
headdim = 64
batch_size = 2
seqlen= 8192


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
