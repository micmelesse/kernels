import torch
from flash_attn.utils.benchmark import benchmark_fwd_bwd
from kernels.flash_attention import attention as attention_triton

# params
repeats = 30 # 1 also fails
device = 'cuda'
dtype = torch.float16
dim = 2048
dropout_p = 0.0
method = "Triton"
causal = False
headdim = 64
batch_size = 2
seqlen= 8192
config = (causal, headdim, batch_size, seqlen)
nheads = dim // headdim

# inputs
q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                    requires_grad=True) for _ in range(3)]

# Sequence Parallel has issues when size is large
sequence_parallel = True
try:
    _, b0 = benchmark_fwd_bwd(
        attention_triton, q, k, v, causal, headdim**(-0.5),
        sequence_parallel, repeats=repeats, verbose=True
    )
except Exception as e:
    print(f"An error occurred: {e}")
    b0 = float('inf')
print("Success!")