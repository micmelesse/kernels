import torch
import torch.utils.benchmark as benchmark
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

# NOTE: source of issue
sequence_parallel = True # Sequence Parallel has issues when size is large


# inputs
q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                    requires_grad=True) for _ in range(3)]

def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )

try:
    _, b0 = benchmark_fwd_bwd(
        attention_triton, q, k, v, causal, headdim**(-0.5),
        sequence_parallel, repeats=repeats, verbose=True
    )
except Exception as e:
    print(f"An error occurred: {e}")
    b0 = float('inf')


print("Success!")