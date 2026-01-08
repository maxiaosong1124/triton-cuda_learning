import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, gamma_ptr, output_ptr, n_rows, n_cols, eps:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(axis=0)
    if row_idx >= n_rows:
        return
    row_start_ptr = x_ptr + row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(row_start_ptr + offsets, mask = mask, other=0.0)
    mean = tl.sum(x*x, axis=0) / n_cols
    rms = tl.sqrt(mean + eps)
    x = x / rms
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    output = x * gamma
    tl.store(output_ptr + row_idx * n_cols + offsets, output, mask=mask)

def rmsnorm(x, gamma, eps=1e-5):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grids = lambda meta: (n_rows,)
    rmsnorm_kernel[grids](x, gamma, output, n_rows, n_cols, eps, BLOCK_SIZE=1024)
    return output

x = torch.randn(10000, 100, device='cuda', dtype=torch.float32)
gamma = torch.ones(x.shape[-1], device='cuda', dtype=torch.float32)  # gamma 初始化为 1
output = rmsnorm(x, gamma)

# 计算期望的 RMSNorm 结果（不使用 gamma）
rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / x.shape[-1] + 1e-5)
expected = x / rms

print(f"RMSNorm 结果正确: {torch.allclose(output, expected, atol=1e-3, rtol=1e-3)}")
