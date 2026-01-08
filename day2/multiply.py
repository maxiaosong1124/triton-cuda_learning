import torch
import triton 
import triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:tl.constexpr,):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grids = lambda meta:(triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grids](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
output = add(x, y)
print(torch.allclose(output, x * y))