import torch
import triton
import triton.language as tl

#以二维矩阵的角度进行矩阵的计算
@triton.jit
def matmul_kernel(
x_ptr, #输入矩阵的首元素指针 
w_ptr, #权重矩阵的首元素指针
z_ptr, #输出结果地址
M, N, K, #矩阵的维度
BLOCK_SIZE_M:tl.constexpr=128, #块大小
BLOCK_SIZE_N:tl.constexpr=128,
BLOCK_SIZE_K:tl.constexpr=64,
):
    # 1. 计算每个线程块处理的范围
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 2.每个triton_block处理的范围都在M，N轴上，并且处理的大小为BLOCK_SIZE_M，BLOCK_SIZE_N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    # 3.构造z
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 4.在K轴上进行分块归约
    for k in range(0, K, BLOCK_SIZE_K):
        # 4.1加载输入矩阵的分块
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other = 0.0)
        # 4.2加载权重矩阵的分块
        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(offs_n < N) & (w_k < K), other = 0.0)
        # 4.3矩阵乘法累加
        z = tl.dot(x, w, acc=z)
    # 5.存储结果
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)
    tl.store(z_ptr + z_offset, z, mask=z_mask)

@torch.no_grad()
def matmul(x, w):
    output_shape_0 = x.shape[:-1]
    x = x.view((-1, x.shape[-1]))
    M, K = x.shape
    N = w.shape[1]
    # 1.分配输出矩阵
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    # 2.配置网格
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    # 3.启动kernel
    matmul_kernel[grid](
        x,
        w,
        z,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return z.view((*output_shape_0, N))

if __name__ == '__main__':
    batch_size = 64
    sequence_len = 128
    hidden_size = 1280

    output_size = 2560

    x = torch.randn((batch_size, sequence_len, hidden_size), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden_size, output_size), device='cuda', dtype=torch.float16)

    output = matmul(x, weight)
    golden = x@weight
    print(f"结果正确: {torch.allclose(output, golden, atol=1e-3, rtol=1e-3)}")