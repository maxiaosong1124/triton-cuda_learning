# MLP算子融合策略对比分析

## 概述

在深度学习推理优化中，MLP（多层感知机）算子的融合策略直接影响性能。本文对比两种主流的融合方案：
- **lite_llama**: silu + mul 融合（简单易维护）
- **triton_mlp**: matmul + silu 融合（性能更优）

---

## 一、实现方式对比

### 1.1 lite_llama 方案（silu + mul 融合）

**计算流程：**
```python
def torch_mlp_silu(x, w1, w2, w3):
    y1 = torch.mm(x, w1)           # 第1步：普通matmul
    y2 = torch.mm(x, w2)           # 第2步：普通matmul
    out = swiglu_forward(y1, y2)   # 第3步：融合 silu(y1) * y2
    mlp_out = torch.mm(out, w3)    # 第4步：最后的matmul
    return mlp_out
```

**kernel 实现：**
```python
@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, ...):
    a_row = tl.load(a_ptr + col_offsets, mask=mask).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask)
    c_row = silu(a_row) * b_row  # 融合点：silu + mul
    tl.store(c_ptr + col_offsets, c_row, mask=mask)
```

**特点：**
- ✅ 代码简洁清晰
- ✅ kernel逻辑简单，易于维护
- ✅ 数值精度高（silu在fp32中计算）
- ❌ 需要从全局内存加载两个中间结果(y1, y2)

---

### 1.2 triton_mlp 方案（matmul + silu 融合）

**计算流程：**
```python
def mlp(inputs, w1t, w2t, w3t):
    w1x = matmul(inputs, w1t, True)   # 第1步：matmul + silu融合
    w3x = matmul(inputs, w3t)         # 第2步：普通matmul
    mul_out = vector_mul(w1x, w3x)    # 第3步：向量乘法
    out = matmul(mul_out, w2t)        # 第4步：最后的matmul
    return out
```

**kernel 实现：**
```python
@triton.jit
def matmul_kernel(..., need_silu):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, ...)
        b = tl.load(b_ptrs, ...)
        accumulator = tl.dot(a, b, accumulator)
    
    if need_silu:
        sigmoid_x = 1. / (1. + tl.exp(-accumulator))
        c = accumulator.to(tl.float16) * sigmoid_x.to(tl.float16)
    else:
        c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)
```

**特点：**
- ✅ 性能更优（避免中间结果的内存往返）
- ✅ 内存带宽利用率高
- ❌ kernel逻辑复杂
- ❌ 代码维护成本高

---

## 二、性能对比分析

### 2.1 内存访问模式

| 方案 | 内存写入 | 内存读取 | 总带宽 |
|------|--------|--------|--------|
| **lite_llama** | y1(MN) + y2(MN) | y1(MN) + y2(MN) | 4MN |
| **triton_mlp** | w1x(MN) | w1x(MN) | 2MN |
| **节省比例** | - | - | **50%** |

**具体分析：**

**lite_llama 的数据流：**
```
inputs → [matmul] → y1 (写入内存)
inputs → [matmul] → y2 (写入内存)
y1, y2 → [swiglu] → (读取内存) → out (写入内存)
out → [matmul] → output
```
- 中间结果 y1, y2 需要往返内存两次

**triton_mlp 的数据流：**
```
inputs → [matmul+silu] → w1x (寄存器→内存)
inputs → [matmul] → w3x (寄存器→内存)
w1x, w3x → [vector_mul] → mul_out (内存→寄存器→内存)
mul_out → [matmul] → output
```
- silu在matmul的accumulator（寄存器）中计算，避免了额外的内存访问

### 2.2 计算效率

**triton_mlp 优势：**
1. **寄存器复用**：silu计算在fp32 accumulator上进行，无需额外的全局内存访问
2. **缓存友好**：w1x直接从matmul kernel流向vector_mul，减少L2缓存压力
3. **指令级并行**：silu计算与后续dot product迭代可以更好地重叠

**lite_llama 优势：**
1. **数值精度**：silu在fp32中计算，精度更高
2. **代码简洁**：kernel逻辑清晰，易于理解和维护

### 2.3 性能估算

假设典型LLM参数：M=4, N=2048, K=4096

**内存带宽节省：**
```
节省带宽 = 2 × (M × N × 2字节) = 4MN字节
         = 4 × 4 × 2048 × 2 = 65,536 字节 ≈ 64KB
```

**在A100 GPU上的时间节省：**
- A100 内存带宽：2TB/s
- 时间节省 ≈ 64KB / 2TB/s ≈ **0.03ms**
- 对于大批量推理，累积效果显著

---

## 三、选择建议

### 3.1 选择 lite_llama 方案的场景：
- ✅ 代码可维护性优先
- ✅ 团队规模小，维护成本高
- ✅ 对性能要求不是极致
- ✅ 需要快速迭代和调试

### 3.2 选择 triton_mlp 方案的场景：
- ✅ 性能优先（推理延迟敏感）
- ✅ 大规模部署（性能收益显著）
- ✅ 团队有Triton优化经验
- ✅ 需要极致性能

---

## 四、总结

| 维度 | lite_llama | triton_mlp |
|------|-----------|-----------|
| **代码复杂度** | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| **维护成本** | ⭐ 低 | ⭐⭐⭐ 高 |
| **性能** | ⭐⭐ 中等 | ⭐⭐⭐ 优秀 |
| **内存带宽** | 4MN | 2MN |
| **数值精度** | ⭐⭐⭐ 高 | ⭐⭐ 中等 |
| **学习曲线** | ⭐ 平缓 | ⭐⭐⭐ 陡峭 |

**核心结论：**
- **lite_llama** 采用 silu + mul 融合，代码简洁易维护，适合快速开发
- **triton_mlp** 采用 matmul + silu 融合，性能更优，适合生产环境
- 两者各有优劣，选择应根据实际需求（性能 vs 可维护性）权衡

