# TokenRain-100M-Base

**版本：** v0.2  
**日期：** 2026-11-06  
**定位：** 面向现代规范简体中文与自研硬件加速器的字符级 Decoder-only 语言模型。  
**状态：** 设计稿，尚未训练。

> 固定字符表 + byte fallback；一个汉字一个 token；不做子词分词器训练。  
> 全模型维度严格对齐 $2^N$；词表 8,192，模型 103.9M 参数，最大上下文 4,096。

## 模型概述

TokenRain 是一个针对硬件计算宽度对齐与后续极低比特量化（如 `Q1_0`）深度优化的 Decoder-only Causal LM。采用 Pre-Norm Transformer 骨架，不引入 MoE、Mamba 或复杂注意力变体，通过轻量微结构（`Token-Shift` + `QK-Norm` + `ReLU²`）在极简硬件算子下维持高表现力。

| 项目 | 设定 | 硬件对齐 ($2^N$) |
|---|---:|---:|
| 模型类型 | Decoder-only Causal LM | - |
| 参数量 | 103,856,640（约 103.9M） | - |
| Transformer 层数 | 14 | - |
| Hidden Size | 1,024 | $2^{10}$ |
| Attention Heads | 16 | $2^4$ |
| KV Heads | 4（GQA，4:1） | $2^2$（扇出比 $2^2$） |
| Head Dimension | 64 | $2^6$ |
| FFN Intermediate Size | 2,048（双矩阵非门控） | $2^{11}$ |
| Vocabulary Size | 8,192（8,156 实际 + 36 预留） | $2^{13}$ |
| 最大上下文长度 | 4,096 tokens | $2^{12}$ |
| Embedding / LM Head | 权重共享（tied） | - |
| 位置编码 | Partial RoPE（前 32 维，theta 10,000） | $2^5$ |
| 局部序列增强 | Token-Shift（1/4 通道时间移位，256 维） | $2^8$ |

## 分词器

不使用训练得到的子词分词器，而是一个严格对齐到 $8,192$ ($2^{13}$) 的固定字符映射：

| 部分 | 数量 | ID 范围 | 说明 |
|---|---:|---|---|
| 特殊 token | 4 | 0–3 | `<pad> <bos> <eos> <unk>` |
| 字节 token | 256 | 4–259 | `0x00`–`0xFF`，UTF-8 fallback |
| 通用规范汉字 | 7,848 | 260–8,107 | 一级 3,500 + 二级 3,000 + 高频三级 1,348 |
| 中文标点/符号 | 48 | 8,108–8,155 | 高频中文标点与全角符号 |
| 预留 | 36 | 8,156–8,191 | 后续扩展与对齐 |
| **合计** | **8,192** | 0–8,191 | $8,192 = 2^{13} = 64 \times 128$ |

- **汉字覆盖与长尾裁剪**：完整保留《通用规范汉字表》一、二级字（共 6,500 字，覆盖现代中文语料 >99.99%），并按语料字频保留前 1,348 个三级字；剔除的 257 个极低频三级生僻字自然走 3 字节 UTF-8 回退，避免低频死 token 污染共享 LM Head，同时使词表精准锁定在 13-bit 满寻址空间（`8,192`）。
- **ASCII 零开销复用**：ASCII 的 128 个字符直接复用字节 token `0x00`–`0x7F`，1 字符 = 1 token，不额外占槽位。
- **无损往返**：表内汉字/符号 1 字符 = 1 token；表外字符按 UTF-8 编码拆成 1–4 个字节 token；任意合法 UTF-8 文本都能无损往返。
- **无词边界符**：空格、换行就是普通字节 token；不使用 `▁` 词边界标记。
- **标点直通**：48 个标点避免 `，。！？` 等高频符号走 3 字节回退。

编码示例：`人工智能正在改变世界。` → 11 tokens；`The quick brown fox.` → 20 tokens；`🙂` → 4 个字节 token。

设计说明：

- 汉字本身就是自然的中文建模单位，不需要分词工具，也不会出现切分歧义。
- 词表严格为 $2^{13}$，硬件片上查表（LUT）无地址空洞，LM Head 的 `Argmax` / `Top-K` 可直接使用对称二叉树归约。
- 代价是中文 token 数约为子词方案的 1.2–2.2 倍，英文/代码为 1 token/字节；繁体、异体和极生僻字按 3 字节回退。

## 网络结构

### 单层结构

1. RMSNorm（`attn_norm`）
2. **Token-Shift**（前 256 维替换为 $t-1$ 时刻特征）
3. Grouped-Query Attention（含 **QK-Norm** 与 **Partial RoPE**）
4. Residual Add
5. RMSNorm（`ffn_norm`）
6. **Token-Shift**（前 256 维替换为 $t-1$ 时刻特征）
7. **ReLU²** Feed-Forward Network（双矩阵结构）
8. Residual Add

14 个相同结构的 Block 堆叠后接 Final RMSNorm，再接与 Token Embedding 共享权重的 LM Head。

### Token-Shift（零参数局部聚合）

在进入 Attention 的 Q/K/V 投影前以及 FFN 的 Up 投影前，对归一化后的特征张量执行通道级时间移位：

$$\text{TokenShift}(x_t) = \text{Concat}\big(x_{t-1}[0:256], \; x_t[256:1024]\big)$$

- **0 参数、0 算力开销**：在自研加速器上仅需一个 256 维的寄存器缓存上一拍向量（Prefill 阶段为简单地址偏移）。
- **解决字符/字节拼词瓶颈**：使第 1 层起每个线性投影天然具备 Bigram 局部感受野，大幅减轻浅层 Attention 头组合多字词与多字节 UTF-8 序列的负担。

### Attention

- **全 $2^N$ 几何对齐**：Query Heads 16，Key / Value Heads 4，每个 KV Head 严格广播对应 **4 个** Query Heads（4:1 GQA）。
- **投影维度**：Head Dimension 64；`q_proj` 输出 1,024，`k_proj`、`v_proj` 输出 256，`o_proj` 输出 1,024；均不使用 bias。
- **QK-Norm**：在施加位置编码前，对每个 Head 的 Q 和 K 独立执行维度为 64 的 RMSNorm（`q_norm` 共 1,024 参数，`k_norm` 共 256 参数），锁死注意力点积动态范围，适配定点化 Softmax 电路与低比特量化。
- **Partial RoPE**：仅对每个 Head 的前 **32 维**（50%）施加旋转位置编码（theta 10,000），后 32 维保持位置无关直通；三角函数 LUT 与复数旋转电路开销减半。
- **KV Cache 占用**：4,096 上下文下，bf16 推理时 KV Cache 为 **56 MiB**（int8 量化下仅 **28 MiB**）。

### FFN（双矩阵 ReLU²）

摒弃 3 矩阵 SwiGLU，采用对硬件与极低比特量化更友好的双矩阵 **Squared ReLU** 结构：

$$\text{FFN}(x) = \text{down\_proj}\big(\text{ReLU}(x \cdot W_{\text{up}})^2\big)$$

- `up_proj`：1,024 → 2,048
- `down_proj`：2,048 → 1,024
- 两个投影均不使用 bias。
- **硬件与量化优势**：
  1. 彻底移除 `SiLU` 所需的指数/除法拟合电路，仅需比较器与一次整数/定点自乘；
  2. 消除门控相乘带来的激活值离群尖峰（Outliers），大幅提升后续 `Q1_0` / `W1A8` 量化的信噪比；
  3. 天然具备 >75% 的激活稀疏度，加速器可直接支持零值跳过（Zero-Skipping），且比 3 矩阵 GLU 减少 1/3 的权重访存流。

### 归一化与其他

| 项目 | 设定 |
|---|---|
| Normalization | RMSNorm，Pre-Norm + Head-wise QK-Norm，epsilon 1e-5 |
| Activation | $\text{ReLU}^2(x) = (\max(0, x))^2$ |
| Temporal Mixing | Token-Shift（1/4 通道，256 维） |
| Residual | 标准残差连接 |
| Attention / Residual / FFN Dropout | 0.0 |
| Linear Bias | 全部关闭 |

## 参数量

| 参数部分 | 维度明细 | 参数量 |
|---|---|---:|
| 单层 Attention：Q、K、V、O | $1024^2 \times 2 + 1024 \times 256 \times 2$ | 2,621,440 |
| 单层 FFN：Up、Down | $1024 \times 2048 \times 2$ | 4,194,304 |
| 单层 RMSNorm：Attn、FFN、Q、K | $1024 + 1024 + 1024 + 256$ | 3,328 |
| **单层 Transformer Block 合计** | - | **6,819,072** |
| **14 层 Transformer Blocks** | $14 \times 6,819,072$ | **95,467,008** |
| Token Embedding（与 LM Head 共享） | $8,192 \times 1,024$ | 8,388,608 |
| Final RMSNorm | $1,024$ | 1,024 |
| **总计** | - | **103,856,640** |

- Embedding 占总参数的 **8.08%**，其余 **91.92%** 均集中在 Transformer Block 中。
- 后续主干网络（95.5M 参数）量化至 `Q1_0` 后权重体积约为 **12 MiB**；Embedding 层（8.4M 参数）保持 `Q8_0` 精度体积仅 **8 MiB**，全模型可轻松常驻小型加速器片上或近存空间。

## 数据管线

```text
raw_dirs/*.txt  ->  jsonl/{train,val}.jsonl  ->  packed/{train.bin,val.bin}
```

- **中文优先**：按文档统计 CJK 字符占比，默认保留 CJK ≥ 50% 的文档；可选保留 5%–10% 英文/代码，其余丢弃。
- **清洗**：统一 UTF-8 解码，去除 NUL 和大多数控制字符，保留 `\n`、`\t`；默认 NFC 规范化。
- **切分与去重**：按空行分段，按 `doc_chars`（建议 2,000–4,000）拼文档；整篇 MD5 去重。
- **划分**：`md5(source:idx) % 10000` 决定 train/val，默认 val_ratio 2%，确定性、重跑不漂移。
- **打包**：文档用 `<bos> … <eos>` 包裹，按 seed 打乱后首尾相接，对齐到 4,096，以 `uint16`（有效位宽 13-bit）存盘。
- **训练采样**：每个 epoch 以 `train.seed + epoch` 为随机种子生成序列级随机排列，按批随机读取；epoch 未跑完时也是全语料的均匀随机子集，续训从 `(epoch, cursor)` 确定性恢复。

## 定位

TokenRain 的核心赌注是：对中文来说，字符本身就是足够好的 token；对端侧与自研硬件来说，规整的 $2^N$ 维度与极简算子就是最高的效率。它用 8,192 的固定词表和 103.9M 参数，换一个简单、硬件极致友好、可复现、可压至极低比特的中文模型，而不是追求英文全能。
