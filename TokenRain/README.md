# TokenRain 代码目录

设计文档见仓库根目录的 [`TokenRain-README.md`](../TokenRain-README.md)。

## 结构

```text
TokenRain/
├── configs/tokenrain_100m.yaml
├── scripts/
│   ├── build_vocab.py      # 生成 7,848 汉字 + 48 符号
│   ├── prepare_data.py     # 中文过滤、JSONL、uint16 打包
│   ├── run_train.py        # 单进程训练
│   └── generate.py         # 从 checkpoint 续写
├── tokenrain/
│   ├── config.py
│   ├── tokenizer.py        # 固定字符映射 + byte fallback
│   ├── data.py
│   ├── model.py            # Token-Shift / QK-Norm / Partial RoPE / ReLU²
│   ├── train.py
│   └── utils.py
└── tests/test_tokenrain.py
```

## 快速开始

```bash
# 1. 生成词表（首次运行；也可在替换 frequence.txt 后重新运行）
python scripts/build_vocab.py

# 2. 准备数据
python scripts/prepare_data.py --config configs/tokenrain_100m.yaml

# 3. 训练（单 GPU / CPU；本精简版不含 DDP）
python scripts/run_train.py --config configs/tokenrain_100m.yaml

# 4. 生成
python scripts/generate.py \
  --checkpoint outputs/tokenrain_100m_v0.2/latest.pt \
  --prompt "从前有一只小乌龟，"

# 测试
python -m pytest tests -q
```

## 说明

- 词表固定为 8,192 = 4 个特殊 token + 256 个字节 token + 7,848 汉字 + 48 符号 + 36 预留。
- ASCII 直接复用 `0x00`–`0x7F` 字节 token；表外字符走 UTF-8 byte fallback。
- `build_vocab.py` 默认使用 `data/tokenizer/tokenrain/frequence.txt`（维基百科字频）挑选高频三级字；如果需要严格按本地语料选字，替换为同格式字频表后重新运行即可。
- 训练循环只支持单进程，去掉了 DDP、按秒日志、checkpoint 归档等非必要逻辑。
- `generate.py` 为了保持代码精简，采用全量重算，没有实现 KV Cache。
