# Linear Probe 训练指南

## 概述

`LinearProbeTrainer` 类用于训练1024个linear probe网络，每个probe对应一个attention head，学习从head输出预测语义先验偏置强度（g_u值）。

## 架构

### 简单线性映射（推荐）

```
输入: head_vector (128维)
  ↓
Linear(128, 1)
  ↓
Sigmoid
  ↓
输出: g_u (标量, 0-1之间)
```

**参数数量**: 129 (128权重 + 1偏置)

### 单隐藏层（可选）

```
输入: head_vector (128维)
  ↓
Linear(128, hidden_dim)  # hidden_dim通常为8
  ↓
ReLU
  ↓
Dropout(0.2)  # 可选
  ↓
Linear(hidden_dim, 1)
  ↓
Sigmoid
  ↓
输出: g_u (标量, 0-1之间)
```

**参数数量**: 128 × hidden_dim + hidden_dim + hidden_dim + 1

## 使用方法

### 1. 基本使用

```python
from linear_probe_trainer import LinearProbeTrainer

# 创建trainer
trainer = LinearProbeTrainer(
    num_layers=32,
    num_heads=32,
    input_dim=128,
    hidden_dim=None,  # None表示使用简单线性映射
    use_dropout=False,
    device="cuda:0"
)

# 训练所有probe
results = trainer.train_all(
    ground_truth_dir="train/coco_train_500_head_ground_truth",
    save_dir="train/linear_probe_models",
    batch_size=64,
    num_epochs=100,
    lr=0.001,
    weight_decay=0.01,
    patience=10
)
```

### 2. 命令行使用

```bash
python train/linear_probe_trainer.py \
    --ground-truth-dir train/coco_train_500_head_ground_truth \
    --save-dir train/linear_probe_models \
    --device cuda:0 \
    --batch-size 64 \
    --num-epochs 100 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --patience 10
```

### 3. 使用单隐藏层

```python
trainer = LinearProbeTrainer(
    num_layers=32,
    num_heads=32,
    input_dim=128,
    hidden_dim=8,  # 使用8维隐藏层
    use_dropout=True,  # 启用Dropout
    device="cuda:0"
)
```

### 4. 加载已训练的模型

```python
# 创建trainer
trainer = LinearProbeTrainer(
    num_layers=32,
    num_heads=32,
    input_dim=128,
    device="cuda:0"
)

# 加载模型
trainer.load_models("train/linear_probe_models")

# 使用模型进行预测
head_vector = torch.tensor([...])  # [128] 或 [batch_size, 128]
g_u = trainer.predict(layer_idx=0, head_idx=0, head_vector=head_vector)
```

## 数据格式

真值对数据应该按以下格式组织：

```
ground_truth_dir/
├── layer_0_head_0.json
├── layer_0_head_1.json
├── ...
└── layer_31_head_31.json
```

每个JSON文件包含一个真值对列表，格式如下：

```json
[
  {
    "case_id": 1,
    "layer": 0,
    "head": 0,
    "head_vector": [0.123, -0.456, ..., 0.789],  // 128维向量
    "g_u": 0.5307,  // 目标标签
    "s_u": 0.1234,
    "delta_log_p_plus": 0.001,
    "delta_log_p_minus": 0.002,
    "case_type": "POPE"
  },
  ...
]
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_ratio` | 0.8 | 训练集比例 |
| `val_ratio` | 0.1 | 验证集比例 |
| `test_ratio` | 0.1 | 测试集比例 |
| `batch_size` | 64 | 批量大小 |
| `num_epochs` | 100 | 最大训练轮数 |
| `lr` | 0.001 | 学习率 |
| `weight_decay` | 0.01 | L2正则化系数 |
| `patience` | 10 | 早停patience |

## 评估指标

训练完成后，会输出以下评估指标：

- **MSE (均方误差)**: 衡量预测值与真实值的差异
- **R² (决定系数)**: 衡量模型解释数据变异的比例（越接近1越好）
- **MAE (平均绝对误差)**: 预测误差的平均值

## 输出文件

训练完成后，会在 `save_dir` 目录下生成：

```
save_dir/
├── layer_0_head_0.pth
├── layer_0_head_1.pth
├── ...
├── layer_31_head_31.pth
└── training_results.json
```

`training_results.json` 包含每个probe的训练统计信息：

```json
{
  "layer_0_head_0": {
    "status": "trained",
    "num_samples": 6000,
    "train_size": 4800,
    "val_size": 600,
    "test_size": 600,
    "best_val_loss": 0.001234,
    "test_loss": 0.001456,
    "test_r2": 0.8523,
    "test_mae": 0.0234,
    "num_epochs": 45
  },
  ...
}
```

## 推荐配置

### 配置1：简单线性映射（首选）

```python
trainer = LinearProbeTrainer(
    num_layers=32,
    num_heads=32,
    input_dim=128,
    hidden_dim=None,  # 简单线性映射
    use_dropout=False,
    device="cuda:0"
)

trainer.train_all(
    ground_truth_dir="...",
    save_dir="...",
    batch_size=64,
    num_epochs=100,
    lr=0.001,
    weight_decay=0.01,  # L2正则化
    patience=10
)
```

**优点**:
- 参数少，不会过拟合
- 训练快速
- 可解释性强
- 符合linear probe的设计理念

### 配置2：单隐藏层（如果简单线性效果不好）

```python
trainer = LinearProbeTrainer(
    num_layers=32,
    num_heads=32,
    input_dim=128,
    hidden_dim=8,  # 8维隐藏层
    use_dropout=True,  # 启用Dropout
    device="cuda:0"
)

trainer.train_all(
    ground_truth_dir="...",
    save_dir="...",
    batch_size=64,
    num_epochs=100,
    lr=0.001,
    weight_decay=0.01,
    patience=10
)
```

**适用场景**:
- 简单线性映射的R² < 0.5
- 需要捕捉非线性关系

## 注意事项

1. **数据量**: 确保每个head有足够的训练样本（建议至少1000个）
2. **设备**: 如果使用GPU，确保有足够的显存（1024个模型需要一定内存）
3. **训练时间**: 训练1024个模型需要一定时间，建议使用GPU加速
4. **早停**: 使用早停机制可以防止过拟合，patience参数控制容忍度

## 故障排除

### 问题1: 某些probe没有数据

**原因**: 某些head在生成真值对时没有对应的数据

**解决**: 检查真值对生成过程，确保所有head都有数据

### 问题2: 训练损失不下降

**原因**: 可能学习率过大或过小

**解决**: 尝试调整学习率（如0.0001或0.01）

### 问题3: 验证损失上升（过拟合）

**原因**: 模型容量过大或正则化不足

**解决**:
- 增加weight_decay（如0.1）
- 如果使用隐藏层，减小hidden_dim
- 启用Dropout

### 问题4: R²值很低

**原因**: 模型无法捕捉head输出与g_u之间的关系

**解决**:
- 尝试使用单隐藏层
- 检查数据质量
- 增加训练样本数
