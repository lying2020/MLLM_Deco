# Linear Probe 设计指南

## 1. 问题分析

### 1.1 当前情况

- **每个 head 的训练样本数**：$N = 500 \times 12 = 6000$ 个真值对
- **Head 维度**：$d_h = 128$
- **目标**：学习映射 $f: \mathbb{R}^{128} \rightarrow [0, 1]$，其中 $f(x) = g_u^{(l,n)}$

### 1.2 过拟合 vs 欠拟合的权衡

**过拟合（Overfitting）**：模型参数过多，在训练集上表现好但泛化能力差
**欠拟合（Underfitting）**：模型参数过少，无法捕捉数据中的模式

对于 linear probe，我们需要在**模型容量**和**训练样本数**之间找到平衡。

## 2. 参数数量与样本数量的关系

### 2.1 经验法则

在机器学习中，常用的经验法则是：

$$\text{样本数} \geq 10 \times \text{参数数}$$

对于更复杂的模型或高维数据，可能需要：

$$\text{样本数} \geq 20 \times \text{参数数}$$

**LaTeX代码：**
```latex
\text{样本数} \geq 10 \times \text{参数数}
```

### 2.2 当前情况分析

**方案1：简单线性映射（128×1）**
- 参数数：$P_1 = 128 + 1 = 129$（权重 + 偏置）
- 样本数：$N = 6000$
- 比例：$\frac{N}{P_1} = \frac{6000}{129} \approx 46.5$

**结论**：比例远大于10，**不会过拟合**，但可能**欠拟合**（模型太简单，无法捕捉复杂模式）

**方案2：单隐藏层（128×64×1）**
- 参数数：$P_2 = 128 \times 64 + 64 \times 1 + 64 + 1 = 8192 + 64 + 64 + 1 = 8321$
- 样本数：$N = 6000$
- 比例：$\frac{N}{P_2} = \frac{6000}{8321} \approx 0.72$

**结论**：比例小于1，**会严重过拟合**

**方案3：单隐藏层（128×32×1）**
- 参数数：$P_3 = 128 \times 32 + 32 \times 1 + 32 + 1 = 4096 + 32 + 32 + 1 = 4161$
- 样本数：$N = 6000$
- 比例：$\frac{N}{P_3} = \frac{6000}{4161} \approx 1.44$

**结论**：比例略大于1，**可能过拟合**

**方案4：单隐藏层（128×16×1）**
- 参数数：$P_4 = 128 \times 16 + 16 \times 1 + 16 + 1 = 2048 + 16 + 16 + 1 = 2081$
- 样本数：$N = 6000$
- 比例：$\frac{N}{P_4} = \frac{6000}{2081} \approx 2.88$

**结论**：比例接近3，**可能轻微过拟合**，但可以尝试

**方案5：单隐藏层（128×8×1）**
- 参数数：$P_5 = 128 \times 8 + 8 \times 1 + 8 + 1 = 1024 + 8 + 8 + 1 = 1041$
- 样本数：$N = 6000$
- 比例：$\frac{N}{P_5} = \frac{6000}{1041} \approx 5.76$

**结论**：比例接近6，**相对安全**，但可能仍需要正则化

## 3. 推荐方案

### 3.1 方案A：简单线性映射 + L2正则化（推荐）

**架构**：
$$f(x) = \sigma(w^T x + b)$$

其中：
- $w \in \mathbb{R}^{128}$：权重向量
- $b \in \mathbb{R}$：偏置
- $\sigma$：Sigmoid激活函数（因为输出是 $[0,1]$）

**参数数**：$P = 129$

**损失函数**（带L2正则化）：
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - y_i)^2 + \lambda \|w\|_2^2$$

其中 $\lambda$ 是正则化系数（建议 $\lambda \in [0.001, 0.01]$）

**LaTeX代码：**
```latex
f(x) = \sigma(w^T x + b)
```

```latex
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - y_i)^2 + \lambda \|w\|_2^2
```

**优点**：
- 参数少，不会过拟合
- 训练快速
- 可解释性强
- Linear probe 的本质就是学习简单的线性映射

**缺点**：
- 可能无法捕捉复杂的非线性关系（但这是 linear probe 的预期行为）

### 3.2 方案B：单隐藏层 + Dropout（如果简单线性不够）

**架构**：
$$h = \text{ReLU}(W_1 x + b_1)$$
$$f(x) = \sigma(w_2^T h + b_2)$$

其中：
- $W_1 \in \mathbb{R}^{8 \times 128}$：第一层权重矩阵
- $b_1 \in \mathbb{R}^{8}$：第一层偏置
- $w_2 \in \mathbb{R}^{8}$：第二层权重向量
- $b_2 \in \mathbb{R}$：第二层偏置

**参数数**：$P = 128 \times 8 + 8 + 8 + 1 = 1041$

**损失函数**（带Dropout和L2正则化）：
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - y_i)^2 + \lambda_1 \|W_1\|_F^2 + \lambda_2 \|w_2\|_2^2$$

**LaTeX代码：**
```latex
h = \text{ReLU}(W_1 x + b_1), \quad f(x) = \sigma(w_2^T h + b_2)
```

**优点**：
- 可以捕捉一些非线性关系
- 参数数量在可接受范围内

**缺点**：
- 需要仔细调优正则化参数
- 训练时间更长

### 3.3 方案C：带特征工程的线性映射

**思路**：在输入特征上添加一些简单的非线性变换，然后使用线性映射

**特征工程**：
$$x' = [x, x^2, \sqrt{|x|}, \text{ReLU}(x)]$$

其中 $x^2$ 和 $\sqrt{|x|}$ 是逐元素操作。

**架构**：
$$f(x) = \sigma(w^T x' + b)$$

**参数数**：$P = 128 \times 4 + 1 = 513$

**LaTeX代码：**
```latex
x' = [x, x^2, \sqrt{|x|}, \text{ReLU}(x)], \quad f(x) = \sigma(w^T x' + b)
```

**优点**：
- 在保持线性映射的同时，增加了模型容量
- 参数数量适中

## 4. 具体实现建议

### 4.1 PyTorch实现（方案A：简单线性映射）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LinearProbe(nn.Module):
    def __init__(self, input_dim=128):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 训练
model = LinearProbe(128)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # weight_decay是L2正则化

# 训练循环
for epoch in range(100):
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

### 4.2 PyTorch实现（方案B：单隐藏层）

```python
class MLPProbe(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=8):
        super(MLPProbe, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout正则化
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        return self.sigmoid(self.fc2(h))

# 训练
model = MLPProbe(128, hidden_dim=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 5. 训练策略

### 5.1 数据划分

建议将6000个样本划分为：
- **训练集**：4800个（80%）
- **验证集**：600个（10%）
- **测试集**：600个（10%）

### 5.2 早停（Early Stopping）

监控验证集损失，如果连续 $K$ 个epoch（如 $K=10$）验证损失不再下降，则停止训练。

### 5.3 学习率调度

使用学习率衰减：
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### 5.4 批量大小

建议批量大小 $B \in [32, 128]$，根据GPU内存调整。

## 6. 评估指标

### 6.1 回归指标

- **MSE（均方误差）**：$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - y_i)^2$
- **MAE（平均绝对误差）**：$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |f(x_i) - y_i|$
- **R²（决定系数）**：$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - f(x_i))^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$

**LaTeX代码：**
```latex
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (f(x_i) - y_i)^2
```

```latex
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - f(x_i))^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
```

### 6.2 分类指标（如果阈值化）

如果设置阈值 $\tau = 0.5$，将输出转换为二分类：
- **准确率（Accuracy）**
- **精确率（Precision）**
- **召回率（Recall）**
- **F1分数**

## 7. 最终推荐

### 7.1 首选方案：简单线性映射 + L2正则化

**理由**：
1. **Linear probe 的本质**：Linear probe 的设计初衷就是学习简单的线性映射，捕捉 head 输出与目标之间的线性关系
2. **样本充足**：6000个样本对于129个参数来说非常充足（比例46.5:1）
3. **不会过拟合**：参数少，即使没有正则化也不容易过拟合
4. **可解释性强**：线性模型的权重可以直接解释
5. **训练快速**：计算量小，训练速度快

**实现要点**：
- 使用 L2 正则化（weight_decay=0.01）
- 使用 Adam 优化器，学习率 0.001
- 早停机制
- 数据划分：80/10/10

### 7.2 备选方案：如果简单线性映射效果不好

如果验证集上表现不佳（如 $R^2 < 0.5$），可以尝试：

1. **单隐藏层（hidden_dim=8）** + Dropout + L2正则化
2. **特征工程**：添加平方项、平方根项等
3. **集成方法**：训练多个简单的 linear probe，然后平均

## 8. 参数数量与样本数量的平衡公式

### 8.1 一般规则

对于线性模型：
$$\text{样本数} \geq 10 \times \text{参数数}$$

对于非线性模型（单隐藏层）：
$$\text{样本数} \geq 20 \times \text{参数数}$$

### 8.2 考虑正则化

如果使用 L2 正则化或 Dropout，可以适当放宽：
$$\text{样本数} \geq 5 \times \text{参数数}$$

### 8.3 当前情况

对于你的情况（6000个样本）：
- **简单线性映射（129参数）**：$\frac{6000}{129} = 46.5 \gg 10$ ✅ **非常安全**
- **单隐藏层8维（1041参数）**：$\frac{6000}{1041} = 5.76 \approx 5$ ⚠️ **需要正则化**
- **单隐藏层16维（2081参数）**：$\frac{6000}{2081} = 2.88 < 5$ ❌ **不推荐**

## 9. 总结

1. **首选**：简单线性映射（128×1）+ L2正则化
2. **如果效果不好**：单隐藏层（128×8×1）+ Dropout + L2正则化
3. **关键**：使用验证集监控过拟合，使用早停机制
4. **原则**：Linear probe 应该保持简单，其目的是学习 head 输出与目标之间的线性关系

## 10. 实验建议

建议按以下顺序实验：

1. **实验1**：简单线性映射（128×1），无正则化
2. **实验2**：简单线性映射（128×1），L2正则化（weight_decay=0.01）
3. **实验3**：简单线性映射（128×1），L2正则化（weight_decay=0.1）
4. **实验4**：单隐藏层（128×8×1），Dropout(0.2) + L2正则化

根据验证集表现选择最佳方案。
