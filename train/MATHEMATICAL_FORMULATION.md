# Head级别真值对构建的数学公式化说明

本文档详细说明 `generate_head_ground_truth.py` 脚本的实现过程，使用数学公式进行形式化描述。

## 1. 问题定义

我们的目标是构建 1024 个线性探针（linear probe）的训练数据，每个探针对应模型的一个 attention head。模型有 $L=32$ 层，每层有 $H=32$ 个 head，总共 $L \times H = 1024$ 个 head。

对于每个 head $(l, n)$（其中 $l \in [0, L-1]$ 是层索引，$n \in [0, H-1]$ 是 head 索引），我们需要构建训练对 $(x_u^{(l,n)}, y_u^{(l,n)})$，其中：
- $x_u^{(l,n)} \in \mathbb{R}^{d_h}$ 是 head 的原始输出向量（$d_h = 128$ 是 head 维度）
- $y_u^{(l,n)} \in [0, 1]$ 是目标标签（SPP输出 $g_u^{(l,n)}$）

## 2. 核心数学公式

### 2.0 Attention层的真实结构

在计算 head 贡献之前，我们需要理解真实的 attention 层结构：

1. **每个 head 计算独立输出**：$\text{head\_out}_{l,n,t} \in \mathbb{R}^{d_h}$（$d_h = 128$ 是 head 维度）
2. **所有 head 拼接**：$[\text{head\_out}_{l,0,t}; \text{head\_out}_{l,1,t}; \ldots; \text{head\_out}_{l,H-1,t}] \in \mathbb{R}^{d}$（$d = H \times d_h = 4096$）
3. **应用共享的线性矩阵 $W_O$**：$W_O \in \mathbb{R}^{d \times d}$

完整的 attention 层输出为：

$$h_{e,t}^{(l)} = h_{e,t}^{(l-1)} + W_O [\text{head\_out}_{l,0,t}; \text{head\_out}_{l,1,t}; \ldots; \text{head\_out}_{l,H-1,t}]$$

**LaTeX代码：**
```latex
h_{e,t}^{(l)} = h_{e,t}^{(l-1)} + W_O [\text{head\_out}_{l,0,t}; \text{head\_out}_{l,1,t}; \ldots; \text{head\_out}_{l,H-1,t}]
```

**单个 head 的贡献**：

为了提取单个 head $(l,n)$ 的贡献，我们：
1. 创建一个只包含该 head 的拼接向量（其他 head 位置为0）：
   $$[\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$
2. 应用 $W_O$ 得到该 head 的完整贡献：
   $$H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$

**LaTeX代码：**
```latex
H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]
```

### 2.1 集合概率计算

对于给定的 logits $\xi \in \mathbb{R}^{|V|}$（$|V|$ 是词汇表大小）和 token 集合 $B$，集合概率定义为：

$$P(B | \xi) = \sum_{b \in B} P(b | \xi) = \sum_{b \in B} \text{softmax}(\xi)_b$$

其中 $\text{softmax}(\xi)_b = \frac{\exp(\xi_b)}{\sum_{v \in V} \exp(\xi_v)}$。

**LaTeX代码：**
```latex
P(B | \xi) = \sum_{b \in B} P(b | \xi) = \sum_{b \in B} \text{softmax}(\xi)_b
```

### 2.2 对数概率增益（Log Probability Gain）

对于 head $(l,n)$，在时间步 $t$，定义：

- $h_{e,t}^{(l-1)}$：第 $l$ 层之前的 hidden state（第 $l-1$ 层的输出）
- $H_{e,t}^{(l,n)}$：第 $l$ 层第 $n$ 个 head 的贡献（经过 $o\_proj$ 后的完整输出）
- $h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}$：加入该 head 后的 hidden state

对于集合 $B_u$，对数概率增益定义为：

$$\Delta \log P_u^{(l,n)}(B_u) = \log P(B_u | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) - \log P(B_u | h_{e,t}^{(l-1)})$$

其中：
- $P(B_u | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) = P(B_u | \text{lm\_head}(\text{norm}(h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)})))$
- $P(B_u | h_{e,t}^{(l-1)}) = P(B_u | \text{lm\_head}(\text{norm}(h_{e,t}^{(l-1)})))$

**LaTeX代码：**
```latex
\Delta \log P_u^{(l,n)}(B_u) = \log P(B_u | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) - \log P(B_u | h_{e,t}^{(l-1)})
```

### 2.3 语义先验偏置分数（Semantic Prior Bias Score）

对于 head $(l,n)$，语义先验偏置分数定义为：

$$s_u^{(l,n)} = \Delta \log P_u^{(l,n)}(B_u^-) - \Delta \log P_u^{(l,n)}(B_u^+)$$

其中：
- $B_u^+$ 是非幻视集合（正确标签集合）
- $B_u^-$ 是幻视集合（错误标签集合）

**LaTeX代码：**
```latex
s_u^{(l,n)} = \Delta \log P_u^{(l,n)}(B_u^-) - \Delta \log P_u^{(l,n)}(B_u^+)
```

### 2.4 SPP输出（Sigmoid-Transformed Prior Probability）

SPP输出通过 sigmoid 函数将语义先验偏置分数映射到 $[0, 1]$ 区间：

$$g_u^{(l,n)} = \sigma(\alpha \cdot s_u^{(l,n)} + \beta) = \frac{1}{1 + \exp(-(\alpha \cdot s_u^{(l,n)} + \beta))}$$

其中：
- $\alpha = 2.0$ 是温度/尺度参数
- $\beta = 0.0$ 是偏置参数

**LaTeX代码：**
```latex
g_u^{(l,n)} = \sigma(\alpha \cdot s_u^{(l,n)} + \beta) = \frac{1}{1 + \exp(-(\alpha \cdot s_u^{(l,n)} + \beta))}
```

## 3. POPE基准的处理流程

### 3.1 问题形式

POPE基准的输入是 Yes/No 问题，例如："Is there a bicycle in the image? Please answer Yes or No."

### 3.2 集合定义

对于 label = "yes" 的 case：
- $B_u^+ = \{\text{"yes"}\}$（正确标签，非幻视集合）
- $B_u^- = \{\text{"no"}\}$（错误标签，幻视集合）

对于 label = "no" 的 case：
- $B_u^+ = \{\text{"no"}\}$（正确标签，非幻视集合）
- $B_u^- = \{\text{"yes"}\}$（错误标签，幻视集合）

### 3.3 计算流程

1. **输入处理**：将图像和文本输入模型，获取最后一个 token 位置的 hidden states
2. **Head输出提取**：对于每个 head $(l,n)$，提取其输出 $H_{e,t}^{(l,n)}$
3. **计算对数概率增益**：
   - $\Delta \log P_u^{(l,n)}(B_u^+) = \log P(B_u^+ | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) - \log P(B_u^+ | h_{e,t}^{(l-1)})$
   - $\Delta \log P_u^{(l,n)}(B_u^-) = \log P(B_u^- | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) - \log P(B_u^- | h_{e,t}^{(l-1)})$
4. **计算语义先验偏置分数**：$s_u^{(l,n)} = \Delta \log P_u^{(l,n)}(B_u^-) - \Delta \log P_u^{(l,n)}(B_u^+)$
5. **计算SPP输出**：$g_u^{(l,n)} = \sigma(2.0 \cdot s_u^{(l,n)} + 0.0)$
6. **构建训练对**：$(x_u^{(l,n)}, y_u^{(l,n)}) = (\text{head\_raw\_output}, g_u^{(l,n)})$

### 3.4 POPE示例

**输入 Case：**
```json
{
  "question_id": 1,
  "image": "COCO_val2014_000000123456.jpg",
  "text": "Is there a bicycle in the image? Please answer Yes or No.",
  "label": ["yes"]
}
```

**处理过程：**

1. **集合定义**：
   - $B_u^+ = \{\text{token\_id("yes")}\}$
   - $B_u^- = \{\text{token\_id("no")}\}$

2. **对于 head (0, 0)**（第0层，第0个head）：
   - 假设 $\Delta \log P_u^{(0,0)}(B_u^+) = 0.15$
   - 假设 $\Delta \log P_u^{(0,0)}(B_u^-) = -0.22$
   - $s_u^{(0,0)} = -0.22 - 0.15 = -0.37$
   - $g_u^{(0,0)} = \sigma(2.0 \times (-0.37) + 0.0) = \frac{1}{1 + \exp(0.74)} \approx 0.32$

3. **输出真值对**：
```json
{
  "case_id": 1,
  "layer": 0,
  "head": 0,
  "head_vector": [0.123, -0.456, ..., 0.234],  // 128维向量
  "s_u": -0.37,
  "g_u": 0.32,
  "delta_log_p_plus": 0.15,
  "delta_log_p_minus": -0.22,
  "case_type": "POPE"
}
```

## 4. CHAIR基准的处理流程

### 4.1 问题形式

CHAIR基准的输入是图像描述请求，例如："Please help me describe the image in detail."

### 4.2 物理词汇识别

1. **文本生成**：模型生成完整描述文本
2. **物理词汇提取**：使用 CHAIR 评估器识别物理词汇（COCO对象类别）
3. **Token映射**：将物理词汇映射到对应的生成步骤（token位置）
4. **限制策略**：
   - 每个物理词汇只使用第一次出现的第一个组成 token
   - 最多使用 6 个非重复的物理词汇（按出现顺序）

### 4.3 集合定义（动态构建）

对于每个生成步骤 $t$，根据当前生成的物理词汇 $y_t$ 和 top-K 候选池 $C_t$，动态构建：

**情况A：$y_t$ 是 Grounded（真实实例）**

$$B_u^+ = \{y_t\} \cup (C_t \cap G)$$

$$B_u^- = C_t \cap H - G$$

其中：
- $G$ 是真实实例词汇集合（ground truth objects）
- $H$ 是 COCO 对象类别词汇表（包括所有同义词）
- $C_t$ 是当前 head 在 top-K（K=20）预测中的物理词汇集合

**情况B：$y_t$ 是 Hallucinated（幻视）**

$$B_u^- = \{y_t\} \cup (C_t \cap H - G)$$

$$B_u^+ = C_t \cap G$$

**LaTeX代码：**
```latex
\text{情况A: } B_u^+ = \{y_t\} \cup (C_t \cap G), \quad B_u^- = C_t \cap H - G
```

```latex
\text{情况B: } B_u^- = \{y_t\} \cup (C_t \cap H - G), \quad B_u^+ = C_t \cap G
```

### 4.4 计算流程

对于每个选中的物理词汇对应的生成步骤 $t$：

1. **获取 hidden states**：
   - $h_{e,t}^{(l-1)}$：第 $l$ 层之前的 hidden state（第 $l-1$ 层的输出）
   - $h_{e,t}^{(l)}$：第 $l$ 层之后的 hidden state，按照真实的 attention 层结构：

   $$h_{e,t}^{(l)} = h_{e,t}^{(l-1)} + W_O [\text{head\_out}_{l,0,t}; \text{head\_out}_{l,1,t}; \ldots; \text{head\_out}_{l,H-1,t}]$$

   其中 $W_O \in \mathbb{R}^{d \times d}$ 是共享的输出投影矩阵（$d = 4096$ 是 hidden_size）。

2. **提取单个 head 的贡献**（按照真实的 attention 层结构）：

   **理想方法**（在 forward pass 中使用 hook 提取）：
   - 首先，每个 head 计算独立的输出：$\text{head\_out}_{l,n,t} \in \mathbb{R}^{d_h}$（维度为 $d_h = 128$）
   - 然后，创建一个只包含该 head 的拼接向量（其他 head 位置为0）：
     $$[\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}] \in \mathbb{R}^{d}$$
   - 最后，应用共享的线性矩阵 $W_O$ 得到该 head 的完整贡献：

   $$H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$

   **LaTeX代码：**
   ```latex
   H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]
   ```

   **近似方法**（在 generate 过程中，hook 可能无法捕获每个步骤）：
   由于 `model.generate()` 过程中 hook 的限制，我们使用近似：

   $$H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H} = \frac{W_O [\text{head\_out}_{l,0,t}; \ldots; \text{head\_out}_{l,H-1,t}]}{H}$$

   这是一个近似，假设所有 head 的贡献相等。理想情况下应该使用 hook 提取的精确 head 输出。

   **LaTeX代码：**
   ```latex
   H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H}
   ```

3. **计算 logits**：
   - $\xi_{\text{before}} = \text{lm\_head}(\text{norm}(h_{e,t}^{(l-1)}))$
   - $\xi_{\text{with\_head}} = \text{lm\_head}(\text{norm}(h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}))$
4. **构建 $B_u^+$ 和 $B_u^-$**：根据当前步骤的 $y_t$ 类型和 top-K 候选池
5. **计算对数概率增益**：
   - $\Delta \log P_u^{(l,n)}(B_u^+) = \log P(B_u^+ | \xi_{\text{with\_head}}) - \log P(B_u^+ | \xi_{\text{before}})$
   - $\Delta \log P_u^{(l,n)}(B_u^-) = \log P(B_u^- | \xi_{\text{with\_head}}) - \log P(B_u^- | \xi_{\text{before}})$
6. **计算语义先验偏置分数和SPP输出**：同POPE流程

### 4.5 CHAIR示例

**输入 Case：**
```json
{
  "question_id": 2,
  "image": "COCO_val2014_000000056752.jpg",
  "text": "Please help me describe the image in detail.",
  "label": ["backpack", "bicycle", "person"]
}
```

**处理过程：**

1. **文本生成**：
   ```
   "The image features a man riding a bicycle on a road next to a river.
   He is wearing a backpack and appears to be enjoying the ride."
   ```

2. **物理词汇识别**（假设识别出8个，限制为前6个）：
   - "man" (node: "person") [Grounded] -> step_4
   - "bicycle" (node: "bicycle") [Grounded] -> step_8
   - "backpack" (node: "backpack") [Grounded] -> step_24
   - "people" (node: "person") [Hallucinated] -> step_57
   - "person" (node: "person") [Hallucinated] -> step_69
   - "road" (node: "road") [Hallucinated] -> step_11

3. **对于 head (5, 10) 在 step_8（生成 "bicycle"）**：
   - **当前词**：$y_t = \text{"bicycle"}$（Grounded）
   - **Top-K候选池**：$C_t = \{\text{"bicycle"}, \text{"bike"}, \text{"person"}, \text{"backpack"}, ...\}$（top-20）
   - **构建集合**：
     - $B_u^+ = \{\text{"bicycle"}\} \cup (C_t \cap G) = \{\text{"bicycle"}, \text{"backpack"}, \text{"person"}\}$
     - $B_u^- = C_t \cap H - G = \{\text{"bike"}, \text{"road"}, ...\}$（不在G中的COCO对象）
   - **计算**：
     - 假设 $\Delta \log P_u^{(5,10)}(B_u^+) = 0.18$
     - 假设 $\Delta \log P_u^{(5,10)}(B_u^-) = -0.25$
     - $s_u^{(5,10)} = -0.25 - 0.18 = -0.43$
     - $g_u^{(5,10)} = \sigma(2.0 \times (-0.43) + 0.0) \approx 0.30$

4. **输出真值对**：
```json
{
  "case_id": 2,
  "step": 8,
  "layer": 5,
  "head": 10,
  "head_vector": [0.0, 0.0, ..., 0.0],  // 128维零向量（占位符）
  "s_u": -0.43,
  "g_u": 0.30,
  "delta_log_p_plus": 0.18,
  "delta_log_p_minus": -0.25,
  "case_type": "CHAIR"
}
```

## 5. 完整流程总结

### 5.1 数据流

$$
\text{Input} \rightarrow \text{Model Forward} \rightarrow \text{Hidden States} \rightarrow \text{Head Outputs} \rightarrow \text{Logits} \rightarrow \text{Probabilities} \rightarrow \text{Log Gains} \rightarrow s_u \rightarrow g_u
$$

**LaTeX代码：**
```latex
\text{Input} \rightarrow \text{Model Forward} \rightarrow \text{Hidden States} \rightarrow \text{Head Outputs} \rightarrow \text{Logits} \rightarrow \text{Probabilities} \rightarrow \text{Log Gains} \rightarrow s_u \rightarrow g_u
```

### 5.2 关键参数

- $L = 32$：transformer层数
- $H = 32$：每层的head数
- $d_h = 128$：head维度（hidden_size / num_heads）
- $K = 20$：top-K候选池大小
- $\alpha = 2.0$：SPP温度参数
- $\beta = 0.0$：SPP偏置参数
- $N_{\text{max}} = 6$：CHAIR基准最大物理词汇数

### 5.3 输出格式

对于每个 head $(l,n)$，生成一个JSON文件 `layer_{l}_head_{n}.json`，包含该 head 的所有训练对：

```json
[
  {
    "case_id": 1,
    "layer": l,
    "head": n,
    "head_vector": [x_1, x_2, ..., x_{128}],
    "s_u": s_u^{(l,n)},
    "g_u": g_u^{(l,n)},
    "delta_log_p_plus": \Delta \log P_u^{(l,n)}(B_u^+),
    "delta_log_p_minus": \Delta \log P_u^{(l,n)}(B_u^-),
    "case_type": "POPE" | "CHAIR",
    "step": t  // 仅CHAIR类型有
  },
  ...
]
```

## 6. 数学公式汇总（LaTeX）

### 6.1 集合概率
```latex
P(B | \xi) = \sum_{b \in B} \frac{\exp(\xi_b)}{\sum_{v \in V} \exp(\xi_v)}
```

### 6.2 对数概率增益
```latex
\Delta \log P_u^{(l,n)}(B_u) = \log P(B_u | h_{e,t}^{(l-1)} + H_{e,t}^{(l,n)}) - \log P(B_u | h_{e,t}^{(l-1)})
```

### 6.3 语义先验偏置分数
```latex
s_u^{(l,n)} = \Delta \log P_u^{(l,n)}(B_u^-) - \Delta \log P_u^{(l,n)}(B_u^+)
```

### 6.4 SPP输出
```latex
g_u^{(l,n)} = \frac{1}{1 + \exp(-(\alpha \cdot s_u^{(l,n)} + \beta))}
```

### 6.5 CHAIR集合构建（情况A：Grounded）
```latex
B_u^+ = \{y_t\} \cup (C_t \cap G), \quad B_u^- = C_t \cap H - G
```

### 6.6 CHAIR集合构建（情况B：Hallucinated）
```latex
B_u^- = \{y_t\} \cup (C_t \cap H - G), \quad B_u^+ = C_t \cap G
```

## 7. 实现细节

### 7.1 Head输出提取

使用 PyTorch forward hooks 精确提取每个 head 的输出：

1. **Pre-hook**：捕获 attention 层的输入 hidden states
2. **Forward hook**：在 attention 层内部：
   - 计算 Q, K, V
   - 计算 attention scores 和 weights
   - 提取每个 head 的 attention output
   - 应用 $o\_proj$ 得到完整的 head 输出

### 7.2 Layer Normalization

在应用 `lm_head` 之前，先通过 layer normalization：

$$\text{norm}(h) = \text{LayerNorm}(h)$$

$$\xi = \text{lm\_head}(\text{norm}(h))$$

### 7.3 Hidden States过滤

`output_hidden_states=True` 返回的 hidden states 包含 embedding 层（索引0）和所有 transformer 层（索引1-32）。我们过滤掉 embedding 层，只保留 transformer 层，使得索引对齐：
- `step_hidden_states[0]` 对应 `layer_0`
- `step_hidden_states[l]` 对应 `layer_l`

## 8. 总结

本脚本实现了从 LLaVA 模型的中间表示中提取 head 级别的真值对，用于训练 1024 个线性探针。核心思想是：

1. **POPE基准**：通过对比正确标签和错误标签的对数概率增益，量化 head 对正确预测的贡献
2. **CHAIR基准**：通过对比真实实例和幻视词汇的对数概率增益，量化 head 对幻觉的贡献
3. **统一输出**：两种基准都输出相同的格式 $(x_u^{(l,n)}, g_u^{(l,n)})$，其中 $g_u^{(l,n)}$ 是 sigmoid 变换后的语义先验偏置分数

最终，每个 head 都会得到一个训练数据集，用于训练对应的线性探针，该探针将学习如何根据 head 的输出预测其语义先验偏置分数。
