# generate_head_ground_truth.py 逻辑流程说明

## 一、整体流程概览

```
输入: coco_train_*.json (训练case文件)
  ↓
加载模型和配置
  ↓
创建HeadOutputExtractor并注册hooks
  ↓
对每个case进行处理:
  ├─ 判断case类型 (POPE/CHAIR)
  ├─ 加载图像，准备输入
  ├─ Forward pass获取hidden states和head输出
  └─ 对每一层、每个head计算真值对
  ↓
保存所有真值对到JSON文件
```

## 二、详细流程说明

### 阶段1: 初始化和准备

#### 1.1 输入数据
- **输入文件**: `coco_train_*.json`
- **格式**: JSON数组，每个元素是一个case
  ```json
  {
    "question_id": 1,
    "image": "COCO_val2014_000000015017.jpg",
    "text": "Is there a person in the image?  Please answer Yes or No.",
    "label": ["yes"]
  }
  ```

#### 1.2 加载模型
- 加载LLaVA模型（默认: llava-v1.5-7b）
- 获取模型配置：`num_layers=32`, `num_heads=32`
- 总共需要处理 **32 × 32 = 1024** 个head

#### 1.3 创建HeadOutputExtractor
- 使用PyTorch hook机制
- 在每层的attention层注册forward hook
- Hook会在forward pass时自动提取每个head的输出

### 阶段2: 处理单个Case

#### 2.1 判断Case类型
```python
case_type = "CHAIR" if "describe" in case["text"].lower() else "POPE"
```

#### 2.2 准备输入
- **加载图像**: 从COCO val2014目录加载图像
- **构建prompt**:
  - POPE: 原始问题文本
  - CHAIR: "Please help me describe the image in detail."
- **处理多模态输入**: 使用`prepare_inputs_labels_for_multimodal`处理图像和文本

#### 2.3 Forward Pass
```python
outputs = model.get_model().forward(
    inputs_embeds=inputs_embeds,
    output_hidden_states=True,  # 获取所有层的hidden states
    return_dict=True
)
```

**Hook自动执行**:
- 在每层的attention计算时，hook会：
  1. 提取该层之前的hidden state: `h_t^(l-1)`
  2. 计算Q, K, V并分割成多个head
  3. 计算每个head的attention输出: `H_t^(l,n)` (未经过o_proj，128维)
  4. 应用o_proj得到完整输出: `ĥ_t^(l,n)` (4096维)

#### 2.4 计算真值对（对每一层、每个head）

##### 对于POPE类型case:

**步骤1: 定义Bu^+和Bu^-**
```python
# 如果label是"yes":
bu_plus_words = ["yes"]  # 正确标签（非幻视集合）
bu_minus_words = ["no"]  # 错误标签（幻视集合）

# 如果label是"no":
bu_plus_words = ["no"]
bu_minus_words = ["yes"]
```

**步骤2: 获取token ID集合**
```python
bu_plus_tokens = get_vocab_tokens_for_words(tokenizer, bu_plus_words)
bu_minus_tokens = get_vocab_tokens_for_words(tokenizer, bu_minus_words)
```

**步骤3: 对每一层、每个head计算**

对于layer_idx=0, head_idx=0:

1. **获取hidden states**:
   ```python
   h_before = h_t^(l-1)  # 该层之前的hidden state [1, 1, 4096]
   head_output = H_t^(l,n) (经过o_proj)  # 该head的完整输出 [1, seq_len, 4096]
   head_raw_output = H_t^(l,n) (未经过o_proj)  # 原始head输出 [1, seq_len, 128]
   ```

2. **计算logits**:
   ```python
   logits_before = lm_head(h_before)  # 未加入head的logits
   h_with_head = h_before + head_output_last  # 加入head后的hidden state
   logits_with_head = lm_head(h_with_head)  # 加入head后的logits
   ```

3. **计算对数概率增益（公式C2）**:
   ```python
   # 计算集合概率
   P(Bu^+ | h_before) = Σ P(token | h_before) for token in bu_plus_tokens
   P(Bu^+ | h_with_head) = Σ P(token | h_with_head) for token in bu_plus_tokens

   # 对数概率增益
   Δlog P_plus = log P(Bu^+ | h_with_head) - log P(Bu^+ | h_before)
   Δlog P_minus = log P(Bu^- | h_with_head) - log P(Bu^- | h_before)
   ```

4. **计算语义先验偏置分数（公式C3）**:
   ```python
   s_u = Δlog P_minus - Δlog P_plus
   ```

5. **计算SPP输出g（公式D1）**:
   ```python
   g_u = sigmoid(alpha * s_u + beta)  # alpha=2, beta=0
   ```

6. **提取head向量**:
   ```python
   head_vector = head_raw_output[:, last_token_idx, :]  # [128] 最后一个token的head输出
   ```

7. **保存真值对**:
   ```json
   {
     "case_id": 1,
     "layer": 0,
     "head": 0,
     "head_vector": [0.123, -0.456, ..., 0.789],  // 128维，训练linear probe的输入x
     "s_u": 0.1234,                                // 语义先验偏置分数
     "g_u": 0.5307,                                // SPP输出，训练linear probe的标签y
     "delta_log_p_plus": 0.001,
     "delta_log_p_minus": 0.002,
     "case_type": "POPE"
   }
   ```

##### 对于CHAIR类型case:

**步骤1: 定义Bu^+和Bu^-**
```python
# Bu^+ = 真实实例词汇集合（从case["label"]获取）
bu_plus_words = case["label"]  # 例如: ["person", "car", "bicycle"]

# Bu^- = 幻视词汇集合（从top-K候选中选择）
# 1. 获取top-K候选池
top_k_indices = topk(logits_after, k=50)

# 2. 从top-K中选择：
#    - 属于COCO对象类别
#    - 但不属于真实实例
bu_minus_tokens = {token_id for token_id in top_k_indices
                   if token_id in all_coco_object_tokens
                   and token_id not in bu_plus_tokens}
```

**步骤2-7**: 与POPE类型相同

### 阶段3: 输出结果

#### 3.1 保存到JSON文件
- 文件名: `{train_file_name}_head_ground_truth.json`
- 格式: JSON数组，每个元素是一个head的真值对

#### 3.2 统计信息
- 总真值对数量
- 覆盖的(layer, head)组合数
- POPE/CHAIR类型分布
- g_u和s_u的统计信息

## 三、完整例子说明

### 例子：POPE类型case

#### 输入Case:
```json
{
  "question_id": 1,
  "image": "COCO_val2014_000000015017.jpg",
  "text": "Is there a person in the image?  Please answer Yes or No.",
  "label": ["yes"]
}
```

#### 处理流程:

**1. 加载图像和准备输入**
```
图像路径: /path/to/coco/val2014/COCO_val2014_000000015017.jpg
Prompt: "<image>\nIs there a person in the image?  Please answer Yes or No."
```

**2. Forward Pass**
```
输入序列长度: 588 tokens (包含图像tokens)
最后一个token位置: 587
```

**3. 对Layer 0, Head 0的处理:**

假设hook提取到：
- `h_before[0, 587, :]`: [1, 1, 4096] - 第0层之前的hidden state
- `head_raw_output[0, 587, :]`: [128] - Head 0的原始输出（未经过o_proj）
- `head_output[0, 587, :]`: [1, 1, 4096] - Head 0经过o_proj后的完整输出

**4. 计算logits:**
```python
logits_before = lm_head(h_before)  # [1, 32000] vocab_size=32000
h_with_head = h_before + head_output  # 加入head后的hidden state
logits_with_head = lm_head(h_with_head)  # [1, 32000]
```

**5. 定义token集合:**
```python
# label是"yes"，所以：
bu_plus_tokens = {tokenizer.encode("yes")}  # 例如: {1234, 5678}
bu_minus_tokens = {tokenizer.encode("no")}  # 例如: {9012, 3456}
```

**6. 计算集合概率:**
```python
# 对logits_before计算
probs_before = softmax(logits_before[0])
P_plus_before = sum(probs_before[t] for t in bu_plus_tokens)  # 例如: 0.3
P_minus_before = sum(probs_before[t] for t in bu_minus_tokens)  # 例如: 0.1

# 对logits_with_head计算
probs_with = softmax(logits_with_head[0])
P_plus_with = sum(probs_with[t] for t in bu_plus_tokens)  # 例如: 0.35
P_minus_with = sum(probs_with[t] for t in bu_minus_tokens)  # 例如: 0.08
```

**7. 计算对数概率增益:**
```python
Δlog P_plus = log(0.35) - log(0.3) = -0.9163 - (-1.2040) = 0.2877
Δlog P_minus = log(0.08) - log(0.1) = -2.5257 - (-2.3026) = -0.2231
```

**8. 计算语义先验偏置分数:**
```python
s_u = Δlog P_minus - Δlog P_plus = -0.2231 - 0.2877 = -0.5108
```

**9. 计算SPP输出:**
```python
g_u = sigmoid(2.0 * (-0.5108) + 0.0) = sigmoid(-1.0216) = 0.2647
```

**10. 提取head向量:**
```python
head_vector = head_raw_output[0, 587, :]  # [128] 例如: [0.123, -0.456, ..., 0.789]
```

**11. 保存真值对:**
```json
{
  "case_id": 1,
  "layer": 0,
  "head": 0,
  "head_vector": [0.123, -0.456, ..., 0.789],  // 128个浮点数
  "s_u": -0.5108,
  "g_u": 0.2647,
  "delta_log_p_plus": 0.2877,
  "delta_log_p_minus": -0.2231,
  "case_type": "POPE"
}
```

#### 重复处理:
- 对32层 × 32个head = 1024个head都执行上述步骤
- 每个case生成1024个真值对

#### 最终输出:
```json
[
  // Case 1, Layer 0, Head 0
  {"case_id": 1, "layer": 0, "head": 0, "head_vector": [...], "g_u": 0.2647, ...},
  // Case 1, Layer 0, Head 1
  {"case_id": 1, "layer": 0, "head": 1, "head_vector": [...], "g_u": 0.3124, ...},
  // ...
  // Case 1, Layer 31, Head 31
  {"case_id": 1, "layer": 31, "head": 31, "head_vector": [...], "g_u": 0.1892, ...},
  // Case 2, Layer 0, Head 0
  {"case_id": 2, "layer": 0, "head": 0, "head_vector": [...], "g_u": 0.4521, ...},
  // ...
]
```

## 四、关键公式回顾

### 公式C2: 对数概率增益
```
Δlog P_u^(l,n)(B_u) = log P(B_u | h + H) - log P(B_u | h)
```
- 衡量加入head后，集合B_u的概率变化（对数域）

### 公式C3: 语义先验偏置分数
```
s_u^(l,n) = Δlog P_u^(l,n)(B_u^-) - Δlog P_u^(l,n)(B_u^+)
```
- 如果s_u > 0: head倾向于推高幻视集合，压低非幻视集合（语义先验通路）
- 如果s_u < 0: head倾向于支持非幻视方向

### 公式D1: SPP输出
```
g_u^(l,n) = σ(α * s_u^(l,n) + β)
```
- 将偏置分数映射到[0,1]区间
- g_u越大，表示该head越像语义先验通路，应该被抑制

## 五、输出用途

生成的真值对用于训练1024个linear probe:
- **输入x**: `head_vector` (128维向量)
- **标签y**: `g_u` (标量，0-1之间)
- **模型**: `Linear(128, 1)` - 128维输入，1维输出
- **目标**: 学习从head输出预测该head的语义先验偏置强度
