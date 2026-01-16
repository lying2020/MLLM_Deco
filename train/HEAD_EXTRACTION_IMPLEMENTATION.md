# Head输出提取实现说明

## 1. 实现位置总结

### 1.1 理想方法实现（使用Hook机制）

**位置**：`generate_head_ground_truth.py` 第 **219-295行**

**类**：`HeadOutputExtractor._make_attn_hook()`

**使用场景**：
- ✅ **POPE基准**：在 `process_case_pope()` 中使用（第339-544行）
- ❌ **CHAIR基准**：在 `process_case_chair()` 中**未使用**（使用近似方法）

### 1.2 近似方法实现

**位置**：`generate_head_ground_truth.py` 第 **976-986行**

**使用场景**：
- ❌ **POPE基准**：不使用
- ✅ **CHAIR基准**：在 `process_case_chair()` 中使用

## 2. 理想方法实现详解（第219-295行）

### 2.1 实现步骤

按照真实的 attention 层结构，在 hook 中实现：

```python
def _make_attn_hook(self, layer_idx: int):
    def attn_hook(module, input_tuple, output):
        # 步骤1: 计算Q, K, V
        Q = module.q_proj(hidden_states)
        K = module.k_proj(hidden_states)
        V = module.v_proj(hidden_states)

        # 步骤2: 重塑为多头格式并计算attention
        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # 计算attention scores和weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)

        # 应用attention到V
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # 步骤3: 提取每个head的输出
        for head_idx in range(self.num_heads):
            # 提取单个head的原始输出（未经过o_proj）
            head_attn_output = attn_output[:, head_idx, :, :]  # [batch, seq_len, head_dim]

            # 保存原始head输出（用于训练linear probe）
            self.head_raw_outputs[(layer_idx, head_idx)] = head_attn_output

            # 步骤4: 创建只包含该head的concat向量（其他head位置为0）
            head_only_concat = torch.zeros_like(attn_output_concat)
            head_start = head_idx * head_dim
            head_end = (head_idx + 1) * head_dim
            head_only_concat[:, :, head_start:head_end] = head_attn_output

            # 步骤5: 应用o_proj得到该head的完整输出
            head_output_full = module.o_proj(head_only_concat)  # [batch, seq_len, hidden_size]

            self.head_outputs[(layer_idx, head_idx)] = head_output_full
```

### 2.2 数学公式对应

**步骤1-2**：计算每个head的独立输出
$$\text{head\_out}_{l,n,t} = \text{Attention}(Q_n, K_n, V_n) \in \mathbb{R}^{d_h}$$

**步骤3-4**：创建只包含该head的拼接向量
$$[\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}] \in \mathbb{R}^{d}$$

**步骤5**：应用共享的线性矩阵 $W_O$（即 `o_proj`）
$$H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$

**LaTeX代码：**
```latex
H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]
```

### 2.3 在POPE中的使用（第472-504行）

```python
# 获取该head的输出 H_t^(l,n)（经过o_proj后的完整输出）
head_output = extractor.get_head_output(layer_idx, head_idx)

# 获取该head的原始输出（未经过o_proj，用于训练linear probe）
head_raw_output = extractor.get_head_raw_output(layer_idx, head_idx)

if head_output is not None:
    # 使用精确的head输出
    head_output_last = head_output[:, last_token_idx:last_token_idx+1, :]
    h_with_head = h_before + head_output_last

    # 提取最后一个token的原始head向量
    if head_raw_output is not None:
        head_raw_vector = head_raw_output[:, last_token_idx, :].cpu().numpy()
```

**这是理想方法** ✅

## 3. 近似方法实现详解（第976-986行）

### 3.1 实现代码

```python
# 遍历每个head
for head_idx in range(num_heads):
    # 获取该head的输出（需要从hook中获取，但hook可能没有捕获到这个步骤）
    # 由于hook是在forward pass时注册的，而generate过程中可能不会触发hook
    # 我们需要使用近似方法：计算head的贡献

    # 方法：使用hidden states的差值来近似head的贡献
    # h_after_all_heads - h_before 是所有head的总贡献
    # 单个head的贡献 = (h_after_all_heads - h_before) / num_heads
    head_contribution_approx = (h_after_all_heads - h_before) / num_heads
    h_with_head = h_before + head_contribution_approx

    # 对于head_raw_vector，我们无法直接从hidden states中提取
    # 使用零向量作为占位符
    head_raw_vector = None
```

### 3.2 数学公式

$$H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H} = \frac{W_O [\text{head\_out}_{l,0,t}; \ldots; \text{head\_out}_{l,H-1,t}]}{H}$$

**LaTeX代码：**
```latex
H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H}
```

### 3.3 为什么使用近似方法？

在 `model.generate()` 过程中：
- Hook 是在 forward pass 时注册的
- 但 `generate()` 的内部实现可能不会触发我们注册的 hook
- 因此无法使用 hook 提取精确的 head 输出
- 只能使用 hidden states 的差值来近似

## 4. test_chair_test.py 中的实现

### 4.1 检查结果

通过搜索 `test_chair_test.py`，发现：
- **没有提取单个head输出的实现**
- 主要关注的是：
  - 提取各层的 logits（`_extract_layer_logits_for_tokens`，第1527行）
  - 提取 attention maps（`_extract_head_layer_attention_data`，第1768行）
  - 计算 head 集中度（`_calculate_head_concentrations`，第2427行）

### 4.2 test_chair_test.py 的处理方式

`test_chair_test.py` 主要使用：
- `all_hidden_states`：直接从 `model.generate()` 获取
- `all_attentions`：直接从 `model.generate()` 获取
- 不单独提取每个 head 的输出贡献

**结论**：`test_chair_test.py` 中没有类似的单个 head 输出提取实现，因为它不需要计算单个 head 对 logits 的贡献。

## 5. 当前实现总结

### 5.1 POPE基准（使用理想方法）

✅ **使用Hook机制提取精确的head输出**
- 位置：`HeadOutputExtractor._make_attn_hook()`（第219-295行）
- 在 `process_case_pope()` 中使用（第472-504行）
- 实现方式：按照真实的 attention 层结构
  1. 提取 `head_out_{l,n,t}`
  2. 创建只包含该head的concat向量
  3. 应用 `o_proj` 得到 $H_{e,t}^{(l,n)}$

### 5.2 CHAIR基准（使用近似方法）

⚠️ **使用hidden states差值近似head贡献**
- 位置：`process_case_chair()`（第976-986行）
- 原因：`model.generate()` 过程中hook可能无法触发
- 近似公式：$H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H}$

## 6. 改进建议

### 6.1 为什么CHAIR不能使用理想方法？

在 `process_case_chair()` 中，我们使用 `model.generate()` 来生成文本，而hook是在 `model.forward()` 时触发的。`generate()` 的内部实现可能：
- 使用缓存的hidden states
- 使用不同的forward路径
- 不触发我们注册的hook

### 6.2 可能的改进方案

**方案1**：尝试在 `generate()` 过程中注册hook
- 可能仍然无法触发（取决于模型实现）

**方案2**：使用 `model.forward()` 逐步生成
- 需要手动实现生成循环
- 可以确保hook被触发
- 但实现复杂度较高

**方案3**：接受近似方法
- 对于CHAIR基准，近似方法可能足够
- 因为主要关注的是相对贡献，而不是绝对精确的值

## 7. 代码位置索引

| 功能 | 位置 | 方法类型 |
|------|------|----------|
| Hook实现（理想方法） | 第219-295行 | 理想方法 ✅ |
| POPE使用hook | 第472-504行 | 理想方法 ✅ |
| CHAIR使用近似 | 第976-986行 | 近似方法 ⚠️ |
| Hook注册 | 第297-312行 | - |
| Hook移除 | 第314-318行 | - |

## 8. 总结

1. **POPE基准**：使用**理想方法**（Hook机制），按照真实的attention层结构提取head输出
2. **CHAIR基准**：使用**近似方法**（hidden states差值），因为`generate()`过程中hook可能无法触发
3. **test_chair_test.py**：没有类似的单个head输出提取实现
4. **建议**：对于CHAIR基准，如果精度要求高，可以考虑方案2（手动实现生成循环），但当前近似方法可能已经足够
