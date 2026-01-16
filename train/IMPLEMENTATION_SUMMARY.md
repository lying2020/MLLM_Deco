# Head输出提取实现总结

## 1. 两种实现方法

### 1.1 理想方法（使用Hook精确提取）

**位置**：`HeadOutputExtractor._make_attn_hook()` 方法，**第219-295行**

**实现原理**（按照真实的attention层结构）：

1. **计算每个head的独立输出**（第265行）：
   ```python
   attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
   ```
   得到 $\text{head\_out}_{l,n,t} \in \mathbb{R}^{d_h}$

2. **提取单个head的输出**（第275行）：
   ```python
   head_attn_output = attn_output[:, head_idx, :, :]  # [batch, seq_len, head_dim]
   ```
   这是单个head的原始输出（未经过 $o\_proj$）

3. **创建只包含该head的concat向量**（第281-284行）：
   ```python
   head_only_concat = torch.zeros_like(attn_output_concat)
   head_start = head_idx * head_dim
   head_end = (head_idx + 1) * head_dim
   head_only_concat[:, :, head_start:head_end] = head_attn_output
   ```
   创建 $[\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$

4. **应用 $o\_proj$ 得到完整贡献**（第288行）：
   ```python
   head_output_full = module.o_proj(head_only_concat)  # [batch, seq_len, hidden_size]
   ```
   得到 $H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$

**数学公式**：
$$H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$

**LaTeX代码：**
```latex
H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]
```

### 1.2 近似方法（使用Hidden States差值）

**位置**：`process_case_chair()` 函数，**第985行**

**实现原理**：
```python
head_contribution_approx = (h_after_all_heads - h_before) / num_heads
h_with_head = h_before + head_contribution_approx
```

**数学公式**：
$$H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H} = \frac{W_O [\text{head\_out}_{l,0,t}; \ldots; \text{head\_out}_{l,H-1,t}]}{H}$$

**LaTeX代码：**
```latex
H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H}
```

**假设**：所有head的贡献相等（这是一个近似）

## 2. 使用场景

### 2.1 POPE基准：使用理想方法

**位置**：`process_case_pope()` 函数

**调用方式**（第427行）：
```python
outputs = model.get_model().forward(
    input_ids=input_ids_processed if inputs_embeds is None else None,
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
    return_dict=True
)
```

**为什么可以使用理想方法**：
- 使用 `model.forward()` 进行单次forward pass
- Hook可以正常工作，能够捕获每个head的输出
- 可以精确提取每个head的贡献

**提取head输出**（第475-489行）：
```python
head_output = extractor.get_head_output(layer_idx, head_idx)  # 从hook中获取
head_raw_output = extractor.get_head_raw_output(layer_idx, head_idx)  # 从hook中获取

if head_output is None:
    # 如果hook没有捕获到，使用近似方法（fallback）
    h_after_all_heads = hidden_states[layer_idx + 1][:, last_token_idx:last_token_idx+1, :]
    head_contribution_approx = (h_after_all_heads - h_before) / num_heads
    h_with_head = h_before + head_contribution_approx
else:
    # 使用精确的head输出（理想方法）
    head_output_last = head_output[:, last_token_idx:last_token_idx+1, :]
    h_with_head = h_before + head_output_last
```

### 2.2 CHAIR基准：使用理想方法 + Fallback

**位置**：`process_case_chair()` 函数

**实现策略**：
1. 先生成完整文本（用于识别物理词汇和步骤）
2. 对于每个目标生成步骤，手动调用 `model.get_model().forward()` 来触发hook
3. 从hook中提取精确的head输出（理想方法）
4. 如果hook未捕获到，使用fallback近似方法

**手动调用forward**（第1001-1019行）：
```python
# 构建到当前步骤为止的完整token序列
if inputs_embeds is not None:
    # 将新生成的token IDs转换为embeddings并拼接
    new_token_embeds = embedding_layer(new_token_ids)
    current_inputs_embeds = torch.cat([inputs_embeds[0], new_token_embeds], dim=0)
else:
    # 直接拼接token IDs
    current_input_ids = torch.cat([input_ids_processed[0], new_token_ids], dim=0)

# 手动调用forward来触发hook
outputs = model.get_model().forward(
    input_ids=current_input_ids,
    inputs_embeds=current_inputs_embeds,
    position_ids=current_position_ids,
    attention_mask=current_attention_mask,
    output_hidden_states=True,
    return_dict=True
)
```

**提取head输出**（第1051-1072行）：
```python
# 尝试从hook中获取精确的head输出（理想方法）
head_output = extractor.get_head_output(layer_idx, head_idx)
head_raw_output = extractor.get_head_raw_output(layer_idx, head_idx)

if head_output is None or head_raw_output is None:
    # Hook没有捕获到，使用fallback方法：从hidden states差值近似
    h_after_all_heads = hidden_states[layer_idx + 1][:, last_token_idx:last_token_idx+1, :]
    head_contribution_approx = (h_after_all_heads - h_before) / num_heads
    h_with_head = h_before + head_contribution_approx
    head_raw_vector = None
else:
    # 使用精确的head输出（理想方法）
    head_output_last = head_output[:, last_token_idx:last_token_idx+1, :]
    h_with_head = h_before + head_output_last
    head_raw_vector = head_raw_output[:, last_token_idx, :].cpu().numpy()
```

## 3. 代码位置总结

| 方法 | 函数/类 | 行号 | 使用场景 |
|------|---------|------|----------|
| **理想方法** | `HeadOutputExtractor._make_attn_hook()` | 219-295 | Hook实现，提取精确head输出 |
| **理想方法使用** | `process_case_pope()` | 475-504 | POPE基准，forward pass |
| **理想方法使用** | `process_case_chair()` | 1001-1072 | CHAIR基准，手动调用forward触发hook |
| **Fallback方法** | `process_case_chair()` | 1055-1064 | CHAIR基准，当hook未捕获到时使用 |

## 4. 关键代码片段

### 4.1 理想方法实现（第280-288行）

```python
# 创建一个只包含该head的完整attn_output（其他head为零）
head_only_concat = torch.zeros_like(attn_output_concat)
head_start = head_idx * head_dim
head_end = (head_idx + 1) * head_dim
head_only_concat[:, :, head_start:head_end] = head_attn_output

# 应用o_proj得到该head的完整输出
if hasattr(module, 'o_proj'):
    head_output_full = module.o_proj(head_only_concat)  # [batch, seq_len, hidden_size]
```

这对应数学公式：
$$H_{e,t}^{(l,n)} = W_O [\mathbf{0}; \ldots; \text{head\_out}_{l,n,t}; \ldots; \mathbf{0}]$$

### 4.2 Fallback方法实现（第1055-1064行）

当hook未捕获到head输出时，使用fallback近似方法：

```python
# Hook没有捕获到，使用fallback方法：从hidden states差值近似
h_after_all_heads = hidden_states[layer_idx + 1][:, last_token_idx:last_token_idx+1, :]
head_contribution_approx = (h_after_all_heads - h_before) / num_heads
h_with_head = h_before + head_contribution_approx
head_raw_vector = None
```

这对应数学公式：
$$H_{e,t}^{(l,n)} \approx \frac{h_{e,t}^{(l)} - h_{e,t}^{(l-1)}}{H}$$

## 5. 总结

- **POPE基准**：使用**理想方法**（hook精确提取），因为使用 `forward()` 进行单次forward pass
- **CHAIR基准**：使用**理想方法**（手动调用forward触发hook），如果hook未捕获到则使用fallback近似方法

理想方法的实现完全按照真实的attention层结构：
1. 提取单个head的原始输出 $\text{head\_out}_{l,n,t}$
2. 创建只包含该head的concat向量（其他位置为0）
3. 应用共享的 $W_O$ 矩阵得到该head的完整贡献

CHAIR基准现在也使用理想方法，通过手动调用forward来确保hook能够正常工作，从而获得精确的head输出。
