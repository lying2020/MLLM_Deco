# 生成步骤（Generation Steps）说明

## 什么是生成步骤？

在文本生成过程中，模型是**逐步生成token**的，每个token的生成都是一个**生成步骤**。

### 例子

假设模型生成文本："A person is riding a bicycle."

生成过程如下：

```
步骤0: 输入prompt → 生成 "A"        (第1个新token)
步骤1: "A" → 生成 " person"        (第2个新token)
步骤2: "A person" → 生成 " is"     (第3个新token)
步骤3: "A person is" → 生成 " riding" (第4个新token)
步骤4: "A person is riding" → 生成 " a" (第5个新token)
步骤5: "A person is riding a" → 生成 " bicycle" (第6个新token)
步骤6: "A person is riding a bicycle" → 生成 "." (第7个新token)
```

总共生成了 **7个token**，所以有 **7个生成步骤**。

## output_dict.hidden_states 的结构

当设置 `output_hidden_states=True` 时，`output_dict.hidden_states` 的结构是：

```python
output_dict.hidden_states = (
    step_0_hidden_states,  # 步骤0的hidden states
    step_1_hidden_states,  # 步骤1的hidden states
    step_2_hidden_states,  # 步骤2的hidden states
    ...
    step_N_hidden_states,  # 步骤N的hidden states（最后一个）
)
```

每个 `step_i_hidden_states` 又是一个tuple，包含所有层的hidden states：

```python
step_i_hidden_states = (
    embedding_layer,      # 索引0: embedding层
    layer_0_hidden,       # 索引1: 第0层transformer
    layer_1_hidden,       # 索引2: 第1层transformer
    ...
    layer_31_hidden,      # 索引32: 第31层transformer
)
```

## 当前实现（只处理最后一个步骤）

当前代码在第708-717行：

```python
# 获取最后一个生成步骤的hidden states
if all_step_hidden_states is not None and len(all_step_hidden_states) > 0:
    last_step_hidden_states = all_step_hidden_states[-1]  # 只取最后一个步骤
    if layer_idx < len(last_step_hidden_states):
        h_after_all_heads = last_step_hidden_states[layer_idx][:, -1:, :]
```

这意味着：
- ✅ **只使用了最后一个步骤**（`all_step_hidden_states[-1]`）
- ❌ **忽略了前面所有步骤**（步骤0到步骤N-1）

### 为什么这样设计？

1. **计算效率**：如果生成了100个token，处理所有步骤需要计算100次，计算量很大
2. **简化实现**：最后一个token通常包含了整个生成过程的累积信息
3. **符合需求**：对于计算语义先验偏置分数，最后一个token的logits已经足够

## 如果要处理所有生成步骤

如果需要处理所有步骤，代码应该是：

```python
# 遍历所有生成步骤
for step_idx, step_hidden_states in enumerate(all_step_hidden_states):
    # 对每个步骤都进行计算
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # 获取该步骤的hidden states
            h_after_all_heads = step_hidden_states[layer_idx][:, -1:, :]

            # 计算logits、s_u、g_u等
            # ...

            # 保存真值对（包含step_idx信息）
            pair = {
                "case_id": case["question_id"],
                "step": step_idx,  # 添加步骤索引
                "layer": layer_idx,
                "head": head_idx,
                # ...
            }
```

### 计算量对比

假设生成了100个token，有32层，每层32个head：

- **当前实现（只处理最后一步）**：
  - 计算次数：32层 × 32head = **1,024次**

- **处理所有步骤**：
  - 计算次数：100步骤 × 32层 × 32head = **102,400次**
  - 计算量增加 **100倍**！

## 什么时候需要处理所有步骤？

1. **需要分析每个token的生成过程**：了解模型在每个步骤的决策
2. **需要找到物理词汇对应的具体步骤**：如果"person"是在步骤1生成的，需要分析步骤1的hidden states
3. **需要更细粒度的真值对**：每个步骤都生成一个真值对

## 当前实现的适用场景

当前实现（只处理最后一步）适用于：
- ✅ 计算整体语义先验偏置分数
- ✅ 训练linear probe（不需要知道具体是哪个步骤）
- ✅ 快速生成大量真值对

## 总结

- **生成步骤** = 每个新token的生成过程
- **当前实现** = 只使用最后一个步骤的hidden states
- **处理所有步骤** = 对每个步骤都进行计算（计算量增加100倍）
- **选择** = 根据具体需求决定是否需要处理所有步骤
