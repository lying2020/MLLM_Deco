# `extract_attention_during_generation` 函数功能与图像 Token 位置计算详解

## 一、函数功能概述

`extract_attention_during_generation` 函数的主要功能是：**在文本生成过程中，提取每个生成步骤的 attention map，并可视化模型对图像 token 的关注度**。

### 核心功能：
1. **准备多模态输入**：将图像和文本提示词组合成模型可处理的格式
2. **执行文本生成**：使用 `model.generate()` 生成文本，同时收集 attention 信息
3. **提取 Attention Map**：对每个生成步骤的每一层，提取 attention 权重
4. **可视化图像关注度**：将 attention map 映射回原图，展示模型在生成每个词时关注的图像区域

---

## 二、图像 Token 位置计算流程

### 阶段 1: 文本 Tokenization (`tokenizer_image_token`)

**位置**: `llava/mm_utils.py:43-62`

**功能**: 将包含 `<image>` 占位符的文本转换为 token IDs，其中 `<image>` 被替换为 `IMAGE_TOKEN_INDEX`（通常是 -200）

**处理流程**:
```python
# 示例输入: "USER: <image>\nPlease describe this image."
# 步骤 1: 按 '<image>' 分割文本
prompt_chunks = [
    tokenizer("USER: ").input_ids,      # [1, 1234, 5678, ...]  (包含 BOS token)
    tokenizer("\nPlease describe this image.").input_ids  # [1234, 5678, ...]
]

# 步骤 2: 在文本块之间插入 IMAGE_TOKEN_INDEX 占位符
# 结果: [BOS, 1234, 5678, ..., IMAGE_TOKEN_INDEX, 1234, 5678, ...]
```

**关键点**:
- `IMAGE_TOKEN_INDEX` 是一个**占位符**，不是真正的 token
- 如果第一个文本块包含 BOS token，会保留它
- 占位符数量 = 文本块数量 - 1

**输出示例**:
```
input_ids = [1, 1234, 5678, ..., -200, 1234, 5678, ...]
              ↑                    ↑
            BOS token        IMAGE_TOKEN_INDEX 占位符
```

---

### 阶段 2: 图像特征提取

**位置**: `extract_attention_during_generation` 函数内 (1163-1186行)

**功能**: 通过 Vision Tower 和 MM Projector 提取图像特征

**处理流程**:
```python
# 1. Vision Tower 提取图像特征
vision_hidden = vision_tower(image_tensor)  # [1, 576, vision_hidden_size]

# 2. MM Projector 投影到语言模型空间
vision_hidden = mm_projector(vision_hidden)  # [1, 576, 4096]

# 3. 确定图像 token 数量
num_image_tokens = vision_hidden.shape[1]  # 通常是 576 (24×24 patches)
```

**关键点**:
- 图像被分割成 576 个 patches（24×24）
- 每个 patch 对应一个图像 token
- 图像特征维度: `[batch, 576, hidden_size]`

---

### 阶段 3: 多模态输入准备 (`prepare_inputs_labels_for_multimodal`)

**位置**: `llava/model/llava_arch.py:99-198`

**功能**: 将 `IMAGE_TOKEN_INDEX` 占位符替换为实际的图像特征 embeddings

**核心逻辑** (第 170-198 行):

```python
# 1. 找到所有 IMAGE_TOKEN_INDEX 的位置
image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
# 示例: [-1, 35, 48]  (35 是第一个 <image> 的位置，48 是序列结束位置)

# 2. 按 IMAGE_TOKEN_INDEX 分割文本块
for i in range(len(image_token_indices) - 1):
    # 提取文本块: cur_input_ids[image_token_indices[i]+1 : image_token_indices[i+1]]
    # i=0: cur_input_ids[0:35]  (第一个 <image> 之前的文本)
    # i=1: cur_input_ids[36:48] (第一个 <image> 之后的文本)
    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])

# 3. 对文本块进行 embedding
cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

# 4. 交替拼接: 文本 emb + 图像特征 + 文本 emb + ...
for i in range(num_images + 1):
    cur_new_input_embeds.append(cur_input_embeds_no_im[i])  # 文本块
    if i < num_images:
        cur_new_input_embeds.append(cur_image_features)      # 图像特征 (576 tokens)
```

**关键点**:
- `image_token_indices[0] = -1` 是起始标记
- `image_token_indices[1]` 是第一个 `IMAGE_TOKEN_INDEX` 的位置
- 文本块从 `image_token_indices[i]+1` 开始（因为 `-1+1=0`）
- 图像特征直接插入，**不经过 embedding 层**

---

### 阶段 4: 图像 Token 位置计算

**位置**: `extract_attention_during_generation` 函数内 (1215-1251行)

**计算逻辑**:

```python
# 1. 找到原始 input_ids 中第一个 IMAGE_TOKEN_INDEX 的位置
first_image_placeholder_pos = image_token_placeholder_positions[0]  # 例如: 35

# 2. 根据 prepare_inputs_labels_for_multimodal 的逻辑:
#    - 第一个文本块: input_ids[0:35] (35 个 tokens)
#    - 这些 tokens 会被 embed，得到 35 个 embeddings
#    - 然后图像特征 (576 tokens) 会被插入
#    - 所以图像 token 的起始位置 = 第一个文本块的长度 = 35

image_token_start = first_image_placeholder_pos  # 35
image_token_end = image_token_start + num_image_tokens  # 35 + 576 = 611
```

**重要理解**:

1. **图像 token 起始位置** = 原始 `input_ids` 中第一个 `IMAGE_TOKEN_INDEX` 的位置
   - 原因：`prepare_inputs_labels_for_multimodal` 中，第一个文本块是 `input_ids[0:first_image_placeholder_pos]`
   - 这个文本块被 embed 后，图像特征直接插入其后
   - 所以图像 token 的起始位置 = 第一个文本块的长度 = `first_image_placeholder_pos`

2. **图像 token 结束位置** = `image_token_start + num_image_tokens`
   - 通常是 `first_image_placeholder_pos + 576`

3. **序列结构**:
   ```
   处理后序列 (inputs_embeds):
   [文本 emb (0:35), 图像特征 (35:611), 文本 emb (611:end)]
   ```

---

## 三、完整流程示例

### 输入:
- **Prompt**: `"USER: <image>\nPlease describe this image."`
- **图像**: 一张 COCO 图像

### 处理步骤:

1. **Tokenization** (`tokenizer_image_token`):
   ```
   input_ids = [1, 1234, 5678, ..., -200, 1234, 5678, ...]
                 ↑                    ↑
               BOS (pos 0)      IMAGE_TOKEN_INDEX (pos 35)
   ```

2. **图像特征提取**:
   ```
   vision_hidden: [1, 576, 4096]
   num_image_tokens = 576
   ```

3. **多模态输入准备** (`prepare_inputs_labels_for_multimodal`):
   ```
   image_token_indices = [-1, 35, 48]

   文本块 1: input_ids[0:35]  → embed → 35 个 embeddings
   图像特征: vision_hidden     → 576 个 embeddings (直接插入)
   文本块 2: input_ids[36:48]  → embed → 12 个 embeddings

   最终 inputs_embeds: [35 文本 emb + 576 图像 emb + 12 文本 emb]
   ```

4. **图像 Token 位置**:
   ```
   image_token_start = 35
   image_token_end = 35 + 576 = 611
   图像 token 位置范围: [35, 610] (共 576 个)
   ```

5. **Attention Map 提取**:
   - 对于每个生成步骤的每一层
   - 提取 attention 矩阵的最后一行（新生成 token 对所有历史 token 的 attention）
   - 从这一行中提取位置 [35:611] 的 attention 值
   - 将这 576 个值 reshape 成 24×24 的 attention map
   - 映射回原图进行可视化

---

## 四、关键代码位置总结

| 功能 | 文件位置 | 关键代码行 |
|------|---------|-----------|
| 文本 Tokenization | `llava/mm_utils.py` | 43-62 |
| 图像特征提取 | `tests/test_llava_v15_7b_attention.py` | 1163-1186 |
| 多模态输入准备 | `llava/model/llava_arch.py` | 99-198 |
| 图像 Token 位置计算 | `tests/test_llava_v15_7b_attention.py` | 1215-1251 |
| Attention 提取 | `tests/test_llava_v15_7b_attention.py` | 1352+ |

---

## 五、注意事项

1. **IMAGE_TOKEN_INDEX 是占位符**：在 `input_ids` 中，`IMAGE_TOKEN_INDEX` 不是真正的 token，只是一个标记，会在 `prepare_inputs_labels_for_multimodal` 中被替换为图像特征。

2. **图像 Token 数量固定**：对于标准的 LLaVA 模型，图像 token 数量通常是 576（24×24 patches），但可能因模型配置而异。

3. **位置计算的假设**：代码假设 `prepare_inputs_labels_for_multimodal` 的处理逻辑是：
   - 第一个文本块: `input_ids[0:first_image_placeholder_pos]`
   - 图像特征直接插入其后
   - 因此图像 token 起始位置 = `first_image_placeholder_pos`

4. **生成过程中的 Attention**：在生成过程中，每次只生成一个 token，所以 attention 形状是 `[batch, num_heads, 1, seq_len]`，其中 `seq_len` 包括所有历史 token（输入 + 已生成的 token）。
