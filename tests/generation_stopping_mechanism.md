# 文本生成停止机制详解

## 一、生成停止的三种方式

### 1. EOS Token（End-of-Sequence Token）

**工作原理**：
- EOS token 是模型在**推理过程中预测生成**的一个特殊 token
- 当模型认为应该结束生成时，会在某个步骤生成 EOS token
- 一旦生成 EOS token，生成过程会**立即停止**

**关键点**：
- ✅ EOS token **是模型生成的**，不是预先设定的
- ✅ 每个生成步骤，模型都会预测下一个 token
- ✅ 如果预测的 token 是 EOS token，生成停止
- ✅ EOS token 会被包含在 `generated_ids` 中

**示例**：
```
步骤 0: 生成 token "Yes" (token_id: 1234)
步骤 1: 生成 token "or" (token_id: 5678)
步骤 2: 生成 token EOS (token_id: 2)  ← 生成停止
```

---

### 2. 停止字符串（Stop Strings / KeywordsStoppingCriteria）

**工作原理**：
- 在每个生成步骤**之后**，检查最新生成的文本是否包含停止字符串
- 如果包含，则停止生成
- 停止字符串**不会**被包含在最终输出中（会被移除）

**实现方式** (`KeywordsStoppingCriteria`):
```python
class KeywordsStoppingCriteria(StoppingCriteria):
    def __call__(self, output_ids, scores, **kwargs) -> bool:
        # 1. 检查最新生成的 token 序列是否匹配停止字符串的 token IDs
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True  # 停止生成

        # 2. 解码最新生成的文本，检查是否包含停止字符串
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True  # 停止生成

        return False  # 继续生成
```

**关键点**：
- ✅ 停止字符串是**预先设定的**（如对话模板的结束标记）
- ✅ 在每个生成步骤**后**检查，不是在生成**前**
- ✅ 如果匹配，停止字符串**不会**出现在最终输出中
- ✅ 需要显式传入 `stopping_criteria` 参数才会生效

**示例**（假设停止字符串是 `"</s>"`）：
```
步骤 0: 生成 token "Yes" → 检查: 不包含 "</s>" → 继续
步骤 1: 生成 token "or" → 检查: 不包含 "</s>" → 继续
步骤 2: 生成 token "No" → 检查: 包含 "</s>" → 停止
最终输出: "Yes or No" (不包含 "</s>")
```

---

### 3. max_new_tokens 限制

**工作原理**：
- 如果生成的 token 数量达到 `max_new_tokens`，即使没有 EOS token 也会停止
- 这是硬性限制，防止生成过长

**关键点**：
- ✅ 这是**硬性限制**，优先级最高
- ✅ 即使模型想继续生成，达到限制也会停止

---

## 二、生成过程的详细流程

### 标准生成循环（简化版）

```python
for step in range(max_new_tokens):
    # 1. 模型预测下一个 token
    logits = model(input_ids)  # [batch, seq_len, vocab_size]
    next_token_logits = logits[0, -1, :]  # [vocab_size]
    next_token_id = next_token_logits.argmax().item()  # 贪婪解码

    # 2. 将新 token 添加到序列
    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    # 3. 检查是否应该停止
    if next_token_id == eos_token_id:
        break  # EOS token，停止生成

    if stopping_criteria(input_ids, scores):
        break  # 停止字符串匹配，停止生成

    if step >= max_new_tokens - 1:
        break  # 达到最大 token 数，停止生成
```

---

## 三、在你的代码中的情况

### 当前代码 (`test_llava_v15_7b_attention.py`)

```python
output_dict = model.generate(
    inputs=input_ids,
    images=image_tensor.unsqueeze(0).half().to(device),
    max_new_tokens=max_new_tokens,  # 最多生成 10 个 token
    output_attentions=True,
    return_dict_in_generate=True,
    do_sample=False,
    num_beams=1
    # 注意: 没有设置 stopping_criteria
)
```

**停止机制**：
1. ✅ **EOS token**: 如果模型生成 EOS token，会停止
2. ❌ **停止字符串**: 未设置 `stopping_criteria`，所以不依赖停止字符串
3. ✅ **max_new_tokens**: 最多生成 `max_new_tokens` 个 token

**为什么只生成了 2 个 token？**

可能的原因：
1. **模型生成了 EOS token**：在第 2 个步骤，模型预测的 token 是 EOS token
2. **模型认为应该结束**：模型在生成 2 个 token 后，认为回答已经完整

---

## 四、EOS Token vs 停止字符串的区别

| 特性 | EOS Token | 停止字符串 |
|------|-----------|------------|
| **来源** | 模型预测生成 | 预先设定 |
| **检查时机** | 生成 token **时** | 生成 token **后** |
| **是否包含在输出** | ✅ 是（但通常会被过滤） | ❌ 否（会被移除） |
| **是否需要设置** | ❌ 否（自动处理） | ✅ 是（需要传入 stopping_criteria） |
| **灵活性** | 模型决定何时结束 | 用户/系统决定何时结束 |

---

## 五、实际示例

### 场景 1: 模型生成 EOS token

```
输入: "Please describe this image."
生成过程:
  步骤 0: 生成 "The" → 继续
  步骤 1: 生成 "image" → 继续
  步骤 2: 生成 "shows" → 继续
  ...
  步骤 N: 生成 EOS → 停止
```

### 场景 2: 停止字符串匹配

```
输入: "USER: <image>\nPlease describe this image."
停止字符串: "</s>" (对话模板的结束标记)
生成过程:
  步骤 0: 生成 "The" → 检查: 不包含 "</s>" → 继续
  步骤 1: 生成 "image" → 检查: 不包含 "</s>" → 继续
  步骤 2: 生成 "shows" → 检查: 不包含 "</s>" → 继续
  ...
  步骤 N: 生成 "description</s>" → 检查: 包含 "</s>" → 停止
最终输出: "The image shows ... description" (不包含 "</s>")
```

---

## 六、如何查看停止原因

代码中已添加停止原因分析，会输出：
- 是否因为 EOS token 停止
- 是否因为达到 max_new_tokens 停止
- 最后生成的 token 信息

这样可以帮助理解为什么生成在某个步骤停止了。
