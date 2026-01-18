# 计算对比逻辑详细说明

## 1. 核心目标

计算**语义先验偏置分数** `s_u`，用于衡量某个 head 对幻视/真实实例的贡献程度。

## 2. 数学公式

### 2.1 语义先验偏置分数

```
s_u = delta_log_p_minus - delta_log_p_plus
```

其中：
- `delta_log_p_plus`：真实实例集合的对数概率增益
- `delta_log_p_minus`：幻视集合的对数概率增益

### 2.2 对数概率增益

```
delta_log_p = log P(Bu | h_with_head) - log P(Bu | h_without_head)
```

其中：
- `P(Bu | h)`：在隐藏状态 `h` 下，集合 `Bu` 的概率
- `P(Bu) = Σ_{b∈Bu} P(b)`：集合中所有 token 的概率和

### 2.3 集合概率计算

```
P(Bu) = Σ_{b∈Bu} P(b)
      = Σ_{b∈Bu} softmax(logits)[b]
```

## 3. 为什么需要同时有 Bu^+ 和 Bu^-？

### 3.1 数学原因

如果 `Bu` 为空集合：
- `P(Bu) = Σ_{b∈∅} P(b) = 0`
- `log P(Bu) = log(0) = -∞`
- 无法计算 `delta_log_p`

因此，必须同时有 `Bu^+` 和 `Bu^-` 才能计算 `s_u`。

### 3.2 语义原因

`s_u` 衡量的是 head 对**真实实例**和**幻视词汇**的**相对贡献**：
- 如果 `s_u > 0`：head 更倾向于生成幻视词汇（`delta_log_p_minus > delta_log_p_plus`）
- 如果 `s_u < 0`：head 更倾向于生成真实实例（`delta_log_p_plus > delta_log_p_minus`）

如果没有对比对象，就无法衡量这种相对贡献。

## 4. 构建规则

### 4.1 Grounded Token（真实实例）

**规则**：
- `Bu^+ = {当前真实词汇}`（例如：`{"dog"}`）
- `Bu^- = top-K 中的所有幻视词汇`（例如：`{"cat", "bird"}`）

**为什么需要 top-K 中的幻视词汇？**
- 当前 token 是真实实例，我们需要找到**对比对象**（幻视词汇）
- 如果 top-K 中没有幻视词汇，则 `Bu^- = ∅`，无法计算对比

### 4.2 Hallucinated Token（幻视词汇）

**规则**：
- `Bu^- = {当前幻视词汇}`（例如：`{"cat"}`）
- `Bu^+ = top-K 中的所有真实词汇`（例如：`{"dog", "person"}`）

**为什么需要 top-K 中的真实实例词汇？**
- 当前 token 是幻视词汇，我们需要找到**对比对象**（真实实例）
- 如果 top-K 中没有真实实例词汇，则 `Bu^+ = ∅`，无法计算对比

## 5. 详细举例

### 例1：Grounded Token，但 top-K 中没有幻视词汇

**场景**：
- 图像中只有：`["dog", "person"]`
- 当前生成步骤：`step_idx = 10`
- 当前 token：`yt = "dog"`（Grounded，真实实例）
- 某个 head 的 top-K 预测：`["dog", "puppy", "animal", "pet", "canine", ...]`

**构建过程**：
1. 从 top-K 中筛选物理词汇：
   - `Ct_physical_tokens = {"dog", "puppy", "animal", "pet", "canine", ...}`
2. 分离真实实例和幻视词汇：
   - `Ct_grounded_tokens = {"dog", "puppy", "canine"}`（都是真实实例或同义词）
   - `Ct_hallucinated_tokens = {}`（没有幻视词汇）
3. 构建 `Bu^+` 和 `Bu^-`：
   - `Bu^+ = {"dog"}` ✓
   - `Bu^- = {}` ✗（空集合）

**结果**：
- 无法计算 `delta_log_p_minus`，因为 `P(Bu^-) = 0`
- 因此，**所有 head 都被跳过**（对于这个 step）

**为什么会出现这种情况？**
- 该 head 的 top-K 预测中，所有物理词汇都是真实实例或同义词
- 没有预测任何幻视词汇，因此无法构建对比

### 例2：Hallucinated Token，但 top-K 中没有真实实例词汇

**场景**：
- 图像中只有：`["dog", "person"]`
- 当前生成步骤：`step_idx = 15`
- 当前 token：`yt = "cat"`（Hallucinated，幻视）
- 某个 head 的 top-K 预测：`["cat", "kitten", "animal", "pet", "feline", ...]`

**构建过程**：
1. 从 top-K 中筛选物理词汇：
   - `Ct_physical_tokens = {"cat", "kitten", "animal", "pet", "feline", ...}`
2. 分离真实实例和幻视词汇：
   - `Ct_grounded_tokens = {}`（没有真实实例）
   - `Ct_hallucinated_tokens = {"cat", "kitten", "feline"}`（都是幻视词汇）
3. 构建 `Bu^+` 和 `Bu^-`：
   - `Bu^- = {"cat"}` ✓
   - `Bu^+ = {}` ✗（空集合）

**结果**：
- 无法计算 `delta_log_p_plus`，因为 `P(Bu^+) = 0`
- 因此，**所有 head 都被跳过**（对于这个 step）

**为什么会出现这种情况？**
- 该 head 的 top-K 预测中，所有物理词汇都是幻视词汇
- 没有预测任何真实实例，因此无法构建对比

### 例3：正常情况（Grounded Token，top-K 中有幻视词汇）

**场景**：
- 图像中只有：`["dog", "person"]`
- 当前生成步骤：`step_idx = 10`
- 当前 token：`yt = "dog"`（Grounded，真实实例）
- 某个 head 的 top-K 预测：`["dog", "cat", "bird", "puppy", "kitten", ...]`

**构建过程**：
1. 从 top-K 中筛选物理词汇：
   - `Ct_physical_tokens = {"dog", "cat", "bird", "puppy", "kitten", ...}`
2. 分离真实实例和幻视词汇：
   - `Ct_grounded_tokens = {"dog", "puppy"}`（真实实例）
   - `Ct_hallucinated_tokens = {"cat", "bird", "kitten"}`（幻视词汇）
3. 构建 `Bu^+` 和 `Bu^-`：
   - `Bu^+ = {"dog"}` ✓
   - `Bu^- = {"cat", "bird", "kitten"}` ✓

**计算过程**：
1. 计算 `delta_log_p_plus`：
   ```
   P(Bu^+ | h_with_head) = P("dog" | h_with_head)
   P(Bu^+ | h_without_head) = P("dog" | h_without_head)
   delta_log_p_plus = log P("dog" | h_with_head) - log P("dog" | h_without_head)
   ```

2. 计算 `delta_log_p_minus`：
   ```
   P(Bu^- | h_with_head) = P("cat" | h_with_head) + P("bird" | h_with_head) + P("kitten" | h_with_head)
   P(Bu^- | h_without_head) = P("cat" | h_without_head) + P("bird" | h_without_head) + P("kitten" | h_without_head)
   delta_log_p_minus = log P(Bu^- | h_with_head) - log P(Bu^- | h_without_head)
   ```

3. 计算 `s_u`：
   ```
   s_u = delta_log_p_minus - delta_log_p_plus
   ```

**结果**：
- 成功计算 `s_u`，生成真值对 ✓

## 6. 为什么不同 head 的 top-K 预测不同？

每个 head 关注不同的特征：
- **Head 0**：可能关注全局特征，top-K 包含多种物体
- **Head 1**：可能关注局部特征，top-K 只包含特定类型的物体
- **Head 2**：可能关注语义关系，top-K 包含语义相关的物体

因此，对于同一个 step，不同 head 的 top-K 预测可能不同：
- 有些 head 的 top-K 中既有真实实例又有幻视词汇 → 可以计算对比
- 有些 head 的 top-K 中只有真实实例或只有幻视词汇 → 无法计算对比

## 7. 总结

1. **必须同时有 Bu^+ 和 Bu^-**：否则无法计算 `s_u`
2. **Grounded token 需要 top-K 中的幻视词汇**：作为对比对象
3. **Hallucinated token 需要 top-K 中的真实实例词汇**：作为对比对象
4. **不同 head 的 top-K 预测不同**：因此有些 head 可能被跳过
5. **这是正常现象**：不是 bug，而是因为不同 head 关注不同的特征
