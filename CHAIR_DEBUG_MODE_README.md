# CHAIR 评估 Debug 模式使用说明

## 概述

已优化 `chair.py` 和 `run_chair_eval.py`，添加了完整的 debug 模式，可以查看每个样本从输入到最终评估的完整过程。

## 主要改进

### 1. 默认测试样本数改为 10

```python
"num_samples": 10  # 默认只处理10个图像（之前是40个）
```

### 2. 添加 Debug 模式

使用 `--debug` 参数启用 debug 模式：

```bash
python run_chair_eval.py --coco-root /path/to/coco --debug
```

### 3. 详细的处理过程输出

在 debug 模式下，每个样本会输出以下信息：

#### [1] 分词和预处理
- 原始分词结果
- 词形还原后的词
- 识别到的 MSCOCO 对象
- 对象数量

#### [2] Ground Truth 对象
- GT 对象集合（排序后）
- GT 对象数量

#### [3] 幻觉检测
- 对每个对象进行检测
- ✓ 正确：对象在 GT 中
- ✗ 幻觉：对象不在 GT 中（显示位置和原因）

#### [4] 结果摘要
- 是否包含幻觉
- 幻觉对象数量
- 正确对象数量
- 总词数
- CHAIRs（句子级别）
- CHAIRi（实例级别）
- Recall（召回率）
- Len（平均长度）

### 4. 详细的 JSON 输出

所有中间结果都会保存到 JSON 文件中，包括：

#### 主要结果文件：`*_chair_results.json`

```json
{
  "overall_metrics": {
    "CHAIRs": 0.3000,
    "CHAIRi": 0.1500,
    "Recall": 0.8500,
    "Len": 0.2500
  },
  "sentences": [
    {
      "image_id": 226988,
      "caption": "The image features a woman...",
      "mscoco_gt_words": ["person", "dining table", "pizza", ...],
      "mscoco_generated_words": ["person", "dining table", "pizza", "backpack", ...],
      "mscoco_hallucinated_words": [
        ["backpack", "backpack"]
      ],
      "hallucination_details": [
        {
          "word": "backpack",
          "node_word": "backpack",
          "position": 15,
          "reason": "'backpack' 不在GT对象集合中"
        }
      ],
      "recall_gt_objects": ["person", "dining table", "pizza"],
      "recall_count": 3,
      "processed_words": ["person", "dining", "table", ...],
      "node_words": ["person", "dining table", "pizza", ...],
      "word_indices": [0, 1, 2, ...],
      "words": ["the", "image", "features", ...],
      "metrics": {
        "CHAIRs": 1,
        "CHAIRi": 0.1250,
        "Recall": 0.7500,
        "Len": 0.3200
      }
    }
  ]
}
```

#### 错误样本文件：`*_chair_errors.json`

只包含包含幻觉的样本：

```json
{
  "error_count": 3,
  "total_samples": 10,
  "error_samples": [
    {
      "image_id": 226988,
      "caption": "...",
      "mscoco_hallucinated_words": [...],
      "hallucination_details": [...],
      ...
    }
  ]
}
```

## 使用示例

### 基本用法（带 Debug 模式）

```bash
python run_chair_eval.py \
    --coco-root /home/liying/Documents/dataset/coco \
    --debug \
    --num-samples 10
```

### 输出示例

```
================================================================================
[样本 1/10] Image ID: 226988
================================================================================
图像: /path/to/COCO_val2014_000000226988.jpg

[输入准备] 图像信息:
  - 图像路径: /path/to/COCO_val2014_000000226988.jpg
  - 图像尺寸: (640, 480)
  - 图像张量形状: torch.Size([3, 224, 224])

[生成结果] 描述:
  - 输出长度: 245 字符
  - 描述预览: The image features a woman sitting at a dining table...

================================================================================
自动计算 CHAIR 指标...
================================================================================
Debug模式: 将输出所有 10 个样本的详细信息

[样本 1/10] Image ID: 226988
================================================================================
原始描述: The image features a woman sitting at a dining table...

[1] 分词和预处理:
  - 原始分词: ['the', 'image', 'features', 'a', 'woman', ...]
  - 词形还原后: ['the', 'image', 'feature', 'a', 'woman', ...]
  - 识别到的MSCOCO对象: ['person', 'dining table', 'pizza', 'backpack']
  - 对象数量: 4

[2] Ground Truth 对象:
  - GT对象集合: ['dining table', 'person', 'pizza']
  - GT对象数量: 3

[3] 幻觉检测:
  ✓ 正确: 'woman' -> 'person' (位置: 4)
  ✓ 正确: 'dining' -> 'dining table' (位置: 7)
  ✓ 正确: 'pizza' -> 'pizza' (位置: 12)
  ✗ 幻觉: 'backpack' -> 'backpack' (位置: 15)

[4] 结果摘要:
  - 是否包含幻觉: 是
  - 幻觉对象数量: 1
  - 正确对象数量: 3
  - 总词数: 32
  - CHAIRs (句子级别): 1
  - CHAIRi (实例级别): 0.1250
  - Recall (召回率): 1.0000
  - Len (平均长度): 0.3200
================================================================================
```

## 输出文件结构

参考 `run_pope_eval.py` 的路径结构：

```
results/chair/
├── chair_captions_20260101_223602.jsonl          # 生成的描述文件
├── chair_captions_20260101_223602_chair_results.json  # 详细评估结果（所有样本）
├── chair_captions_20260101_223602_chair_errors.json   # 错误样本（包含幻觉的样本）
└── chair_evaluator.pkl                           # 评估器缓存（加速后续运行）
```

## JSON 文件字段说明

### 每个样本的详细信息

| 字段 | 说明 |
|------|------|
| `image_id` | COCO 图像 ID |
| `caption` | 原始生成的描述 |
| `mscoco_gt_words` | Ground Truth 对象列表（从 COCO annotations 提取） |
| `mscoco_generated_words` | 生成描述中识别到的对象列表 |
| `mscoco_hallucinated_words` | 幻觉对象列表（格式：`[原始词, 标准化对象名]`） |
| `hallucination_details` | 详细幻觉信息（包含位置和原因） |
| `recall_gt_objects` | 正确识别的 GT 对象列表 |
| `recall_count` | 正确识别的对象数量 |
| `processed_words` | 处理后的词列表（词形还原后） |
| `node_words` | 标准化后的对象名列表 |
| `word_indices` | 词在原始描述中的位置索引 |
| `words` | 原始分词结果 |
| `metrics` | CHAIRs, CHAIRi, Recall, Len 指标 |

### 总体指标

| 指标 | 说明 |
|------|------|
| `CHAIRs` | 包含幻觉的描述比例（句子级别） |
| `CHAIRi` | 幻觉对象占所有对象的比例（实例级别） |
| `Recall` | 真实对象的覆盖率 |
| `Len` | 平均描述长度（以 0.01 为单位） |

## Debug 模式 vs 普通模式

### Debug 模式 (`--debug`)

- 输出所有样本的详细处理过程
- 显示每个步骤的中间结果
- 适合：调试、分析、理解评估过程

### 普通模式（默认）

- 最多输出 10 个样本的详细信息（均匀分布）
- 其他样本只显示进度
- 适合：快速评估、批量处理

## 性能优化

### 缓存机制

- 首次运行：创建评估器（5-15 分钟）
- 后续运行：从缓存加载（几秒钟）
- 缓存文件：`results/chair/chair_evaluator.pkl`

### 按需处理

- 如果只评估少量图像（如 10 个），评估器仍然会处理所有 COCO annotations
- 但评估过程很快（只是从字典中查找）
- 缓存后，后续运行会非常快

## 故障排除

### 问题：Debug 输出太多

**解决方案**：不使用 `--debug` 参数，系统会自动选择 10 个样本输出详细信息

### 问题：JSON 文件太大

**解决方案**：这是正常的，因为包含了所有中间结果。如果只需要总体指标，可以只查看 `overall_metrics`

### 问题：某些样本没有详细信息

**解决方案**：确保使用 `--debug` 参数，或者样本索引在自动选择的范围内

## 示例工作流

```bash
# 1. 生成描述并自动评估（Debug模式）
python run_chair_eval.py \
    --coco-root /home/liying/Documents/dataset/coco \
    --debug \
    --num-samples 10

# 2. 查看详细结果
cat results/chair/chair_captions_*_chair_results.json | jq '.overall_metrics'

# 3. 查看错误样本
cat results/chair/chair_captions_*_chair_errors.json | jq '.error_samples[0]'
```

## 与 run_pope_eval.py 的对比

| 特性 | run_pope_eval.py | run_chair_eval.py |
|------|-----------------|-------------------|
| 输出格式 | Yes/No | 详细描述 |
| Debug 模式 | ✓ | ✓（新增） |
| 详细 JSON | ✓ | ✓（增强） |
| 错误样本文件 | ✓ | ✓（新增） |
| 中间结果 | 部分 | 完整（新增） |

现在你可以完整地追踪每个样本从输入到最终评估的整个过程！
