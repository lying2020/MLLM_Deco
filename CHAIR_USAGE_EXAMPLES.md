# CHAIR 评估函数使用示例


### CHAIR 指标说明

- **CHAIR-s**: 包含幻觉对象的描述比例（sentence-level）
- **CHAIR-i**: 幻觉对象占所有对象的比例（instance-level）
- **Recall**: 真实对象的覆盖率
- **Len**: 平均描述长度

本文档展示如何使用封装后的 `chair.py` 中的函数接口。

## 函数接口说明

### 1. `evaluate_chair()` - 从文件计算 CHAIR 指标

这是最常用的函数，从 JSONL 或 JSON 文件读取描述并计算 CHAIR 指标。

```python
from chair import evaluate_chair

# 基本用法
results = evaluate_chair(
    cap_file="results/chair/captions.jsonl",
    coco_path="/path/to/coco/annotations_trainval2014/annotations/",
    image_id_key="image_id",
    caption_key="caption"
)

# 查看结果
print("CHAIRs:", results['overall_metrics']['CHAIRs'])
print("CHAIRi:", results['overall_metrics']['CHAIRi'])
print("Recall:", results['overall_metrics']['Recall'])
print("Len:", results['overall_metrics']['Len'])
```

### 2. `evaluate_chair_from_dict()` - 从字典列表计算 CHAIR 指标

如果描述数据已经在内存中（字典列表），可以直接使用此函数，无需先保存到文件。

```python
from chair import evaluate_chair_from_dict

# 准备描述数据
captions = [
    {"image_id": 226988, "caption": "The image features a woman sitting at a dining table..."},
    {"image_id": 337443, "caption": "The image depicts a busy city street..."},
    # ... 更多描述
]

# 计算 CHAIR 指标
results = evaluate_chair_from_dict(
    captions_dict=captions,
    coco_path="/path/to/coco/annotations_trainval2014/annotations/",
    image_id_key="image_id",
    caption_key="caption"
)

# 查看结果
print("CHAIRs:", results['overall_metrics']['CHAIRs'])
```

### 3. `get_chair_evaluator()` - 获取评估器对象

如果需要多次评估，可以重用评估器对象以提高效率。

```python
from chair import get_chair_evaluator

# 获取评估器（支持缓存）
evaluator = get_chair_evaluator(
    coco_path="/path/to/coco/annotations_trainval2014/annotations/",
    cache_file="chair_evaluator.pkl",  # 缓存文件路径
    use_cache=True
)

# 使用评估器计算多个文件的 CHAIR 指标
results1 = evaluator.compute_chair("captions1.jsonl", "image_id", "caption")
results2 = evaluator.compute_chair("captions2.jsonl", "image_id", "caption")
```

## 完整示例

### 示例 1: 在 run_chair_eval.py 中使用

```python
# 在 run_chair_eval.py 中已经集成了自动评估功能
# 使用 --auto-evaluate 参数即可自动计算 CHAIR 指标

python run_chair_eval.py \
    --coco-root /path/to/coco \
    --output-file results/chair/captions.jsonl \
    --auto-evaluate
```

### 示例 2: 在 Python 脚本中使用

```python
from chair import evaluate_chair
import project as project

# 计算 CHAIR 指标
results = evaluate_chair(
    cap_file="results/chair/captions.jsonl",
    coco_path=project.coco_annotations_path,
    image_id_key="image_id",
    caption_key="caption",
    cache_file="results/chair/chair_evaluator.pkl",  # 使用缓存加速
    use_cache=True,
    save_path="results/chair/chair_results.json",  # 保存详细结果
    verbose=True
)

# 访问结果
metrics = results['overall_metrics']
print(f"CHAIRs: {metrics['CHAIRs']:.4f}")
print(f"CHAIRi: {metrics['CHAIRi']:.4f}")
print(f"Recall: {metrics['Recall']:.4f}")
print(f"Len: {metrics['Len']:.4f}")

# 访问每个句子的详细信息
for sentence in results['sentences'][:5]:  # 前5个句子
    print(f"Image ID: {sentence['image_id']}")
    print(f"Caption: {sentence['caption'][:100]}...")
    print(f"Hallucinated words: {sentence['mscoco_hallucinated_words']}")
    print(f"CHAIRi: {sentence['metrics']['CHAIRi']:.4f}")
    print()
```

### 示例 3: 批量评估多个模型

```python
from chair import get_chair_evaluator
import project as project

# 获取评估器（只创建一次，可重复使用）
evaluator = get_chair_evaluator(
    coco_path=project.coco_annotations_path,
    cache_file="chair_evaluator.pkl",
    use_cache=True
)

# 评估多个模型的结果
models = ["llava_baseline", "llava_deco", "instructblip"]
for model_name in models:
    cap_file = f"results/chair/{model_name}_captions.jsonl"
    results = evaluator.compute_chair(cap_file, "image_id", "caption")

    print(f"\n{model_name}:")
    print(f"  CHAIRs: {results['overall_metrics']['CHAIRs']:.4f}")
    print(f"  CHAIRi: {results['overall_metrics']['CHAIRi']:.4f}")
    print(f"  Recall: {results['overall_metrics']['Recall']:.4f}")
```

## 参数说明

### `evaluate_chair()` 参数

- `cap_file` (str): 描述文件路径（JSON 或 JSONL 格式）
- `coco_path` (str): COCO annotations 目录路径
- `image_id_key` (str, 默认="image_id"): 描述文件中图像 ID 的键名
- `caption_key` (str, 默认="caption"): 描述文件中描述的键名
- `cache_file` (str, 可选): 缓存文件路径，用于加速重复评估
- `use_cache` (bool, 默认=True): 是否使用缓存
- `save_path` (str, 可选): 保存详细结果的路径（JSON 格式）
- `verbose` (bool, 默认=True): 是否输出详细信息

### 返回值

返回一个字典，包含：
- `overall_metrics`: 总体指标字典
  - `CHAIRs`: 包含幻觉对象的描述比例（sentence-level）
  - `CHAIRi`: 幻觉对象占所有对象的比例（instance-level）
  - `Recall`: 真实对象的覆盖率
  - `Len`: 平均描述长度
- `sentences`: 每个句子的详细结果列表
- `evaluator`: CHAIR 评估器对象（可用于后续评估）

## 注意事项

1. **首次运行较慢**：首次创建 CHAIR 评估器时需要加载 COCO annotations，可能需要几分钟时间。后续使用缓存会快很多。

2. **缓存文件**：建议使用缓存文件（`cache_file` 参数）来加速重复评估。缓存文件可以在多次运行之间共享。

3. **内存使用**：CHAIR 评估器会加载所有 COCO annotations 到内存，确保有足够的内存空间。

4. **文件格式**：描述文件可以是 JSON 或 JSONL 格式，每行/每个对象应包含 `image_id` 和 `caption` 字段。

## 与命令行接口的对比

### 命令行方式（原有方式）

```bash
python chair.py \
    --cap_file results/chair/captions.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --coco_path /path/to/coco/annotations_trainval2014/annotations/ \
    --save_path results/chair/chair_results.json \
    --cache chair_evaluator.pkl
```

### Python 接口方式（新方式）

```python
from chair import evaluate_chair

results = evaluate_chair(
    cap_file="results/chair/captions.jsonl",
    coco_path="/path/to/coco/annotations_trainval2014/annotations/",
    image_id_key="image_id",
    caption_key="caption",
    save_path="results/chair/chair_results.json",
    cache_file="chair_evaluator.pkl"
)
```

两种方式功能相同，Python 接口更适合在脚本中集成使用。
