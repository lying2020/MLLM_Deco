# POPE 评估 - 直接运行版本

## 简介

`run_pope_eval.py` 是一个可以直接运行的 POPE 评估脚本，**无需任何参数**即可运行。它会自动：

- ✅ 检测并使用 `pope_visualized` 目录中的数据集
- ✅ 使用 `project.py` 中配置的模型路径
- ✅ 自动生成问题文件（如果不存在）
- ✅ 自动设置输出路径
- ✅ 使用默认的 Deco 参数

## 快速开始

### 最简单的方式（零参数）

```bash
# 直接运行，使用所有默认值
python3 run_pope_eval.py
```

这将：
- 使用 `Full/adversarial` split
- 自动生成问题文件 `pope_questions_adversarial.jsonl`
- 将结果保存到 `results/pope/pope_adversarial_YYYYMMDD_HHMMSS.jsonl`
- 使用 GPU 0（如果可用）
- 使用默认 Deco 参数

### 评估不同的 split

```bash
# 评估 popular split
python3 run_pope_eval.py --split popular

# 评估 random split
python3 run_pope_eval.py --split random

# 评估 default config 的 test split
python3 run_pope_eval.py --config default --split test
```

### 自定义参数（可选）

```bash
# 使用不同的 Deco 参数
python3 run_pope_eval.py --alpha 0.7 --start-layer 22 --end-layer 29

# 禁用 Deco 早退机制
python3 run_pope_eval.py --no-deco

# 使用 CPU（如果 GPU 不可用）
python3 run_pope_eval.py --device -1

# 使用指定的问题文件（跳过自动生成）
python3 run_pope_eval.py --question-file my_questions.jsonl

# 指定输出文件
python3 run_pope_eval.py --answers-file my_results.jsonl
```

## 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `Full` | 数据集 config |
| `--split` | `adversarial` | 数据集 split |
| `--model-path` | `project.py` 中的路径 | 模型路径 |
| `--device` | `0` (自动检测) | GPU 设备 |
| `--use-deco` | `True` | 启用 Deco |
| `--alpha` | `0.6` | Deco 参数 |
| `--start-layer` | `20` | 早退起始层 |
| `--end-layer` | `29` | 早退结束层 |
| `--temperature` | `-1` | 贪婪生成 |
| `--seed` | `42` | 随机种子 |

## 输出说明

### 自动生成的文件

1. **问题文件**: `pope_questions_{split}.jsonl`
   - 如果已存在，将直接使用
   - 如果不存在，将自动从 `pope_visualized` 生成

2. **答案文件**: `results/pope/pope_{split}_{timestamp}.jsonl`
   - 自动创建带时间戳的文件名
   - 避免覆盖之前的评估结果

### 输出格式

答案文件每行一个 JSON 对象：
```json
{
  "question_id": 0,
  "prompt": "Is there a person in the image?",
  "text": "Yes",
  "model_id": "llava-v1.5-7b",
  "image": "/absolute/path/to/image.png",
  "metadata": {}
}
```

## 完整工作流程示例

### 1. 评估所有 Full config 的 splits

```bash
# 评估 adversarial
python3 run_pope_eval.py --split adversarial

# 评估 popular
python3 run_pope_eval.py --split popular

# 评估 random
python3 run_pope_eval.py --split random
```

### 2. 评估结果

```bash
# 评估 adversarial 结果
python3 eval_tool/eval_pope.py \
    --gt_files probe_exp/train_set/coco_pope_adversarial.json \
    --gen_files results/pope/pope_adversarial_*.jsonl

# 评估 popular 结果
python3 eval_tool/eval_pope.py \
    --gt_files probe_exp/train_set/coco_pope_popular.json \
    --gen_files results/pope/pope_popular_*.jsonl

# 评估 random 结果
python3 eval_tool/eval_pope.py \
    --gt_files probe_exp/train_set/coco_pope_random.json \
    --gen_files results/pope/pope_random_*.jsonl
```

## 与原始脚本的区别

| 特性 | `pope_llava.py` | `run_pope_eval.py` |
|------|----------------|-------------------|
| 参数要求 | 需要 `--question-file` 和 `--answers-file` | 所有参数可选 |
| 问题文件 | 需要手动生成 | 自动生成 |
| 输出路径 | 需要手动指定 | 自动生成带时间戳 |
| 数据集检测 | 无 | 自动检测 `pope_visualized` |
| 使用难度 | 需要了解参数 | 直接运行即可 |

## 常见问题

### Q: 如何知道使用了哪些默认值？

A: 运行时会打印所有配置信息：
```
================================================================================
POPE 数据集评估
================================================================================
模型路径: /home/liying/Documents/llava-v1.5-7b
设备: cuda:0
问题文件: pope_questions_adversarial.jsonl
答案文件: results/pope/pope_adversarial_20241201_120000.jsonl
Deco 参数: alpha=0.6, layers=20-29
================================================================================
```

### Q: 如何评估所有 splits？

A: 使用循环：
```bash
for split in adversarial popular random; do
    python3 run_pope_eval.py --split $split
done
```

### Q: 如何只测试少量样本？

A: 先生成问题文件，然后手动编辑：
```bash
# 生成问题文件
python3 run_pope_eval.py --split adversarial

# 只保留前10个问题
head -10 pope_questions_adversarial.jsonl > test_questions.jsonl

# 使用测试问题文件
python3 run_pope_eval.py --question-file test_questions.jsonl
```

### Q: 如何查看所有可用参数？

A: 运行帮助命令：
```bash
python3 run_pope_eval.py --help
```

## 注意事项

1. **首次运行**: 会自动生成问题文件，可能需要几秒钟
2. **GPU 内存**: 如果 GPU 内存不足，可以尝试 `--device -1` 使用 CPU（较慢）
3. **输出目录**: 结果会自动保存到 `results/pope/` 目录
4. **时间戳**: 每次运行都会生成新的答案文件，不会覆盖之前的

## 推荐使用方式

对于日常使用，推荐直接运行：
```bash
python3 run_pope_eval.py
```

如果需要评估不同的配置，只需添加 `--split` 参数即可。
