# LLaVA v1.5 7B 测试脚本使用说明

## 功能特性

1. **参数化管理**: 所有参数都可通过命令行参数配置，当前设置作为默认值
2. **双模式输出**: 同时输出原生 LLaVA 和 Deco 策略的推理结果
3. **批量处理**: 支持从 JSON 文件批量处理多张图片
4. **结果保存**: 自动将结果保存到 JSON 文件

## 使用方法

### 1. 单张图片处理

```bash
python tests/test_llava_v15_7b.py --image-file /path/to/image.png
```

### 2. 批量处理

首先创建一个 JSON 文件，格式如下：

```json
[
  {
    "image_path": "/absolute/path/to/image1.png",
    "prompt": "Please describe this image in detail."
  },
  {
    "image_path": "/absolute/path/to/image2.jpg",
    "prompt": "What objects are in this image?"
  },
  {
    "image_path": "/absolute/path/to/image3.png",
    "prompt": ""
  }
]
```

注意：
- `image_path`: 图片的绝对路径（必需）
- `prompt`: 提示词，如果为空字符串，将使用默认提示词（可选）

然后运行：

```bash
python tests/test_llava_v15_7b.py --batch-file tests/example_batch.json
```

### 3. 自定义参数

```bash
python tests/test_llava_v15_7b.py \
  --image-file /path/to/image.png \
  --model-path /path/to/model \
  --device cuda \
  --temperature -1 \
  --max-new-tokens 512 \
  --use-deco \
  --alpha 0.5 \
  --threshold-top-p 0.9 \
  --threshold-top-k 20 \
  --early-exit-layers "20,21,22,23,24,25,26,27,28" \
  --output-dir tests/output
```

## 参数说明

### 模型参数
- `--model-path`: 模型路径（默认: `/home/liying/Documents/llava-v1.5-7b`）
- `--device`: 设备，cuda 或 cpu（默认: `cuda`）
- `--conv-mode`: 对话模式（默认: `llava_v1`）

### 输入参数
- `--image-file`: 单张图片路径（与 `--batch-file` 二选一）
- `--batch-file`: 批量处理 JSON 文件路径（与 `--image-file` 二选一）
- `--default-prompt`: 默认提示词（默认: `"Please describe this image in detail."`）

### 生成参数
- `--temperature`: 生成温度，-1 表示不使用采样（默认: `-1`）
- `--top-p`: Top-p 采样参数（默认: `None`）
- `--num-beams`: Beam search 数量（默认: `1`）
- `--max-new-tokens`: 最大生成 token 数（默认: `512`）

### Deco 参数
- `--use-deco`: 启用 Deco 早退机制（默认: 启用）
- `--no-deco`: 禁用 Deco 早退机制
- `--alpha`: Deco 置信度阈值参数（默认: `0.5`）
- `--threshold-top-p`: 早退判断的 top-p 阈值（默认: `0.9`）
- `--threshold-top-k`: 早退判断的 top-k 阈值（默认: `20`）
- `--early-exit-layers`: 允许早退的层索引列表，用逗号分隔（默认: `"20,21,22,23,24,25,26,27,28"`）

### 输出参数
- `--output-dir`: 输出目录（默认: `tests/output`）

## 输出格式

结果会保存为 JSON 文件，格式如下：

```json
{
  "image_path": "/path/to/image.png",
  "prompt": "Please describe this image in detail.",
  "image_size": [1024, 768],
  "timestamp": "2024-01-01T12:00:00",
  "native_llava": {
    "output": "生成的文本...",
    "input_tokens": 100,
    "output_tokens": 200,
    "total_tokens": 300,
    "generation_time": 5.23,
    "tokens_per_second": 38.24
  },
  "deco": {
    "output": "生成的文本...",
    "input_tokens": 100,
    "output_tokens": 200,
    "total_tokens": 300,
    "generation_time": 3.45,
    "tokens_per_second": 57.97,
    "alpha": 0.5,
    "threshold_top_p": 0.9,
    "threshold_top_k": 20,
    "early_exit_layers": [20, 21, 22, 23, 24, 25, 26, 27, 28]
  },
  "speedup": 1.52
}
```

批量处理的结果是一个数组，每个元素都是上述格式的对象。

## 注意事项

1. Deco 策略仅在贪婪生成模式下生效（`num_beams=1` 且 `do_sample=False`）
2. 批量处理时，如果某个任务失败，会在结果中记录错误信息，不会中断整个流程
3. 输出文件会自动添加时间戳，避免覆盖之前的结果
4. 图片路径必须是绝对路径
