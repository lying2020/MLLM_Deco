#!/usr/bin/env python3
"""
POPE Probing 实验脚本
基于层级隐藏状态的物体存在性分类器训练和评估

功能：
1. 为每一层训练一个独立的线性分类器
2. 使用该层的隐藏状态判断物体是否存在
3. 评估每层分类器的准确率
4. 可视化结果
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import numpy as np
import matplotlib.pyplot as plt

# 抑制常见的无害警告
warnings.filterwarnings('ignore', message='.*You are using a model of type llava to instantiate a model of type llava_llama.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'MLLM_Deco' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from project import llava_v15_7b_path
from eval_tool.eval_pope import evaluate_pope

from PIL import Image
import re
from transformers import set_seed


class LayerClassifier(nn.Module):
    """
    单层线性分类器
    输入：隐藏状态向量 (hidden_size,)
    输出：2维 logits (存在/不存在)
    """
    def __init__(self, hidden_size: int):
        super(LayerClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, 2)  # 2分类：存在/不存在

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch_size, hidden_size] 或 [hidden_size]
        Returns:
            logits: [batch_size, 2] 或 [2]
        """
        return self.classifier(hidden_state)


class ProbingDataset(Dataset):
    """
    Probing 数据集
    每个样本包含：图像路径、物体词、真实标签、所有层的隐藏状态
    """
    def __init__(self, data: List[Dict], hidden_states: Dict[int, torch.Tensor]):
        """
        Args:
            data: List of {image, text (object word), label (0/1)}
            hidden_states: Dict[layer_idx, tensor] where tensor shape is [num_samples, hidden_size]
        """
        self.data = data
        self.hidden_states = hidden_states
        self.num_layers = len(hidden_states)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(sample['label'], dtype=torch.long)

        # 获取所有层的隐藏状态，确保断开计算图
        layer_hidden_states = {}
        for layer_idx in range(self.num_layers):
            hidden_state = self.hidden_states[layer_idx][idx]
            # 确保断开计算图（虽然应该已经 detach，但为了安全再检查）
            if hidden_state.requires_grad:
                hidden_state = hidden_state.detach()
            layer_hidden_states[layer_idx] = hidden_state

        return {
            'label': label,
            'hidden_states': layer_hidden_states,
            'image': sample.get('image', ''),
            'text': sample.get('text', '')
        }


def load_image(image_file):
    """加载图像文件，支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"图像文件不存在: {image_file}")
        image = Image.open(image_file).convert("RGB")
    return image


def auto_detect_questions(probe_exp_dir="probe_exp/train_set", split="adversarial", coco_root="/home/liying/Documents/dataset/coco"):
    """从 probe_exp/train_set 目录读取问题文件"""
    probe_exp_dir = Path(probe_exp_dir)
    coco_root = Path(coco_root)

    question_file = probe_exp_dir / f"coco_pope_{split}.json"
    if not question_file.exists():
        raise FileNotFoundError(f"找不到问题文件: {question_file}")

    all_questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            image_relative_path = sample['image']
            image_path = coco_root / image_relative_path
            image_path = image_path.resolve()

            if not image_path.exists():
                print(f"⚠️  警告: 图像文件不存在: {image_path}")
                continue

            # 提取物体词（从问题中提取，例如 "Is there a dog in the image?" -> "dog"）
            question_text = sample['text']
            # 使用正则表达式提取物体词
            import re
            # 匹配 "Is there a/an {object} in the image?" 格式
            pattern = r'is there (?:a|an) (.+?)(?:\s+in\s+(?:the|this)\s+image)?\s*\??$'
            match = re.search(pattern, question_text.lower())
            if match:
                object_word = match.group(1).strip()
            else:
                # 如果正则匹配失败，使用简单方法
                object_word = question_text.lower()
                for prefix in ["is there a ", "is there an ", "is there "]:
                    if object_word.startswith(prefix):
                        object_word = object_word[len(prefix):]
                        break
                for suffix in [" in the image?", " in this image?", "?"]:
                    if object_word.endswith(suffix):
                        object_word = object_word[:-len(suffix)]
                object_word = object_word.strip()

            # 转换标签：Yes -> 1 (存在), No -> 0 (不存在)
            raw_label = sample.get('label', '').lower().strip()
            if raw_label == 'yes':
                label = 1  # 物体存在
            elif raw_label == 'no':
                label = 0  # 物体不存在
            else:
                # 如果标签格式异常，给出警告并使用默认值
                print(f"⚠️  警告: 未知标签格式 '{sample.get('label', '')}'，默认设为 0")
                label = 0

            all_questions.append({
                "question_id": sample['question_id'],
                "image": str(image_path),
                "text": object_word,  # 物体词
                "label": label,  # 0/1: 0=不存在(No), 1=存在(Yes)
                "question": question_text,  # 保留原始问题
                "raw_label": sample.get('label', '')  # 保留原始标签用于调试
            })

    all_questions.sort(key=lambda x: x['question_id'])
    return all_questions


def extract_hidden_states(model, tokenizer, image_processor, image_file: str, prompt: str,
                         conv_mode: str, device: str, num_layers: int):
    """
    提取所有层的隐藏状态

    Returns:
        返回最后一个 token 的隐藏状态: [num_layers, hidden_size]
    """
    # 加载图像
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    # 添加问题格式
    qs = qs + " Please answer with Yes or No only."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 准备多模态输入（参考 llava_llama.py 的 generate 方法）
    images = image_tensor.unsqueeze(0).half().to(device)

    # 调用 prepare_inputs_labels_for_multimodal 处理多模态输入
    (
        input_ids_processed,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,  # position_ids
        None,  # attention_mask
        None,  # past_key_values
        None,  # labels
        images
    )

    # Forward pass 获取隐藏状态
    with torch.no_grad():
        outputs = model.get_model().forward(
            input_ids=input_ids_processed if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    # 获取所有层的隐藏状态
    all_hidden_states = outputs.hidden_states  # List of [batch_size, seq_len, hidden_size]

    # 提取最后一个 token 的隐藏状态（用于分类）
    last_token_hidden_states = []
    for layer_idx in range(len(all_hidden_states)):
        # 取最后一个 token: [batch_size, hidden_size]
        last_token_hidden = all_hidden_states[layer_idx][:, -1, :]  # [1, hidden_size]
        # 断开计算图（probing 不需要梯度）
        last_token_hidden = last_token_hidden.squeeze(0).detach()  # [hidden_size]
        last_token_hidden_states.append(last_token_hidden)

    # Stack: [num_layers, hidden_size]
    return torch.stack(last_token_hidden_states, dim=0)


def collect_probing_data(model, tokenizer, image_processor, questions: List[Dict],
                        conv_mode: str, device: str, num_layers: int, verbose: bool = False):
    """
    收集所有样本的隐藏状态

    Returns:
        data: List of samples with labels
        hidden_states: Dict[layer_idx, tensor] where tensor shape is [num_samples, hidden_size]
    """
    print(f"\n收集 {len(questions)} 个样本的隐藏状态...")

    all_hidden_states_by_layer = {i: [] for i in range(num_layers)}
    valid_data = []

    for idx, sample in enumerate(tqdm(questions, desc="提取隐藏状态")):
        try:
            # 提取隐藏状态
            hidden_states = extract_hidden_states(
                model, tokenizer, image_processor,
                sample['image'], sample['text'],  # 使用物体词作为 prompt
                conv_mode, device, num_layers
            )

            # 分离各层的隐藏状态，转换为 float32 并移到 CPU，断开计算图
            for layer_idx in range(num_layers):
                # 断开计算图（probing 不需要梯度），转换为 float32，移到 CPU
                hidden_state = hidden_states[layer_idx].detach().cpu()
                if hidden_state.dtype != torch.float32:
                    hidden_state = hidden_state.float()
                all_hidden_states_by_layer[layer_idx].append(hidden_state)

            valid_data.append(sample)

        except Exception as e:
            if verbose:
                print(f"⚠️  样本 {idx} 处理失败: {e}")
            continue

    # 转换为张量
    hidden_states_dict = {}
    for layer_idx in range(num_layers):
        # Stack: [num_samples, hidden_size]
        hidden_states_dict[layer_idx] = torch.stack(all_hidden_states_by_layer[layer_idx], dim=0)

    print(f"✓ 成功收集 {len(valid_data)} 个样本的隐藏状态")
    return valid_data, hidden_states_dict


def split_dataset(dataset: ProbingDataset, train_ratio: float = 0.5):
    """
    将数据集分割为训练集和测试集

    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例（默认 0.5，即前一半）

    Returns:
        train_dataset: 训练集
        test_dataset: 测试集
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    # 分割数据
    train_data = dataset.data[:train_size]
    test_data = dataset.data[train_size:]

    # 分割隐藏状态
    train_hidden_states = {}
    test_hidden_states = {}
    for layer_idx in range(dataset.num_layers):
        train_hidden_states[layer_idx] = dataset.hidden_states[layer_idx][:train_size]
        test_hidden_states[layer_idx] = dataset.hidden_states[layer_idx][train_size:]

    # 创建新的数据集
    train_dataset = ProbingDataset(train_data, train_hidden_states)
    test_dataset = ProbingDataset(test_data, test_hidden_states)

    return train_dataset, test_dataset


def train_classifiers(dataset: ProbingDataset, num_layers: int, hidden_size: int,
                     device: str, num_epochs: int = 10, batch_size: int = 32,
                     learning_rate: float = 0.001, weight_decay: float = 0.01, verbose: bool = False):
    """
    训练所有层的分类器

    Returns:
        classifiers: Dict[layer_idx, LayerClassifier]
        train_losses: Dict[layer_idx, List[float]]
    """
    print(f"\n训练 {num_layers} 个分类器...")

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 为每层创建分类器和优化器
    classifiers = {}
    optimizers = {}
    criterion = nn.CrossEntropyLoss()
    train_losses = {i: [] for i in range(num_layers)}

    for layer_idx in range(num_layers):
        classifier = LayerClassifier(hidden_size).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifiers[layer_idx] = classifier
        optimizers[layer_idx] = optimizer

    # 训练循环
    for epoch in range(num_epochs):
        epoch_losses = {i: [] for i in range(num_layers)}

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            labels = batch['label'].to(device)

            # 对每层分别训练
            for layer_idx in range(num_layers):
                classifier = classifiers[layer_idx]
                optimizer = optimizers[layer_idx]

                # 获取该层的隐藏状态，并确保是 float32 类型且断开计算图
                hidden_states = batch['hidden_states'][layer_idx].to(device)  # [batch_size, hidden_size]
                # 确保断开计算图（虽然应该已经 detach，但为了安全再检查）
                if hidden_states.requires_grad:
                    hidden_states = hidden_states.detach()
                # 转换为 float32（分类器使用 float32）
                if hidden_states.dtype != torch.float32:
                    hidden_states = hidden_states.float()

                # Forward
                logits = classifier(hidden_states)  # [batch_size, 2]
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses[layer_idx].append(loss.item())

        # 记录平均损失
        for layer_idx in range(num_layers):
            avg_loss = np.mean(epoch_losses[layer_idx])
            train_losses[layer_idx].append(avg_loss)

        # 记录并显示训练进度
        avg_loss = np.mean([np.mean(epoch_losses[i]) for i in range(num_layers)])
        if verbose or (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: 平均损失 = {avg_loss:.4f}")
            # 显示损失最高的几层
            layer_avg_losses = {i: np.mean(epoch_losses[i]) for i in range(num_layers)}
            top_loss_layers = sorted(layer_avg_losses.items(), key=lambda x: x[1], reverse=True)[:3]
            if verbose:
                print(f"  损失最高的3层: {', '.join([f'Layer {l}: {loss:.4f}' for l, loss in top_loss_layers])}")

    print("✓ 分类器训练完成")

    # 显示最终训练损失
    final_losses = {i: train_losses[i][-1] if train_losses[i] else 0.0 for i in range(num_layers)}
    print(f"最终训练损失范围: {min(final_losses.values()):.4f} - {max(final_losses.values()):.4f}")

    return classifiers, train_losses


def evaluate_classifiers(classifiers: Dict[int, LayerClassifier], dataset: ProbingDataset,
                        device: str, verbose: bool = False):
    """
    评估所有层的分类器

    Returns:
        accuracies: Dict[layer_idx, float]
    """
    print(f"\n评估 {len(classifiers)} 个分类器...")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    accuracies = {}

    for layer_idx, classifier in classifiers.items():
        classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                labels = batch['label'].to(device)
                hidden_states = batch['hidden_states'][layer_idx].to(device)
                # 确保是 float32 类型
                if hidden_states.dtype != torch.float32:
                    hidden_states = hidden_states.float()

                logits = classifier(hidden_states)
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        accuracies[layer_idx] = accuracy

        if verbose:
            print(f"Layer {layer_idx:2d}: Accuracy = {accuracy:.4f} ({correct}/{total})")

    print("✓ 评估完成")
    return accuracies


def visualize_results(accuracies: Dict[int, float], output_file: str):
    """可视化每层的准确率"""
    layers = sorted(accuracies.keys())
    accs = [accuracies[l] for l in layers]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, accs, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Probing Accuracy by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    # 标记最高准确率
    max_acc_idx = np.argmax(accs)
    max_acc = accs[max_acc_idx]
    max_layer = layers[max_acc_idx]
    plt.annotate(f'Max: {max_acc:.4f} @ Layer {max_layer}',
                xy=(max_layer, max_acc),
                xytext=(max_layer + len(layers)*0.1, max_acc + 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化结果已保存到: {output_file}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="POPE Probing 实验")

    # 数据集参数
    parser.add_argument("--probe-exp-dir", type=str, default="probe_exp/train_set",
                       help="probe_exp/train_set 目录路径")
    parser.add_argument("--split", type=str, default="adversarial",
                       choices=["adversarial", "popular", "random"], help="数据集 split")
    parser.add_argument("--coco-root", type=str, default="/home/liying/Documents/dataset/coco",
                       help="COCO 数据集根目录")
    parser.add_argument("--num-samples", type=int, default=3000,
                       help="使用的样本数量（0表示全部）")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=llava_v15_7b_path,
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备")

    # 训练参数
    parser.add_argument("--num-epochs", type=int, default=50,
                       help="训练轮数（默认: 50，增加轮数以获得更好的效果）")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="权重衰减（L2正则化）")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认: results/pope_probing）")

    # 其他参数
    parser.add_argument("--seed", type=int, default=30, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()
    set_seed(args.seed)

    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(project_root, "results", "pope_probing")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("POPE Probing 实验")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"数据集: {args.split}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 加载模型
    print("\n[1/6] 加载模型...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, device=args.device
    )
    model.eval()

    # 获取模型层数
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"✓ 模型加载完成: {model_name}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 隐藏状态维度: {hidden_size}")
    print(f"  - 每层分类器参数量: {(hidden_size + 1) * 2} ≈ {((hidden_size + 1) * 2) / 1000:.1f}K")
    print(f"  - 总分类器参数量: {num_layers * (hidden_size + 1) * 2 / 1000:.1f}K")

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 加载数据
    print(f"\n[2/6] 加载数据...")
    questions = auto_detect_questions(args.probe_exp_dir, args.split, args.coco_root)
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    print(f"✓ 加载了 {len(questions)} 个样本")

    # 收集隐藏状态
    print(f"\n[3/6] 收集隐藏状态...")
    data, hidden_states = collect_probing_data(
        model, tokenizer, image_processor, questions,
        conv_mode, args.device, num_layers, verbose=args.verbose
    )

    # 创建数据集
    full_dataset = ProbingDataset(data, hidden_states)
    print(f"✓ 数据集创建完成: {len(full_dataset)} 个样本")

    # 分割数据集（前一半训练，后一半测试）
    print(f"\n[4/6] 分割数据集...")
    train_dataset, test_dataset = split_dataset(full_dataset, train_ratio=0.5)
    print(f"  - 训练集: {len(train_dataset)} 个样本")
    print(f"  - 测试集: {len(test_dataset)} 个样本")

    # 检查数据标签分布
    train_labels = [sample['label'] for sample in train_dataset.data]
    test_labels = [sample['label'] for sample in test_dataset.data]
    train_pos = sum(train_labels)  # label=1 (Yes, 存在)
    train_neg = len(train_labels) - train_pos  # label=0 (No, 不存在)
    test_pos = sum(test_labels)
    test_neg = len(test_labels) - test_pos
    print(f"\n数据标签分布:")
    print(f"  标签定义: 0 = No (不存在), 1 = Yes (存在)")
    print(f"  - 训练集: 正样本(Yes/1)={train_pos} ({train_pos/len(train_labels)*100:.1f}%), 负样本(No/0)={train_neg} ({train_neg/len(train_labels)*100:.1f}%)")
    print(f"  - 测试集: 正样本(Yes/1)={test_pos} ({test_pos/len(test_labels)*100:.1f}%), 负样本(No/0)={test_neg} ({test_neg/len(test_labels)*100:.1f}%)")

    # 如果数据严重不平衡，给出警告
    train_imbalance = abs(train_pos - train_neg) / len(train_labels)
    test_imbalance = abs(test_pos - test_neg) / len(test_labels)
    if train_imbalance > 0.3:
        print(f"  ⚠️  警告: 训练集标签分布不平衡 (差异={train_imbalance*100:.1f}%)，可能影响训练效果")
        print(f"     建议: 使用类别权重或平衡采样")
    if test_imbalance > 0.3:
        print(f"  ⚠️  警告: 测试集标签分布不平衡 (差异={test_imbalance*100:.1f}%)")

    # 如果数据完全平衡，也给出提示
    if train_imbalance < 0.05 and test_imbalance < 0.05:
        print(f"  ✓ 数据标签分布较为平衡")

    # 训练分类器（使用训练集）
    print(f"\n[5/6] 训练分类器（使用训练集）...")
    classifiers, train_losses = train_classifiers(
        train_dataset, num_layers, hidden_size, args.device,
        args.num_epochs, args.batch_size, args.learning_rate,
        args.weight_decay, verbose=args.verbose
    )

    # 在训练集上评估，检查是否过拟合
    print(f"\n在训练集上评估（检查过拟合）...")
    train_accuracies = evaluate_classifiers(classifiers, train_dataset, args.device, verbose=False)
    print(f"训练集平均准确率: {np.mean(list(train_accuracies.values())):.4f}")
    print(f"训练集最高准确率: {max(train_accuracies.values()):.4f} @ Layer {max(train_accuracies, key=train_accuracies.get)}")

    # 评估分类器（使用测试集）
    print(f"\n[6/6] 评估分类器（使用测试集）...")
    accuracies = evaluate_classifiers(classifiers, test_dataset, args.device, verbose=args.verbose)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"probing_results_{args.split}_{timestamp}.json")

    results = {
        "model": model_name,
        "split": args.split,
        "num_samples": len(data),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "accuracies": accuracies,
        "train_losses": {str(k): v for k, v in train_losses.items()},
        "train_accuracies": train_accuracies,
        "config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_ratio": 0.5
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {results_file}")

    # 可视化
    viz_file = os.path.join(output_dir, f"probing_accuracy_{args.split}_{timestamp}.png")
    visualize_results(accuracies, viz_file)

    # 打印摘要
    print("\n" + "=" * 80)
    print("实验结果摘要（测试集）")
    print("=" * 80)
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"层数范围: 0 - {num_layers-1}")
    print(f"\n测试集结果:")
    print(f"  最高准确率: {max(accuracies.values()):.4f} @ Layer {max(accuracies, key=accuracies.get)}")
    print(f"  最低准确率: {min(accuracies.values()):.4f} @ Layer {min(accuracies, key=accuracies.get)}")
    print(f"  平均准确率: {np.mean(list(accuracies.values())):.4f}")
    print(f"\n训练集结果（参考）:")
    print(f"  最高准确率: {max(train_accuracies.values()):.4f} @ Layer {max(train_accuracies, key=train_accuracies.get)}")
    print(f"  平均准确率: {np.mean(list(train_accuracies.values())):.4f}")

    # 分析结果
    test_avg = np.mean(list(accuracies.values()))
    train_avg = np.mean(list(train_accuracies.values()))
    if test_avg < 0.55:
        print(f"\n⚠️  警告: 测试集准确率 ({test_avg:.4f}) 接近随机猜测 (0.5)，可能的原因:")
        print(f"   1. 训练轮数不足（当前: {args.num_epochs}）")
        print(f"   2. 隐藏状态特征区分度不够")
        print(f"   3. 数据标签分布不平衡")
        print(f"   4. 学习率或正则化参数需要调整")
    elif abs(train_avg - test_avg) > 0.1:
        print(f"\n⚠️  警告: 训练集和测试集准确率差距较大，可能存在过拟合")
    print("=" * 80)

    # 打印每层准确率（表格形式）
    print("\n每层准确率:")
    print("-" * 40)
    for layer_idx in sorted(accuracies.keys()):
        print(f"Layer {layer_idx:2d}: {accuracies[layer_idx]:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
