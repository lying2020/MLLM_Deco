#!/usr/bin/env python3
"""
CHAIR 评估脚本 - 生成图像描述并保存为 JSONL 格式
参考 run_pope_eval.py 的实现, 针对 CHAIR benchmark 优化
自动检测数据集和模型, 使用默认参数, 无需输入参数即可运行

CHAIR 评估需要:
1. COCO 2014 val2014 图像目录
2. 生成的描述文件(JSONL 格式): {"image_id": int, "caption": str}
3. COCO annotations 目录(用于后续的 chair.py 评估)

使用步骤:
1. 运行此脚本生成描述文件:
   python run_chair_eval.py --coco-root /path/to/coco --output-file results/chair/captions.jsonl

2. 使用 chair.py 计算 CHAIR 指标:
   python chair.py --cap_file results/chair/captions.jsonl --image_id_key image_id --caption_key caption \
                   --coco_path /path/to/coco/annotations_trainval2014/annotations/
"""

import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import warnings

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

import project as project
from PIL import Image
import requests
from io import BytesIO
from transformers import set_seed
from eval_tool.chair import evaluate_chair
from train.linear_probe_trainer import LinearProbeTrainer, LinearProbe


class LinearProbeManager:
    """管理所有 linear probe 网络，用于在生成时应用权重"""

    def __init__(self, model_dir: str, num_layers: int = 32, num_heads: int = 32,
                 input_dim: int = 128, device: str = "cuda:0"):
        """
        Args:
            model_dir: linear probe 模型保存目录
            num_layers: 模型层数
            num_heads: 每层的head数
            input_dim: head维度
            device: 设备
        """
        self.model_dir = model_dir
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.device = device
        self.probes = {}
        self.original_forwards = {}  # 保存原始的forward方法
        self.is_patched = False

        # 加载所有 linear probe
        self._load_probes()

    def _load_probes(self):
        """加载所有 linear probe 模型"""
        from pathlib import Path
        model_dir = Path(self.model_dir)

        loaded_count = 0
        detected_hidden_dim = None
        detected_use_dropout = None

        # 首先尝试检测模型架构（从第一个存在的模型文件）
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                key = f"layer_{layer_idx}_head_{head_idx}"
                model_path = model_dir / f"{key}.pth"

                if model_path.exists():
                    # 加载state_dict来检测架构
                    state_dict = torch.load(model_path, map_location=self.device)

                    # 检测架构类型
                    if "linear.weight" in state_dict:
                        # 简单线性映射架构
                        detected_hidden_dim = None
                        detected_use_dropout = False
                    elif "fc1.weight" in state_dict:
                        # 有隐藏层的架构
                        # 从fc1.weight的形状推断hidden_dim
                        fc1_weight_shape = state_dict["fc1.weight"].shape
                        detected_hidden_dim = fc1_weight_shape[0]  # [hidden_dim, input_dim]
                        # 检测是否有dropout（通过检查是否有dropout层，但通常dropout不保存参数）
                        # 这里假设如果有fc2，则可能有dropout，但dropout参数不保存在state_dict中
                        # 我们默认不使用dropout，因为dropout在eval模式下不影响
                        detected_use_dropout = False
                    else:
                        # 未知架构，使用默认
                        detected_hidden_dim = None
                        detected_use_dropout = False

                    print(f"✓ 检测到模型架构: hidden_dim={detected_hidden_dim}, use_dropout={detected_use_dropout}")
                    break

            if detected_hidden_dim is not None or detected_use_dropout is not None:
                break

        # 如果没找到任何模型，使用默认架构
        if detected_hidden_dim is None and detected_use_dropout is None:
            detected_hidden_dim = None
            detected_use_dropout = False
            print(f"⚠️  未找到模型文件，将使用默认架构: hidden_dim=None")

        # 加载所有模型
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                key = f"layer_{layer_idx}_head_{head_idx}"
                model_path = model_dir / f"{key}.pth"

                if model_path.exists():
                    # 使用检测到的架构创建probe
                    probe = LinearProbe(
                        input_dim=self.input_dim,
                        hidden_dim=detected_hidden_dim,
                        use_dropout=detected_use_dropout
                    )
                    probe.load_state_dict(torch.load(model_path, map_location=self.device))
                    probe.to(self.device)
                    probe.eval()
                    self.probes[(layer_idx, head_idx)] = probe
                    loaded_count += 1
                else:
                    # 如果模型不存在，创建一个默认的probe（输出0，即权重为1）
                    # 使用检测到的架构（如果已检测到），否则使用简单架构
                    probe = LinearProbe(
                        input_dim=self.input_dim,
                        hidden_dim=detected_hidden_dim if detected_hidden_dim is not None else None,
                        use_dropout=detected_use_dropout if detected_use_dropout is not None else False
                    )
                    probe.to(self.device)
                    probe.eval()
                    self.probes[(layer_idx, head_idx)] = probe

        print(f"✓ 加载了 {loaded_count}/{self.num_layers * self.num_heads} 个 linear probe 模型")
        if loaded_count < self.num_layers * self.num_heads:
            print(f"  ⚠️  警告: 部分 linear probe 模型不存在，将使用默认权重（1.0）")

    def get_weight(self, layer_idx: int, head_idx: int, head_vector: torch.Tensor) -> float:
        """
        获取指定 head 的权重

        Args:
            layer_idx: 层索引
            head_idx: head索引
            head_vector: head向量 [head_dim] 或 [batch, head_dim]

        Returns:
            float: 权重 lambda，范围在 [-1, 1] 之间（经过 tanh）
        """
        key = (layer_idx, head_idx)
        probe = self.probes.get(key)

        if probe is None:
            # 如果没有对应的probe，返回0（即权重为1.0）
            return 0.0

        probe.eval()
        with torch.no_grad():
            if head_vector.dim() == 1:
                head_vector = head_vector.unsqueeze(0)

            # 获取 probe 的参数数据类型（通常是 float32）
            # 从第一个参数获取 dtype
            probe_dtype = next(probe.parameters()).dtype

            # 将 head_vector 转换为与 probe 相同的数据类型和设备
            head_vector = head_vector.to(device=self.device, dtype=probe_dtype)

            # LinearProbe 输出经过 tanh，范围在 [-1, 1]
            # 直接使用输出作为 lambda（已经是 tanh 的结果）
            output = probe(head_vector)
            if output.size(0) == 1:
                lambda_value = output.cpu().item()
            else:
                lambda_value = output.cpu().item() if output.numel() == 1 else output.cpu()

            return lambda_value

    def patch_attention_layers(self, model):
        """修改 attention 层的 forward 方法，应用 linear probe 权重"""
        if self.is_patched:
            return

        lang_model = model.get_model()
        if not hasattr(lang_model, 'layers'):
            raise ValueError("模型没有layers属性")

        # 为每一层创建修改后的 forward 方法
        for layer_idx in range(self.num_layers):
            if layer_idx >= len(lang_model.layers):
                continue

            layer = lang_model.layers[layer_idx]
            if not hasattr(layer, 'self_attn'):
                continue

            attn_module = layer.self_attn
            original_forward = attn_module.forward

            # 保存原始的forward方法
            self.original_forwards[layer_idx] = original_forward

            # 创建修改后的forward方法
            # 使用模型内部的 attn_output，通过 head_weights 参数应用权重
            def make_patched_forward(layer_idx, original_forward):
                def patched_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    output_attn_output=False,  # 添加 output_attn_output 参数
                ):
                    # 获取attention层参数
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    num_heads = self.num_heads
                    head_dim = hidden_size // num_heads

                    # 先调用原始 forward 获取 attn_output（使用 output_attn_output=True）
                    # 这样可以获取 head_vector 用于计算权重
                    attn_result = original_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        output_attn_output=True,  # 启用 attn_output 输出以获取 head_vector
                    )

                    # 解包返回值（应该是4个元素：output, attn_weights, past_key_value, attn_output_before_reshape）
                    if len(attn_result) == 4:
                        _, attn_weights, past_key_value_for_return, attn_output = attn_result
                    else:
                        raise ValueError(
                            f"Expected 4 return values from attention forward (with output_attn_output=True), "
                            f"but got {len(attn_result)}. This indicates the model's attention layer "
                            f"does not support output_attn_output parameter."
                        )

                    if attn_output is None:
                        raise RuntimeError(
                            f"Failed to get attn_output from layer {layer_idx}. "
                            f"This should not happen if output_attn_output is enabled."
                        )

                    # 确保 attn_output 的形状正确: [batch, num_heads, seq_len, head_dim]
                    if attn_output.shape != (batch_size, num_heads, seq_len, head_dim):
                        raise ValueError(
                            f"attn_output shape mismatch: expected {(batch_size, num_heads, seq_len, head_dim)}, "
                            f"got {attn_output.shape}"
                        )

                    # 对每个head，使用最后一个token的head向量预测权重
                    # head_weights 形状: [num_heads]，默认所有权重为 1.0
                    head_weights = torch.ones(num_heads, device=hidden_states.device, dtype=hidden_states.dtype)

                    last_token_idx = seq_len - 1

                    # 即使没有权重变更，也要走完整流程以验证链路正确性
                    for head_idx in range(num_heads):
                        # 获取最后一个token的head向量（用于预测权重）
                        head_vector = attn_output[:, head_idx, last_token_idx, :]  # [batch, head_dim]

                        # 对于 batch 中的每个样本，计算权重（取平均值或使用第一个样本）
                        # 为了简化，我们使用第一个样本的 head_vector
                        lambda_val = self.get_weight(layer_idx, head_idx, head_vector[0])
                        # 使用 tanh 约束 lambda 在 [-1, 1] 之间（已经在 get_weight 中完成）
                        weight = 1.0 + lambda_val  # weight = 1 + lambda
                        head_weights[head_idx] = weight

                    # 无论是否有权重调整，都使用原始 forward 并传递 head_weights 参数
                    # 这样可以确保与原始模型完全一致，同时验证完整链路
                    # 参考 test_spp_head_weighting.py 的实现方式
                    # 注意：即使所有权重都是 1.0，也传递 head_weights 以验证链路
                    return original_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        output_attn_output=output_attn_output,  # 使用传入的原始值
                        head_weights=head_weights,  # 传递计算好的权重（即使都是1.0，也传递以验证链路）
                    )

                return patched_forward

            # 替换forward方法
            attn_module.forward = make_patched_forward(layer_idx, original_forward)

        self.is_patched = True
        print(f"✓ 已修改 {self.num_layers} 个 attention 层的 forward 方法")

    def unpatch_attention_layers(self, model):
        """恢复原始的 attention 层 forward 方法"""
        if not self.is_patched:
            return

        lang_model = model.get_model()
        if not hasattr(lang_model, 'layers'):
            return

        for layer_idx in range(self.num_layers):
            if layer_idx >= len(lang_model.layers):
                continue

            layer = lang_model.layers[layer_idx]
            if not hasattr(layer, 'self_attn'):
                continue

            if layer_idx in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[layer_idx]

        self.is_patched = False
        print(f"✓ 已恢复 {self.num_layers} 个 attention 层的原始 forward 方法")


def load_image(image_file):
    """加载图像文件, 支持本地文件和 URL"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"图像文件不存在: {image_file}")
        image = Image.open(image_file).convert("RGB")
    return image


def get_coco_val2014_images(coco_root: str, image_id_list: Optional[List[int]] = None, max_images: int = 0):
    """
    获取 COCO val2014 图像列表

    Args:
        coco_root: COCO 数据集根目录(包含 val2014 子目录)
        image_id_list: 可选的图像 ID 列表, 如果提供则只返回这些图像
        max_images: 最大图像数量(0 表示全部)

    Returns:
        List[Dict]: 包含 image_id 和 image_path 的字典列表
    """
    coco_root = Path(coco_root)
    val2014_dir = coco_root / "val2014"

    if not val2014_dir.exists():
        raise FileNotFoundError(f"COCO val2014 目录不存在: {val2014_dir}")

    images = []

    if image_id_list is not None:
        # 如果提供了图像 ID 列表, 只处理这些图像
        for image_id in image_id_list:
            image_filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            image_path = val2014_dir / image_filename
            if image_path.exists():
                images.append({
                    "image_id": image_id,
                    "image_path": str(image_path)
                })
            else:
                print(f"⚠️  警告: 图像文件不存在: {image_path}")
    else:
        # 扫描 val2014 目录中的所有图像
        image_files = sorted(val2014_dir.glob("COCO_val2014_*.jpg"))
        for image_file in image_files:
            # 从文件名提取 image_id
            # 格式: COCO_val2014_000000123456.jpg
            filename = image_file.stem  # 去掉 .jpg
            image_id = int(filename.split("_")[-1])
            images.append({
                "image_id": image_id,
                "image_path": str(image_file)
            })

    # 限制图像数量
    if max_images > 0:
        images = images[:max_images]

    # 按 image_id 排序
    images.sort(key=lambda x: x['image_id'])

    return images


def prepare_inputs(model, tokenizer, image_processor, image_file: str, prompt: str, conv_mode: str, device: str, verbose: bool = False):
    """
    准备模型输入

    Returns:
        input_ids, image_tensor, stopping_criteria, stop_str
    """
    # 加载图像
    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    if verbose:
        print(f"\n  [输入准备] 图像信息:")
        print(f"    - 图像路径: {image_file}")
        print(f"    - 图像尺寸: {image.size}")
        print(f"    - 图像张量形状: {image_tensor.shape}")

    # 准备文本输入
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    if verbose:
        print(f"  [输入准备] 文本信息:")
        print(f"    - 原始提示词: {prompt}")
        print(f"    - 完整提示词长度: {len(full_prompt)} 字符")

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, image_tensor, stopping_criteria, stop_str


def generate_response(model, tokenizer, input_ids, image_tensor, stopping_criteria,
                     temperature, top_p, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, num_beams=1, verbose: bool = False,
                     use_linear_probe=False, linear_probe_manager=None):
    """
    生成回答

    Returns:
        outputs: 生成的文本
        output_token_len: 生成的 token 长度
        input_token_len: 输入的 token 长度
    """
    do_sample = True if temperature > 0 else False

    # 准备生成参数
    generate_kwargs = {
        "inputs": input_ids,
        "images": image_tensor.unsqueeze(0).half().to(device),
        "do_sample": do_sample,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "return_dict": True,
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "stopping_criteria": [stopping_criteria]
    }

    if use_deco:
        generate_kwargs.update({
            "use_deco": True,
            "alpha": alpha,
            "threshold_top_p": threshold_top_p,
            "threshold_top_k": threshold_top_k,
            "early_exit_layers": early_exit_layers,
        })

    with torch.inference_mode():
        with torch.no_grad():
            output_dict = model.generate(**generate_kwargs)

    # 解码输出
    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]

    # 检查 output_ids 是否包含 input_ids（前缀匹配）
    prefix_match = False
    if output_ids.shape[1] >= input_token_len:
        prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()

    # 根据前缀匹配情况决定如何处理
    if prefix_match:
        # 如果 output_ids 包含 input_ids，需要剔除 input_ids 部分
        generated_ids = output_ids[:, input_token_len:]
        output_token_len = generated_ids.shape[1]
    else:
        # 如果 output_ids 不包含 input_ids，直接使用 output_ids 作为生成的 token
        generated_ids = output_ids
        output_token_len = generated_ids.shape[1]

    # 获取新生成的 token
    if output_token_len > 0:
        # 如果新生成的 token 以 BOS token 开头, 跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0 and generated_ids[0, 0].item() == bos_token_id:
            generated_ids = generated_ids[:, 1:]
            if generated_ids.shape[1] > 0:
                outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                outputs = ""
        else:
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    else:
        outputs = ""

    return outputs, output_token_len, input_token_len


def simplify_array_field(value):
    """
    简化数组字段, 如果是数组, 只显示最后一行

    Args:
        value: 要简化的值

    Returns:
        简化后的值
    """
    if isinstance(value, list):
        if len(value) == 0:
            return []
        # 如果是嵌套数组, 取最后一个元素
        if len(value) > 0 and isinstance(value[0], list):
            return value[-1] if len(value) > 0 else []
        # 如果是普通数组, 只取最后一个元素
        return value[-1] if len(value) > 0 else []
    return value


def get_sentence_by_image_id(results, image_id):
    """
    从结果中根据 image_id 获取句子信息

    Args:
        results: 评估结果字典
        image_id: 图像ID

    Returns:
        句子字典, 如果不存在返回 None
    """
    if results and 'sentences' in results:
        for s in results['sentences']:
            if s.get('image_id') == image_id:
                return s
    return None


def simplify_sentence_data(sentence):
    """
    简化句子数据, 将数组字段只保留最后一行

    Args:
        sentence: 句子字典

    Returns:
        简化后的句子字典
    """
    if not sentence:
        return None

    # 将 image_id 转换为完整的图片文件名
    image_id = sentence.get("image_id")
    if image_id is not None:
        # 如果 image_id 是数字，转换为文件名
        if isinstance(image_id, (int, str)) and str(image_id).isdigit():
            image_id = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        # 如果已经是文件名格式，保持不变
        elif isinstance(image_id, str) and image_id.endswith('.jpg'):
            image_id = image_id
        else:
            # 尝试转换
            try:
                image_id = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            except:
                image_id = str(image_id)

    simplified = {
        "image_id": image_id,  # 使用完整的图片文件名
        "caption": sentence.get("caption"),
        "metrics": sentence.get("metrics", {})
    }

    # 简化数组字段
    array_fields = [
        'mscoco_hallucinated_words', 'mscoco_gt_words', 'mscoco_generated_words',
        'hallucination_idxs', 'words', 'processed_words', 'node_words',
        'word_indices', 'recall_gt_objects'
    ]

    for field in array_fields:
        if field in sentence:
            simplified[field] = simplify_array_field(sentence[field])

    # 保留其他非数组字段
    other_fields = ['hallucination_details', 'recall_count']
    for field in other_fields:
        if field in sentence:
            simplified[field] = sentence[field]

    return simplified


def compare_deco_vs_vanilla(deco_results, vanilla_results, deco_captions_file, vanilla_captions_file,
                            output_file):
    """
    对比 Deco 和 Vanilla 的结果, 生成对比表格和 CHAIRs/CHAIRi 不一致 case 的 JSON 文件

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        deco_captions_file: Deco 版本的描述文件路径
        vanilla_captions_file: Vanilla 版本的描述文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载描述文件
    deco_captions = {}
    with open(deco_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                deco_captions[image_id] = item

    vanilla_captions = {}
    with open(vanilla_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                vanilla_captions[image_id] = item

    # 找到 CHAIRs 或 CHAIRi 不一致的 case
    inconsistent_cases = []
    common_image_ids = set(deco_captions.keys()) & set(vanilla_captions.keys())

    for image_id in common_image_ids:
        # 获取两个版本的句子数据
        deco_sentence = get_sentence_by_image_id(deco_results, image_id)
        vanilla_sentence = get_sentence_by_image_id(vanilla_results, image_id)

        if not deco_sentence or not vanilla_sentence:
            continue

        deco_metrics = deco_sentence.get('metrics', {})
        vanilla_metrics = vanilla_sentence.get('metrics', {})

        deco_chairs = deco_metrics.get('CHAIRs', 0)
        deco_chairi = deco_metrics.get('CHAIRi', 0)
        vanilla_chairs = vanilla_metrics.get('CHAIRs', 0)
        vanilla_chairi = vanilla_metrics.get('CHAIRi', 0)

        # 检查 CHAIRs 或 CHAIRi 是否不一致
        chairs_different = deco_chairs != vanilla_chairs
        chairi_different = abs(deco_chairi - vanilla_chairi) > 1e-6  # 浮点数比较

        if chairs_different or chairi_different:
            # 获取图片文件名
            image_filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"

            # 判断谁的效果更好(幻视率更低)
            # CHAIRs 和 CHAIRi 都是越小越好
            # 优先使用 CHAIRi 作为主要判断标准, 如果 CHAIRi 相同则使用 CHAIRs
            if deco_chairi < vanilla_chairi:
                better_method = "deco"
            elif deco_chairi > vanilla_chairi:
                better_method = "vanilla"
            else:
                # CHAIRi 相同, 使用 CHAIRs 判断
                if deco_chairs < vanilla_chairs:
                    better_method = "deco"
                elif deco_chairs > vanilla_chairs:
                    better_method = "vanilla"
                else:
                    # 两者都相同, 默认标记为 deco (实际上应该不会出现这种情况)
                    better_method = "equal"

            # 简化句子数据
            simplified_deco = simplify_sentence_data(deco_sentence)
            simplified_vanilla = simplify_sentence_data(vanilla_sentence)

            case_info = {
                "image_id": image_filename,  # 使用完整的图片文件名
                "better_method": better_method,  # 新增字段: 说明谁的效果更好
                "vanilla_data": simplified_vanilla,
                "deco_data": simplified_deco,
                "difference": {
                    "CHAIRs": deco_chairs - vanilla_chairs,
                    "CHAIRi": deco_chairi - vanilla_chairi,
                    "Recall": deco_metrics.get('Recall', 0) - vanilla_metrics.get('Recall', 0),
                    "Len": deco_metrics.get('Len', 0) - vanilla_metrics.get('Len', 0)
                }
            }
            inconsistent_cases.append(case_info)

    # 保存不一致的 case 到 JSON 文件
    comparison_result = {
        "summary": {
            "total_cases": len(common_image_ids),
            "inconsistent_cases": len(inconsistent_cases),
            "consistent_cases": len(common_image_ids) - len(inconsistent_cases),
            "inconsistency_rate": len(inconsistent_cases) / len(common_image_ids) if len(common_image_ids) > 0 else 0
        },
        "metrics_comparison": {
            "vanilla": {
                "CHAIRs": vanilla_results.get('overall_metrics', {}).get('CHAIRs', 0) if vanilla_results else 0,
                "CHAIRi": vanilla_results.get('overall_metrics', {}).get('CHAIRi', 0) if vanilla_results else 0,
                "Recall": vanilla_results.get('overall_metrics', {}).get('Recall', 0) if vanilla_results else 0,
                "Len": vanilla_results.get('overall_metrics', {}).get('Len', 0) if vanilla_results else 0
            },
            "deco": {
                "CHAIRs": deco_results.get('overall_metrics', {}).get('CHAIRs', 0) if deco_results else 0,
                "CHAIRi": deco_results.get('overall_metrics', {}).get('CHAIRi', 0) if deco_results else 0,
                "Recall": deco_results.get('overall_metrics', {}).get('Recall', 0) if deco_results else 0,
                "Len": deco_results.get('overall_metrics', {}).get('Len', 0) if deco_results else 0
            },
            "difference": {
                "CHAIRs": (deco_results.get('overall_metrics', {}).get('CHAIRs', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('CHAIRs', 0) if vanilla_results else 0),
                "CHAIRi": (deco_results.get('overall_metrics', {}).get('CHAIRi', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('CHAIRi', 0) if vanilla_results else 0),
                "Recall": (deco_results.get('overall_metrics', {}).get('Recall', 0) if deco_results else 0) -
                          (vanilla_results.get('overall_metrics', {}).get('Recall', 0) if vanilla_results else 0),
                "Len": (deco_results.get('overall_metrics', {}).get('Len', 0) if deco_results else 0) -
                       (vanilla_results.get('overall_metrics', {}).get('Len', 0) if vanilla_results else 0)
            }
        },
        "inconsistent_cases": inconsistent_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    return comparison_result


def save_both_hallucinated_errors(deco_results, vanilla_results, deco_captions_file, vanilla_captions_file,
                                   output_file):
    """
    保存两个方法都出现幻视的 error 例子

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        deco_captions_file: Deco 版本的描述文件路径
        vanilla_captions_file: Vanilla 版本的描述文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载描述文件
    deco_captions = {}
    with open(deco_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                deco_captions[image_id] = item

    vanilla_captions = {}
    with open(vanilla_captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_id = item.get("image_id")
            if image_id is not None:
                vanilla_captions[image_id] = item

    # 找到两个方法都出现幻视的 case
    both_hallucinated_cases = []
    common_image_ids = set(deco_captions.keys()) & set(vanilla_captions.keys())

    for image_id in common_image_ids:
        # 获取两个版本的句子数据
        deco_sentence = get_sentence_by_image_id(deco_results, image_id)
        vanilla_sentence = get_sentence_by_image_id(vanilla_results, image_id)

        if not deco_sentence or not vanilla_sentence:
            continue

        deco_metrics = deco_sentence.get('metrics', {})
        vanilla_metrics = vanilla_sentence.get('metrics', {})

        # 检查是否都出现幻视 (CHAIRs > 0 表示有幻视)
        deco_has_hallucination = deco_metrics.get('CHAIRs', 0) > 0
        vanilla_has_hallucination = vanilla_metrics.get('CHAIRs', 0) > 0

        if deco_has_hallucination and vanilla_has_hallucination:
            # 获取图片文件名
            image_filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"

            # 简化句子数据
            simplified_deco = simplify_sentence_data(deco_sentence)
            simplified_vanilla = simplify_sentence_data(vanilla_sentence)

            # 判断谁的效果更好(幻视率更低)
            # CHAIRs 和 CHAIRi 都是越小越好
            # 优先使用 CHAIRi 作为主要判断标准, 如果 CHAIRi 相同则使用 CHAIRs
            deco_chairs = deco_metrics.get('CHAIRs', 0)
            deco_chairi = deco_metrics.get('CHAIRi', 0)
            vanilla_chairs = vanilla_metrics.get('CHAIRs', 0)
            vanilla_chairi = vanilla_metrics.get('CHAIRi', 0)

            if deco_chairi < vanilla_chairi:
                better_method = "deco"
            elif deco_chairi > vanilla_chairi:
                better_method = "vanilla"
            else:
                # CHAIRi 相同, 使用 CHAIRs 判断
                if deco_chairs < vanilla_chairs:
                    better_method = "deco"
                elif deco_chairs > vanilla_chairs:
                    better_method = "vanilla"
                else:
                    # 两者都相同, 默认标记为 equal (实际上应该不会出现这种情况)
                    better_method = "equal"

            case_info = {
                "image_id": image_filename,  # 使用完整的图片文件名
                "better_method": better_method,  # 说明谁的效果更好
                "vanilla_data": simplified_vanilla,
                "deco_data": simplified_deco,
                "difference": {
                    "CHAIRs": deco_chairs - vanilla_chairs,
                    "CHAIRi": deco_chairi - vanilla_chairi,
                    "Recall": deco_metrics.get('Recall', 0) - vanilla_metrics.get('Recall', 0),
                    "Len": deco_metrics.get('Len', 0) - vanilla_metrics.get('Len', 0)
                }
            }
            both_hallucinated_cases.append(case_info)

    # 保存结果
    result = {
        "summary": {
            "total_cases": len(common_image_ids),
            "both_hallucinated_cases": len(both_hallucinated_cases),
            "both_hallucinated_rate": len(both_hallucinated_cases) / len(common_image_ids) if len(common_image_ids) > 0 else 0
        },
        "both_hallucinated_cases": both_hallucinated_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def print_comparison_table(deco_results, vanilla_results):
    """
    打印 Deco vs Vanilla 的对比表格

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
    """
    deco_metrics = deco_results.get('overall_metrics', {}) if deco_results else {}
    vanilla_metrics = vanilla_results.get('overall_metrics', {}) if vanilla_results else {}

    print("\n" + "=" * 80)
    print("Deco vs Vanilla 对比")
    print("=" * 80)
    print(f"{'指标':<15} {'Vanilla':<12} {'Deco':<12} {'差异':<12} {'变化':<10}")
    print("-" * 80)

    metrics_list = [
        ('CHAIRs', 'CHAIRs'),
        ('CHAIRi', 'CHAIRi'),
        ('Recall', 'Recall'),
        ('Len', 'Len')
    ]

    for metric_name, metric_key in metrics_list:
        vanilla_val = vanilla_metrics.get(metric_key, 0)
        deco_val = deco_metrics.get(metric_key, 0)
        diff = deco_val - vanilla_val
        change = f"{diff:+.4f}" if diff != 0 else "0.0000"
        change_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="

        # 对于 CHAIRs 和 CHAIRi, 越小越好, 所以符号相反
        if metric_key in ['CHAIRs', 'CHAIRi']:
            change_symbol = "↓" if diff > 0 else "↑" if diff < 0 else "="

        print(f"{metric_name:<15} {vanilla_val:<12.4f} {deco_val:<12.4f} {diff:<12.4f} {change_symbol} {change}")

    print("=" * 80)


def save_summary_to_file(summary_file, args, output_file, chair_results_file=None,
                         chair_errors_file=None, results=None, model_name=None, error=None):
    """
    保存 CHAIR 评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        output_file: 输出描述文件路径
        chair_results_file: CHAIR 详细结果文件路径(可选)
        chair_errors_file: CHAIR 错误样本文件路径(可选)
        results: 评估结果字典(如果评估成功)
        model_name: 模型名称
        error: 错误信息(如果评估失败)
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHAIR 评估结果总结\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("【基本信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {args.model_path}\n")
        if model_name:
            f.write(f"模型名称: {model_name}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"COCO 根目录: {args.coco_root}\n")
        f.write(f"评测样本数: {args.num_samples if args.num_samples > 0 else '全部'}\n")
        f.write("\n")

        # Deco配置
        f.write("【Deco 配置】\n")
        f.write("-" * 80 + "\n")
        f.write(f"使用 Deco: {'是' if args.use_deco else '否'}\n")
        if args.use_deco:
            f.write(f"  - Alpha: {args.alpha}\n")
            f.write(f"  - Threshold Top-p: {args.threshold_top_p}\n")
            f.write(f"  - Threshold Top-k: {args.threshold_top_k}\n")
            f.write(f"  - Early Exit Layers: {args.start_layer}-{args.end_layer}\n")
        f.write("\n")

        # 生成参数
        f.write("【生成参数】\n")
        f.write("-" * 80 + "\n")
        f.write(f"Temperature: {args.temperature if args.temperature > 0 else 'None (greedy)'}\n")
        f.write(f"Top-p: {args.top_p if args.top_p else 'None'}\n")
        f.write(f"Max New Tokens: {args.max_new_tokens}\n")
        f.write(f"Num Beams: {args.num_beams}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write("\n")

        # 文件路径
        f.write("【文件路径】\n")
        f.write("-" * 80 + "\n")
        f.write(f"输出描述文件: {output_file}\n")
        if chair_results_file:
            f.write(f"CHAIR 详细结果文件: {chair_results_file}\n")
        if chair_errors_file:
            f.write(f"CHAIR 错误样本文件: {chair_errors_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if results is not None:
            if 'overall_metrics' in results:
                metrics = results['overall_metrics']
                f.write(f"CHAIRs (句子级别): {metrics.get('CHAIRs', 0):.4f}\n")
                f.write(f"CHAIRi (实例级别): {metrics.get('CHAIRi', 0):.4f}\n")
                f.write(f"Recall (召回率):   {metrics.get('Recall', 0):.4f}\n")
                f.write(f"Len (平均长度):    {metrics.get('Len', 0):.4f}\n")

            # 统计错误样本
            if 'sentences' in results:
                total_samples = len(results['sentences'])
                error_samples = [
                    s for s in results['sentences']
                    if s.get('metrics', {}).get('CHAIRs', 0) > 0
                ]
                error_count = len(error_samples)
                f.write(f"\n总样本数: {total_samples}\n")
                f.write(f"包含幻觉的样本数: {error_count}\n")
                if total_samples > 0:
                    f.write(f"幻觉样本比例: {error_count / total_samples * 100:.2f}%\n")
        elif error:
            f.write(f"评估失败: {error}\n")
        else:
            f.write("评估结果未生成\n")
        f.write("\n")

        # 分隔线
        f.write("=" * 80 + "\n")
        f.write("总结文件生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n")


def eval_model(args):
    """评估模型, 生成图像描述"""
    print("=" * 80)
    print("CHAIR 评估 - 生成图像描述")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"COCO 根目录: {args.coco_root}")
    print(f"输出文件: {args.output_file}")
    if args.use_deco:
        print(f"Deco 参数: use_deco={args.use_deco}, alpha={args.alpha}, layers={args.start_layer}-{args.end_layer}")
    else:
        print(f"使用原生 LLaVA 模型(Deco 已禁用)")
    if args.use_linear_probe if hasattr(args, 'use_linear_probe') else False:
        print(f"Linear Probe: 已启用, 模型目录={args.linear_probe_dir}")
    else:
        print(f"Linear Probe: 已禁用")
    print("=" * 80)

    # 加载模型
    print("\n[1/3] 正在加载模型...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    device = args.device if isinstance(args.device, str) else f"cuda:{args.device}" if args.device >= 0 else "cpu"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )
    print(f"✓ 模型加载完成: {model_name}")

    # 加载 linear probe（如果启用）
    linear_probe_manager = None
    if args.use_linear_probe:
        print(f"\n[1.5/3] 正在加载 Linear Probe 网络...")
        if not args.linear_probe_dir:
            raise ValueError("使用 --use-linear-probe 时必须指定 --linear-probe-dir")

        # 获取模型配置
        num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
        num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32
        hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
        head_dim = hidden_size // num_heads

        linear_probe_manager = LinearProbeManager(
            model_dir=args.linear_probe_dir,
            num_layers=num_layers,
            num_heads=num_heads,
            input_dim=head_dim,
            device=device
        )

        # 修改 attention 层的 forward 方法
        linear_probe_manager.patch_attention_layers(model)
        print(f"✓ Linear Probe 已启用，权重计算方式: weight = 1 + tanh(lambda)")

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 获取图像列表
    print(f"\n[2/3] 正在获取图像列表...")

    # 如果提供了图像 ID 列表文件, 读取它
    image_id_list = None
    if args.image_id_list_file:
        image_id_list_file = os.path.expanduser(args.image_id_list_file)
        if not os.path.exists(image_id_list_file):
            raise FileNotFoundError(f"图像 ID 列表文件不存在: {image_id_list_file}")

        # 检测文件格式: JSON 数组或文本文件(每行一个 ID)
        with open(image_id_list_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 尝试解析为 JSON(支持 JSON 数组格式, 如 ["COCO_val2014_000000001171.jpg", ...])
        if content.startswith('[') and content.endswith(']'):
            # JSON 数组格式
            image_names = json.loads(content)
            image_id_list = []
            for name in image_names:
                if isinstance(name, str):
                    # 从文件名提取 image_id
                    # 格式: "COCO_val2014_000000001171.jpg" 或 "COCO_val2014_000000001171"
                    if name.endswith('.jpg'):
                        name = name[:-4]  # 移除 .jpg 后缀
                    # 提取最后的数字部分
                    parts = name.split('_')
                    if len(parts) > 0:
                        try:
                            image_id = int(parts[-1])
                            image_id_list.append(image_id)
                        except ValueError:
                            print(f"⚠️  警告: 无法从文件名提取 image_id: {name}")
                elif isinstance(name, int):
                    # 直接是数字 ID
                    image_id_list.append(name)
                else:
                    print(f"⚠️  警告: 无法处理的数据类型: {type(name)}, 值: {name}")
            print(f"✓ 从 JSON 文件读取了 {len(image_id_list)} 个图像 ID")
        else:
            # 文本文件格式(每行一个 ID)
            with open(image_id_list_file, 'r', encoding='utf-8') as f:
                image_id_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 尝试解析为整数
                    try:
                        image_id = int(line)
                        image_id_list.append(image_id)
                    except ValueError:
                        # 如果不是数字, 尝试从文件名格式提取
                        if 'COCO_val2014_' in line:
                            if line.endswith('.jpg'):
                                line = line[:-4]
                            parts = line.split('_')
                            if len(parts) > 0:
                                try:
                                    image_id = int(parts[-1])
                                    image_id_list.append(image_id)
                                except ValueError:
                                    print(f"⚠️  警告: 无法从行提取 image_id: {line}")
                        else:
                            print(f"⚠️  警告: 无法解析行: {line}")
            print(f"✓ 从文本文件读取了 {len(image_id_list)} 个图像 ID")

    images = get_coco_val2014_images(
        coco_root=args.coco_root,
        image_id_list=image_id_list,
        max_images=args.num_samples if args.num_samples > 0 else 0
    )
    print(f"✓ 找到 {len(images)} 个图像")

    # 准备输出文件
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    output_f = open(output_file, "w", encoding="utf-8")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个图像
    print(f"\n[3/3] 开始生成描述...")
    # prompt = "Please describe this image in detail."
    prompt = "Please help me describe the image in detail."


    # 计算需要输出详细信息的样本索引(如果启用debug模式)
    debug_mode = getattr(args, 'debug', False)
    total_samples = len(images)
    debug_indices = set()

    if debug_mode:
        # Debug模式: 输出所有样本的详细信息(因为只有10个样本)
        if total_samples > 0:
            debug_indices = set(range(total_samples))
            print(f"Debug模式: 将输出所有 {len(debug_indices)} 个样本的详细信息")
    else:
        # 非Debug模式: 最多输出10个样本的详细信息(均匀分布)
        max_debug_samples = min(10, total_samples)
        if total_samples > 0:
            if total_samples <= max_debug_samples:
                debug_indices = set(range(total_samples))
            else:
                step = total_samples / max_debug_samples
                for i in range(max_debug_samples):
                    idx = int(i * step)
                    debug_indices.add(idx)
            if len(debug_indices) > 0:
                print(f"将输出 {len(debug_indices)} 个样本的详细信息用于调试(样本索引: {sorted(debug_indices)})")

    for sample_idx, image_info in enumerate(tqdm(images, desc="处理进度")):
        image_id = image_info["image_id"]
        image_file = image_info["image_path"]

        # 判断是否需要输出详细信息(debug模式或选中的样本)
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Image ID: {image_id}")
            print("=" * 80)
            print(f"图像: {image_file}")

        # 准备输入
        input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
            model, tokenizer, image_processor, image_file, prompt, conv_mode, device, verbose=verbose
        )

        # 生成回答
        outputs, output_token_len, input_token_len = generate_response(
            model, tokenizer, input_ids, image_tensor, stopping_criteria,
            args.temperature, args.top_p, args.max_new_tokens, device,
            use_deco=args.use_deco,
            alpha=args.alpha,
            threshold_top_p=args.threshold_top_p,
            threshold_top_k=args.threshold_top_k,
            early_exit_layers=early_exit_layers,
            num_beams=args.num_beams,
            verbose=verbose,
            use_linear_probe=args.use_linear_probe if hasattr(args, 'use_linear_probe') else False,
            linear_probe_manager=linear_probe_manager
        )

        # 移除停止字符串
        if outputs and outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 如果输出为空, 记录警告
        if not outputs:
            if verbose:
                print(f"\n  [Warning] 图像 {image_id} 生成结果为空, output_token_len={output_token_len}")
            else:
                print(f"  [Warning] 图像 {image_id} 生成结果为空, output_token_len={output_token_len}")

        if verbose:
            print(f"\n  [生成结果] 描述:")
            print(f"    - 输出长度: {len(outputs)} 字符")
            print(f"    - 描述预览: {outputs[:200]}...")
            print("=" * 80)

        # 保存结果(CHAIR 格式: image_id 和 caption)
        result = {
            "image_id": image_id,
            "caption": outputs
        }
        output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_f.flush()

    output_f.close()
    print(f"\n✓ 描述生成完成！结果已保存到: {output_file}")

    # 如果使用了 linear probe，恢复原始的 forward 方法
    if linear_probe_manager is not None and linear_probe_manager.is_patched:
        linear_probe_manager.unpatch_attention_layers(model)

    # 自动计算 CHAIR 指标(默认启用)
    auto_evaluate = getattr(args, 'auto_evaluate', True)  # 默认为 True
    if auto_evaluate:
        # 构建 annotations 路径
        coco_annotations_path = os.path.join(args.coco_root, "annotations_trainval2014", "annotations")
        if not os.path.exists(coco_annotations_path):
            # 尝试使用 project 中的路径
            if hasattr(project, 'coco_annotations_path'):
                coco_annotations_path = project.coco_annotations_path
            else:
                raise FileNotFoundError(f"找不到 COCO annotations 目录: {coco_annotations_path}")

        print("\n" + "=" * 80)
        print("自动计算 CHAIR 指标...")
        print("=" * 80)

        # 生成结果文件路径(参考run_pope_eval.py的路径结构)
        results_dir = os.path.dirname(output_file)
        # 保存详细结果(包含所有中间信息)
        chair_results_file = output_file.replace('.jsonl', '_chair_results.json')
        # 保存错误样本(如果有)
        chair_errors_file = output_file.replace('.jsonl', '_chair_errors.json')
        # 生成总结文件路径
        summary_file = output_file.replace('.jsonl', '_summary.txt')

        # 计算需要输出详细信息的样本索引(如果启用debug模式)
        debug_indices = None
        if getattr(args, 'debug', False):
            # 读取生成的描述文件, 确定样本数量
            with open(output_file, 'r', encoding='utf-8') as f:
                total_samples = sum(1 for _ in f)

            if total_samples > 0:
                # 如果样本数少于等于10个, 全部输出详细信息
                if total_samples <= 10:
                    debug_indices = set(range(total_samples))
                else:
                    # 均匀分布选择样本(最多10个)
                    max_debug_samples = min(10, total_samples)
                    step = total_samples / max_debug_samples
                    debug_indices = set()
                    for i in range(max_debug_samples):
                        idx = int(i * step)
                        debug_indices.add(idx)

                print(f"Debug模式: 将输出 {len(debug_indices)} 个样本的详细信息(样本索引: {sorted(debug_indices)})")

        # 调用 evaluate_chair 函数
        results = evaluate_chair(
            cap_file=output_file,
            coco_path=coco_annotations_path,
            image_id_key="image_id",
            caption_key="caption",
            cache_file=os.path.join(results_dir, "chair_evaluator.pkl"),
            use_cache=True,
            save_path=chair_results_file,
            verbose=True,
            debug=getattr(args, 'debug', False),
            debug_indices=debug_indices
        )

        print("\n" + "=" * 80)
        print("✓ CHAIR 指标计算完成！")
        print("=" * 80)
        print(f"详细结果文件: {chair_results_file}")
        print(f"输出文件: {output_file}")

        # 保存错误样本(包含幻觉的样本)
        error_count = 0
        if results and 'sentences' in results:
            error_samples = [
                s for s in results['sentences']
                if s.get('metrics', {}).get('CHAIRs', 0) > 0
            ]
            error_count = len(error_samples)
            if error_samples:
                with open(chair_errors_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'error_count': len(error_samples),
                        'total_samples': len(results['sentences']),
                        'error_samples': error_samples
                    }, f, indent=2, ensure_ascii=False)
                print(f"错误样本文件: {chair_errors_file} ({len(error_samples)} 个包含幻觉的样本)")

        # 保存总结到txt文件
        model_name = get_model_name_from_path(args.model_path)
        save_summary_to_file(
            summary_file=summary_file,
            args=args,
            output_file=output_file,
            chair_results_file=chair_results_file,
            chair_errors_file=chair_errors_file if error_count > 0 else None,
            results=results,
            model_name=model_name
        )
        print(f"\n✓ 结果总结已保存到: {summary_file}")
    else:
        print(f"\n下一步: 使用 chair.py 计算 CHAIR 指标")
        print(f"  python chair.py --cap_file {output_file} --image_id_key image_id --caption_key caption \\")
        print(f"                  --coco_path {args.coco_root}/annotations_trainval2014/annotations/")


def main():
    """主函数 - 自动检测并使用默认配置"""
    # 项目根目录
    project_root = Path(__file__).parent

    # 自动检测可用 GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("⚠ 未检测到 CUDA, 将使用 CPU(速度较慢)")

    # 默认配置
    default_config = {
        "model_path": project.llava_v15_7b_path,
        "device": device,
        "coco_root": project.coco_data_path,  # 需要根据实际情况修改
        "image_id_list_file": "pope_coco/coco_baseline_500.json",
        "use_linear_probe": True,
        "linear_probe_dir": "train/ckpt/coco_train_200_generate_spp_gt_pair_np_log",
        "use_deco": False,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": -1,
        "top_p": None,
        "max_new_tokens": 512,  # CHAIR 需要详细描述
        "num_beams": 10,
        "num_samples": 0,  # 0 表示处理所有图像
        "seed": 42
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="CHAIR 评估 - 生成图像描述(所有参数可选)")

    # 数据集参数
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径(包含 val2014 子目录)")
    parser.add_argument("--image_id_list_file", type=str, default=default_config["image_id_list_file"],
                       help="图像 ID 列表文件, 支持两种格式: 1) JSON 数组格式(如 [\"COCO_val2014_000000001171.jpg\", ...]);2) 文本文件(每行一个 image_id 或图像文件名)。如果提供则只处理这些图像")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="处理图像数量(0表示处理所有图像, 非零表示只处理前N个)")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出描述文件路径(JSONL 格式, 如果不指定, 将自动生成)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度(-1表示贪婪生成)")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--num-beams", type=int, default=default_config["num_beams"],
                       help="Beam search 的 beam 数量")

    # Deco 参数(默认不使用 Deco, 只使用原生 LLaVA 模型)
    parser.add_argument("--use-deco", action="store_true", default=default_config["use_deco"],
                       help="启用 Deco 早退机制(默认: False, 使用原生 LLaVA 模型)")
    parser.add_argument("--alpha", type=float, default=default_config["alpha"],
                       help="Deco 置信度阈值参数")
    parser.add_argument("--threshold-top-p", type=float, default=default_config["threshold_top_p"],
                       help="早退判断的 top-p 阈值")
    parser.add_argument("--threshold-top-k", type=int, default=default_config["threshold_top_k"],
                       help="早退判断的 top-k 阈值")
    parser.add_argument("--start-layer", type=int, default=default_config["start_layer"],
                       help="允许早退的起始层索引")
    parser.add_argument("--end-layer", type=int, default=default_config["end_layer"],
                       help="允许早退的结束层索引")

    # Linear Probe 参数
    parser.add_argument("--use-linear-probe", default=default_config["use_linear_probe"],
                       help="启用 Linear Probe 网络进行加权(默认: False)")
    parser.add_argument("--linear-probe-dir", type=str, default=default_config["linear_probe_dir"],
                       help="Linear Probe 模型保存目录(例如: train/ckpt/)")

    # 其他参数
    parser.add_argument("--seed", type=int, default=default_config["seed"], help="随机种子")
    parser.add_argument("--no-auto-evaluate", action="store_true", default=False,
                       help="禁用自动计算 CHAIR 指标(默认会自动计算)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="启用debug模式, 输出每个样本的详细处理过程")

    args = parser.parse_args()
    set_seed(args.seed)

    # 设置 auto_evaluate 参数(默认启用, 除非指定 --no-auto-evaluate)
    args.auto_evaluate = not args.no_auto_evaluate

    # 自动生成输出文件路径(如果未指定)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", "chair")
    os.makedirs(output_dir, exist_ok=True)

    # 如果使用 Linear Probe, 需要同时运行 vanilla 版本进行对比
    vanilla_output_file = None
    vanilla_results = None

    if args.use_linear_probe:
        print("\n" + "=" * 80)
        print("检测到使用 Linear Probe, 将同时运行 Vanilla 版本进行对比")
        print("=" * 80)

        # 先运行 Linear Probe 版本
        print("\n" + "-" * 80)
        print("[1/2] 先运行 Linear Probe 版本")
        print("-" * 80)
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"chair_captions_linear_probe_{timestamp}.jsonl")

        eval_model(args)

        # 再运行 Vanilla 版本（不使用 linear probe）
        print("\n" + "-" * 80)
        print("[2/2] 再运行 Vanilla 版本（不使用 Linear Probe）")
        print("-" * 80)
        vanilla_args = argparse.Namespace(**vars(args))
        vanilla_args.use_linear_probe = False

        vanilla_args.output_file = os.path.join(output_dir, f"chair_captions_vanilla_{timestamp}.jsonl")
        vanilla_args.auto_evaluate = args.auto_evaluate

        eval_model(vanilla_args)
        vanilla_output_file = vanilla_args.output_file

        # 如果两个版本都完成了评估, 进行对比
        if vanilla_args.auto_evaluate and args.auto_evaluate:
            print("\n" + "=" * 80)
            print("对比 Linear Probe vs Vanilla")
            print("=" * 80)

            # 加载两个版本的结果
            vanilla_results_file = vanilla_output_file.replace('.jsonl', '_chair_results.json')
            linear_probe_results_file = args.output_file.replace('.jsonl', '_chair_results.json')

            if os.path.exists(vanilla_results_file) and os.path.exists(linear_probe_results_file):
                with open(vanilla_results_file, 'r', encoding='utf-8') as f:
                    vanilla_results = json.load(f)
                with open(linear_probe_results_file, 'r', encoding='utf-8') as f:
                    linear_probe_results = json.load(f)

                # 生成对比 JSON 文件(CHAIRs/CHAIRi 不一致的 case)
                comparison_file = args.output_file.replace('.jsonl', '_comparison.json')
                comparison_result = compare_deco_vs_vanilla(
                    deco_results=linear_probe_results,
                    vanilla_results=vanilla_results,
                    deco_captions_file=args.output_file,
                    vanilla_captions_file=vanilla_output_file,
                    output_file=comparison_file
                )

                # 保存两个方法都出现幻视的 error 例子
                both_hallucinated_file = args.output_file.replace('.jsonl', '_both_hallucinated_errors.json')
                both_hallucinated_result = save_both_hallucinated_errors(
                    deco_results=linear_probe_results,
                    vanilla_results=vanilla_results,
                    deco_captions_file=args.output_file,
                    vanilla_captions_file=vanilla_output_file,
                    output_file=both_hallucinated_file
                )

                # 打印对比表格
                print("\n" + "=" * 80)
                print("Linear Probe vs Vanilla 对比")
                print("=" * 80)
                print_comparison_table(deco_results=linear_probe_results, vanilla_results=vanilla_results)

                print(f"\n✓ 对比结果已保存到: {comparison_file}")
                print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
                print(f"  - CHAIRs/CHAIRi 不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
                print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")

                print(f"\n✓ 两个方法都出现幻视的错误例子已保存到: {both_hallucinated_file}")
                print(f"  - 总样本数: {both_hallucinated_result['summary']['total_cases']}")
                print(f"  - 都出现幻视的样本数: {both_hallucinated_result['summary']['both_hallucinated_cases']}")
                print(f"  - 都出现幻视的比例: {both_hallucinated_result['summary']['both_hallucinated_rate']:.2%}")
            else:
                print("⚠️  无法找到评估结果文件, 跳过对比")
    elif args.use_deco:
        print("\n" + "=" * 80)
        print("检测到使用 Deco, 将同时运行 Vanilla 版本进行对比")
        print("=" * 80)

        # 先运行 Vanilla 版本
        print("\n" + "-" * 80)
        print("[1/2] 运行 Vanilla 版本")
        print("-" * 80)
        vanilla_args = argparse.Namespace(**vars(args))
        vanilla_args.use_deco = False
        vanilla_args.output_file = os.path.join(output_dir, f"chair_captions_vanilla_{timestamp}.jsonl")
        vanilla_args.auto_evaluate = args.auto_evaluate  # 保持相同的 auto_evaluate 设置

        eval_model(vanilla_args)
        vanilla_output_file = vanilla_args.output_file

        # 然后运行 Deco 版本
        print("\n" + "-" * 80)
        print("[2/2] 运行 Deco 版本")
        print("-" * 80)
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"chair_captions_deco_{timestamp}.jsonl")

        eval_model(args)

        # 如果两个版本都完成了评估, 进行对比
        if vanilla_args.auto_evaluate and args.auto_evaluate:
            print("\n" + "=" * 80)
            print("对比 Deco vs Vanilla")
            print("=" * 80)

            # 加载两个版本的结果
            vanilla_results_file = vanilla_output_file.replace('.jsonl', '_chair_results.json')
            deco_results_file = args.output_file.replace('.jsonl', '_chair_results.json')

            if os.path.exists(vanilla_results_file) and os.path.exists(deco_results_file):
                with open(vanilla_results_file, 'r', encoding='utf-8') as f:
                    vanilla_results = json.load(f)
                with open(deco_results_file, 'r', encoding='utf-8') as f:
                    deco_results = json.load(f)

                # 生成对比 JSON 文件(CHAIRs/CHAIRi 不一致的 case)
                comparison_file = args.output_file.replace('.jsonl', '_comparison.json')
                comparison_result = compare_deco_vs_vanilla(
                    deco_results=deco_results,
                    vanilla_results=vanilla_results,
                    deco_captions_file=args.output_file,
                    vanilla_captions_file=vanilla_output_file,
                    output_file=comparison_file
                )

                # 保存两个方法都出现幻视的 error 例子
                both_hallucinated_file = args.output_file.replace('.jsonl', '_both_hallucinated_errors.json')
                both_hallucinated_result = save_both_hallucinated_errors(
                    deco_results=deco_results,
                    vanilla_results=vanilla_results,
                    deco_captions_file=args.output_file,
                    vanilla_captions_file=vanilla_output_file,
                    output_file=both_hallucinated_file
                )

                # 打印对比表格
                print_comparison_table(deco_results=deco_results, vanilla_results=vanilla_results)

                print(f"\n✓ 对比结果已保存到: {comparison_file}")
                print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
                print(f"  - CHAIRs/CHAIRi 不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
                print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")

                print(f"\n✓ 两个方法都出现幻视的错误例子已保存到: {both_hallucinated_file}")
                print(f"  - 总样本数: {both_hallucinated_result['summary']['total_cases']}")
                print(f"  - 都出现幻视的样本数: {both_hallucinated_result['summary']['both_hallucinated_cases']}")
                print(f"  - 都出现幻视的比例: {both_hallucinated_result['summary']['both_hallucinated_rate']:.2%}")
            else:
                print("⚠️  无法找到评估结果文件, 跳过对比")
    else:
        # 不使用 Deco, 正常处理
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"chair_captions_vanilla_{timestamp}.jsonl")

        # 运行评估
        eval_model(args)

    print("\n" + "=" * 80)
    print("✓ 所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
