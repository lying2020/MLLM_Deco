#!/usr/bin/env python3
"""
POPE 评估脚本 - 使用 Linear Probe 进行 POPE 基准测试
支持使用或不使用 linear probe 网络进行对比评估

POPE 评估需要:
1. POPE 测试文件 (JSONL 格式): {"question_id": int, "image": str, "text": str, "label": "yes"/"no"}
2. COCO 2014 val2014 图像目录
3. 生成的答案文件(JSONL 格式): {"question_id": int, "text": str, ...}

使用步骤:
1. 运行此脚本生成答案文件:
   python train/linear_probe_pope_eval.py --pope-file pope_coco/coco_pope_random.json --use-linear-probe --linear-probe-dir train/ckpt

2. 脚本会自动计算 POPE 指标 (accuracy, precision, recall, F1)
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
from eval_tool.eval_pope import evaluate_pope
from train.linear_probe_trainer import LinearProbeTrainer, LinearProbe
import re


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
            float: 处理后的 lambda 值
        """
        # 只对深层（layer_idx >= 16）使用 linear probe，前16层直接返回 0.0
        if layer_idx < 33:
            return 0.0

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
            output = probe(head_vector)
            if output.size(0) == 1:
                lambda_value = output.cpu().item()
            else:
                lambda_value = output.cpu().item() if output.numel() == 1 else output.cpu()

            # 对 lambda 进行转换处理
            # 1. 如果 lambda 在 [-0.3, 0.5] 之间，直接置为 0
            if -0.3 <= lambda_value <= 0.5:
                lambda_value = 0.0
            # 2. 如果 lambda > 0.5，则将 lambda 置为 2*lambda - 1
            elif lambda_value > 0.5:
                lambda_value = 2.0 * lambda_value - 1.0
            # 3. 如果 lambda < -0.3，则将 lambda 置为 (1.0/0.7)*lambda + 0.43
            elif lambda_value < -0.3:
                lambda_value = (1.0 / 0.7) * lambda_value + 0.43

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
            # 使用修改后的源码，通过 head_weights 参数应用权重
            def make_patched_forward(layer_idx, original_forward):
                def patched_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                ):
                    # 获取attention层参数
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    num_heads = self.num_heads
                    head_dim = hidden_size // num_heads

                    # 为了获取最后一个 token 的 head 向量来计算权重，我们需要先计算到 attn_output
                    # 但为了使用原始实现的正确性，我们使用一个临时调用
                    # 方案：先调用原始 forward 获取中间结果（通过 hook），或者手动计算到 attn_output
                    # 为了保持代码简洁和正确性，我们手动计算到 attn_output，然后使用 head_weights 参数

                    # 计算Q, K, V
                    Q = attn_module.q_proj(hidden_states)
                    K = attn_module.k_proj(hidden_states)
                    V = attn_module.v_proj(hidden_states)

                    # 重塑为多头格式
                    num_key_value_heads = getattr(attn_module, 'num_key_value_heads', num_heads)
                    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    K = K.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
                    V = V.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

                    # 计算 kv_seq_len
                    kv_seq_len = K.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]

                    # 应用 RoPE
                    cos, sin = attn_module.rotary_emb(V, seq_len=kv_seq_len)
                    if position_ids is None:
                        position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

                    # 处理past_key_value
                    if past_key_value is not None:
                        K = torch.cat([past_key_value[0], K], dim=2)
                        V = torch.cat([past_key_value[1], V], dim=2)

                    past_key_value_for_return = (K, V) if use_cache else None

                    # repeat k/v heads if n_kv_heads < n_heads (GQA)
                    from transformers.models.llama.modeling_llama import repeat_kv
                    num_key_value_groups = getattr(attn_module, 'num_key_value_groups', 1)
                    K = repeat_kv(K, num_key_value_groups)
                    V = repeat_kv(V, num_key_value_groups)

                    kv_seq_len = K.shape[-2]

                    # 计算attention scores
                    import math
                    attn_weights = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(head_dim)

                    if attention_mask is not None:
                        if attention_mask.size() != (batch_size, 1, seq_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(batch_size, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask

                    # 应用softmax
                    import torch.nn.functional as F
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

                    # 应用attention到V
                    attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

                    # 对每个head，使用最后一个token的head向量预测权重
                    # head_weights 形状: [num_heads]，默认所有权重为 1.0
                    head_weights = torch.ones(num_heads, device=hidden_states.device, dtype=hidden_states.dtype)

                    last_token_idx = seq_len - 1
                    for head_idx in range(num_heads):
                        # 获取最后一个token的head向量（用于预测权重）
                        head_vector = attn_output[:, head_idx, last_token_idx, :]  # [batch, head_dim]

                        # 对于 batch 中的每个样本，计算权重（取平均值或使用第一个样本）
                        # 为了简化，我们使用第一个样本的 head_vector
                        lambda_val = self.get_weight(layer_idx, head_idx, head_vector[0])
                        # lambda 已经经过转换处理
                        # 将 head 的原始系数（值为 1）与 lambda 系数相减：weight = 1 - lambda
                        weight = 1.0 - lambda_val
                        head_weights[head_idx] = weight

                    # 应用权重到 attn_output（在 reshape 之前）
                    head_weights = head_weights.view(1, num_heads, 1, 1)  # [1, num_heads, 1, 1] 用于广播
                    attn_output = attn_output * head_weights

                    # 现在调用原始 forward，但传递 head_weights=None（因为我们已经应用了权重）
                    # 或者，我们直接 reshape 和 o_proj
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
                    output = attn_module.o_proj(attn_output)

                    if output_attentions:
                        return output, attn_weights, past_key_value_for_return
                    else:
                        return output, None, past_key_value_for_return

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


def load_pope_questions(pope_file: str, coco_root: str) -> List[Dict]:
    """
    从 POPE JSONL 文件加载测试用例

    Args:
        pope_file: POPE 测试文件路径 (JSONL 格式)
        coco_root: COCO 数据集根目录

    Returns:
        List[Dict]: 测试用例列表，每个包含 question_id, image, text, label
    """
    pope_file = Path(pope_file)
    coco_root = Path(coco_root)

    if not pope_file.exists():
        raise FileNotFoundError(f"POPE 文件不存在: {pope_file}")

    questions = []
    with open(pope_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                # 处理图像路径
                image_path = case.get("image", "")
                if image_path.startswith("val2014/"):
                    # 相对路径，需要拼接 coco_root
                    image_path = coco_root / image_path
                elif not os.path.isabs(image_path):
                    # 相对路径，尝试拼接
                    image_path = coco_root / "val2014" / os.path.basename(image_path)
                else:
                    # 绝对路径，直接使用
                    image_path = Path(image_path)

                # 确保图像文件存在
                if not image_path.exists():
                    print(f"⚠️  警告: 图像文件不存在: {image_path}")
                    continue

                case["image_path"] = str(image_path)
                questions.append(case)
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 无法解析 JSON 行: {line[:100]}... 错误: {e}")
                continue

    print(f"✓ 从 {pope_file} 加载了 {len(questions)} 个测试用例")
    return questions


def recorder(out):
    """将输出转换为 Yes/No"""
    if not out or not out.strip():
        return "No"

    out_lower = out.lower().strip()
    word_list = re.split(r'[^\w]+', out_lower)

    # 检查是否包含 "yes"
    if "yes" in word_list:
        return "Yes"
    # 检查是否包含 "no"
    elif "no" in word_list:
        return "No"
    else:
        # 如果既没有 "yes" 也没有 "no"，默认返回 "No"
        return "No"


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

    # 准备文本输入（POPE 风格）
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    # 对于 Yes/No 问题，添加明确的输出格式说明
    qs = qs + " Please answer with Yes or No only."

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


def save_summary_to_file(summary_file, args, output_file, pope_results=None, model_name=None, error=None):
    """
    保存 POPE 评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        output_file: 输出答案文件路径
        pope_results: POPE 评估结果字典(如果评估成功)
        model_name: 模型名称
        error: 错误信息(如果评估失败)
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("POPE 评估结果总结\n")
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
        f.write(f"POPE 测试文件: {args.pope_file}\n")
        f.write(f"评测样本数: {args.num_samples if args.num_samples > 0 else '全部'}\n")
        f.write("\n")

        # Linear Probe 配置
        f.write("【Linear Probe 配置】\n")
        f.write("-" * 80 + "\n")
        f.write(f"使用 Linear Probe: {'是' if args.use_linear_probe else '否'}\n")
        if args.use_linear_probe:
            f.write(f"  - Linear Probe 目录: {args.linear_probe_dir}\n")
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
        f.write(f"输出答案文件: {output_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if pope_results is not None:
            if 'metrics' in pope_results:
                metrics = pope_results['metrics']
                f.write(f"Accuracy (准确率):    {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Precision (精确率):   {metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall (召回率):      {metrics.get('recall', 0):.4f}\n")
                f.write(f"F1 Score:             {metrics.get('f1', 0):.4f}\n")
                f.write(f"Yes Proportion:       {metrics.get('yes_proportion', 0):.4f}\n")

            # 统计信息
            if 'statistics' in pope_results:
                stats = pope_results['statistics']
                f.write(f"\n详细统计:\n")
                f.write(f"  总问题数: {stats.get('total_questions', 0)}\n")
                f.write(f"  True Positives (TP):  {stats.get('true_positives', 0)}\n")
                f.write(f"  True Negatives (TN):  {stats.get('true_negatives', 0)}\n")
                f.write(f"  False Positives (FP): {stats.get('false_positives', 0)}\n")
                f.write(f"  False Negatives (FN): {stats.get('false_negatives', 0)}\n")
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
    """评估模型, 生成 POPE 答案"""
    print("=" * 80)
    print("POPE 评估 - 生成 Yes/No 答案")
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
        print(f"✓ Linear Probe 已启用，权重计算方式: weight = 1 - lambda")
        print(f"  Lambda 转换规则:")
        print(f"    - 如果 lambda 在 [-0.3, 0.5] 之间: lambda = 0")
        print(f"    - 如果 lambda > 0.5: lambda = 2*lambda - 1")
        print(f"    - 如果 lambda < -0.3: lambda = (1.0/0.7)*lambda + 0.43")

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 加载 POPE 测试用例
    print(f"\n[2/3] 正在加载 POPE 测试用例...")
    if not args.pope_file:
        raise ValueError("必须指定 --pope-file 参数（例如: pope_coco/coco_pope_random.json）")

    questions = load_pope_questions(args.pope_file, args.coco_root)
    if len(questions) == 0:
        raise ValueError(f"从 {args.pope_file} 加载的测试用例为空")

    # 限制问题数量（如果指定）
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"✓ 限制为前 {len(questions)} 个测试用例")

    # 准备输出文件
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    output_f = open(output_file, "w", encoding="utf-8")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个测试用例
    print(f"\n[3/3] 开始生成答案...")


    # 计算需要输出详细信息的样本索引(如果启用debug模式)
    debug_mode = getattr(args, 'debug', False)
    total_samples = len(questions)
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

    for sample_idx, question in enumerate(tqdm(questions, desc="处理进度")):
        question_id = question["question_id"]
        image_file = question["image_path"]
        prompt_text = question["text"]
        gt_label = question.get("label", "")

        # 判断是否需要输出详细信息(debug模式或选中的样本)
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Question ID: {question_id}")
            print("=" * 80)
            print(f"图像: {image_file}")
            print(f"问题: {prompt_text}")
            print(f"真值标签: {gt_label}")

        # 准备输入
        input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
            model, tokenizer, image_processor, image_file, prompt_text, conv_mode, device, verbose=verbose
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

        # 转换为 Yes/No 格式
        answer = recorder(outputs)

        # 如果输出为空, 记录警告
        if not outputs:
            if verbose:
                print(f"\n  [Warning] Question {question_id} 生成结果为空, output_token_len={output_token_len}")
            else:
                print(f"  [Warning] Question {question_id} 生成结果为空, output_token_len={output_token_len}")

        if verbose:
            print(f"\n  [生成结果] 答案:")
            print(f"    - 原始输出: {outputs}")
            print(f"    - 转换后答案: {answer}")
            print(f"    - 真值标签: {gt_label}")
            print(f"    - 是否正确: {answer.lower() == gt_label.lower()}")
            print("=" * 80)

        # 保存结果(POPE 格式)
        result = {
            "question_id": question_id,
            "text": answer,
            "prompt": prompt_text,
            "image": question.get("image", ""),
            "model_id": get_model_name_from_path(args.model_path),
            "metadata": {
                "raw_output": outputs,
                "gt_label": gt_label
            }
        }
        output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_f.flush()

    output_f.close()
    print(f"\n✓ 答案生成完成！结果已保存到: {output_file}")

    # 如果使用了 linear probe，恢复原始的 forward 方法
    if linear_probe_manager is not None and linear_probe_manager.is_patched:
        linear_probe_manager.unpatch_attention_layers(model)

    # 自动计算 POPE 指标(默认启用)
    auto_evaluate = getattr(args, 'auto_evaluate', True)  # 默认为 True
    if auto_evaluate:
        print("\n" + "=" * 80)
        print("自动计算 POPE 指标...")
        print("=" * 80)

        # 生成结果文件路径
        results_dir = os.path.dirname(output_file)
        # 保存错误样本(如果有)
        pope_errors_file = output_file.replace('.jsonl', '_pope_errors.json')
        # 生成总结文件路径
        summary_file = output_file.replace('.jsonl', '_summary.txt')

        # 调用 evaluate_pope 函数
        results = evaluate_pope(
            gt_files_path=args.pope_file,
            gen_files_path=output_file,
            output_errors_path=pope_errors_file,
            verbose=True
        )

        print("\n" + "=" * 80)
        print("✓ POPE 指标计算完成！")
        print("=" * 80)
        print(f"输出文件: {output_file}")
        if os.path.exists(pope_errors_file):
            print(f"错误样本文件: {pope_errors_file}")

        # 保存总结到txt文件
        model_name = get_model_name_from_path(args.model_path)
        save_summary_to_file(
            summary_file=summary_file,
            args=args,
            output_file=output_file,
            pope_results=results,
            model_name=model_name
        )
        print(f"\n✓ 结果总结已保存到: {summary_file}")
    else:
        print(f"\n下一步: 使用 eval_pope.py 计算 POPE 指标")
        print(f"  python eval_tool/eval_pope.py --gt_files {args.pope_file} --gen_files {output_file}")


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
        "coco_root": project.coco_data_path,
        "pope_file": "pope_coco/coco_pope_random.json",
        "use_linear_probe": True,
        "linear_probe_dir": "train/ckpt/coco_train_20_generate_spp_gt_pair",
        "use_deco": False,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": -1,
        "top_p": None,
        "max_new_tokens": 10,  # POPE 只需要 Yes/No，不需要太多 tokens
        "num_beams": 1,
        "num_samples": 200,  # 0 表示处理所有问题
        "seed": 42
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="POPE 评估 - 生成 Yes/No 答案(所有参数可选)")

    # 数据集参数
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径(包含 val2014 子目录)")
    parser.add_argument("--pope-file", type=str, default=default_config["pope_file"],
                       help="POPE 测试文件路径 (JSONL 格式, 例如: pope_coco/coco_pope_random.json)")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="处理问题数量(0表示处理所有问题, 非零表示只处理前N个)")

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
                       help="最大生成 token 数 (POPE 通常只需要 1-2 个 token)")
    parser.add_argument("--num-beams", type=int, default=default_config["num_beams"],
                       help="Beam search 的 beam 数量 (POPE 通常使用 1)")

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
                       help="禁用自动计算 POPE 指标(默认会自动计算)")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="启用debug模式, 输出每个样本的详细处理过程")

    args = parser.parse_args()
    set_seed(args.seed)

    # 设置 auto_evaluate 参数(默认启用, 除非指定 --no-auto-evaluate)
    args.auto_evaluate = not args.no_auto_evaluate

    # 自动生成输出文件路径(如果未指定)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", "pope")
    os.makedirs(output_dir, exist_ok=True)

    # 如果使用 Linear Probe, 需要同时运行 vanilla 版本进行对比
    vanilla_output_file = None
    vanilla_results = None

    if args.use_linear_probe:
        print("\n" + "=" * 80)
        print("检测到使用 Linear Probe, 将同时运行 Vanilla 版本进行对比")
        print("=" * 80)

        # 从 pope_file 提取 split 名称（例如: coco_pope_random.json -> random）
        pope_basename = os.path.basename(args.pope_file)
        if 'random' in pope_basename:
            split_name = 'random'
        elif 'popular' in pope_basename:
            split_name = 'popular'
        elif 'adversarial' in pope_basename:
            split_name = 'adversarial'
        else:
            split_name = 'unknown'

        # 先运行 Linear Probe 版本
        print("\n" + "-" * 80)
        print("[1/2] 先运行 Linear Probe 版本")
        print("-" * 80)
        if args.output_file is None:
            args.output_file = os.path.join(output_dir, f"pope_{split_name}_linear_probe_{timestamp}.jsonl")

        eval_model(args)

        # 再运行 Vanilla 版本（不使用 linear probe）
        print("\n" + "-" * 80)
        print("[2/2] 再运行 Vanilla 版本（不使用 Linear Probe）")
        print("-" * 80)
        vanilla_args = argparse.Namespace(**vars(args))
        vanilla_args.use_linear_probe = False

        vanilla_args.output_file = os.path.join(output_dir, f"pope_{split_name}_vanilla_{timestamp}.jsonl")
        vanilla_args.auto_evaluate = args.auto_evaluate

        eval_model(vanilla_args)
        vanilla_output_file = vanilla_args.output_file

        # 如果两个版本都完成了评估, 进行对比
        if vanilla_args.auto_evaluate and args.auto_evaluate:
            print("\n" + "=" * 80)
            print("对比 Linear Probe vs Vanilla")
            print("=" * 80)

            # 加载两个版本的结果
            vanilla_results_file = vanilla_output_file.replace('.jsonl', '_summary.txt')
            linear_probe_results_file = args.output_file.replace('.jsonl', '_summary.txt')

            # 重新评估以获取详细结果
            vanilla_results = evaluate_pope(
                gt_files_path=args.pope_file,
                gen_files_path=vanilla_output_file,
                output_errors_path=None,
                verbose=False
            )

            linear_probe_results = evaluate_pope(
                gt_files_path=args.pope_file,
                gen_files_path=args.output_file,
                output_errors_path=None,
                verbose=False
            )

            # 打印对比表格
            print("\n" + "=" * 80)
            print("Linear Probe vs Vanilla 对比")
            print("=" * 80)
            print(f"{'指标':<20} {'Vanilla':<12} {'Linear Probe':<12} {'差异':<12} {'变化':<10}")
            print("-" * 80)

            metrics_list = ['accuracy', 'precision', 'recall', 'f1']
            for metric_name in metrics_list:
                vanilla_val = vanilla_results.get('metrics', {}).get(metric_name, 0)
                lp_val = linear_probe_results.get('metrics', {}).get(metric_name, 0)
                diff = lp_val - vanilla_val
                change_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"{metric_name.capitalize():<20} {vanilla_val:<12.4f} {lp_val:<12.4f} {diff:<12.4f} {change_symbol}")

            print("=" * 80)
    else:
        # 不使用 Linear Probe, 正常处理
        if args.output_file is None:
            # 从 pope_file 提取 split 名称
            pope_basename = os.path.basename(args.pope_file)
            if 'random' in pope_basename:
                split_name = 'random'
            elif 'popular' in pope_basename:
                split_name = 'popular'
            elif 'adversarial' in pope_basename:
                split_name = 'adversarial'
            else:
                split_name = 'unknown'
            args.output_file = os.path.join(output_dir, f"pope_{split_name}_vanilla_{timestamp}.jsonl")

        # 运行评估
        eval_model(args)

    print("\n" + "=" * 80)
    print("✓ 所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
