#!/usr/bin/env python3
"""
生成 head 级别的真值对（ground truth pairs）
用于训练 1024 个 linear probe（32层 x 32个head）

根据截图中的公式：
1. 计算每个head的概率增益（使用指数函数，避免log计算时的数值不稳定问题）
2. 计算语义先验偏置分数（公式C3）
3. 计算SPP输出g（公式D1，alpha=1.0, beta=0.0）
"""

import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import sys
import warnings

warnings.filterwarnings('ignore')

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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

current_dir = os.path.dirname(os.path.abspath(__file__))
coco_train_json_dir = os.path.join(current_dir, "coco_train_json")
os.makedirs(coco_train_json_dir, exist_ok=True)


# SPP 参数
ALPHA = 1.0  # 温度/尺度参数
BETA = 0.0   # 偏置参数
TOP_K = 40   # 用于构建候选池的top-K值

EXP_GAIN_COEFF=2.0
USE_NP_EXP = True  # 默认使用 np.exp，如果为 False 则使用 np.log

def tanh(x):
    """Tanh函数"""
    return np.tanh(x)


def load_image(image_file):
    """加载图像文件"""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"图像文件不存在: {image_file}")
        image = Image.open(image_file).convert("RGB")
    return image


def get_token_ids_for_text(tokenizer, text: str) -> Set[int]:
    """
    获取文本对应的token ID集合

    Args:
        tokenizer: tokenizer对象
        text: 文本字符串

    Returns:
        Set[int]: token ID集合
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return set(tokens)


def filter_hidden_states(all_hidden_states_raw):
    """
    过滤掉embedding层（第一个隐藏层），只保留transformer层的隐藏状态
    参考 test_chair_test.py 的 generate_response 函数

    Args:
        all_hidden_states_raw: 原始hidden states，每个步骤包含33个元素（索引0是embedding层，索引1-32是transformer层）

    Returns:
        过滤后的hidden states，每个步骤只包含32个transformer层的隐藏状态
    """
    if all_hidden_states_raw is None:
        return None

    if isinstance(all_hidden_states_raw, (tuple, list)):
        # 对每个步骤的hidden_states，去掉第一个元素（embedding层）
        all_hidden_states = []
        for step_hidden_states in all_hidden_states_raw:
            if step_hidden_states is None:
                all_hidden_states.append(None)
            elif isinstance(step_hidden_states, (tuple, list)) and len(step_hidden_states) > 0:
                # 跳过第一个元素（embedding层），只保留transformer层（索引1-32）
                transformer_hidden_states = step_hidden_states[1:] if len(step_hidden_states) > 1 else step_hidden_states
                all_hidden_states.append(transformer_hidden_states)
            else:
                all_hidden_states.append(step_hidden_states)
        return tuple(all_hidden_states) if isinstance(all_hidden_states_raw, tuple) else all_hidden_states
    else:
        return all_hidden_states_raw


def get_vocab_tokens_for_words(tokenizer, words: List[str]) -> Set[int]:
    """
    获取多个词汇对应的token ID集合

    Args:
        tokenizer: tokenizer对象
        words: 词汇列表

    Returns:
        Set[int]: token ID集合
    """
    token_ids = set()
    for word in words:
        # 尝试不同的编码方式
        word_lower = word.lower()
        word_upper = word.upper()
        word_capitalize = word.capitalize()

        for w in [word, word_lower, word_upper, word_capitalize]:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            token_ids.update(tokens)

            # 如果词汇包含空格，也尝试单独编码每个部分
            if ' ' in w:
                parts = w.split()
                for part in parts:
                    part_tokens = tokenizer.encode(part, add_special_tokens=False)
                    token_ids.update(part_tokens)

    return token_ids


def format_tokens_with_probs(tokenizer, logits: torch.Tensor, token_set: Set[int], max_display: int = 10) -> str:
    """
    格式化token集合，显示token ID、对应的词汇和概率值

    Args:
        tokenizer: tokenizer对象
        logits: [vocab_size] 的logits
        token_set: token ID集合
        max_display: 最多显示的token数量

    Returns:
        str: 格式化后的字符串，例如 "2022('person':0.023), 3094('bicycle':0.015), ..."
    """
    if len(token_set) == 0:
        return "[] (空)"

    # 转换为概率分布
    probs = torch.softmax(logits, dim=-1)

    # 获取token列表（限制显示数量）
    token_list = list(token_set)[:max_display]

    # 格式化每个token
    formatted_tokens = []
    for token_id in token_list:
        if token_id < len(probs):
            prob = probs[token_id].item()
            # 尝试解码token（可能包含特殊字符，需要处理）
            try:
                word = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                # 清理特殊字符和空白
                word = word.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                if not word or word.isspace():
                    word = f"<token_{token_id}>"
            except:
                word = f"<token_{token_id}>"
            formatted_tokens.append(f"{token_id}('{word}':{prob:.4f})")
        else:
            formatted_tokens.append(f"{token_id}(<invalid>)")

    # 如果还有更多token，添加省略号
    if len(token_set) > max_display:
        formatted_tokens.append(f"... (共{len(token_set)}个)")

    return f"[{', '.join(formatted_tokens)}]"


def compute_set_probability(logits: torch.Tensor, token_set: Set[int]) -> float:
    """
    计算集合概率 P(B | ξ) = Σ_{b∈B} P(b | ξ)

    Args:
        logits: [vocab_size] 的logits
        token_set: token ID集合

    Returns:
        float: 集合概率（如果计算失败或包含nan/inf，返回0.0）
    """
    # 检查logits是否包含nan或inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        return 0.0

    # 转换为概率分布
    probs = torch.softmax(logits, dim=-1)

    # 检查softmax结果是否包含nan或inf
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return 0.0

    # 计算集合中所有token的概率和
    set_probs = [probs[token_id].item() for token_id in token_set if token_id < len(probs)]

    result = sum(set_probs) if set_probs else 0.0

    # 检查结果是否为nan或inf
    if np.isnan(result) or np.isinf(result):
        return 0.0

    return result


def compute_log_probability_gain(
    logits_with_head: torch.Tensor,
    logits_without_head: torch.Tensor,
    token_set: Set[int],
    use_np_exp: bool = None
) -> float:
    """
    计算概率增益

    支持两种计算方法：
    1. use_np_exp=True (默认): 使用指数函数
       ΔP_u^(l,n)(B_u) = exp(coeff*P(B_u | h_with)) - exp(coeff*P(B_u | h_without))
       优势：避免log(0)导致的数值不稳定问题，数值范围更稳定
    2. use_np_exp=False: 使用对数函数（原始公式）
       Δlog P_u^(l,n)(B_u) = log P(B_u | h_with) - log P(B_u | h_without)
       注意：当prob=0时，log(0)=-inf，需要特殊处理

    Args:
        logits_with_head: 加入head后的logits [vocab_size]
        logits_without_head: 未加入head的logits [vocab_size]
        token_set: token ID集合（可以为空，空集合时自动返回0.0）
        use_np_exp: 是否使用np.exp计算（默认使用全局变量USE_NP_EXP）

    Returns:
        float: 概率增益
    """
    # 如果没有指定，使用全局变量
    if use_np_exp is None:
        use_np_exp = USE_NP_EXP

    if token_set is None or len(token_set) == 0:
        return 0.0

    # 计算集合概率（范围在 [0, 1]）
    # 注意：如果token_set为空，compute_set_probability会返回0.0
    prob_with = compute_set_probability(logits_with_head, token_set)
    prob_without = compute_set_probability(logits_without_head, token_set)

    # 检查概率值是否为nan或inf
    if np.isnan(prob_with) or np.isinf(prob_with):
        prob_with = 0.0
    if np.isnan(prob_without) or np.isinf(prob_without):
        prob_without = 0.0

    if use_np_exp:
        # 方法1：使用指数函数计算增益
        # exp(coeff*prob) 的范围是 [1, e^coeff]，其中 e^coeff ≈ e^2 ≈ 7.389
        # 当 prob 接近 0 时，exp(coeff*prob) ≈ 1，不会趋向 -inf
        # 当 prob 接近 1 时，exp(coeff*prob) ≈ e^coeff，不会趋向 +inf
        # 当 prob = 0（空集合）时，exp(0) - exp(0) = 0，自动处理空集合情况
        exp_gain = np.exp(EXP_GAIN_COEFF * prob_with) - np.exp(EXP_GAIN_COEFF * prob_without)
        # 检查结果是否为nan或inf
        if np.isnan(exp_gain) or np.isinf(exp_gain):
            return 0.0
        return exp_gain
    else:
        # 方法2：使用对数函数计算增益（原始公式）
        # log P(B_u | h_with) - log P(B_u | h_without)
        prob_with = max(prob_with, 1e-12)
        prob_without = max(prob_without, 1e-12)

        # 计算对数增益
        log_gain = np.log(prob_with) - np.log(prob_without)
        return log_gain


class HeadOutputExtractor:
    """使用hook机制提取每个head的输出"""

    def __init__(self, model, num_layers: int, num_heads: int):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hooks = []
        self.head_outputs = {}  # {(layer_idx, head_idx): output_tensor} (经过o_proj后的完整输出)
        self.head_raw_outputs = {}  # {(layer_idx, head_idx): raw_output_tensor} (原始head输出，未经过o_proj，维度为head_dim)
        self.hidden_states_before = {}  # {layer_idx: hidden_state}

    def _make_attn_pre_hook(self, layer_idx: int):
        """创建attention pre-hook来获取输入hidden_states

        注意：当前代码已改用手动计算head输出，不再依赖hook机制。
        此hook保留作为备用方案，但不会产生警告信息。
        """
        def attn_pre_hook(module, input_tuple):
            # input_tuple是forward的参数，第一个是hidden_states
            # 结构应该是: (hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
            hidden_states = None

            if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
                # 第一个参数应该是hidden_states
                first_arg = input_tuple[0]
                if isinstance(first_arg, torch.Tensor):
                    hidden_states = first_arg
                elif isinstance(first_arg, tuple) and len(first_arg) > 0:
                    # 有时候可能是嵌套的tuple
                    hidden_states = first_arg[0] if isinstance(first_arg[0], torch.Tensor) else None

            # 如果成功获取hidden_states，保存它；否则静默失败（不打印警告）
            if isinstance(hidden_states, torch.Tensor):
                self.hidden_states_before[layer_idx] = hidden_states
            # 注意：不再打印警告，因为代码已改用手动计算方式，不依赖hook
        return attn_pre_hook

    def _make_attn_hook(self, layer_idx: int):
        """创建attention hook来提取head输出"""
        def attn_hook(module, input_tuple, output):
            # 尝试从input_tuple获取hidden_states
            # input_tuple的结构应该是: (hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
            hidden_states = None

            if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
                # 第一个参数应该是hidden_states
                first_arg = input_tuple[0]
                if isinstance(first_arg, torch.Tensor):
                    hidden_states = first_arg
                elif isinstance(first_arg, tuple) and len(first_arg) > 0:
                    # 有时候可能是嵌套的tuple
                    hidden_states = first_arg[0] if isinstance(first_arg[0], torch.Tensor) else None

            # 如果input_tuple为空，尝试从之前保存的hidden_states_before获取
            if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
                hidden_states = self.hidden_states_before.get(layer_idx)

            # 如果仍然无法获取，尝试从output中获取（某些实现可能返回hidden_states）
            if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
                if isinstance(output, tuple) and len(output) > 0:
                    # output的结构通常是: (attn_output, attn_weights, past_key_value)
                    # 但attn_output是经过o_proj的，不是我们需要的
                    pass  # 不能从output获取，因为已经经过o_proj了

            # 如果仍然无法获取，跳过这一层（静默失败，不打印警告）
            # 注意：当前代码已改用手动计算head输出，不再依赖hook机制
            if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
                return

            # 确保hidden_states是正确的格式
            if len(hidden_states.shape) != 3:
                return

            # 获取attention层参数
            batch_size, seq_len, hidden_size = hidden_states.shape
            head_dim = hidden_size // self.num_heads

            # 检查head_dim是否合理
            if head_dim <= 0 or hidden_size % self.num_heads != 0:
                return

            # 计算Q, K, V
            Q = module.q_proj(hidden_states)  # [batch, seq_len, hidden_size]
            K = module.k_proj(hidden_states)
            V = module.v_proj(hidden_states)

            # 重塑为多头格式
            Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

            # 计算attention scores
            scale = 1.0 / np.sqrt(head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, seq_len]

            # 应用softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # 应用attention到V
            attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

            # 提取每个head的输出
            # 重塑attn_output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
            attn_output_reshaped = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            # Concatenate所有head: [batch, seq_len, num_heads * head_dim] = [batch, seq_len, hidden_size]
            attn_output_concat = attn_output_reshaped.contiguous().view(batch_size, seq_len, hidden_size)

            for head_idx in range(self.num_heads):
                # 提取单个head的输出（原始输出，未经过o_proj）
                head_attn_output = attn_output[:, head_idx, :, :]  # [batch, seq_len, head_dim]

                # 保存原始head输出（用于训练linear probe）
                self.head_raw_outputs[(layer_idx, head_idx)] = head_attn_output

                # 创建一个只包含该head的完整attn_output（其他head为零）
                head_only_concat = torch.zeros_like(attn_output_concat)
                head_start = head_idx * head_dim
                head_end = (head_idx + 1) * head_dim
                head_only_concat[:, :, head_start:head_end] = head_attn_output

                # 应用o_proj得到该head的完整输出
                if hasattr(module, 'o_proj'):
                    head_output_full = module.o_proj(head_only_concat)  # [batch, seq_len, hidden_size]
                else:
                    # 如果没有o_proj，直接使用head_only_concat
                    head_output_full = head_only_concat

                self.head_outputs[(layer_idx, head_idx)] = head_output_full

        return attn_hook

    def register_hooks(self):
        """注册所有层的hooks"""
        lang_model = self.model.get_model()
        if not hasattr(lang_model, 'layers'):
            raise ValueError("模型没有layers属性")

        for layer_idx in range(self.num_layers):
            if layer_idx < len(lang_model.layers):
                layer = lang_model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    # 注册pre-hook来获取输入hidden_states
                    pre_hook = layer.self_attn.register_forward_pre_hook(self._make_attn_pre_hook(layer_idx))
                    self.hooks.append(pre_hook)
                    # 注册forward hook来提取head输出
                    hook = layer.self_attn.register_forward_hook(self._make_attn_hook(layer_idx))
                    self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_head_output(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """获取特定head的输出（经过o_proj后的完整输出）"""
        return self.head_outputs.get((layer_idx, head_idx))

    def get_head_raw_output(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """获取特定head的原始输出（未经过o_proj，维度为head_dim）"""
        return self.head_raw_outputs.get((layer_idx, head_idx))

    def get_hidden_state_before(self, layer_idx: int) -> Optional[torch.Tensor]:
        """获取该层之前的hidden state"""
        return self.hidden_states_before.get(layer_idx)

    def clear(self):
        """清空缓存"""
        self.head_outputs.clear()
        self.head_raw_outputs.clear()
        self.hidden_states_before.clear()


def enable_attn_output_extraction(model, enable: bool = True):
    """
    启用或禁用 attention output 提取功能

    Args:
        model: LLaVA模型
        enable: 是否启用
    """
    lang_model = model.get_model()
    if hasattr(lang_model, 'layers'):
        for layer in lang_model.layers:
            if hasattr(layer, 'self_attn'):
                layer._output_attn_output = enable


def get_attn_output_from_layer(layer):
    """
    从层中获取最后一次 forward 的 attn_output（在 reshape 之前）

    Args:
        layer: LlamaDecoderLayer 实例

    Returns:
        torch.Tensor 或 None: attn_output [batch, num_heads, seq_len, head_dim]，如果未启用则返回 None
    """
    if hasattr(layer, '_last_attn_output_before_reshape'):
        return layer._last_attn_output_before_reshape
    return None


def process_case_pope(
    model, tokenizer, image_processor, case: Dict, coco_root: str,
    device: str, conv_mode: str, num_layers: int, num_heads: int,
    extractor: HeadOutputExtractor
) -> List[Dict]:
    """
    处理POPE类型的case（Yes/No问题）

    Args:
        model: LLaVA模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        case: case字典，包含image, text, label
        coco_root: COCO数据集根目录
        device: 设备
        conv_mode: 对话模式
        num_layers: 层数
        num_heads: 每层的head数

    Returns:
        List[Dict]: head真值对列表
    """
    # 加载图像
    image_path = os.path.join(coco_root, "val2014", case["image"])
    if not os.path.exists(image_path):
        print(f"⚠️  图像文件不存在: {image_path}")
        return []

    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    prompt = case["text"]
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 准备多模态输入
    images = image_tensor.unsqueeze(0).half().to(device)

    (
        input_ids_processed,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        images
    )

    # 获取label
    label = case["label"][0].lower() if isinstance(case["label"], list) else case["label"].lower()
    is_yes = label == "yes"

    # 定义Bu^+和Bu^-
    # Bu^+ = 正确标签（非幻视集合）
    # Bu^- = 错误标签（幻视集合）
    if is_yes:
        bu_plus_words = ["yes"]
        bu_minus_words = ["no"]
    else:
        bu_plus_words = ["no"]
        bu_minus_words = ["yes"]

    # 获取token ID集合
    bu_plus_tokens = get_vocab_tokens_for_words(tokenizer, bu_plus_words)
    bu_minus_tokens = get_vocab_tokens_for_words(tokenizer, bu_minus_words)

    # 清空extractor缓存
    extractor.clear()

    # Forward pass获取所有层的hidden states和head输出
    with torch.no_grad():
        outputs = model.get_model().forward(
            input_ids=input_ids_processed if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states  # 包含所有层的hidden states
    lm_head = model.lm_head

    # 获取layer normalization层（如果存在），用于在应用lm_head之前归一化hidden states
    # 参考 test_chair_test.py 的 _get_predicted_token_for_layer 函数
    lang_model = model.get_model()
    norm_layer = lang_model.norm if hasattr(lang_model, 'norm') else None

    # 获取最后一个token的位置（用于生成预测）
    seq_len = hidden_states[0].shape[1]
    last_token_idx = seq_len - 1

    # 存储所有head的真值对
    ground_truth_pairs = []

    # 启用 attention output 提取功能
    enable_attn_output_extraction(model, enable=True)

    # 遍历每一层，使用模型内部计算的 attn_output（而不是手动计算）
    for layer_idx in range(num_layers):
        # 获取该层之前的hidden state（h_t^(l-1)）- 使用完整的序列
        if layer_idx == 0:
            h_before_full = hidden_states[0]  # [batch, seq_len, hidden_size]
        else:
            h_before_full = hidden_states[layer_idx]  # [batch, seq_len, hidden_size]

        # 获取最后一个token的hidden state（用于计算logits）
        h_before = h_before_full[:, last_token_idx:last_token_idx+1, :]  # [batch, 1, hidden_size]

        # 计算未加入head的logits（参考 test_chair_test.py，在应用lm_head之前先通过norm_layer）
        h_before_processed = h_before.squeeze(1)  # [batch, hidden_size]
        if norm_layer is not None:
            h_before_processed = norm_layer(h_before_processed.to(device))
        else:
            h_before_processed = h_before_processed.to(device)
        logits_before = lm_head(h_before_processed)  # [batch, vocab_size]

        # 获取该层的attention模块和层对象
        layer = lang_model.layers[layer_idx]
        attn_module = layer.self_attn

        # 从模型内部获取 attn_output（在 reshape 之前）
        # 注意：我们需要重新调用 forward 来触发 output_attn_output
        # 但为了效率，我们可以直接从层的属性中获取（如果已经计算过）
        attn_output_full = get_attn_output_from_layer(layer)

        # 如果无法从层属性获取，需要手动调用一次 forward
        if attn_output_full is None:
            # 手动调用 forward 来获取 attn_output
            with torch.no_grad():
                # 只对当前层调用 forward，传入 h_before_full
                attn_result = attn_module.forward(
                    hidden_states=h_before_full,
                    attention_mask=None,  # POPE case 通常不需要 attention_mask
                    position_ids=position_ids if position_ids is not None and position_ids.shape[1] == h_before_full.shape[1] else None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    output_attn_output=True,
                )
                # 解包返回值（应该是4个元素）
                if len(attn_result) == 4:
                    _, _, _, attn_output_full = attn_result
                else:
                    raise ValueError(
                        f"Expected 4 return values from attention forward (with output_attn_output=True), "
                        f"but got {len(attn_result)}. This indicates the model's attention layer "
                        f"does not support output_attn_output parameter."
                    )

        # 如果仍然无法获取，直接报错（不再回退到手动计算）
        if attn_output_full is None:
            raise RuntimeError(
                f"Failed to get attn_output from layer {layer_idx}. "
                f"This should not happen if output_attn_output is enabled. "
                f"Please check if the model's attention layer supports output_attn_output parameter."
            )

        # 确保 attn_output_full 的形状正确
        batch_size, seq_len_for_attn, hidden_size = h_before_full.shape
        head_dim = hidden_size // num_heads
        if attn_output_full.shape != (batch_size, num_heads, seq_len_for_attn, head_dim):
            raise ValueError(
                f"attn_output_full shape mismatch: expected {(batch_size, num_heads, seq_len_for_attn, head_dim)}, "
                f"got {attn_output_full.shape}"
            )

        # 获取数据类型（用于后续创建张量）
        dtype = attn_output_full.dtype

        # 遍历每个head
        for head_idx in range(num_heads):
            # 提取单个head的输出（原始输出，未经过o_proj）- 只取最后一个token
            head_attn_output_full = attn_output_full[:, head_idx, :, :]  # [batch, seq_len, head_dim]
            head_attn_output = head_attn_output_full[:, last_token_idx:last_token_idx+1, :]  # [batch, 1, head_dim]

            # 保存原始head输出（用于训练linear probe）- 最后一个token
            # 需要先detach()，因为张量可能requires_grad
            head_raw_vector = head_attn_output[:, 0, :].detach().cpu().numpy()  # [head_dim]

            # 创建一个只包含该head的完整attn_output（其他head为零）- 只对最后一个token
            # 使用与模型相同的数据类型
            head_only_concat = torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype)
            head_start = head_idx * head_dim
            head_end = (head_idx + 1) * head_dim
            head_only_concat[:, :, head_start:head_end] = head_attn_output

            # 应用o_proj得到该head的完整输出
            head_output_full = attn_module.o_proj(head_only_concat)  # [batch, 1, hidden_size]

            # 使用精确的head输出
            h_with_head = h_before + head_output_full

            # 计算加入该head后的logits（参考 test_chair_test.py，在应用lm_head之前先通过norm_layer）
            h_with_head_processed = h_with_head.squeeze(1)  # [batch, hidden_size]
            if norm_layer is not None:
                h_with_head_processed = norm_layer(h_with_head_processed.to(device))
            else:
                h_with_head_processed = h_with_head_processed.to(device)
            logits_with_head = lm_head(h_with_head_processed)

            # 计算概率增益（使用指数函数，避免log计算时的数值不稳定问题）
            delta_log_p_plus = compute_log_probability_gain(
                logits_with_head[0], logits_before[0], bu_plus_tokens
            )
            delta_log_p_minus = compute_log_probability_gain(
                logits_with_head[0], logits_before[0], bu_minus_tokens
            )

            # 计算语义先验偏置分数（公式C3）
            # s_u^(l,n) = ΔP_u^(l,n)(B_u^-) - ΔP_u^(l,n)(B_u^+)
            # 注意：这里使用的是指数函数计算的概率增益，而不是对数概率增益
            s_u = delta_log_p_minus - delta_log_p_plus

            # 检查s_u是否为nan或inf
            if np.isnan(s_u) or np.isinf(s_u):
                # 如果s_u是nan或inf，跳过这个head
                continue

            # 计算SPP输出g（公式D1）
            # g_u^(l,n) = tanh(α * s_u^(l,n) + β)
            g_u = tanh(ALPHA * s_u + BETA)

            # 检查g_u是否为nan或inf
            if np.isnan(g_u) or np.isinf(g_u):
                # 如果g_u是nan或inf，跳过这个head
                continue

            # 保存真值对
            pair = {
                "case_id": case["question_id"],
                "layer": layer_idx,
                "head": head_idx,
                "s_u": float(s_u),
                "g_u": float(g_u),
                "delta_log_p_plus": float(delta_log_p_plus),
                "delta_log_p_minus": float(delta_log_p_minus),
                "case_type": "POPE"
            }

            # 添加head的原始输出向量（用于训练linear probe）
            if head_raw_vector is not None:
                pair["head_vector"] = head_raw_vector.tolist()  # 转换为列表以便JSON序列化
            else:
                # 如果无法获取，使用零向量
                head_dim = model.config.hidden_size // num_heads
                pair["head_vector"] = [0.0] * head_dim

            ground_truth_pairs.append(pair)

    return ground_truth_pairs


def process_case_chair(
    model, tokenizer, image_processor, case: Dict, coco_root: str,
    device: str, conv_mode: str, num_layers: int, num_heads: int,
    extractor: HeadOutputExtractor, chair_evaluator=None
) -> List[Dict]:
    """
    处理CHAIR类型的case（图像描述）

    优化后的流程：
    1. 先让模型生成完整文本
    2. 使用CHAIR接口筛选出物理词汇，找到对应的token和推理步
    3. 从物理词汇中，真实实例作为Bu^+，幻视词汇作为Bu^-的一部分
    4. 对每个head，在每个生成步骤计算logits，取top-K，筛选物理词汇，不在Bu^+中的纳入Bu^-，计算概率值得到s_u和g_u

    Args:
        model: LLaVA模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        case: case字典，包含image, text, label（对象列表）
        coco_root: COCO数据集根目录
        device: 设备
        conv_mode: 对话模式
        num_layers: 层数
        num_heads: 每层的head数

    Returns:
        List[Dict]: head真值对列表
    """
    # 加载图像
    image_path = os.path.join(coco_root, "val2014", case["image"])
    if not os.path.exists(image_path):
        print(f"⚠️  图像文件不存在: {image_path}")
        return []

    # 打印case信息
    case_id = case.get("question_id", case.get("image_id", "N/A"))
    case_text = case.get("text", "N/A")
    case_label = case.get("label", "N/A")
    print(f"\n{'='*80}")
    print(f"处理 Case #{case_id}")
    print(f"  图像路径: {image_path}")
    print(f"  文本 (text): {case_text}")
    print(f"  标签 (label): {case_label}")
    print(f"{'='*80}")

    image = load_image(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # 准备文本输入
    prompt = case["text"]
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    # 准备多模态输入
    images = image_tensor.unsqueeze(0).half().to(device)

    # 获取真实实例词汇集合（Bu^+）
    gt_objects = case["label"] if isinstance(case["label"], list) else [case["label"]]
    bu_plus_words = gt_objects

    # 使用传入的CHAIR评估器（如果为None，则创建新的）
    if chair_evaluator is None:
        from eval_tool.chair import CHAIR
        coco_annotations_path = os.path.join(coco_root, "annotations")
        chair_evaluator = CHAIR(coco_annotations_path)

    # 步骤1: 生成完整文本
    print(f"  生成文本...")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 注意：LLaVA的generate方法不支持inputs_embeds参数
    # 它会内部调用prepare_inputs_labels_for_multimodal来处理输入
    # 所以直接传递input_ids和images即可
    # 使用 torch.inference_mode() 和 torch.no_grad() 来优化性能（参考 chair_eval_train_and_test.py）
    with torch.inference_mode():
        with torch.no_grad():
            output_dict = model.generate(
                inputs=input_ids,
                images=images,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria]
            )

    # 获取生成的token序列
    # 注意：LLaVA的output_ids不包含input_ids，直接使用output_ids作为生成的token
    output_ids = output_dict.sequences
    generated_ids = output_ids
    output_token_len = generated_ids.shape[1]

    # 处理 BOS token 和最终解码（参考 test_chair_test.py）
    if output_token_len > 0:
        # 如果新生成的 token 以 BOS token 开头, 跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0:
            first_token = generated_ids[0, 0].item()
            if first_token == bos_token_id:
                generated_ids = generated_ids[:, 1:]
                output_token_len = generated_ids.shape[1]

        if generated_ids.shape[1] > 0:
            generated_token_ids = generated_ids[0].cpu().tolist()
        else:
            generated_token_ids = []
    else:
        generated_token_ids = []

    # 解码生成的文本
    if len(generated_token_ids) > 0:
        generated_text = tokenizer.batch_decode([generated_token_ids], skip_special_tokens=True)[0].strip()
    else:
        generated_text = ""

    # 移除停止字符串
    if generated_text and generated_text.endswith(stop_str):
        generated_text = generated_text[:-len(stop_str)].strip()

    if not generated_text:
        print(f"  ⚠️  警告: 生成文本为空，跳过此case")
        return []

    print(f"  生成文本: {generated_text[:100]}...")

    # 步骤2: 使用CHAIR接口筛选物理词汇，找到对应的token位置和生成步骤
    # 参考 test_chair_test.py 的 identify_object_tokens_in_caption 函数，使用精确的token匹配方法
    words, node_words, word_indices, raw_words = chair_evaluator.caption_to_words(generated_text)

    # 构建物理词汇到token位置和生成步骤的映射
    # physical_word_to_steps: {word: [step_indices]} - 物理词汇对应的生成步骤（可能多个）
    # physical_word_to_node: {word: node_word} - 物理词汇对应的标准COCO类别
    # step_to_physical_word: {step_idx: word} - 每个步骤对应的完整物理词汇
    physical_word_to_steps = {}  # {word: [step_indices]}
    physical_word_to_node = {}    # {word: node_word}
    step_to_physical_word = {}    # {step_idx: word} - 每个步骤对应的完整物理词汇

    # 使用精确的token匹配方法（参考 test_chair_test.py）
    # generated_token_ids[0] 对应 step_0（生成第1个token）
    # generated_token_ids[1] 对应 step_1（生成第2个token）
    # 所以 token_idx 就是 step_idx

    def _is_singular_plural_match(word1, word2):
        """检查两个单词是否是单复数关系（参考 test_chair_test.py）"""
        word1_lower = word1.lower().strip()
        word2_lower = word2.lower().strip()
        if word1_lower == word2_lower:
            return True
        # 简单的单复数检查
        if word1_lower + 's' == word2_lower or word1_lower == word2_lower + 's':
            return True
        if word1_lower + 'es' == word2_lower or word1_lower == word2_lower + 'es':
            return True
        if word1_lower.endswith('y') and word1_lower[:-1] + 'ies' == word2_lower:
            return True
        if word2_lower.endswith('y') and word2_lower[:-1] + 'ies' == word1_lower:
            return True
        return False

    # 使用node_word作为key，确保相同node的词汇只保留第一次出现
    # node_word_to_first_word: {node_word: first_word} - 记录每个node第一次出现的词汇
    node_word_to_first_word = {}  # {node_word: first_word}

    for word, node_word, word_idx in zip(words, node_words, word_indices):
        # 如果这个node_word已经出现过，跳过（只保留第一次出现的词汇）
        if node_word in node_word_to_first_word:
            continue

        # 记录这个node_word第一次出现的词汇
        node_word_to_first_word[node_word] = word

        if word not in physical_word_to_steps:
            physical_word_to_steps[word] = []
            physical_word_to_node[word] = node_word

        word_lower = word.lower()
        token_positions = []
        matched_ranges = []  # 记录所有匹配的token范围 [(start_idx, end_idx), ...]

        # 方法：精确匹配，只匹配真正组成目标词汇的token
        # 1. 首先检查单个token是否精确等于目标词汇（去除前后空格）
        # 2. 如果单个token匹配失败，使用滑动窗口匹配多个token的组合（处理被分解的单词）
        # 3. 对于多词短语，也使用滑动窗口匹配

        # 1. 检查单个token精确匹配（快速路径，支持单复数匹配）
        single_token_matched = False
        for token_idx, token_id in enumerate(generated_token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip().lower()
            # 检查精确匹配或单复数匹配
            if _is_singular_plural_match(token_text, word_lower):
                if token_idx not in token_positions:
                    token_positions.append(token_idx)
                    # 单个token作为一个匹配范围
                    matched_ranges.append((token_idx, token_idx))
                single_token_matched = True

        # 2. 如果单个token匹配失败，或者目标词汇是多词（如 "traffic light"），使用滑动窗口匹配
        # 对于单个单词，也尝试多token组合匹配（处理被tokenizer分解的情况）
        if not single_token_matched or ' ' in word_lower:
            # 确定滑动窗口的最大大小
            if ' ' in word_lower:
                words_in_phrase = word_lower.split()
                max_window_size = min(len(words_in_phrase) + 2, len(generated_token_ids))
                min_window_size = len(words_in_phrase)
            else:
                # 单个单词：尝试2-5个token的组合
                max_window_size = min(5, len(generated_token_ids))
                min_window_size = 2

            # 使用滑动窗口匹配
            for window_size in range(min_window_size, max_window_size + 1):
                for start_idx in range(len(generated_token_ids) - window_size + 1):
                    # 检查这个窗口是否与已匹配的范围有重叠
                    end_idx = start_idx + window_size - 1
                    is_overlapping = False
                    for matched_start, matched_end in matched_ranges:
                        if not (end_idx < matched_start or start_idx > matched_end):
                            is_overlapping = True
                            break

                    if is_overlapping:
                        continue

                    # 获取窗口内的token序列
                    window_tokens = generated_token_ids[start_idx:start_idx + window_size]
                    window_text = tokenizer.decode(window_tokens, skip_special_tokens=False).strip().lower()

                    # 关键检查：只有当窗口解码后的文本正好等于目标词汇或单复数匹配时，才认为是有效匹配
                    if _is_singular_plural_match(window_text, word_lower):
                        # 精确匹配成功，记录这个匹配范围
                        matched_ranges.append((start_idx, end_idx))

                        # 添加窗口内的所有token
                        for i in range(window_size):
                            token_idx = start_idx + i
                            if token_idx not in token_positions:
                                token_positions.append(token_idx)

        # 将token位置转换为step索引（token_idx 就是 step_idx）
        # 对于相同的物理词汇，只保留第一次出现的第一个组成token（只取第一个token，不取整个token组）
        if matched_ranges:
            # 找到第一次出现的匹配范围（最小的start_idx）
            first_range = min(matched_ranges, key=lambda x: x[0])
            first_start, first_end = first_range

            # 只使用第一次出现的第一个组成token（只取first_start，不取整个范围）
            step_idx = first_start
            if step_idx not in physical_word_to_steps[word]:
                physical_word_to_steps[word].append(step_idx)
            # 标记该步骤对应的完整物理词汇
            step_to_physical_word[step_idx] = word

    # 对于CHAIR类型，限制物理词汇数量：最多使用6个非重复的词汇
    # 这样可以保持POPE和CHAIR类型的case生成的真值对数量级相同
    MAX_PHYSICAL_WORDS_FOR_CHAIR = 6
    if len(physical_word_to_steps) > MAX_PHYSICAL_WORDS_FOR_CHAIR:
        # 按step_idx排序，优先选择较早出现的物理词汇
        word_step_pairs = [(word, steps[0]) for word, steps in physical_word_to_steps.items() if steps]
        word_step_pairs.sort(key=lambda x: x[1])  # 按step_idx排序

        # 只保留前MAX_PHYSICAL_WORDS_FOR_CHAIR个词汇
        selected_words = [word for word, _ in word_step_pairs[:MAX_PHYSICAL_WORDS_FOR_CHAIR]]

        # 更新physical_word_to_steps和step_to_physical_word，只保留选中的词汇
        physical_word_to_steps_filtered = {}
        step_to_physical_word_filtered = {}
        for word in selected_words:
            if word in physical_word_to_steps:
                steps = physical_word_to_steps[word]
                physical_word_to_steps_filtered[word] = steps
                for step_idx in steps:
                    step_to_physical_word_filtered[step_idx] = word

        physical_word_to_steps = physical_word_to_steps_filtered
        step_to_physical_word = step_to_physical_word_filtered

        print(f"  ⚠️  物理词汇数量超过限制，已筛选为前 {MAX_PHYSICAL_WORDS_FOR_CHAIR} 个词汇（按出现顺序）")

    # 步骤3: 识别每个生成步骤的token类型（Grounded/Hallucinated/Neutral）
    # 根据截图规则：
    # - Grounded（真实实例）: node_word在gt_objects中
    # - Hallucinated（幻视）: node_word不在gt_objects中
    # - Neutral（非物理词汇）: 不在物理词汇列表中，直接忽略

    # 构建G（当前图可支持的词表）= 真实实例词汇集合
    G_tokens = get_vocab_tokens_for_words(tokenizer, gt_objects)

    # 打印详细的物理词汇和生成步骤信息（包含Grounded/Hallucinated标注）
    print(f"  找到 {len(physical_word_to_steps)} 个物理词汇，对应 {sum(len(steps) for steps in physical_word_to_steps.values())} 个生成步骤")
    grounded_words = []
    hallucinated_words = []

    for word, steps in sorted(physical_word_to_steps.items()):
        node_word = physical_word_to_node.get(word, word)
        # 判断是Grounded还是Hallucinated
        is_grounded = node_word.lower() in [obj.lower() for obj in gt_objects]
        word_type = "Grounded" if is_grounded else "Hallucinated"

        if is_grounded:
            grounded_words.append(word)
        else:
            hallucinated_words.append(word)

        steps_str = ", ".join([f"step_{s}" for s in sorted(steps)])
        # 获取每个步骤对应的token文本
        token_texts = []
        for step_idx in sorted(steps):
            if step_idx < len(generated_token_ids):
                token_id = generated_token_ids[step_idx]
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                token_texts.append(f"'{token_text}'")
        tokens_str = ", ".join(token_texts) if token_texts else "N/A"
        print(f"    - 物理词汇: '{word}' (node: '{node_word}') [{word_type}] -> 步骤: [{steps_str}] -> tokens: [{tokens_str}] (仅使用第一次出现)")

    # 打印汇总信息
    grounded_str = ', '.join([f"'{w}'" for w in grounded_words]) if grounded_words else '无'
    hallucinated_str = ', '.join([f"'{w}'" for w in hallucinated_words]) if hallucinated_words else '无'
    print(f"  - Grounded物理词汇 ({len(grounded_words)}个): {grounded_str}")
    print(f"  - Hallucinated物理词汇 ({len(hallucinated_words)}个): {hallucinated_str}")

    # 检查是否有幻视词汇（用于调试）
    if len(hallucinated_words) == 0:
        print(f"  ⚠️  警告: 生成的 caption 中没有幻视词汇！")
        print(f"     这可能导致无法构建有效的对比（Bu^- 为空），所有 head 都会被跳过")
        print(f"     可能原因:")
        print(f"     1. 模型生成的 caption 非常准确，没有产生幻视")
        print(f"     2. 筛选时生成的 caption 与当前生成的 caption 不一致（模型生成的不确定性）")
        print(f"     3. 图片在补充阶段被随机选中，没有经过 caption 检查")
        print(f"     建议: 检查 build_coco_train_dataset.py 的筛选逻辑，确保所有图片都经过 caption 检查")

    # 构建H（幻觉候选词表）= COCO对象类别词汇表
    from eval_tool.chair import synonyms_txt
    synonyms = synonyms_txt.strip().splitlines()
    synonyms = [s.strip().split(', ') for s in synonyms if s.strip()]
    all_coco_object_words = []
    for synonym_group in synonyms:
        all_coco_object_words.extend(synonym_group)
    all_coco_object_words = list(set(all_coco_object_words))
    H_tokens = get_vocab_tokens_for_words(tokenizer, all_coco_object_words)

    # 为每个生成步骤标记token类型
    # step_token_type: {step_idx: 'grounded' | 'hallucinated'}
    # step_token_word: {step_idx: word} - 该步骤对应的完整物理词汇（即使单个token只是部分）
    step_token_type = {}  # {step_idx: 'grounded' | 'hallucinated'}
    step_token_word = {}  # {step_idx: word} - 该步骤对应的完整物理词汇

    # 根据step_to_physical_word标记每个步骤的类型
    for step_idx, word in step_to_physical_word.items():
        if word in physical_word_to_node:
            node_word = physical_word_to_node[word]
            # 判断是Grounded还是Hallucinated
            is_grounded = node_word.lower() in [obj.lower() for obj in gt_objects]
            token_type = 'grounded' if is_grounded else 'hallucinated'

            step_token_type[step_idx] = token_type
            step_token_word[step_idx] = word  # 即使单个token只是部分，也标记为完整物理词汇

    # 收集所有需要处理的生成步骤（只处理物理词汇对应的步骤，忽略Neutral）
    target_steps = sorted(list(step_token_type.keys()))

    # 获取所有生成步骤的hidden states
    # output_dict.hidden_states 是一个tuple，每个元素是每一层的hidden states
    # 格式: (step_0_hidden_states, step_1_hidden_states, ...)
    # 每个step的hidden_states是一个tuple，包含所有层的hidden states（索引0是embedding层，索引1-32是transformer层）
    all_hidden_states_raw = output_dict.hidden_states if hasattr(output_dict, 'hidden_states') else None

    if all_hidden_states_raw is None:
        print(f"  ⚠️  无法获取生成步骤的hidden states，跳过")
        return []

    # 过滤掉embedding层，只保留transformer层的hidden states（参考 test_chair_test.py）
    # 过滤后：每个步骤只包含32个transformer层的隐藏状态（索引0-31对应layer_0到layer_31）
    all_step_hidden_states = filter_hidden_states(all_hidden_states_raw)

    if len(target_steps) == 0:
        print(f"  ⚠️  没有找到物理词汇对应的生成步骤，跳过")
        return []

    print(f"  将处理 {len(target_steps)} 个生成步骤: {target_steps}")
    print(f"  - Grounded步骤: {sum(1 for s in target_steps if step_token_type.get(s) == 'grounded')}")
    print(f"  - Hallucinated步骤: {sum(1 for s in target_steps if step_token_type.get(s) == 'hallucinated')}")

    # 存储所有head的真值对
    ground_truth_pairs = []
    lm_head = model.lm_head

    # 获取layer normalization层（如果存在），用于在应用lm_head之前归一化hidden states
    # 参考 test_chair_test.py 的 _get_predicted_token_for_layer 函数
    lang_model = model.get_model()
    norm_layer = lang_model.norm if hasattr(lang_model, 'norm') else None

    # 准备输入（用于后续手动调用forward）
    # 我们需要使用prepare_inputs_labels_for_multimodal来处理输入
    (
        input_ids_processed,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        images
    )

    # 获取embedding层（如果需要将新生成的token IDs转换为embeddings）
    lang_model = model.get_model()
    embedding_layer = lang_model.embed_tokens if hasattr(lang_model, 'embed_tokens') else None

    # 统计信息：记录每个step的处理情况（用于调试）
    step_processed_count = {}  # {step_idx: count} - 记录每个step成功处理的head数量
    step_skipped_reasons = {}  # {step_idx: [reasons]} - 记录每个step被跳过的原因

    # 遍历每个目标生成步骤
    for step_idx in target_steps:
        # 检查步骤是否有效
        if step_idx >= len(generated_token_ids):
            if step_idx not in step_skipped_reasons:
                step_skipped_reasons[step_idx] = []
            step_skipped_reasons[step_idx].append(f"step_idx >= len(generated_token_ids): {step_idx} >= {len(generated_token_ids)}")
            continue

        # 方案1：尝试使用理想方法 - 手动调用forward来触发hook
        # 构建到当前步骤为止的完整token序列
        # step_idx=0 表示生成第1个token，所以需要 input_ids + generated_token_ids[0:1]
        # step_idx=1 表示生成第2个token，所以需要 input_ids + generated_token_ids[0:2]
        # 注意：generated_token_ids[step_idx] 是当前步骤生成的token

        # 清空extractor缓存，准备捕获新的head输出
        # 注意：确保hook已经注册，并且每次forward前清空缓存
        extractor.clear()

        # 调试：检查hook是否已注册（只在第一个step检查一次）
        if step_idx == target_steps[0]:
            lang_model = model.get_model()
            if hasattr(lang_model, 'layers') and len(lang_model.layers) > 0:
                first_layer = lang_model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    # 检查是否有hook注册
                    hooks = first_layer.self_attn._forward_hooks
                    pre_hooks = first_layer.self_attn._forward_pre_hooks
                    if not hooks and not pre_hooks:
                        print(f"  ⚠️  警告: Layer 0 的attention层没有注册hook！")
                    else:
                        print(f"  ✓ Layer 0 的attention层已注册hook（forward_hooks: {len(hooks)}, pre_hooks: {len(pre_hooks)}）")

        # 构建当前步骤的完整序列
        # 如果inputs_embeds不为None，说明原始输入已经被转换为embeddings
        # 我们需要将新生成的token IDs转换为embeddings并拼接
        if inputs_embeds is not None:
            # 原始输入已经是embeddings
            # 将新生成的token IDs转换为embeddings
            new_token_ids = torch.tensor(generated_token_ids[:step_idx+1], device=device, dtype=torch.long)
            if embedding_layer is not None:
                new_token_embeds = embedding_layer(new_token_ids)  # [step_idx+1, hidden_size]
            else:
                # 如果没有embedding层，使用零向量（不应该发生）
                hidden_size = inputs_embeds.shape[-1]
                new_token_embeds = torch.zeros((step_idx+1, hidden_size), device=device, dtype=inputs_embeds.dtype)

            # 拼接原始embeddings和新生成的token embeddings
            current_inputs_embeds = torch.cat([
                inputs_embeds[0],  # [original_seq_len, hidden_size]
                new_token_embeds   # [step_idx+1, hidden_size]
            ], dim=0).unsqueeze(0)  # [1, total_seq_len, hidden_size]

            current_input_ids = None
        else:
            # 原始输入是token IDs
            current_input_ids = torch.cat([
                input_ids_processed[0],  # 原始input_ids
                torch.tensor(generated_token_ids[:step_idx+1], device=device, dtype=torch.long)  # 已生成的tokens（包括当前步骤）
            ], dim=0).unsqueeze(0)  # [1, seq_len]
            current_inputs_embeds = None

        # 手动调用forward来触发hook
        # 注意：需要确保hook已经注册，并且forward会触发hook
        with torch.no_grad():
            # 计算position_ids和attention_mask
            if current_inputs_embeds is not None:
                current_seq_len = current_inputs_embeds.shape[1]
            else:
                current_seq_len = current_input_ids.shape[1]
            current_position_ids = torch.arange(current_seq_len, device=device, dtype=torch.long).unsqueeze(0)
            current_attention_mask = torch.ones((1, current_seq_len), device=device, dtype=torch.long)

            # 调用forward（hook会自动触发）
            # 注意：直接调用model.get_model().forward()会触发hook
            outputs = model.get_model().forward(
                input_ids=current_input_ids,
                inputs_embeds=current_inputs_embeds,
                position_ids=current_position_ids,
                attention_mask=current_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # 注意：我们不再依赖hook，而是直接使用hidden_states手动计算每个head的输出
            # 因此hook相关的调试信息已不再需要

        # 获取hidden states（用于获取h_before）
        hidden_states = outputs.hidden_states  # 包含所有层的hidden states

        # 获取最后一个token的位置（用于生成预测）
        seq_len = hidden_states[0].shape[1]
        last_token_idx = seq_len - 1

        # 获取语言模型
        lang_model = model.get_model()

        # 遍历每一层，手动计算每个head的输出
        for layer_idx in range(num_layers):
            # 获取该层之前的hidden state（h_t^(l-1)）- 使用完整的序列
            if layer_idx == 0:
                h_before_full = hidden_states[0]  # [batch, seq_len, hidden_size]
            else:
                h_before_full = hidden_states[layer_idx]  # [batch, seq_len, hidden_size]

            # 获取最后一个token的hidden state（用于计算logits）
            h_before = h_before_full[:, last_token_idx:last_token_idx+1, :]  # [batch, 1, hidden_size]

            # 计算未加入head的logits（参考 test_chair_test.py，在应用lm_head之前先通过norm_layer）
            h_before_processed = h_before.squeeze(1)  # [batch, hidden_size]
            if norm_layer is not None:
                h_before_processed = norm_layer(h_before_processed.to(device))
            else:
                h_before_processed = h_before_processed.to(device)
            logits_before = lm_head(h_before_processed)  # [batch, vocab_size]

            # 获取该层的attention模块和层对象
            layer = lang_model.layers[layer_idx]
            attn_module = layer.self_attn

            # 从模型内部获取 attn_output（在 reshape 之前）
            # 注意：我们已经在前面调用了 forward，所以应该可以从层的属性中获取
            attn_output_full = get_attn_output_from_layer(layer)

            # 如果无法从层属性获取，需要手动调用一次 forward
            if attn_output_full is None:
                # 手动调用 forward 来获取 attn_output
                with torch.no_grad():
                    # 只对当前层调用 forward，传入 h_before_full
                    attn_result = attn_module.forward(
                        hidden_states=h_before_full,
                        attention_mask=current_attention_mask if 'current_attention_mask' in locals() else None,
                        position_ids=current_position_ids if 'current_position_ids' in locals() and current_position_ids.shape[1] == h_before_full.shape[1] else None,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        output_attn_output=True,
                    )
                    # 解包返回值（应该是4个元素）
                    if len(attn_result) == 4:
                        _, _, _, attn_output_full = attn_result
                    else:
                        raise ValueError(
                            f"Expected 4 return values from attention forward (with output_attn_output=True), "
                            f"but got {len(attn_result)}. This indicates the model's attention layer "
                            f"does not support output_attn_output parameter."
                        )

            # 如果仍然无法获取，直接报错（不再回退到手动计算）
            if attn_output_full is None:
                raise RuntimeError(
                    f"Failed to get attn_output from layer {layer_idx} at step {step_idx}. "
                    f"This should not happen if output_attn_output is enabled. "
                    f"Please check if the model's attention layer supports output_attn_output parameter."
                )

            # 确保 attn_output_full 的形状正确
            batch_size, seq_len_for_attn, hidden_size = h_before_full.shape
            head_dim = hidden_size // num_heads
            if attn_output_full.shape != (batch_size, num_heads, seq_len_for_attn, head_dim):
                raise ValueError(
                    f"attn_output_full shape mismatch at layer {layer_idx}, step {step_idx}: "
                    f"expected {(batch_size, num_heads, seq_len_for_attn, head_dim)}, "
                    f"got {attn_output_full.shape}"
                )

            # 获取数据类型（用于后续创建张量）
            dtype = attn_output_full.dtype

            # 遍历每个head
            for head_idx in range(num_heads):
                # 提取单个head的输出（原始输出，未经过o_proj）- 只取最后一个token
                head_attn_output_full = attn_output_full[:, head_idx, :, :]  # [batch, seq_len, head_dim]
                head_attn_output = head_attn_output_full[:, last_token_idx:last_token_idx+1, :]  # [batch, 1, head_dim]

                # 保存原始head输出（用于训练linear probe）- 最后一个token
                # 需要先detach()，因为张量可能requires_grad
                head_raw_vector = head_attn_output[:, 0, :].detach().cpu().numpy()  # [head_dim]

                # 创建一个只包含该head的完整attn_output（其他head为零）- 只对最后一个token
                # 使用与模型相同的数据类型
                head_only_concat = torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype)
                head_start = head_idx * head_dim
                head_end = (head_idx + 1) * head_dim
                head_only_concat[:, :, head_start:head_end] = head_attn_output

                # 应用o_proj得到该head的完整输出
                head_output_full = attn_module.o_proj(head_only_concat)  # [batch, 1, hidden_size]

                # 使用精确的head输出
                h_with_head = h_before + head_output_full

                # 计算加入该head后的logits（参考 test_chair_test.py，在应用lm_head之前先通过norm_layer）
                h_with_head_processed = h_with_head.squeeze(1)  # [batch, hidden_size]
                if norm_layer is not None:
                    h_with_head_processed = norm_layer(h_with_head_processed.to(device))
                else:
                    h_with_head_processed = h_with_head_processed.to(device)
                logits_with_head = lm_head(h_with_head_processed)

                # 步骤4: 根据截图规则构建Bu^+和Bu^-
                # 获取当前步骤的token类型
                current_token_type = step_token_type.get(step_idx)
                if current_token_type is None:
                    continue

                # 获取当前步骤生成的token ID
                current_word = step_token_word.get(step_idx)
                if current_word is None:
                    continue

                # 获取当前词汇的大小写变体的token IDs
                # 包括：原始形式、小写、大写、首字母大写等（如 bus, Bus, BUS等）
                # 这样可以处理tokenizer可能以不同方式编码同一个概念的情况
                current_token_ids = get_vocab_tokens_for_words(tokenizer, [current_word])

                # 同时，如果当前词汇在CHAIR的同义词组中，也应该包含同义词的token
                # 例如：如果current_word是"people"，而CHAIR认为"people"和"person"是同义词（都映射到"person" node）
                # 那么也应该包含"person"的token
                # 优化：只包含常见的直接同义词（如 people/person），而不是整个同义词组的所有词
                # 这样可以避免token数量过多（如"person"同义词组有60+个词，会产生200+个token）
                if current_word in physical_word_to_node:
                    node_word = physical_word_to_node[current_word]
                    # 从CHAIR的synonyms_txt中查找所有属于同一个node的词汇
                    from eval_tool.chair import synonyms_txt
                    synonyms = synonyms_txt.strip().splitlines()
                    synonyms = [s.strip().split(', ') for s in synonyms if s.strip()]
                    # 找到包含node_word的同义词组
                    for synonym_group in synonyms:
                        if node_word.lower() in [s.lower() for s in synonym_group]:
                            # 优化：只包含常见的直接同义词（前5个最常见的同义词）
                            # 例如：person -> person, people, man, woman, child
                            # 这样可以避免包含整个同义词组（60+个词）导致token数量过多
                            common_synonyms = synonym_group[:5]  # 只取前5个最常见的同义词

                            # 同时，如果当前词不在前5个中，也要包含它
                            if current_word.lower() not in [s.lower() for s in common_synonyms]:
                                common_synonyms = [current_word] + common_synonyms[:4]

                            # 只对这几个常见同义词获取token
                            for synonym_word in common_synonyms:
                                # 跳过已经处理过的当前词（避免重复）
                                if synonym_word.lower() == current_word.lower():
                                    continue
                                synonym_tokens = get_vocab_tokens_for_words(tokenizer, [synonym_word])
                                current_token_ids.update(synonym_tokens)
                            break

                # 确保至少包含实际生成的token（如果step_idx有效）
                if step_idx < len(generated_token_ids):
                    actual_token_id = generated_token_ids[step_idx]
                    current_token_ids.add(actual_token_id)  # 确保包含实际生成的token

                # ========================================================================
                # 步骤4: 根据简化规则构建Bu^+和Bu^-，并计算语义先验偏置分数
                # ========================================================================
                #
                # 【计算对比逻辑说明】
                #
                # 1. 目标：计算语义先验偏置分数 s_u，用于衡量head对幻视/真实实例的贡献
                #    s_u = delta_log_p_minus - delta_log_p_plus
                #    其中：
                #    - delta_log_p_plus = exp(P(Bu^+ | h_with_head)) - exp(P(Bu^+ | h_without_head))
                #    - delta_log_p_minus = exp(P(Bu^- | h_with_head)) - exp(P(Bu^- | h_without_head))
                #    - P(Bu) = Σ_{b∈Bu} P(b)，即集合中所有token的概率和
                #    注意：使用指数函数而不是对数函数，避免log(0)导致的数值不稳定问题
                #
                # 2. 为什么需要同时有 Bu^+ 和 Bu^-？
                #    - 如果 Bu^+ 为空：P(Bu^+) = 0，exp(0) = 1，delta_log_p_plus = 0（中性假设）
                #    - 如果 Bu^- 为空：P(Bu^-) = 0，exp(0) = 1，delta_log_p_minus = 0（中性假设）
                #    - 因此，即使单个集合为空，也可以计算 s_u（使用中性假设）
                #
                # 3. 构建规则：
                #    - 如果当前推理步是真实词汇（Grounded）：
                #      * Bu^+ = {当前真实词汇}（例如：{"dog"}）
                #      * Bu^- = top-K中的所有幻视词汇（例如：{"cat", "bird"}）
                #      * 如果 top-K 中没有幻视词汇，则 Bu^- 为空，无法计算对比
                #
                #    - 如果当前推理步是幻视词汇（Hallucinated）：
                #      * Bu^- = {当前幻视词汇}（例如：{"cat"}）
                #      * Bu^+ = top-K中的所有真实词汇（例如：{"dog", "person"}）
                #      * 如果 top-K 中没有真实实例词汇，则 Bu^+ 为空，无法计算对比
                #
                # 4. 举例说明：
                #    例1：Grounded token "dog"，但 top-K 中没有幻视词汇
                #    - 当前词汇：yt = "dog"（真实实例）
                #    - top-K 预测：["dog", "puppy", "animal", "pet", ...]（都是真实实例或非物理词汇）
                #    - Bu^+ = {"dog"} ✓
                #    - Bu^- = {} ✗（空集合）
                #    - 结果：无法计算 delta_log_p_minus，因为 P(Bu^-) = 0
                #
                #    例2：Hallucinated token "cat"，但 top-K 中没有真实实例词汇
                #    - 当前词汇：yt = "cat"（幻视，因为图像中只有"dog"）
                #    - top-K 预测：["cat", "kitten", "animal", "pet", ...]（都是幻视或非物理词汇）
                #    - Bu^- = {"cat"} ✓
                #    - Bu^+ = {} ✗（空集合）
                #    - 结果：无法计算 delta_log_p_plus，因为 P(Bu^+) = 0
                #
                #    例3：正常情况（Grounded token "dog"，top-K 中有幻视词汇）
                #    - 当前词汇：yt = "dog"（真实实例）
                #    - top-K 预测：["dog", "cat", "bird", "puppy", ...]
                #    - Bu^+ = {"dog"} ✓
                #    - Bu^- = {"cat", "bird"} ✓（从 top-K 中筛选出的幻视词汇）
                #    - 结果：可以计算 s_u = delta_log_p_minus - delta_log_p_plus
                # ========================================================================

                # 获取top-K候选池 Ct（针对当前head的logits）
                # 注意：不同head的top-K预测可能不同，因此每个head需要单独计算
                # 只考虑概率值在 top-K 以内的 token
                top_k_logits, top_k_indices = torch.topk(logits_with_head[0], k=TOP_K)
                Ct_tokens = set(top_k_indices.detach().cpu().numpy().tolist())

                # 从top-K中筛选物理词汇（只保留在H中的token）
                Ct_physical_tokens = Ct_tokens & H_tokens

                # 检查top-K中的token是否能映射到完整物理词汇
                # 方法：尝试将token解码后，使用CHAIR接口检查是否是物理词汇
                # 如果无法映射到完整物理词汇或无法被CHAIR识别，就当做中性词汇（忽略）
                Ct_valid_physical_tokens = set()  # 能映射到完整物理词汇的token

                # 获取所有已知物理词汇的token IDs（用于快速检查）
                known_physical_word_tokens = set()
                for word in physical_word_to_node.keys():
                    word_tokens = get_vocab_tokens_for_words(tokenizer, [word])
                    known_physical_word_tokens.update(word_tokens)

                for token_id in Ct_physical_tokens:
                    # 方法1: 检查token是否在已知物理词汇的token集合中
                    if token_id in known_physical_word_tokens:
                        Ct_valid_physical_tokens.add(token_id)
                        continue

                    # 方法2: 解码token，检查是否能被CHAIR识别为物理词汇
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                    token_lower = token_text.lower()

                    # 检查token文本是否直接是物理词汇
                    if token_lower in all_coco_object_words:
                        Ct_valid_physical_tokens.add(token_id)
                        continue

                    # 方法3: 检查token是否是已知物理词汇的一部分
                    # 如果token是物理词汇的子串，也认为是有效的（因为可能是多token词汇的一部分）
                    for word in physical_word_to_node.keys():
                        word_lower = word.lower()
                        # 检查token是否是物理词汇的一部分，或者物理词汇是token的一部分
                        if token_lower in word_lower or word_lower in token_lower:
                            Ct_valid_physical_tokens.add(token_id)
                            break

                # 从有效的物理词汇中，分离出真实实例和幻视词汇
                Ct_grounded_tokens = Ct_valid_physical_tokens & G_tokens  # top-K中的真实实例物理词汇
                Ct_hallucinated_tokens = Ct_valid_physical_tokens - G_tokens  # top-K中的非真实实例物理词汇（幻视词汇）

                # 初始化Bu^+和Bu^-
                step_bu_plus_tokens = set()
                step_bu_minus_tokens = set()

                # 调试信息：记录top-K的详细信息（仅在第一个head和layer记录，避免输出过多）
                debug_info = None
                if head_idx == 0 and layer_idx == 0:
                    # 解码top-K tokens用于调试
                    top_k_words = [tokenizer.decode([tid], skip_special_tokens=False).strip()
                                   for tid in list(Ct_tokens)[:10]]  # 只显示前10个
                    debug_info = {
                        'top_k_tokens': list(Ct_tokens)[:10],
                        'top_k_words': top_k_words,
                        'Ct_physical_count': len(Ct_valid_physical_tokens),
                        'Ct_grounded_count': len(Ct_grounded_tokens),
                        'Ct_hallucinated_count': len(Ct_hallucinated_tokens),
                        'Ct_grounded_tokens': list(Ct_grounded_tokens),
                        'Ct_hallucinated_tokens': list(Ct_hallucinated_tokens)
                    }

                # 根据token类型构建Bu^+和Bu^-
                # 优化：如果单个集合为空，不再跳过，而是使用 delta_log_p = 0（中性假设）
                # 如果两者都为空，仍然跳过（因为无法确定 head 的真实倾向）
                is_approximated = False  # 标记是否为近似值（使用了空集合的 delta_log_p = 0）

                if current_token_type == 'grounded':
                    # 如果yt是Grounded（真实实例），例如"dog"
                    # Bu^+ = {yt}，包含当前真实词汇的大小写变体token及其同义词token，且这些token必须在top-K内
                    # 注意：可以包含同义词，但只保留在 top-K 候选池中的 token
                    step_bu_plus_tokens = current_token_ids & Ct_tokens

                    # Bu^- = top-K中的所有幻视词汇
                    # 注意：这里仍然从top-K中筛选幻视词汇，用于对比（可以包含同义词）
                    step_bu_minus_tokens = Ct_hallucinated_tokens.copy()

                    # 如果top-K中没有幻视词汇，不再跳过，而是使用 delta_log_p_minus = 0（中性假设）
                    if len(step_bu_minus_tokens) == 0:
                        is_approximated = True
                        # 记录调试信息（只在第一个head记录，避免重复）
                        if head_idx == 0 and layer_idx == 0:
                            print(f"      ℹ️  [优化] Step {step_idx}, Layer {layer_idx}, Head {head_idx}: Bu^-为空，使用 delta_log_p_minus = 0（中性假设）")
                            print(f"         当前token: '{current_word}' (Grounded)")
                            # logits_with_head 已经在第1286行计算好了，可以直接使用
                            bu_plus_str = format_tokens_with_probs(tokenizer, logits_with_head[0], step_bu_plus_tokens)
                            print(f"         Bu^+ = {bu_plus_str}")
                            print(f"         Bu^- = [] (空，将使用 delta_log_p_minus = 0)")

                elif current_token_type == 'hallucinated':
                    # 如果yt是Hallucinated（幻视），例如"cat"
                    # Bu^- = {yt}，包含当前幻视词汇的大小写变体token及其同义词token，且这些token必须在top-K内
                    # 注意：可以包含同义词，但只保留在 top-K 候选池中的 token
                    step_bu_minus_tokens = current_token_ids & Ct_tokens

                    # Bu^+ = top-K中的所有真实词汇
                    # 注意：这里仍然从top-K中筛选真实词汇，用于对比（可以包含同义词）
                    step_bu_plus_tokens = Ct_grounded_tokens.copy()

                    # 如果top-K中没有真实实例词汇，不再跳过，而是使用 delta_log_p_plus = 0（中性假设）
                    if len(step_bu_plus_tokens) == 0:
                        is_approximated = True
                        # 记录调试信息（只在第一个head记录，避免重复）
                        if head_idx == 0 and layer_idx == 0:
                            print(f"      ℹ️  [优化] Step {step_idx}, Layer {layer_idx}, Head {head_idx}: Bu^+为空，使用 delta_log_p_plus = 0（中性假设）")
                            print(f"         当前token: '{current_word}' (Hallucinated)")
                            # logits_with_head 已经在第1286行计算好了，可以直接使用
                            bu_minus_str = format_tokens_with_probs(tokenizer, logits_with_head[0], step_bu_minus_tokens)
                            print(f"         Bu^- = {bu_minus_str}")
                            print(f"         Bu^+ = [] (空，将使用 delta_log_p_plus = 0)")
                else:
                    # Neutral词汇应该已经被过滤掉了
                    continue

                # 确保Bu^+和Bu^-不重叠（虽然理论上不应该重叠，但为了安全）
                # 注意：如果当前token同时出现在Bu^+和Bu^-中，需要从Bu^-中移除
                step_bu_minus_tokens = step_bu_minus_tokens - step_bu_plus_tokens

                # 如果Bu^+和Bu^-都为空，仍然跳过（因为无法确定 head 的真实倾向）
                if len(step_bu_plus_tokens) == 0 and len(step_bu_minus_tokens) == 0:
                    # 记录跳过原因（只在第一个head记录，避免重复）
                    if head_idx == 0 and layer_idx == 0:
                        if step_idx not in step_skipped_reasons:
                            step_skipped_reasons[step_idx] = []
                        reason = f"'{current_word}': Bu^+和Bu^-都为空（所有head都被跳过）"
                        step_skipped_reasons[step_idx].append(reason)

                        # 打印详细的调试信息
                        print(f"      ⚠️  [调试] Step {step_idx}, Layer {layer_idx}, Head {head_idx}: Bu^+和Bu^-都为空（去重后）")
                        print(f"         当前token: '{current_word}' (类型: {current_token_type})")
                        print(f"         Bu^+ = {list(step_bu_plus_tokens)} (数量: {len(step_bu_plus_tokens)})")
                        print(f"         Bu^- = {list(step_bu_minus_tokens)} (数量: {len(step_bu_minus_tokens)})")
                        print(f"         原因：两者都为空，无法确定 head 的真实倾向，跳过")
                    continue

                # 成功处理了这个head，记录统计
                if step_idx not in step_processed_count:
                    step_processed_count[step_idx] = 0
                step_processed_count[step_idx] += 1

                # 计算概率增益（使用指数函数，避免log计算时的数值不稳定问题）
                # 注意：如果 Bu 为空集合，compute_log_probability_gain 会自动返回 0（中性假设）
                # 需要分别把Bu^+和Bu^-中所有词汇的概率加起来，然后使用指数函数计算增益
                delta_log_p_plus = compute_log_probability_gain(
                    logits_with_head[0], logits_before[0], step_bu_plus_tokens
                )
                delta_log_p_minus = compute_log_probability_gain(
                    logits_with_head[0], logits_before[0], step_bu_minus_tokens
                )

                # 计算语义先验偏置分数
                s_u = delta_log_p_minus - delta_log_p_plus

                # 检查s_u是否为nan或inf
                if np.isnan(s_u) or np.isinf(s_u):
                    # 如果s_u是nan或inf，跳过这个head
                    if head_idx == 0 and layer_idx == 0:
                        print(f"      ⚠️  [警告] Step {step_idx}, Layer {layer_idx}, Head {head_idx}: s_u为nan/inf，跳过")
                        print(f"         delta_log_p_plus: {delta_log_p_plus}, delta_log_p_minus: {delta_log_p_minus}")
                    continue

                # 计算SPP输出g
                g_u = tanh(ALPHA * s_u + BETA)

                # 检查g_u是否为nan或inf
                if np.isnan(g_u) or np.isinf(g_u):
                    # 如果g_u是nan或inf，跳过这个head
                    if head_idx == 0 and layer_idx == 0:
                        print(f"      ⚠️  [警告] Step {step_idx}, Layer {layer_idx}, Head {head_idx}: g_u为nan/inf，跳过")
                        print(f"         s_u: {s_u}, ALPHA: {ALPHA}, BETA: {BETA}")
                    continue

                # 保存真值对
                pair = {
                    "case_id": case["question_id"],
                    "step": step_idx,  # 添加生成步骤信息
                    "layer": layer_idx,
                    "head": head_idx,
                    "s_u": float(s_u),
                    "g_u": float(g_u),
                    "delta_log_p_plus": float(delta_log_p_plus),
                    "delta_log_p_minus": float(delta_log_p_minus),
                    "is_approximated": is_approximated,  # 标记是否为近似值（使用了空集合的 delta_log_p = 0）
                    "case_type": "CHAIR"
                }

                # 添加head的原始输出向量
                # head_raw_vector 已经在循环开始时计算（第1170行）
                # 确保它被正确保存
                if 'head_raw_vector' in locals() and head_raw_vector is not None:
                    # head_raw_vector 已经是numpy数组，直接转换为list
                    if isinstance(head_raw_vector, np.ndarray):
                        pair["head_vector"] = head_raw_vector.tolist()
                    else:
                        pair["head_vector"] = list(head_raw_vector)
                else:
                    # 如果head_raw_vector不存在或为None，重新计算
                    head_raw_vector = head_attn_output[:, 0, :].detach().cpu().numpy()
                    pair["head_vector"] = head_raw_vector.tolist()

                ground_truth_pairs.append(pair)

    # 统计信息
    if len(ground_truth_pairs) == 0:
        print(f"  ⚠️  警告: 未生成任何真值对")
        print(f"     可能原因:")
        print(f"     1. 所有生成步骤的Bu^+或Bu^-都为空")
        print(f"     2. top-K中没有找到有效的物理词汇")
        print(f"     3. 所有步骤都被continue跳过")

        # 打印详细的步骤统计信息
        if step_skipped_reasons:
            print(f"     详细统计:")
            for step_idx in sorted(step_skipped_reasons.keys()):
                reasons = step_skipped_reasons[step_idx]
                processed = step_processed_count.get(step_idx, 0)
                print(f"       Step {step_idx}: 处理了 {processed} 个head")
                # 只显示前3个唯一原因，避免输出过长
                unique_reasons = list(set(reasons))[:3]
                for reason in unique_reasons:
                    print(f"          - {reason}")
    else:
        print(f"  ✓ 生成了 {len(ground_truth_pairs)} 个真值对")
        # 打印简要统计
        if step_processed_count:
            total_processed = sum(step_processed_count.values())
            expected_total = len(target_steps) * num_layers * num_heads
            total_skipped = expected_total - total_processed
            print(f"     步骤统计: 总共处理了 {total_processed} 个head（预期: {expected_total}，跳过: {total_skipped}）")
            if step_skipped_reasons:
                total_skipped_steps = len(step_skipped_reasons)
                print(f"     有 {total_skipped_steps} 个步骤的部分head被跳过:")
                for step_idx in sorted(step_skipped_reasons.keys()):
                    reasons = step_skipped_reasons[step_idx]
                    processed = step_processed_count.get(step_idx, 0)
                    expected = num_layers * num_heads
                    skipped = expected - processed
                    # 只显示第一个唯一原因，避免输出过长
                    unique_reason = list(set(reasons))[0] if reasons else "未知原因"
                    print(f"       Step {step_idx}: 处理了 {processed}/{expected} 个head，跳过 {skipped} 个 - {unique_reason}")

    # 禁用 attention output 提取功能
    enable_attn_output_extraction(model, enable=False)

    return ground_truth_pairs


def main():
    parser = argparse.ArgumentParser(description="生成head级别的真值对")
    parser.add_argument("--train-file", type=str, default=None,
                       help="训练case文件路径（coco_train_*.json）")
    parser.add_argument("--model-path", type=str, default=project.llava_v15_7b_path,
                       help="模型路径")
    parser.add_argument("--coco-root", type=str, default=project.coco_data_path,
                       help="COCO数据集根目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备")
    parser.add_argument("--num-samples", type=int, default=0,
                       help="处理的样本数量（0表示全部）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--chair-cache", type=str, default=None,
                       help="CHAIR评估器缓存文件路径（用于加速重复运行）")
    parser.add_argument("--use-np-exp", type=lambda x: (str(x).lower() == 'true'), default=False,
                       help="是否使用np.exp计算概率增益（默认True，使用np.exp；False则使用np.log）")

    args = parser.parse_args()
    set_seed(args.seed)

    # 如果通过命令行指定了use_np_exp，更新全局变量
    global USE_NP_EXP
    if args.use_np_exp is not None:
        USE_NP_EXP = args.use_np_exp

    if args.train_file is None:
        args.train_file = os.path.join(coco_train_json_dir, f"coco_train_200.json")

    print("=" * 80)
    print("生成 Head 级别真值对")
    print("=" * 80)
    print(f"训练文件: {args.train_file}")
    print(f"模型路径: {args.model_path}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"设备: {args.device}")
    print(f"概率增益计算方法: {'np.exp' if USE_NP_EXP else 'np.log'}")
    if USE_NP_EXP:
        print(f"指数增益系数: {EXP_GAIN_COEFF}")
    print("=" * 80)

    # 加载训练cases
    print("\n[1/5] 加载训练cases...")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    if args.num_samples > 0:
        cases = cases[:args.num_samples]

    print(f"✓ 加载了 {len(cases)} 个cases")

    # 加载模型
    print("\n[2/5] 加载模型...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    device = args.device

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device
    )
    print(f"✓ 模型加载完成: {model_name}")

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 获取模型配置
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32

    print(f"✓ 模型配置: {num_layers} 层, 每层 {num_heads} 个head, 总共 {num_layers * num_heads} 个head")

    # 创建head输出提取器
    print(f"\n[3/5] 初始化head输出提取器...")
    extractor = HeadOutputExtractor(model, num_layers, num_heads)
    extractor.register_hooks()
    print(f"✓ 已注册 {len(extractor.hooks)} 个hooks")

    # 初始化CHAIR评估器（使用缓存机制加速）
    print(f"\n[3.5/5] 初始化CHAIR评估器...")
    from eval_tool.chair import get_chair_evaluator
    coco_annotations_path = os.path.join(args.coco_root, "annotations")

    # 设置缓存文件路径
    if args.chair_cache is None:
        # 默认使用 eval_tool 目录下已存在的缓存文件
        eval_tool_dir = os.path.join(project_root, "eval_tool")
        default_cache = os.path.join(eval_tool_dir, "chair_evaluator.pkl")
        # 如果默认缓存文件不存在，使用 train 目录下的缓存
        if not os.path.exists(default_cache):
            cache_dir = Path(__file__).parent
            default_cache = os.path.join(cache_dir, "chair_evaluator_cache.pkl")
        args.chair_cache = default_cache

    chair_evaluator = get_chair_evaluator(
        coco_path=coco_annotations_path,
        cache_file=args.chair_cache,
        use_cache=True
    )
    print(f"✓ CHAIR评估器已就绪（缓存文件: {args.chair_cache}）")

    # 处理每个case
    print(f"\n[4/5] 处理cases并生成真值对...")
    all_ground_truth_pairs = []

    # 统计信息：追踪哪些case返回了空列表
    empty_case_stats = {
        'total_cases': len(cases),
        'processed_cases': 0,
        'empty_cases': 0,
        'cases_with_pairs': 0,
        'total_pairs': 0
    }

    for idx, case in enumerate(tqdm(cases, desc="处理进度")):
        # 打印case基本信息（索引从0开始，但显示时从1开始）
        case_id = case.get("question_id", case.get("image_id", idx))
        case_text = case.get("text", "N/A")
        case_label = case.get("label", "N/A")
        case_image = case.get("image", "N/A")
        image_path = os.path.join(args.coco_root, "val2014", case_image) if case_image != "N/A" else "N/A"

        print(f"\n{'='*80}")
        print(f"处理 Case #{case_id} (索引: {idx + 1}/{len(cases)})")
        print(f"  图像路径: {image_path}")
        print(f"  文本 (text): {case_text}")
        print(f"  标签 (label): {case_label}")
        print(f"{'='*80}")

        # 判断case类型
        case_type = "CHAIR" if "describe" in case["text"].lower() else "POPE"

        try:
            empty_case_stats['processed_cases'] += 1

            if case_type == "POPE":
                pairs = process_case_pope(
                    model, tokenizer, image_processor, case, args.coco_root,
                    device, conv_mode, num_layers, num_heads, extractor
                )
            else:
                pairs = process_case_chair(
                    model, tokenizer, image_processor, case, args.coco_root,
                    device, conv_mode, num_layers, num_heads, extractor, chair_evaluator
                )

            # 记录处理结果
            if len(pairs) == 0:
                empty_case_stats['empty_cases'] += 1
                print(f"  ⚠️  警告: Case #{case_id} 处理完成，但未生成任何真值对（可能所有步骤都被跳过）")
            else:
                empty_case_stats['cases_with_pairs'] += 1
                empty_case_stats['total_pairs'] += len(pairs)
                print(f"  ✓ Case #{case_id} 生成了 {len(pairs)} 个真值对")

            all_ground_truth_pairs.extend(pairs)
        except Exception as e:
            empty_case_stats['empty_cases'] += 1
            print(f"  ❌ 错误: Case #{case_id} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 继续处理下一个case，不中断整个流程

    # 移除hooks
    extractor.remove_hooks()
    print(f"✓ 已移除所有hooks")

    # 打印处理统计信息
    print(f"\n{'='*80}")
    print(f"处理统计:")
    print(f"  总case数: {empty_case_stats['total_cases']}")
    print(f"  已处理case数: {empty_case_stats['processed_cases']}")
    print(f"  生成真值对的case数: {empty_case_stats['cases_with_pairs']}")
    print(f"  未生成真值对的case数: {empty_case_stats['empty_cases']}")
    print(f"  总真值对数量: {empty_case_stats['total_pairs']}")
    print(f"{'='*80}")

    if len(all_ground_truth_pairs) == 0:
        print(f"\n⚠️  警告: 所有case都未生成真值对！")
        print(f"  可能原因:")
        print(f"  1. 所有图像文件都不存在")
        print(f"  2. 所有case的生成文本都为空")
        print(f"  3. 所有case都无法获取hidden states")
        print(f"  4. 所有case都没有找到物理词汇对应的生成步骤")
        print(f"  5. 所有case的所有步骤的Bu^+或Bu^-都为空")
        print(f"  请检查上面的警告信息，找出具体原因")
    else:
        print(f"\n✓ 生成了 {len(all_ground_truth_pairs)} 个head真值对")

    # 保存结果（按layer和head分文件保存）
    print(f"\n[5/5] 保存结果...")
    train_file_name = os.path.basename(args.train_file).split(".")[0]
    script_name = "generate_spp_gt_pair" + ("_np_exp" if USE_NP_EXP else "_np_log")  # 脚本名字
    output_dir = os.path.join(coco_train_json_dir, f"{train_file_name}_{script_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")

    # 按(layer, head)分组真值对
    pairs_by_layer_head = defaultdict(list)
    for pair in all_ground_truth_pairs:
        layer_idx = pair["layer"]
        head_idx = pair["head"]
        pairs_by_layer_head[(layer_idx, head_idx)].append(pair)

    # 保存每个(layer, head)的真值对到单独的文件
    saved_files = 0
    for (layer_idx, head_idx), pairs in pairs_by_layer_head.items():
        filename = f"layer_{layer_idx}_head_{head_idx}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        saved_files += 1

    print(f"✓ 已保存 {saved_files} 个文件（每个文件对应一个layer-head组合）")
    print(f"✓ 结果已保存到目录: {output_dir}")

    # 打印统计信息
    print("\n统计信息")
    print("=" * 80)
    print(f"总真值对数量: {len(all_ground_truth_pairs)}")

    # 按层和head统计
    layer_head_count = defaultdict(int)
    for pair in all_ground_truth_pairs:
        key = (pair["layer"], pair["head"])
        layer_head_count[key] += 1

    print(f"覆盖的 (layer, head) 组合数: {len(layer_head_count)}")
    print(f"理论上的 (layer, head) 组合数: {num_layers * num_heads}")
    print(f"已保存文件数: {saved_files}")

    # 每个head的真值对数量统计
    head_pair_counts = list(layer_head_count.values())
    if head_pair_counts:
        print(f"\n每个head的真值对数量统计:")
        print(f"  最小值: {min(head_pair_counts)}")
        print(f"  最大值: {max(head_pair_counts)}")
        print(f"  平均值: {np.mean(head_pair_counts):.2f}")
        print(f"  中位数: {np.median(head_pair_counts):.2f}")
        print(f"  标准差: {np.std(head_pair_counts):.2f}")

        # 显示分布情况（按数量范围分组）
        count_ranges = {
            "0-50": 0,
            "51-100": 0,
            "101-150": 0,
            "151-200": 0,
            "201+": 0
        }
        for count in head_pair_counts:
            if count <= 50:
                count_ranges["0-50"] += 1
            elif count <= 100:
                count_ranges["51-100"] += 1
            elif count <= 150:
                count_ranges["101-150"] += 1
            elif count <= 200:
                count_ranges["151-200"] += 1
            else:
                count_ranges["201+"] += 1

        print(f"  分布情况:")
        for range_name, count in count_ranges.items():
            if count > 0:
                print(f"    {range_name}: {count} 个head")

    # 按case类型统计
    pope_count = sum(1 for p in all_ground_truth_pairs if p["case_type"] == "POPE")
    chair_count = sum(1 for p in all_ground_truth_pairs if p["case_type"] == "CHAIR")
    print(f"\n按case类型统计:")
    print(f"  POPE类型真值对: {pope_count}")
    print(f"  CHAIR类型真值对: {chair_count}")

    # 检查head_vector字段
    has_head_vector = sum(1 for p in all_ground_truth_pairs if "head_vector" in p)
    if has_head_vector > 0:
        # 检查head_vector的维度
        sample_vector = next((p["head_vector"] for p in all_ground_truth_pairs if "head_vector" in p), None)
        if sample_vector:
            head_dim = len(sample_vector)
            print(f"\nHead向量信息:")
            print(f"  包含head_vector的真值对: {has_head_vector}/{len(all_ground_truth_pairs)}")
            print(f"  Head向量维度: {head_dim}")

    # g_u的统计
    g_values_raw = [p["g_u"] for p in all_ground_truth_pairs if "g_u" in p]
    # 过滤掉nan和inf值
    g_values = [v for v in g_values_raw if not (np.isnan(v) or np.isinf(v))]
    nan_count_g = len(g_values_raw) - len(g_values)

    if len(g_values) > 0:
        print(f"\ng_u 统计:")
        print(f"  总数据点: {len(g_values_raw)}")
        if nan_count_g > 0:
            print(f"  ⚠️  nan/inf值数量: {nan_count_g}")
        print(f"  有效数据点: {len(g_values)}")
        print(f"  平均值: {np.mean(g_values):.4f}")
        print(f"  标准差: {np.std(g_values):.4f}")
        print(f"  最小值: {np.min(g_values):.4f}")
        print(f"  最大值: {np.max(g_values):.4f}")
    else:
        print(f"\ng_u 统计: 无有效数据")
        if len(g_values_raw) > 0:
            print(f"  总数据点: {len(g_values_raw)}")
            print(f"  ⚠️  所有值都是nan/inf")
        else:
            print(f"  未生成任何真值对")

    # s_u的统计
    s_values_raw = [p["s_u"] for p in all_ground_truth_pairs if "s_u" in p]
    # 过滤掉nan和inf值
    s_values = [v for v in s_values_raw if not (np.isnan(v) or np.isinf(v))]
    nan_count_s = len(s_values_raw) - len(s_values)

    if len(s_values) > 0:
        print(f"\ns_u 统计:")
        print(f"  总数据点: {len(s_values_raw)}")
        if nan_count_s > 0:
            print(f"  ⚠️  nan/inf值数量: {nan_count_s}")
        print(f"  有效数据点: {len(s_values)}")
        print(f"  平均值: {np.mean(s_values):.4f}")
        print(f"  标准差: {np.std(s_values):.4f}")
        print(f"  最小值: {np.min(s_values):.4f}")
        print(f"  最大值: {np.max(s_values):.4f}")
    else:
        print(f"\ns_u 统计: 无有效数据")
        if len(s_values_raw) > 0:
            print(f"  总数据点: {len(s_values_raw)}")
            print(f"  ⚠️  所有值都是nan/inf")
        else:
            print(f"  未生成任何真值对")

    # 文件大小统计
    total_size = 0
    for (layer_idx, head_idx), pairs in pairs_by_layer_head.items():
        filename = f"layer_{layer_idx}_head_{head_idx}.json"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            total_size += os.path.getsize(filepath)

    print(f"\n文件大小统计:")
    print(f"  总文件大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"  平均每个文件: {total_size / saved_files / 1024:.2f} KB" if saved_files > 0 else "  平均每个文件: N/A")
    print("=" * 80)


if __name__ == "__main__":
    main()
