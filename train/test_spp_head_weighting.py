#!/usr/bin/env python3
"""
测试 SPP head 权重调整对 CHAIR 基准的影响

功能：
1. 只处理 CHAIR 基准的问题
2. 对每个 CHAIR 问题：
   - 生成 caption 并计算每个 head 的 g_u
   - 注意：g_u 的计算只针对物理词汇（physical words）对应的生成步骤
   - 非物理词汇（如 "the", "a", "is" 等）对应的步骤会被跳过
   - 如果 caption 中有幻视，找出 g_u > 0.5 的 head（或 > 0.3），记为 SPP_HEAD
3. 对出现幻视增益的图片 case：
   - 再次过一遍 LLaVA，但修改 SPP_HEAD 的权重系数为 1 - g_u
   - 对比原始 caption 和修改后的 caption
   - 记录修改前后的 caption 中的幻视词汇和真实词汇（都是物理词汇）
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

# 获取当前目录（在导入之前）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入 generate_spp_gt_pair.py 中的函数和类
# 注意：需要确保 generate_spp_gt_pair.py 在同一目录下
from generate_spp_gt_pair import (
    load_image, get_vocab_tokens_for_words, filter_hidden_states,
    format_tokens_with_probs, compute_set_probability, compute_log_probability_gain,
    tanh, ALPHA, BETA, TOP_K, EXP_GAIN_COEFF, USE_NP_EXP, HeadOutputExtractor, process_case_chair
)

coco_train_json_dir = os.path.join(current_dir, "coco_train_json")
os.makedirs(coco_train_json_dir, exist_ok=True)

# SPP 阈值
SPP_THRESHOLD_HIGH = 0.5  # 高阈值
SPP_THRESHOLD_LOW = 0.3   # 低阈值


class SPPHeadWeightManager:
    """管理 SPP head 权重调整"""

    def __init__(self, model, num_layers: int, num_heads: int):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.spp_heads = {}  # {(layer_idx, head_idx): g_u} - 需要调整权重的 head
        self.original_forwards = {}  # 保存原始的 forward 方法
        self.is_patched = False

    def set_spp_heads(self, spp_heads: Dict[Tuple[int, int], float]):
        """
        设置需要调整权重的 SPP head

        Args:
            spp_heads: {(layer_idx, head_idx): g_u} - 需要调整权重的 head 及其 g_u 值
        """
        self.spp_heads = spp_heads
        print(f"  设置了 {len(spp_heads)} 个 SPP head 需要调整权重")
        if len(spp_heads) > 0:
            # 打印前10个 head 的信息
            sorted_heads = sorted(spp_heads.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"  前10个 SPP head (按 g_u 降序):")
            for (layer_idx, head_idx), g_u in sorted_heads:
                lambda_val = g_u  # lambda = g_u
                weight = 1.0 - lambda_val * 0.5
                print(f"    Layer {layer_idx}, Head {head_idx}: g_u={g_u:.4f}, lambda={lambda_val:.4f}, weight={weight:.4f}")

    def patch_attention_layers(self):
        """修改 attention 层的 forward 方法，应用 SPP head 权重调整"""
        if self.is_patched:
            print("  ⚠️  警告: Attention 层已经被 patch，跳过")
            return

        lang_model = self.model.get_model()
        if not hasattr(lang_model, 'layers'):
            raise ValueError("模型没有layers属性")

        for layer_idx in range(self.num_layers):
            if layer_idx < len(lang_model.layers):
                layer = lang_model.layers[layer_idx]
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    original_forward = attn_module.forward

                    # 保存原始 forward 方法
                    self.original_forwards[layer_idx] = original_forward

                    # 创建新的 forward 方法
                    # 使用修改后的源码，直接传递 head_weights 参数
                    # 这样就不需要手动实现整个 attention 计算了
                    def make_patched_forward(layer_idx, original_forward, num_heads_ref, spp_heads_ref):
                        def patched_forward(
                            hidden_states,
                            attention_mask=None,
                            position_ids=None,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                        ):
                            # 检查是否有需要调整权重的 head
                            has_weight_adjustment = any(
                                (layer_idx, head_idx) in spp_heads_ref
                                for head_idx in range(num_heads_ref)
                            )

                            # 如果没有权重调整，直接调用原始 forward（不传递 head_weights）
                            if not has_weight_adjustment:
                                return original_forward(
                                    hidden_states,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_value=past_key_value,
                                    output_attentions=output_attentions,
                                    use_cache=use_cache,
                                )

                            # 如果有权重调整，构建 head_weights 并传递给原始 forward
                            # head_weights 形状: [num_heads]，默认所有权重为 1.0
                            head_weights = torch.ones(num_heads_ref, device=hidden_states.device, dtype=hidden_states.dtype)

                            # 对 SPP head 应用调整后的权重
                            for head_idx in range(num_heads_ref):
                                if (layer_idx, head_idx) in spp_heads_ref:
                                    g_u = spp_heads_ref[(layer_idx, head_idx)]
                                    lambda_val = g_u
                                    weight = 1.0  - lambda_val * 0.5
                                    head_weights[head_idx] = weight

                            # 调用原始 forward，传递 head_weights 参数
                            # 注意：我们已经修改了 transformers/models/llama/modeling_llama.py，
                            # 添加了 head_weights 参数支持
                            return original_forward(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                head_weights=head_weights,
                            )

                        return patched_forward

                    # 替换forward方法（传递必要的引用）
                    attn_module.forward = make_patched_forward(layer_idx, original_forward, self.num_heads, self.spp_heads)

        self.is_patched = True
        print(f"  ✓ 已 patch {self.num_layers} 个 attention 层")

    def unpatch_attention_layers(self):
        """恢复原始的 forward 方法"""
        if not self.is_patched:
            return

        lang_model = self.model.get_model()
        if not hasattr(lang_model, 'layers'):
            return

        for layer_idx in range(self.num_layers):
            if layer_idx < len(lang_model.layers):
                layer = lang_model.layers[layer_idx]
                if hasattr(layer, 'self_attn') and layer_idx in self.original_forwards:
                    layer.self_attn.forward = self.original_forwards[layer_idx]

        self.is_patched = False
        self.original_forwards.clear()
        print(f"  ✓ 已恢复所有 attention 层的原始 forward 方法")


def generate_caption_with_spp_analysis(
    model, tokenizer, image_processor, case: Dict, coco_root: str,
    device: str, conv_mode: str, num_layers: int, num_heads: int,
    chair_evaluator
) -> Dict:
    """
    生成 caption 并分析 SPP head

    Returns:
        Dict: {
            'caption': str,
            'spp_heads': {(layer_idx, head_idx): g_u},
            'ground_truth_pairs': List[Dict],
            'has_hallucination': bool,
            'hallucinated_words': List[str],
            'grounded_words': List[str]
        }
    """
    from generate_spp_gt_pair import process_case_chair

    # 加载图像
    image_path = os.path.join(coco_root, "val2014", case["image"])
    if not os.path.exists(image_path):
        print(f"⚠️  图像文件不存在: {image_path}")
        return None

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

    # 获取真实实例词汇集合
    gt_objects = case["label"] if isinstance(case["label"], list) else [case["label"]]

    # 生成完整文本
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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

    # 解码生成的文本
    output_ids = output_dict.sequences
    generated_ids = output_ids
    output_token_len = generated_ids.shape[1]

    if output_token_len > 0:
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

    if len(generated_token_ids) > 0:
        generated_text = tokenizer.batch_decode([generated_token_ids], skip_special_tokens=True)[0].strip()
    else:
        generated_text = ""

    if generated_text and generated_text.endswith(stop_str):
        generated_text = generated_text[:-len(stop_str)].strip()

    if not generated_text:
        print(f"  ⚠️  警告: 生成文本为空")
        return None

    # 使用 CHAIR 评估器分析生成的 caption
    words, node_words, word_indices, raw_words = chair_evaluator.caption_to_words(generated_text)

    # 识别幻视词汇和真实词汇
    hallucinated_words = []
    grounded_words = []

    for word, node_word in zip(words, node_words):
        is_grounded = node_word.lower() in [obj.lower() for obj in gt_objects]
        if is_grounded:
            grounded_words.append(word)
        else:
            hallucinated_words.append(word)

    has_hallucination = len(hallucinated_words) > 0

    # 计算 SPP head（如果有幻视）
    # 注意：g_u 的计算只针对物理词汇（physical words）对应的生成步骤
    # process_case_chair 函数会：
    # 1. 使用 CHAIR 评估器识别 caption 中的物理词汇
    # 2. 只处理这些物理词汇对应的生成步骤（target_steps）
    # 3. 对于每个物理词汇的生成步骤，计算每个 head 的 g_u
    # 4. 非物理词汇（如 "the", "a", "is" 等）对应的步骤会被跳过
    spp_heads = {}
    ground_truth_pairs = []

    if has_hallucination:
        # 使用 generate_spp_gt_pair.py 中的 process_case_chair 计算 g_u
        # 注意：这里需要创建一个 HeadOutputExtractor，但实际上 process_case_chair 内部会手动计算
        # process_case_chair 只处理物理词汇对应的生成步骤，非物理词汇会被忽略
        extractor = HeadOutputExtractor(model, num_layers, num_heads)
        extractor.register_hooks()

        try:
            ground_truth_pairs = process_case_chair(
                model, tokenizer, image_processor, case, coco_root,
                device, conv_mode, num_layers, num_heads, extractor, chair_evaluator
            )
        finally:
            extractor.remove_hooks()

        # 从 ground_truth_pairs 中提取 g_u > 0.5 的 head（或 > 0.3）
        # 注意：这些 g_u 值都是基于物理词汇的生成步骤计算的
        candidate_spp_heads = {}  # 临时存储所有 g_u > 0.5 的 head

        for pair in ground_truth_pairs:
            layer_idx = pair["layer"]
            head_idx = pair["head"]
            g_u = pair["g_u"]

            # 优先选择 g_u > 0.5 的 head
            if g_u > SPP_THRESHOLD_HIGH:
                candidate_spp_heads[(layer_idx, head_idx)] = g_u
            elif g_u > SPP_THRESHOLD_LOW and len(candidate_spp_heads) == 0:
                # 如果没有 g_u > 0.5 的 head，则选择 g_u > 0.3 的 head
                candidate_spp_heads[(layer_idx, head_idx)] = g_u

        # 如果候选 head 数量 > 20，只选择 top-20（按 g_u 值降序）
        if len(candidate_spp_heads) > 20:
            sorted_heads = sorted(candidate_spp_heads.items(), key=lambda x: x[1], reverse=True)
            spp_heads = dict(sorted_heads[:20])
            print(f"  找到 {len(candidate_spp_heads)} 个候选 SPP head，选择 top-20")
        else:
            spp_heads = candidate_spp_heads

    return {
        'caption': generated_text,
        'spp_heads': spp_heads,
        'ground_truth_pairs': ground_truth_pairs,
        'has_hallucination': has_hallucination,
        'hallucinated_words': hallucinated_words,
        'grounded_words': grounded_words
    }


def generate_caption_with_weighted_heads(
    model, tokenizer, image_processor, case: Dict, coco_root: str,
    device: str, conv_mode: str, weight_manager: SPPHeadWeightManager,
    chair_evaluator
) -> Dict:
    """
    使用调整后的 head 权重生成 caption

    Returns:
        Dict: {
            'caption': str,
            'hallucinated_words': List[str],
            'grounded_words': List[str]
        }
    """
    # 加载图像
    image_path = os.path.join(coco_root, "val2014", case["image"])
    if not os.path.exists(image_path):
        print(f"⚠️  图像文件不存在: {image_path}")
        return None

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

    # 获取真实实例词汇集合
    gt_objects = case["label"] if isinstance(case["label"], list) else [case["label"]]

    # 生成完整文本（使用调整后的 head 权重）
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        with torch.no_grad():
            output_dict = model.generate(
                inputs=input_ids,
                images=images,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                use_cache=True,
                output_hidden_states=False,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria]
            )

    # 解码生成的文本
    output_ids = output_dict.sequences
    generated_ids = output_ids
    output_token_len = generated_ids.shape[1]

    if output_token_len > 0:
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

    if len(generated_token_ids) > 0:
        generated_text = tokenizer.batch_decode([generated_token_ids], skip_special_tokens=True)[0].strip()
    else:
        generated_text = ""

    if generated_text and generated_text.endswith(stop_str):
        generated_text = generated_text[:-len(stop_str)].strip()

    if not generated_text:
        print(f"  ⚠️  警告: 生成文本为空")
        return None

    # 使用 CHAIR 评估器分析生成的 caption
    words, node_words, word_indices, raw_words = chair_evaluator.caption_to_words(generated_text)

    # 识别幻视词汇和真实词汇
    hallucinated_words = []
    grounded_words = []

    for word, node_word in zip(words, node_words):
        is_grounded = node_word.lower() in [obj.lower() for obj in gt_objects]
        if is_grounded:
            grounded_words.append(word)
        else:
            hallucinated_words.append(word)

    return {
        'caption': generated_text,
        'hallucinated_words': hallucinated_words,
        'grounded_words': grounded_words
    }


def main():
    parser = argparse.ArgumentParser(description="测试 SPP head 权重调整对 CHAIR 基准的影响")
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
                       help="CHAIR评估器缓存文件路径")
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出结果文件路径（JSON格式）")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.train_file is None:
        args.train_file = os.path.join(coco_train_json_dir, f"coco_train_200.json")

    if args.output_file is None:
        args.output_file = os.path.join(coco_train_json_dir, "spp_head_weighting_test_results.json")

    print("=" * 80)
    print("测试 SPP Head 权重调整对 CHAIR 基准的影响")
    print("=" * 80)
    print(f"训练文件: {args.train_file}")
    print(f"模型路径: {args.model_path}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"设备: {args.device}")
    print(f"输出文件: {args.output_file}")
    print("=" * 80)

    # 加载训练cases
    print("\n[1/6] 加载训练cases...")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    # 只处理 CHAIR 基准的问题
    chair_cases = [case for case in cases if "describe" in case["text"].lower()]
    print(f"✓ 总共有 {len(cases)} 个cases，其中 {len(chair_cases)} 个是 CHAIR 基准")

    if args.num_samples > 0:
        chair_cases = chair_cases[:args.num_samples]

    print(f"✓ 将处理 {len(chair_cases)} 个 CHAIR cases")

    # 加载模型
    print("\n[2/6] 加载模型...")
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

    print(f"✓ 模型配置: {num_layers} 层, 每层 {num_heads} 个head")

    # 初始化CHAIR评估器
    print(f"\n[3/6] 初始化CHAIR评估器...")
    from eval_tool.chair import get_chair_evaluator
    coco_annotations_path = os.path.join(args.coco_root, "annotations")

    if args.chair_cache is None:
        eval_tool_dir = os.path.join(project_root, "eval_tool")
        default_cache = os.path.join(eval_tool_dir, "chair_evaluator.pkl")
        if not os.path.exists(default_cache):
            cache_dir = Path(__file__).parent
            default_cache = os.path.join(cache_dir, "chair_evaluator_cache.pkl")
        args.chair_cache = default_cache

    chair_evaluator = get_chair_evaluator(
        coco_path=coco_annotations_path,
        cache_file=args.chair_cache,
        use_cache=True
    )
    print(f"✓ CHAIR评估器已就绪")

    # 创建 SPP head 权重管理器
    weight_manager = SPPHeadWeightManager(model, num_layers, num_heads)

    # 处理每个 CHAIR case
    print(f"\n[4/6] 处理 CHAIR cases 并分析 SPP head...")
    results = []

    for idx, case in enumerate(tqdm(chair_cases, desc="处理进度")):
        case_id = case.get("question_id", case.get("image_id", idx))
        print(f"\n{'='*80}")
        print(f"处理 Case #{case_id} ({idx + 1}/{len(chair_cases)})")
        print(f"{'='*80}")

        # 步骤1: 生成 caption 并分析 SPP head
        print(f"\n[步骤1] 生成 caption 并分析 SPP head...")
        analysis_result = generate_caption_with_spp_analysis(
            model, tokenizer, image_processor, case, args.coco_root,
            device, conv_mode, num_layers, num_heads, chair_evaluator
        )

        if analysis_result is None:
            print(f"  ⚠️  跳过此 case（生成失败）")
            continue

        original_caption = analysis_result['caption']
        spp_heads = analysis_result['spp_heads']
        has_hallucination = analysis_result['has_hallucination']
        original_hallucinated_words = analysis_result['hallucinated_words']
        original_grounded_words = analysis_result['grounded_words']

        print(f"  原始 caption: {original_caption[:200]}...")
        print(f"  是否有幻视: {has_hallucination}")
        print(f"  幻视词汇: {original_hallucinated_words}")
        print(f"  真实词汇: {original_grounded_words}")
        print(f"  找到 {len(spp_heads)} 个 SPP head (g_u > {SPP_THRESHOLD_HIGH if len(spp_heads) > 0 and max(spp_heads.values()) > SPP_THRESHOLD_HIGH else SPP_THRESHOLD_LOW})")

        # 如果没有幻视或没有找到 SPP head，跳过
        if not has_hallucination or len(spp_heads) == 0:
            print(f"  ⚠️  跳过此 case（无幻视或无 SPP head）")
            result = {
                'case_id': case_id,
                'image': case['image'],
                'text': case['text'],
                'label': case['label'],
                # 原始 caption 及其分析结果
                'original_caption': original_caption,
                'original_hallucinated_words': original_hallucinated_words,  # 原始 caption 中的幻视词汇（物理词汇）
                'original_grounded_words': original_grounded_words,  # 原始 caption 中的真实词汇（物理词汇）
                'has_hallucination': has_hallucination,
                # SPP head 信息（基于物理词汇的生成步骤计算）
                'spp_heads_count': len(spp_heads),
                'spp_heads': {f"layer_{l}_head_{h}": float(g_u) for (l, h), g_u in spp_heads.items()},
                'note_spp_heads': 'SPP head 的 g_u 值只基于物理词汇（physical words）的生成步骤计算，非物理词汇对应的步骤被忽略',
                # 调整后的 caption 及其分析结果（跳过，未生成）
                'weighted_caption': None,
                'weighted_hallucinated_words': None,
                'weighted_grounded_words': None,
                'skipped': True,
                'skip_reason': 'no_hallucination' if not has_hallucination else 'no_spp_heads'
            }
            results.append(result)
            continue

        # 步骤2: 使用调整后的 head 权重生成 caption
        print(f"\n[步骤2] 使用调整后的 head 权重生成 caption...")
        weight_manager.set_spp_heads(spp_heads)
        weight_manager.patch_attention_layers()

        try:
            weighted_result = generate_caption_with_weighted_heads(
                model, tokenizer, image_processor, case, args.coco_root,
                device, conv_mode, weight_manager, chair_evaluator
            )
        finally:
            weight_manager.unpatch_attention_layers()

        if weighted_result is None:
            print(f"  ⚠️  跳过此 case（生成失败）")
            continue

        weighted_caption = weighted_result['caption']
        weighted_hallucinated_words = weighted_result['hallucinated_words']
        weighted_grounded_words = weighted_result['grounded_words']

        print(f"  调整后 caption: {weighted_caption[:200]}...")
        print(f"  调整后幻视词汇: {weighted_hallucinated_words}")
        print(f"  调整后真实词汇: {weighted_grounded_words}")

        # 对比结果
        hallucination_reduced = len(weighted_hallucinated_words) < len(original_hallucinated_words)
        hallucination_removed = len(weighted_hallucinated_words) == 0 and len(original_hallucinated_words) > 0

        print(f"\n[对比结果]")
        print(f"  原始幻视词汇数: {len(original_hallucinated_words)}")
        print(f"  调整后幻视词汇数: {len(weighted_hallucinated_words)}")
        print(f"  幻视是否减少: {hallucination_reduced}")
        print(f"  幻视是否完全消除: {hallucination_removed}")

        result = {
            'case_id': case_id,
            'image': case['image'],
            'text': case['text'],
            'label': case['label'],
            # 原始 caption 及其分析结果
            'original_caption': original_caption,
            'original_hallucinated_words': original_hallucinated_words,  # 原始 caption 中的幻视词汇（物理词汇）
            'original_grounded_words': original_grounded_words,  # 原始 caption 中的真实词汇（物理词汇）
            'has_hallucination': has_hallucination,
            # SPP head 信息（基于物理词汇的生成步骤计算）
            'spp_heads_count': len(spp_heads),
            'spp_heads': {f"layer_{l}_head_{h}": float(g_u) for (l, h), g_u in spp_heads.items()},
            'note_spp_heads': 'SPP head 的 g_u 值只基于物理词汇（physical words）的生成步骤计算，非物理词汇对应的步骤被忽略',
            # 调整后的 caption 及其分析结果
            'weighted_caption': weighted_caption,
            'weighted_hallucinated_words': weighted_hallucinated_words,  # 调整后 caption 中的幻视词汇（物理词汇）
            'weighted_grounded_words': weighted_grounded_words,  # 调整后 caption 中的真实词汇（物理词汇）
            # 对比结果
            'hallucination_reduced': hallucination_reduced,
            'hallucination_removed': hallucination_removed,
            'skipped': False
        }
        results.append(result)

    # 保存结果
    print(f"\n[5/6] 保存结果...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {args.output_file}")

    # 打印统计信息
    print(f"\n[6/6] 统计信息")
    print("=" * 80)
    total_cases = len(results)
    processed_cases = sum(1 for r in results if not r.get('skipped', False))
    skipped_cases = total_cases - processed_cases

    print(f"总case数: {total_cases}")
    print(f"处理case数: {processed_cases}")
    print(f"跳过case数: {skipped_cases}")

    if processed_cases > 0:
        hallucination_reduced_count = sum(1 for r in results if r.get('hallucination_reduced', False))
        hallucination_removed_count = sum(1 for r in results if r.get('hallucination_removed', False))

        print(f"\n幻视减少的case数: {hallucination_reduced_count}/{processed_cases}")
        print(f"幻视完全消除的case数: {hallucination_removed_count}/{processed_cases}")

        # 计算平均 SPP head 数量
        avg_spp_heads = np.mean([r['spp_heads_count'] for r in results if not r.get('skipped', False)])
        print(f"平均 SPP head 数量: {avg_spp_heads:.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
