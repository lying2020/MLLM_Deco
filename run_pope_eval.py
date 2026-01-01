#!/usr/bin/env python3
"""
POPE 评估脚本 - 直接运行版本
基于 test_llava_v15_7b.py 的实现，针对 POPE benchmark 优化
自动检测数据集和模型，使用默认参数，无需输入参数即可运行
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
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

from project import llava_v15_7b_path
from eval_tool.eval_pope import evaluate_pope

from PIL import Image
import re
from transformers import set_seed


def recorder(out):
    """将输出转换为 Yes/No"""
    if not out or not out.strip():
        # 如果输出为空，返回 "No"（但应该记录警告）
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
        # 但这种情况应该被记录
        return "No"


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
    """
    从 probe_exp/train_set 目录读取问题文件

    Args:
        probe_exp_dir: probe_exp 目录路径
        split: split 名称 (adversarial, popular, random)
        coco_root: COCO 数据集根目录
    """
    probe_exp_dir = Path(probe_exp_dir)
    coco_root = Path(coco_root)

    # 构建问题文件路径
    question_file = probe_exp_dir / f"coco_pope_{split}.json"

    if not question_file.exists():
        raise FileNotFoundError(f"找不到问题文件: {question_file}")

    all_questions = []

    # 读取 JSONL 格式的文件（每行一个 JSON）
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # 构建完整的图像路径
            # sample['image'] 格式: "val2014/COCO_val2014_000000031041.jpg"
            image_relative_path = sample['image']
            image_path = coco_root / image_relative_path
            image_path = image_path.resolve()

            # 检查图像文件是否存在
            if not image_path.exists():
                print(f"⚠️  警告: 图像文件不存在: {image_path}")
                continue

            all_questions.append({
                "question_id": sample['question_id'],
                "image": str(image_path),
                "text": sample['text']
            })

    # 按 question_id 排序
    all_questions.sort(key=lambda x: x['question_id'])

    return all_questions


def auto_generate_gt_file(probe_exp_dir="probe_exp/train_set", split="adversarial", coco_root="/home/liying/Documents/dataset/coco", output_file=None, results_dir=None):
    """
    从 probe_exp/train_set 目录读取真值（Ground Truth）文件

    Args:
        probe_exp_dir: probe_exp 目录路径
        split: split 名称 (adversarial, popular, random)
        coco_root: COCO 数据集根目录
        output_file: 输出文件路径（如果不指定，将自动生成）
        results_dir: 结果目录（如果不指定，将使用默认路径）

    Returns:
        真值文件路径
    """
    probe_exp_dir = Path(probe_exp_dir)
    coco_root = Path(coco_root)

    # 构建问题文件路径（GT 数据也在同一个文件中）
    gt_file = probe_exp_dir / f"coco_pope_{split}.json"

    if not gt_file.exists():
        raise FileNotFoundError(f"找不到真值文件: {gt_file}")

    if output_file is None:
        # 如果未指定输出文件，保存到 results 目录
        if results_dir is None:
            project_root = Path(__file__).parent
            results_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"pope_gt_{split}.json")
    else:
        # 如果指定了输出文件，确保目录存在
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        os.makedirs(output_dir, exist_ok=True)

    all_gt_data = []

    # 读取 JSONL 格式的文件（每行一个 JSON）
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # 构建完整的图像路径
            # sample['image'] 格式: "val2014/COCO_val2014_000000031041.jpg"
            image_relative_path = sample['image']
            image_path = coco_root / image_relative_path
            image_path = image_path.resolve()

            all_gt_data.append({
                "question_id": sample['question_id'],
                "image": str(image_path),
                "text": sample['text'],
                "label": sample['label'].lower().strip()  # 确保 label 是小写
            })

    # 按 question_id 排序
    all_gt_data.sort(key=lambda x: x['question_id'])

    # 保存真值文件（JSONL 格式，每行一个 JSON）
    with open(output_file, 'w', encoding='utf-8') as f:
        for gt_item in all_gt_data:
            f.write(json.dumps(gt_item, ensure_ascii=False) + '\n')

    print(f"✓ 已生成真值文件: {output_file} ({len(all_gt_data)} 个样本)")
    return output_file


def auto_generate_question_file(probe_exp_dir="probe_exp/train_set", split="adversarial", coco_root="/home/liying/Documents/dataset/coco", output_file=None, results_dir=None):
    """
    自动生成问题文件

    Returns:
        问题文件路径
    """
    if output_file is None:
        # 如果未指定输出文件，保存到 results 目录
        if results_dir is None:
            project_root = Path(__file__).parent
            results_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"pope_questions_{split}.jsonl")
    else:
        # 如果指定了输出文件，确保目录存在
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        os.makedirs(output_dir, exist_ok=True)

    questions = auto_detect_questions(probe_exp_dir, split, coco_root)

    # 保存问题文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"✓ 已生成问题文件: {output_file} ({len(questions)} 个问题)")
    return output_file


def prepare_inputs(model, tokenizer, image_processor, image_file: str, prompt: str, conv_mode: str, device: str, verbose: bool = False):
    """
    准备模型输入（参考 test_llava_v15_7b.py）

    Returns:
        input_ids, image_tensor, stopping_criteria
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

    # 对于 Yes/No 问题，添加明确的输出格式说明（参考 pope_llava.py）
    # 这有助于模型生成更简洁的回答
    qs = qs + " Please answer with Yes or No only."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    if verbose:
        print(f"  [输入准备] 文本信息:")
        print(f"    - 原始提示词: {prompt}")
        print(f"    - 完整提示词长度: {len(full_prompt)} 字符")
        print(f"    - 完整提示词预览: {full_prompt[:200]}...")

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors='pt').unsqueeze(0).to(device)

    if verbose:
        print(f"  [输入准备] Token 信息:")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - input_ids 长度: {input_ids.shape[1]} tokens")
        # 解码前几个 token 看看
        decoded_input = tokenizer.decode(input_ids[0, :20], skip_special_tokens=False)
        print(f"    - 前20个 tokens 解码: {decoded_input}")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if verbose:
        print(f"  [输入准备] 停止条件:")
        print(f"    - 停止字符串: '{stop_str}'")

    return input_ids, image_tensor, stopping_criteria, stop_str


def generate_response(model, tokenizer, input_ids, image_tensor, stopping_criteria,
                     temperature, top_p, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, num_beams=1, verbose: bool = False):
    """
    生成回答（参考 test_llava_v15_7b.py）

    Returns:
        outputs: 生成的文本
        output_token_len: 生成的 token 长度
        input_token_len: 输入的 token 长度
    """
    do_sample = True if temperature > 0 else False

    if verbose:
        print(f"\n  [生成参数] 配置信息:")
        print(f"    - use_deco: {use_deco}")
        print(f"    - do_sample: {do_sample}")
        print(f"    - temperature: {temperature if temperature > 0 else 'None (greedy)'}")
        print(f"    - top_p: {top_p}")
        print(f"    - num_beams: {num_beams}")
        print(f"    - max_new_tokens: {max_new_tokens}")
        if use_deco:
            print(f"    - alpha: {alpha}")
            print(f"    - threshold_top_p: {threshold_top_p}")
            print(f"    - threshold_top_k: {threshold_top_k}")
            print(f"    - early_exit_layers: {early_exit_layers}")

    # 准备生成参数（完全参考 test_llava_v15_7b.py 的实现）
    # LLaVA 的 generate 方法使用 inputs 作为关键字参数
    generate_kwargs = {
        "inputs": input_ids,  # 注意：LLaVA 使用 inputs 参数名（与 test_llava_v15_7b.py 保持一致）
        "images": image_tensor.unsqueeze(0).half().to(device),
        "do_sample": do_sample,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "num_beams": num_beams,  # 添加 num_beams 参数（重要！）
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

    if verbose:
        print(f"  [生成过程] 开始生成...")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - images 形状: {image_tensor.unsqueeze(0).half().to(device).shape}")

    with torch.inference_mode():
        with torch.no_grad():
            # 使用 **generate_kwargs 展开所有参数（与 test_llava_v15_7b.py 保持一致）
            output_dict = model.generate(**generate_kwargs)

    # 解码输出（完全参考 test_llava_v15_7b.py 的实现）
    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]

    # 检查 output_ids 是否包含 input_ids（与 test_llava_v15_7b.py 保持一致）
    if verbose:
        print(f"\n  [生成过程] 检查 output_ids 内容:")
        print(f"    - output_ids 形状: {output_ids.shape}")
        print(f"    - input_ids 形状: {input_ids.shape}")
        print(f"    - output_ids 前几个 token: {output_ids[0, :min(5, output_ids.shape[1])].tolist()}")
        print(f"    - input_ids 前几个 token: {input_ids[0, :min(5, input_ids.shape[1])].tolist()}")

        # 检查 output_ids 是否以 input_ids 开头
        if output_ids.shape[1] >= input_token_len:
            prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
            print(f"    - output_ids 前 {input_token_len} 个 token 是否与 input_ids 匹配: {prefix_match}")
        else:
            print(f"    - ⚠️  警告: output_ids 长度 ({output_ids.shape[1]}) < input_ids 长度 ({input_token_len})")
            print(f"    - 这说明 output_ids 可能只包含新生成的 token，而不是完整序列")
            print(f"    - 需要手动拼接 input_ids 和 output_ids")

    # 如果 output_ids 不包含 input_ids，手动拼接（与 test_llava_v15_7b.py 保持一致）
    if output_ids.shape[1] < input_token_len:
        # output_ids 只包含新生成的 token，需要拼接 input_ids
        if verbose:
            print(f"\n  [修复] 手动拼接 input_ids 和 output_ids:")
            print(f"    - input_ids: {input_ids.shape}")
            print(f"    - output_ids (仅新生成的): {output_ids.shape}")
        output_ids = torch.cat([input_ids, output_ids], dim=1)
        if verbose:
            print(f"    - 拼接后的 output_ids: {output_ids.shape}")
    elif output_ids.shape[1] >= input_token_len:
        # 检查前 input_token_len 个 token 是否与 input_ids 匹配
        prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
        if not prefix_match:
            if verbose:
                print(f"\n  [修复] output_ids 前缀与 input_ids 不匹配，使用 input_ids 替换前缀")
            # 替换前缀为 input_ids
            output_ids = torch.cat([input_ids, output_ids[:, input_token_len:]], dim=1)

    output_token_len = output_ids.shape[1] - input_token_len

    if verbose:
        print(f"\n  [生成结果] Token 信息:")
        print(f"    - input_ids length: {input_token_len}")
        print(f"    - output_ids 形状: {output_ids.shape}")
        print(f"    - output_ids length: {output_ids.shape[1]}")
        print(f"    - new generated tokens length: {output_token_len}")

        # 显示新生成的 token IDs
        if output_token_len > 0:
            generated_ids = output_ids[:, input_token_len:]
            print(f"    - 生成的 token IDs 形状: {generated_ids.shape}")
            print(f"    - 生成的 token IDs: {generated_ids[0].tolist()}")
            try:
                generated_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
                print(f"    - 生成的 token 解码（带特殊token）: {repr(generated_decoded)}")
            except Exception as e:
                print(f"    - 生成的 token 解码失败: {e}")
        else:
            print(f"    - ⚠️  警告: 没有生成新的 token！")

    # 获取新生成的 token（跳过可能的 BOS token，与 test_llava_v15_7b.py 保持一致）
    if output_token_len > 0:
        generated_ids = output_ids[:, input_token_len:]
        # 如果新生成的 token 以 BOS token 开头，跳过它
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
        if bos_token_id is not None and generated_ids.shape[1] > 0 and generated_ids[0, 0].item() == bos_token_id:
            if verbose:
                print(f"\n  [生成结果] 检测到新生成的 token 以 BOS token ({bos_token_id}) 开头，跳过它")
            generated_ids = generated_ids[:, 1:]  # 跳过第一个 BOS token
            if generated_ids.shape[1] > 0:
                outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            else:
                outputs = ""
        else:
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    else:
        outputs = ""

    if verbose:
        print(f"\n  [生成结果] 最终输出:")
        raw_output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0] if output_token_len > 0 else '(empty)'
        print(f"    - 原始输出 (带特殊token): {raw_output}")
        print(f"    - 最终输出 (去特殊token): {repr(outputs)}")
        print(f"    - 输出长度: {len(outputs)} 字符")

    return outputs, output_token_len, input_token_len


def compare_deco_vs_vanilla(deco_results, vanilla_results, deco_answers_file, vanilla_answers_file,
                            gt_file, output_file):
    """
    对比 Deco 和 Vanilla 的结果，生成对比表格和不一致 case 的 JSON 文件

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        deco_answers_file: Deco 版本的答案文件路径
        vanilla_answers_file: Vanilla 版本的答案文件路径
        gt_file: 真值文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载答案文件
    deco_answers = {item["question_id"]: item for item in [json.loads(line) for line in open(deco_answers_file, 'r', encoding='utf-8')]}
    vanilla_answers = {item["question_id"]: item for item in [json.loads(line) for line in open(vanilla_answers_file, 'r', encoding='utf-8')]}
    gt_data = {item["question_id"]: item for item in [json.loads(line) for line in open(gt_file, 'r', encoding='utf-8')]}

    # 找到结果不一致的 case
    inconsistent_cases = []
    common_question_ids = set(deco_answers.keys()) & set(vanilla_answers.keys())

    for qid in common_question_ids:
        deco_answer = deco_answers[qid].get("text", "").strip()
        vanilla_answer = vanilla_answers[qid].get("text", "").strip()

        if deco_answer != vanilla_answer:
            # 获取图片文件名（不包含路径）
            image_path = deco_answers[qid].get("image", "")
            image_filename = os.path.basename(image_path) if image_path else ""

            # 获取 GT 答案
            gt_answer = gt_data.get(qid, {}).get("label", "").strip().lower()

            case_info = {
                "question_id": qid,
                "question": deco_answers[qid].get("prompt", ""),
                "image": image_filename,  # 只保存文件名
                "gt_answer": gt_answer,
                "vanilla_answer": vanilla_answer,
                "deco_answer": deco_answer,
                "vanilla_correct": vanilla_answer.lower() == gt_answer,
                "deco_correct": deco_answer.lower() == gt_answer,
                "vanilla_raw_output": vanilla_answers[qid].get("metadata", {}).get("raw_output", ""),
                "deco_raw_output": deco_answers[qid].get("metadata", {}).get("raw_output", "")
            }
            inconsistent_cases.append(case_info)

    # 保存不一致的 case 到 JSON 文件
    comparison_result = {
        "summary": {
            "total_cases": len(common_question_ids),
            "inconsistent_cases": len(inconsistent_cases),
            "consistent_cases": len(common_question_ids) - len(inconsistent_cases),
            "inconsistency_rate": len(inconsistent_cases) / len(common_question_ids) if len(common_question_ids) > 0 else 0
        },
        "metrics_comparison": {
            "vanilla": {
                "accuracy": vanilla_results.get('metrics', {}).get('accuracy', 0),
                "precision": vanilla_results.get('metrics', {}).get('precision', 0),
                "recall": vanilla_results.get('metrics', {}).get('recall', 0),
                "f1": vanilla_results.get('metrics', {}).get('f1', 0)
            },
            "deco": {
                "accuracy": deco_results.get('metrics', {}).get('accuracy', 0),
                "precision": deco_results.get('metrics', {}).get('precision', 0),
                "recall": deco_results.get('metrics', {}).get('recall', 0),
                "f1": deco_results.get('metrics', {}).get('f1', 0)
            },
            "difference": {
                "accuracy": deco_results.get('metrics', {}).get('accuracy', 0) - vanilla_results.get('metrics', {}).get('accuracy', 0),
                "precision": deco_results.get('metrics', {}).get('precision', 0) - vanilla_results.get('metrics', {}).get('precision', 0),
                "recall": deco_results.get('metrics', {}).get('recall', 0) - vanilla_results.get('metrics', {}).get('recall', 0),
                "f1": deco_results.get('metrics', {}).get('f1', 0) - vanilla_results.get('metrics', {}).get('f1', 0)
            }
        },
        "inconsistent_cases": inconsistent_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    return comparison_result


def print_comparison_table(deco_results, vanilla_results, split_name=""):
    """
    打印 Deco vs Vanilla 的对比表格

    Args:
        deco_results: Deco 版本的评估结果
        vanilla_results: Vanilla 版本的评估结果
        split_name: Split 名称（可选）
    """
    deco_metrics = deco_results.get('metrics', {})
    vanilla_metrics = vanilla_results.get('metrics', {})

    title = f"Deco vs Vanilla 对比{' - ' + split_name if split_name else ''}"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'指标':<15} {'Vanilla':<12} {'Deco':<12} {'差异':<12} {'变化':<10}")
    print("-" * 80)

    metrics_list = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1')
    ]

    for metric_name, metric_key in metrics_list:
        vanilla_val = vanilla_metrics.get(metric_key, 0)
        deco_val = deco_metrics.get(metric_key, 0)
        diff = deco_val - vanilla_val
        change = f"{diff:+.4f}" if diff != 0 else "0.0000"
        change_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="

        print(f"{metric_name:<15} {vanilla_val:<12.4f} {deco_val:<12.4f} {diff:<12.4f} {change_symbol} {change}")

    print("=" * 80)


def save_summary_to_file(summary_file, args, gt_file, question_file, answers_file, errors_file,
                         results=None, model_name=None, error=None):
    """
    保存评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        gt_file: 真值文件路径
        question_file: 问题文件路径
        answers_file: 答案文件路径
        errors_file: 错误样本文件路径
        results: 评估结果字典（如果评估成功）
        model_name: 模型名称
        error: 错误信息（如果评估失败）
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("POPE 评估结果总结\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("【基本信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集 Split: {args.split}\n")
        f.write(f"模型路径: {args.model_path}\n")
        if model_name:
            f.write(f"模型名称: {model_name}\n")
        f.write(f"设备: {args.device}\n")
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
        f.write(f"Random Seed: {args.seed}\n")
        f.write("\n")

        # 文件路径
        f.write("【文件路径】\n")
        f.write("-" * 80 + "\n")
        f.write(f"真值文件 (GT): {gt_file}\n")
        f.write(f"问题文件: {question_file}\n")
        f.write(f"答案文件: {answers_file}\n")
        f.write(f"错误样本文件: {errors_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if results is not None:
            metrics = results.get('metrics', {})
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:  {metrics.get('f1', 0):.4f}\n")

            # 如果有错误样本信息
            if 'error_samples' in results:
                error_count = len(results['error_samples'])
                total_count = results.get('total_count', 0)
                f.write(f"\n错误样本数: {error_count} / {total_count}\n")
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
    """评估模型"""
    print("=" * 80)
    print("POPE 数据集评估")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"问题文件: {args.question_file}")
    print(f"答案文件: {args.answers_file}")
    if args.use_deco:
        print(f"Deco 参数: use_deco={args.use_deco}, alpha={args.alpha}, layers={args.start_layer}-{args.end_layer}")
    else:
        print(f"使用原生 LLaVA 模型（Deco 已禁用）")
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

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 加载问题
    print(f"\n[2/3] 正在加载问题文件: {args.question_file}")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    total_questions = len(questions)

    # 如果指定了评测数量，则只使用前 N 个
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"✓ 加载了 {total_questions} 个问题，将评测前 {len(questions)} 个")
    else:
        print(f"✓ 加载了 {len(questions)} 个问题，将评测全部")

    # 准备输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) if os.path.dirname(answers_file) else ".", exist_ok=True)
    ans_file = open(answers_file, "w")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个问题
    print(f"\n[3/3] 开始评估...")

    # 计算需要输出详细信息的样本索引（最多10个，均匀分布）
    total_samples = len(questions)
    max_debug_samples = min(10, total_samples)
    if total_samples > 0:
        debug_indices = set()
        if total_samples <= max_debug_samples:
            # 如果样本数少于等于10个，全部输出详细信息
            debug_indices = set(range(total_samples))
        else:
            # 均匀分布选择样本
            step = total_samples / max_debug_samples
            for i in range(max_debug_samples):
                idx = int(i * step)
                debug_indices.add(idx)

        print(f"将输出 {len(debug_indices)} 个样本的详细信息用于调试（样本索引: {sorted(debug_indices)}）")

    for sample_idx, line in enumerate(tqdm(questions, desc="处理进度")):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        # 判断是否需要输出详细信息
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_samples}] Question ID: {idx}")
            print("=" * 80)
            print(f"问题: {qs}")
            print(f"图像: {image_file}")

        try:
            # 准备输入
            if verbose:
                print("\n" + "-" * 80)
                print("[准备输入]")
                print("-" * 80)
            input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
                model, tokenizer, image_processor, image_file, qs, conv_mode, device, verbose=verbose
            )

            # 生成回答
            if verbose:
                print("\n" + "-" * 80)
                print("[生成回答]")
                print("-" * 80)
            outputs, output_token_len, input_token_len = generate_response(
                model, tokenizer, input_ids, image_tensor, stopping_criteria,
                args.temperature, args.top_p, args.max_new_tokens, device,
                use_deco=args.use_deco,
                alpha=args.alpha,
                threshold_top_p=args.threshold_top_p,
                threshold_top_k=args.threshold_top_k,
                early_exit_layers=early_exit_layers,
                num_beams=1,  # 添加 num_beams 参数，默认值为 1（贪婪搜索）
                verbose=verbose
            )

            # 移除停止字符串
            if outputs and outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # 如果输出为空，记录警告
            if not outputs:
                if verbose:
                    print(f"\n  [Warning] 问题 {idx} 生成结果为空，output_token_len={output_token_len}")
                else:
                    print(f"  [Warning] 问题 {idx} 生成结果为空，output_token_len={output_token_len}")

            # 转换为 Yes/No
            answer = recorder(outputs)

            if verbose:
                print(f"\n  [后处理] 结果转换:")
                print(f"    - 原始输出: '{outputs}'")
                print(f"    - 转换后答案: '{answer}'")
                print("=" * 80)

            # 保存结果
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": answer,
                "model_id": model_name,
                "image": image_file,
                "metadata": {
                    "output_token_len": output_token_len,
                    "input_token_len": input_token_len,
                    "raw_output": outputs
                }
            }, ensure_ascii=False) + "\n")
            ans_file.flush()

        except Exception as e:
            print(f"\n[Error] 处理问题 {idx} 时出错: {e}")
            # 保存错误信息
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": "Error",
                "model_id": model_name,
                "image": image_file,
                "metadata": {"error": str(e)}
            }, ensure_ascii=False) + "\n")
            ans_file.flush()
            continue

    ans_file.close()
    print(f"\n✓ 评估完成！结果已保存到: {answers_file}")


def main():
    """主函数 - 自动检测并使用默认配置"""
    # 项目根目录
    project_root = Path(__file__).parent

    # 自动检测可用 GPU
    if torch.cuda.is_available():
        device = 0
        device_str = "cuda:0"
    else:
        device = -1
        device_str = "cpu"
        print("⚠ 未检测到 CUDA，将使用 CPU（速度较慢）")

    # 默认配置
    default_config = {
        "model_path": llava_v15_7b_path,
        "device": device_str,
        "probe_exp_dir": str(project_root / "probe_exp" / "train_set"),
        "coco_root": "/home/liying/Documents/dataset/coco",
        "split": ["adversarial", "popular", "random"],  # 默认评估 adversarial split
        "use_deco": True,
        "alpha": 0.8,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "temperature": -1,
        "top_p": None,
        "max_new_tokens": 15,  # POPE 只需要 Yes/No，但给一些缓冲
        "num_samples": 500,
        "seed": 42
    }

    # 解析参数（所有参数都有默认值）
    parser = argparse.ArgumentParser(description="POPE 评估 - 直接运行版本（所有参数可选）")

    # 数据集参数
    # 注意：如果默认值是列表，argparse 需要特殊处理
    default_split = default_config["split"]
    if isinstance(default_split, list):
        # 如果默认值是列表，转换为逗号分隔的字符串
        default_split_str = ','.join(default_split)
    else:
        default_split_str = str(default_split)

    parser.add_argument("--split", type=str, default=default_split_str,
                       help="数据集 split，可以是单个值或逗号分隔的多个值（例如: adversarial,popular,random）")
    parser.add_argument("--probe-exp-dir", type=str, default=default_config["probe_exp_dir"],
                       help="probe_exp/train_set 目录路径")
    parser.add_argument("--coco-root", type=str, default=default_config["coco_root"],
                       help="COCO 数据集根目录路径")
    parser.add_argument("--question-file", type=str, default=None,
                       help="问题文件路径（如果不指定，将自动生成）")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--answers-file", type=str, default=None,
                       help="输出答案文件路径（如果不指定，将自动生成）")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度（-1表示贪婪生成）")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="评测数量（0表示评测所有case，非零表示只评测前N个）")

    # Deco 参数（默认不使用 Deco，只使用原生 LLaVA 模型）
    parser.add_argument("--use-deco", type=bool, default=default_config["use_deco"],
                       help="启用 Deco 早退机制（默认：False，使用原生 LLaVA 模型）")
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

    # 其他参数
    parser.add_argument("--seed", type=int, default=default_config["seed"], help="随机种子")

    args = parser.parse_args()
    set_seed(args.seed)

    # 解析 split 参数（支持单个值、逗号分隔的多个值，或列表）
    if isinstance(args.split, list):
        # 如果已经是列表，直接使用
        splits = [s.strip() if isinstance(s, str) else str(s) for s in args.split]
    elif isinstance(args.split, str):
        # 如果是字符串，检查是否包含逗号
        split_input = args.split.strip()
        if ',' in split_input:
            splits = [s.strip() for s in split_input.split(',')]
        else:
            splits = [split_input]
    else:
        # 其他类型，转换为字符串列表
        splits = [str(args.split)]

    # 验证 split 值
    valid_splits = ["adversarial", "popular", "random"]
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"无效的 split 值: {split}。有效值: {valid_splits}")

    print("=" * 80)
    print(f"将处理 {len(splits)} 个 split: {', '.join(splits)}")
    print("=" * 80)

    # 准备 results 目录
    results_dir = os.path.join(project_root, "results", "pope")

    # 存储所有结果
    all_results = []

    # 循环处理每个 split
    for split_idx, current_split in enumerate(splits, 1):
        print("\n" + "=" * 80)
        print(f"[{split_idx}/{len(splits)}] 处理 split: {current_split}")
        print("=" * 80)

        # 为当前 split 创建 args 副本
        split_args = argparse.Namespace(**vars(args))
        split_args.split = current_split

        # 自动生成真值文件（GT 文件）
        print("\n" + "-" * 80)
        print(f"自动生成真值文件（Ground Truth）- {current_split}")
        print("-" * 80)
        gt_file = auto_generate_gt_file(
            probe_exp_dir=split_args.probe_exp_dir,
            split=current_split,
            coco_root=split_args.coco_root,
            output_file=None,  # 使用默认路径（results/pope/pope_gt_{split}.json）
            results_dir=results_dir
        )

        # 自动生成问题文件（如果未指定）
        if split_args.question_file is None:
            print("\n" + "-" * 80)
            print(f"自动生成问题文件 - {current_split}")
            print("-" * 80)
            question_file = auto_generate_question_file(
                probe_exp_dir=split_args.probe_exp_dir,
                split=current_split,
                coco_root=split_args.coco_root,
                output_file=None,  # 使用默认路径（results/pope/pope_questions_{split}.jsonl）
                results_dir=results_dir
            )
            split_args.question_file = question_file
        else:
            # 如果指定了问题文件，只对第一个 split 使用，其他 split 会报错
            if split_idx > 1:
                raise ValueError(f"当处理多个 split 时，不能指定 --question-file。请移除该参数以自动生成问题文件。")
            # 检查问题文件是否存在
            if not os.path.exists(split_args.question_file):
                raise FileNotFoundError(f"问题文件不存在: {split_args.question_file}")

        # 自动生成答案文件路径（如果未指定）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        answers_dir = os.path.join(project_root, "results", "pope")
        os.makedirs(answers_dir, exist_ok=True)

        # 如果使用 Deco，需要同时运行 vanilla 版本进行对比
        vanilla_answers_file = None
        if split_args.use_deco:
            print("\n" + "=" * 80)
            print(f"检测到使用 Deco，将同时运行 Vanilla 版本进行对比")
            print("=" * 80)

            # 先运行 Vanilla 版本
            print("\n" + "-" * 80)
            print(f"[1/2] 运行 Vanilla 版本 - {current_split}")
            print("-" * 80)
            vanilla_args = argparse.Namespace(**vars(split_args))
            vanilla_args.use_deco = False
            vanilla_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_vanilla_{timestamp}.jsonl")

            eval_model(vanilla_args)
            vanilla_answers_file = vanilla_args.answers_file

            # 然后运行 Deco 版本
            print("\n" + "-" * 80)
            print(f"[2/2] 运行 Deco 版本 - {current_split}")
            print("-" * 80)
            if split_args.answers_file is None:
                split_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_deco_{timestamp}.jsonl")

            eval_model(split_args)
        else:
            # 不使用 Deco，正常处理
            if split_args.answers_file is None:
                split_args.answers_file = os.path.join(answers_dir, f"pope_{current_split}_vanilla_{timestamp}.jsonl")
            else:
                # 如果指定了答案文件，只对第一个 split 使用，其他 split 会报错
                if split_idx > 1:
                    raise ValueError(f"当处理多个 split 时，不能指定 --answers-file。请移除该参数以自动生成答案文件。")

            # 运行评估
            eval_model(split_args)

        print("\n" + "=" * 80)
        print(f"模型评估完成 - {current_split}")
        print("=" * 80)
        print(f"真值文件（GT）: {gt_file}")
        print(f"问题文件: {split_args.question_file}")
        print(f"答案文件: {split_args.answers_file}")

        # 自动执行评估
        print("\n" + "=" * 80)
        print(f"自动执行结果评估 - {current_split}")
        print("=" * 80)

        # 生成错误文件路径（保持与答案文件相同的命名规则）
        errors_file = split_args.answers_file.replace('.jsonl', '_errors.json')
        # 生成总结文件路径
        summary_file = split_args.answers_file.replace('.jsonl', '_summary.txt')

        try:
            # 直接调用评估函数
            results = evaluate_pope(
                gt_files_path=gt_file,
                gen_files_path=split_args.answers_file,
                output_errors_path=errors_file,
                verbose=True
            )

            print("\n" + "=" * 80)
            print(f"✓ 结果评估完成 - {current_split}")
            print("=" * 80)
            print(f"错误样本文件: {errors_file}")
            print(f"\n关键指标:")
            print(f"  - Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"  - Precision: {results['metrics']['precision']:.4f}")
            print(f"  - Recall: {results['metrics']['recall']:.4f}")
            print(f"  - F1: {results['metrics']['f1']:.4f}")

            # 保存总结到txt文件
            save_summary_to_file(
                summary_file=summary_file,
                args=split_args,
                gt_file=gt_file,
                question_file=split_args.question_file,
                answers_file=split_args.answers_file,
                errors_file=errors_file,
                results=results,
                model_name=get_model_name_from_path(split_args.model_path)
            )
            print(f"\n✓ 结果总结已保存到: {summary_file}")

            # 如果使用 Deco，进行对比
            if split_args.use_deco and vanilla_answers_file:
                print("\n" + "=" * 80)
                print(f"对比 Deco vs Vanilla - {current_split}")
                print("=" * 80)

                # 评估 Vanilla 版本
                vanilla_errors_file = vanilla_answers_file.replace('.jsonl', '_errors.json')
                vanilla_results = evaluate_pope(
                    gt_files_path=gt_file,
                    gen_files_path=vanilla_answers_file,
                    output_errors_path=vanilla_errors_file,
                    verbose=False  # 不重复打印详细信息
                )

                # 生成对比 JSON 文件
                comparison_file = split_args.answers_file.replace('.jsonl', '_comparison.json')
                comparison_result = compare_deco_vs_vanilla(
                    deco_results=results,
                    vanilla_results=vanilla_results,
                    deco_answers_file=split_args.answers_file,
                    vanilla_answers_file=vanilla_answers_file,
                    gt_file=gt_file,
                    output_file=comparison_file
                )

                # 打印对比表格
                print_comparison_table(deco_results=results, vanilla_results=vanilla_results, split_name=current_split)

                print(f"\n✓ 对比结果已保存到: {comparison_file}")
                print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
                print(f"  - 不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
                print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")

                # 保存结果到列表（包含对比信息）
                all_results.append({
                    'split': current_split,
                    'gt_file': gt_file,
                    'question_file': split_args.question_file,
                    'answers_file': split_args.answers_file,
                    'vanilla_answers_file': vanilla_answers_file,
                    'errors_file': errors_file,
                    'summary_file': summary_file,
                    'comparison_file': comparison_file,
                    'metrics': results['metrics'],
                    'vanilla_metrics': vanilla_results['metrics'],
                    'comparison': comparison_result
                })
            else:
                # 不使用 Deco，只保存当前结果
                all_results.append({
                    'split': current_split,
                    'gt_file': gt_file,
                    'question_file': split_args.question_file,
                    'answers_file': split_args.answers_file,
                    'errors_file': errors_file,
                    'summary_file': summary_file,
                    'metrics': results['metrics']
                })

        except Exception as e:
            print(f"\n✗ 执行评估时出错 - {current_split}: {e}")
            import traceback
            traceback.print_exc()
            print("\n可以手动运行以下命令:")
            print(f"  python3 eval_tool/eval_pope.py --gt_files {gt_file} --gen_files {split_args.answers_file} --output-errors {errors_file}")

            # 即使评估失败，也保存基本信息到总结文件
            try:
                save_summary_to_file(
                    summary_file=summary_file,
                    args=split_args,
                    gt_file=gt_file,
                    question_file=split_args.question_file,
                    answers_file=split_args.answers_file,
                    errors_file=errors_file,
                    results=None,
                    model_name=get_model_name_from_path(split_args.model_path),
                    error=str(e)
                )
                print(f"✓ 基本信息已保存到: {summary_file}")
            except Exception as save_error:
                print(f"⚠️  保存总结文件时出错: {save_error}")

            # 记录失败的结果
            all_results.append({
                'split': current_split,
                'gt_file': gt_file,
                'question_file': split_args.question_file,
                'answers_file': split_args.answers_file,
                'errors_file': errors_file,
                'summary_file': summary_file,
                'error': str(e)
            })

        print("=" * 80)

    # 打印所有结果的总结
    if len(splits) > 1:
        print("\n" + "=" * 80)
        print("所有 Split 评估总结")
        print("=" * 80)
        print(f"{'Split':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 80)
        for result in all_results:
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{result['split']:<15} {metrics.get('accuracy', 0):<12.4f} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1', 0):<12.4f}")
            else:
                print(f"{result['split']:<15} {'ERROR':<12}")
        print("=" * 80)

        # 计算平均指标（如果所有都成功）
        successful_results = [r for r in all_results if 'metrics' in r]
        if successful_results:
            avg_metrics = {
                'accuracy': sum(r['metrics']['accuracy'] for r in successful_results) / len(successful_results),
                'precision': sum(r['metrics']['precision'] for r in successful_results) / len(successful_results),
                'recall': sum(r['metrics']['recall'] for r in successful_results) / len(successful_results),
                'f1': sum(r['metrics']['f1'] for r in successful_results) / len(successful_results),
            }
            print(f"\n平均指标（{len(successful_results)} 个 split）:")
            print(f"  - Accuracy:  {avg_metrics['accuracy']:.4f}")
            print(f"  - Precision: {avg_metrics['precision']:.4f}")
            print(f"  - Recall:    {avg_metrics['recall']:.4f}")
            print(f"  - F1 Score:  {avg_metrics['f1']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
