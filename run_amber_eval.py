#!/usr/bin/env python3
"""
AMBER 评估脚本 - 生成答案并保存为 JSONL 格式
参考 run_chair_eval.py 和 run_mme_eval.py 的实现，针对 AMBER benchmark 优化
自动检测数据集和模型，使用默认参数，无需输入参数即可运行

AMBER (A Multimodal Benchmark for Evaluating Hallucination) 是一个评估幻觉的基准
包含生成式和判别式任务
"""

import argparse
import torch
import os
import json
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

from eval_tool.amber_infer import evaluate_amber_from_files


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
                     temperature, top_p, num_beams, max_new_tokens, device,
                     use_deco=False, alpha=None, threshold_top_p=None,
                     threshold_top_k=None, early_exit_layers=None, verbose: bool = False):
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

    # 如果 output_ids 不包含 input_ids，手动拼接
    if output_ids.shape[1] < input_token_len:
        output_ids = torch.cat([input_ids, output_ids], dim=1)
    elif output_ids.shape[1] >= input_token_len:
        # 检查前 input_token_len 个 token 是否与 input_ids 匹配
        prefix_match = (input_ids[0] == output_ids[0, :input_token_len]).all().item()
        if not prefix_match:
            output_ids = torch.cat([input_ids, output_ids[:, input_token_len:]], dim=1)

    output_token_len = output_ids.shape[1] - input_token_len

    # 获取新生成的 token
    if output_token_len > 0:
        generated_ids = output_ids[:, input_token_len:]
        # 如果新生成的 token 以 BOS token 开头，跳过它
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


def compare_deco_vs_vanilla(deco_answers_file, vanilla_answers_file, output_file):
    """
    对比 Deco 和 Vanilla 的结果，生成对比表格和不一致 case 的 JSON 文件

    Args:
        deco_answers_file: Deco 版本的答案文件路径
        vanilla_answers_file: Vanilla 版本的答案文件路径
        output_file: 输出 JSON 文件路径
    """
    # 加载答案文件
    deco_answers = {}
    with open(deco_answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("id")
            if qid is not None:
                deco_answers[qid] = item

    vanilla_answers = {}
    with open(vanilla_answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("id")
            if qid is not None:
                vanilla_answers[qid] = item

    # 找到结果不一致的 case（比较 response 字段）
    inconsistent_cases = []
    common_question_ids = set(deco_answers.keys()) & set(vanilla_answers.keys())

    for qid in common_question_ids:
        deco_response = deco_answers[qid].get("response", "").strip()
        vanilla_response = vanilla_answers[qid].get("response", "").strip()

        # 如果响应不同，记录这个 case
        if deco_response != vanilla_response:
            # 获取图片文件名（不包含路径）
            image_path = deco_answers[qid].get("image", "")
            image_filename = os.path.basename(image_path) if image_path else ""

            case_info = {
                "id": qid,
                "query": deco_answers[qid].get("query", ""),
                "image": image_filename,  # 只保存文件名
                "vanilla_response": vanilla_response,
                "deco_response": deco_response,
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
        "inconsistent_cases": inconsistent_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    return comparison_result


def print_comparison_table(deco_answers_file, vanilla_answers_file):
    """
    打印 Deco vs Vanilla 的对比表格

    Args:
        deco_answers_file: Deco 版本的答案文件路径
        vanilla_answers_file: Vanilla 版本的答案文件路径
    """
    # 加载答案文件
    deco_answers = {item["id"]: item for item in [json.loads(line) for line in open(deco_answers_file, 'r', encoding='utf-8')]}
    vanilla_answers = {item["id"]: item for item in [json.loads(line) for line in open(vanilla_answers_file, 'r', encoding='utf-8')]}

    common_question_ids = set(deco_answers.keys()) & set(vanilla_answers.keys())

    # 统计不一致的数量
    inconsistent_count = sum(1 for qid in common_question_ids
                            if deco_answers[qid].get("response", "").strip() != vanilla_answers[qid].get("response", "").strip())

    print("\n" + "=" * 80)
    print("Deco vs Vanilla 对比")
    print("=" * 80)
    print(f"总样本数: {len(common_question_ids)}")
    print(f"一致样本数: {len(common_question_ids) - inconsistent_count}")
    print(f"不一致样本数: {inconsistent_count}")
    if len(common_question_ids) > 0:
        print(f"不一致率: {inconsistent_count / len(common_question_ids):.2%}")
    print("=" * 80)


def evaluate_amber(answers_file, amber_data_path, similarity_score=0.8, evaluation_type='a'):
    """
    评估 AMBER 答案，返回评估结果
    现在调用 eval_tool/amber_infer.py 中的方法

    Args:
        answers_file: 答案文件路径（JSONL 格式）
        amber_data_path: AMBER 数据根目录
        similarity_score: 相似度阈值
        evaluation_type: 评估类型 ('a': all, 'g': generative, 'd': discriminative, 'de': existence, 'da': attribute, 'dr': relation)

    Returns:
        tuple: (results_dict, metrics_dict)
    """

    # 调用 amber_infer.py 中的评估函数
    results, metrics = evaluate_amber_from_files(
        answers_file=answers_file,
        amber_data_path=amber_data_path,
        similarity_score=similarity_score,
        evaluation_type=evaluation_type,
        answers_format='jsonl',  # run_amber_eval.py 使用 JSONL 格式
        verbose=True
    )

    return results, metrics


def print_amber_results_table(results, model_name="Model"):
    """
    打印格式化的 AMBER 评估结果表格

    Args:
        results: 评估结果字典
        model_name: 模型名称
    """
    print("\n" + "=" * 100)
    print(f"AMBER 评估结果 - {model_name}")
    print("=" * 100)

    # 生成式任务结果
    if 'generative' in results:
        gen = results['generative']
        print("\nGenerative Task:")
        print(f"  CHAIR↓:\t{gen['CHAIR']}")
        print(f"  Cover↑:\t{gen['Cover']}")
        print(f"  Hal↓:\t\t{gen['Hal']}")
        print(f"  Cog↓:\t\t{gen['Cog']}")

    # 判别式任务结果
    if 'discriminative' in results:
        disc = results['discriminative']
        print("\nDiscriminative Task:")
        print(f"  Accuracy:\t{disc['Accuracy']}")
        print(f"  Precision:\t{disc['Precision']}")
        print(f"  Recall:\t{disc['Recall']}")
        print(f"  F1:\t\t{disc['F1']}")

    print("=" * 100)


def save_summary_to_file(summary_file, args, answers_file, model_name=None, error=None, amber_results=None):
    """
    保存 AMBER 评估结果总结到txt文件

    Args:
        summary_file: 总结文件路径
        args: 命令行参数
        answers_file: 答案文件路径
        model_name: 模型名称
        error: 错误信息（如果评估失败）
    """
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AMBER 评估结果总结\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        f.write("【基本信息】\n")
        f.write("-" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {args.model_path}\n")
        if model_name:
            f.write(f"模型名称: {model_name}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"AMBER 数据路径: {args.amber_data_path}\n")
        f.write(f"问题文件: {args.question_file}\n")
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
        f.write(f"Num Beams: {args.num_beams}\n")
        f.write(f"Max New Tokens: {args.max_new_tokens}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write("\n")

        # 文件路径
        f.write("【文件路径】\n")
        f.write("-" * 80 + "\n")
        f.write(f"答案文件: {answers_file}\n")
        f.write(f"总结文件: {summary_file}\n")
        f.write("\n")

        # 评估结果
        f.write("【评估结果】\n")
        f.write("-" * 80 + "\n")
        if error:
            f.write(f"评估失败: {error}\n")
        else:
            # 统计答案数量
            try:
                with open(answers_file, 'r', encoding='utf-8') as af:
                    total_answers = sum(1 for _ in af)
                f.write(f"总答案数: {total_answers}\n")
            except Exception as e:
                f.write(f"无法读取答案文件: {e}\n")

            # AMBER 评估结果
            if amber_results:
                f.write("\n【AMBER 评估指标】\n")
                f.write("-" * 80 + "\n")

                if 'generative' in amber_results:
                    gen = amber_results['generative']
                    f.write("Generative Task:\n")
                    f.write(f"  CHAIR↓:\t{gen['CHAIR']}\n")
                    f.write(f"  Cover↑:\t{gen['Cover']}\n")
                    f.write(f"  Hal↓:\t\t{gen['Hal']}\n")
                    f.write(f"  Cog↓:\t\t{gen['Cog']}\n")
                    f.write("\n")

                if 'discriminative' in amber_results:
                    disc = amber_results['discriminative']
                    f.write("Discriminative Task:\n")
                    f.write(f"  Accuracy:\t{disc['Accuracy']}\n")
                    f.write(f"  Precision:\t{disc['Precision']}\n")
                    f.write(f"  Recall:\t{disc['Recall']}\n")
                    f.write(f"  F1:\t\t{disc['F1']}\n")
                    f.write("\n")
        f.write("\n")

        # 分隔线
        f.write("=" * 80 + "\n")
        f.write("总结文件生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n")


def eval_model(args):
    """评估模型，生成答案"""
    print("=" * 80)
    print("AMBER 评估 - 生成答案")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"AMBER 数据路径: {args.amber_data_path}")
    print(f"问题文件: {args.question_file}")
    print(f"答案文件: {args.answers_file}")
    if args.use_deco:
        print(f"Deco 参数: use_deco={args.use_deco}, alpha={args.alpha}, layers={args.start_layer}-{args.end_layer}")
    else:
        print(f"使用原生 LLaVA 模型(Deco 已禁用)")
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
    question_file_path = os.path.expanduser(args.question_file)

    with open(question_file_path, "r", encoding="utf-8") as f:
        query_list_data = json.load(f)

    total_questions = len(query_list_data)

    # 如果指定了 num_samples，只处理前 N 个样本（用于快速测试）
    if hasattr(args, 'num_samples') and args.num_samples > 0:
        query_list_data = query_list_data[:args.num_samples]
        print(f"✓ 加载了 {total_questions} 个问题，将评估前 {len(query_list_data)} 个（快速测试模式）")
    else:
        print(f"✓ 加载了 {total_questions} 个问题，将评估全部")

    total_questions = len(query_list_data)  # 更新为实际要处理的数量

    # 准备输出文件
    answers_file = os.path.expanduser(args.answers_file)
    output_dir = os.path.dirname(answers_file)
    if output_dir:  # 如果 output_dir 不为空字符串，创建目录
        os.makedirs(output_dir, exist_ok=True)
    # 如果 output_dir 为空字符串，说明文件在当前目录，不需要创建目录
    ans_file = open(answers_file, "w", encoding="utf-8")

    # 准备 Deco 参数
    early_exit_layers = None
    if args.use_deco:
        early_exit_layers = [i for i in range(args.start_layer, args.end_layer)]

    # 处理每个问题
    print(f"\n[3/3] 开始评估...")

    # 计算需要输出详细信息的样本索引（最多10个，均匀分布）
    max_debug_samples = min(10, total_questions)
    debug_indices = set()
    if total_questions > 0:
        if total_questions <= max_debug_samples:
            debug_indices = set(range(total_questions))
        else:
            step = total_questions / max_debug_samples
            for i in range(max_debug_samples):
                idx = int(i * step)
                debug_indices.add(idx)
        if len(debug_indices) > 0:
            print(f"将输出 {len(debug_indices)} 个样本的详细信息用于调试（样本索引: {sorted(debug_indices)}）")

    for sample_idx, line in enumerate(tqdm(query_list_data, desc="处理进度")):
        idx = line["id"]
        image_file = line["image"]
        qs = line["query"]
        cur_prompt = qs

        # 判断是否需要输出详细信息
        verbose = sample_idx in debug_indices

        if verbose:
            print("\n" + "=" * 80)
            print(f"[样本 {sample_idx + 1}/{total_questions}] ID: {idx}")
            print("=" * 80)
            print(f"问题: {qs}")
            print(f"图像: {image_file}")

        try:
            # 构建完整图像路径
            if os.path.isabs(image_file):
                image_path = image_file
                image_file_for_output = os.path.basename(image_file)
            else:
                # 图像文件在 image/ 目录下
                image_path = os.path.join(args.amber_data_path, "image", image_file)
                image_file_for_output = image_file  # 只保存文件名

            # 准备输入
            input_ids, image_tensor, stopping_criteria, stop_str = prepare_inputs(
                model, tokenizer, image_processor, image_path, qs, conv_mode, device, verbose=verbose
            )

            # 生成回答
            outputs, output_token_len, input_token_len = generate_response(
                model, tokenizer, input_ids, image_tensor, stopping_criteria,
                args.temperature, args.top_p, args.num_beams, args.max_new_tokens, device,
                use_deco=args.use_deco,
                alpha=args.alpha,
                threshold_top_p=args.threshold_top_p,
                threshold_top_k=args.threshold_top_k,
                early_exit_layers=early_exit_layers,
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

            if verbose:
                print(f"\n  [生成结果] 回答:")
                print(f"    - 输出长度: {len(outputs)} 字符")
                print(f"    - 回答预览: {outputs[:200]}...")
                print("=" * 80)

            # 保存结果
            ans_file.write(json.dumps({
                "id": idx,
                "query": cur_prompt,
                "response": outputs,
                "model_id": model_name,
                "image": image_file_for_output,  # 只保存文件名
                "metadata": {
                    "output_token_len": output_token_len,
                    "input_token_len": input_token_len,
                    "raw_output": outputs
                }
            }, ensure_ascii=False) + "\n")
            ans_file.flush()

        except Exception as e:
            print(f"\n[Error] 处理问题 {idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 保存错误信息
            ans_file.write(json.dumps({
                "id": idx,
                "query": cur_prompt,
                "response": "Error",
                "model_id": model_name,
                "image": image_file_for_output if 'image_file_for_output' in locals() else image_file,
                "metadata": {"error": str(e)}
            }, ensure_ascii=False) + "\n")
            ans_file.flush()
            continue

    ans_file.close()
    print(f"\n✓ 评估完成！结果已保存到: {answers_file}")

    # 运行 AMBER 评估
    amber_results = None
    try:
        print("\n" + "=" * 80)
        print("开始 AMBER 评估...")
        print("=" * 80)

        # 根据问题文件类型确定评估类型
        question_file_basename = os.path.basename(args.question_file)
        if 'generative' in question_file_basename:
            eval_type = 'g'
        elif 'discriminative' in question_file_basename:
            eval_type = 'd'
        else:
            eval_type = 'a'  # 全部评估

        amber_results, _ = evaluate_amber(
            answers_file=answers_file,
            amber_data_path=args.amber_data_path,
            similarity_score=0.8,
            evaluation_type=eval_type
        )

        # 打印结果表格
        model_name_for_display = get_model_name_from_path(args.model_path)
        print_amber_results_table(amber_results, model_name=model_name_for_display)

        print("\n✓ AMBER 评估完成！")
    except Exception as e:
        print(f"\n⚠️  AMBER 评估失败: {e}")
        import traceback
        traceback.print_exc()
        amber_results = None

    # 保存总结到txt文件
    summary_file = answers_file.replace('.jsonl', '_summary.txt')
    model_name = get_model_name_from_path(args.model_path)
    save_summary_to_file(
        summary_file=summary_file,
        args=args,
        answers_file=answers_file,
        model_name=model_name,
        amber_results=amber_results
    )
    print(f"✓ 结果总结已保存到: {summary_file}")

    return amber_results


def main():
    """主函数 - 自动检测并使用默认配置"""
    # 项目根目录
    project_root = Path(__file__).parent

    # 自动检测可用 GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("⚠ 未检测到 CUDA，将使用 CPU(速度较慢)")

    # 默认配置
    default_config = {
        "model_path": project.llava_v15_7b_path,
        "device": device,
        "amber_data_path": project.amber_data_path,
        "question_file": os.path.join(project.amber_data_path, "query", "query_generative.json"),
        "use_deco": True,
        "alpha": 0.6,
        "threshold_top_p": 0.9,
        "threshold_top_k": 20,
        "start_layer": 20,
        "end_layer": 29,
        "num_samples": 0,
        "temperature": 1.0,
        "top_p": 0.9,
        "num_beams": 1,
        "max_new_tokens": 1024,
        "seed": 42
    }

    # 解析参数(所有参数都有默认值)
    parser = argparse.ArgumentParser(description="AMBER 评估 - 生成答案(所有参数可选)")

    # 数据集参数
    parser.add_argument("--amber-data-path", type=str, default=default_config["amber_data_path"],
                       help="AMBER 数据根目录路径（包含 image 和 query 目录）")
    parser.add_argument("--question-file", type=str, default=default_config["question_file"],
                       help="问题文件路径（JSON 格式，包含问题数组）")

    # 模型参数
    parser.add_argument("--model-path", type=str, default=default_config["model_path"],
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--device", type=str, default=default_config["device"],
                       help="设备 (cuda:0/cpu)")

    # 输出参数
    parser.add_argument("--answers-file", type=str, default=None,
                       help="输出答案文件路径(JSONL 格式，如果不指定，将自动生成)")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=default_config["temperature"],
                       help="生成温度(>0表示采样，<=0表示贪婪生成)")
    parser.add_argument("--top-p", type=float, default=default_config["top_p"], help="Top-p采样")
    parser.add_argument("--num-beams", type=int, default=default_config["num_beams"],
                       help="Beam search 的 beam 数量")
    parser.add_argument("--max-new-tokens", type=int, default=default_config["max_new_tokens"],
                       help="最大生成 token 数")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="对话模式")

    # Deco 参数
    parser.add_argument("--use-deco", action="store_true", default=default_config["use_deco"],
                       help="启用 Deco 早退机制")
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
    parser.add_argument("--num-samples", type=int, default=default_config["num_samples"],
                       help="评估样本数量（0表示评估全部，>0表示只评估前N个样本，用于快速测试）")

    args = parser.parse_args()
    set_seed(args.seed)

    # 自动生成输出文件路径(如果未指定)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", "amber")
    os.makedirs(output_dir, exist_ok=True)

    # 从问题文件名提取查询类型（用于输出文件名）
    question_file_basename = os.path.basename(args.question_file)
    query_type = question_file_basename.replace("query_", "").replace(".json", "")

    # 如果使用 Deco，需要同时运行 vanilla 版本进行对比
    vanilla_answers_file = None

    if args.use_deco:
        print("\n" + "=" * 80)
        print("检测到使用 Deco，将同时运行 Vanilla 版本进行对比")
        print("=" * 80)

        # 先运行 Vanilla 版本
        print("\n" + "-" * 80)
        print("[1/2] 运行 Vanilla 版本")
        print("-" * 80)
        vanilla_args = argparse.Namespace(**vars(args))
        vanilla_args.use_deco = False
        vanilla_args.answers_file = os.path.join(output_dir, f"amber_{query_type}_vanilla_{timestamp}.jsonl")

        vanilla_results = eval_model(vanilla_args)
        vanilla_answers_file = vanilla_args.answers_file

        # 然后运行 Deco 版本
        print("\n" + "-" * 80)
        print("[2/2] 运行 Deco 版本")
        print("-" * 80)
        if args.answers_file is None:
            args.answers_file = os.path.join(output_dir, f"amber_{query_type}_deco_{timestamp}.jsonl")

        deco_results = eval_model(args)

        # 进行对比
        print("\n" + "=" * 80)
        print("对比 Deco vs Vanilla")
        print("=" * 80)

        try:
            # 打印答案对比表格
            print_comparison_table(deco_answers_file=args.answers_file, vanilla_answers_file=vanilla_answers_file)

            # 生成对比 JSON 文件
            comparison_file = args.answers_file.replace('.jsonl', '_comparison.json')
            comparison_result = compare_deco_vs_vanilla(
                deco_answers_file=args.answers_file,
                vanilla_answers_file=vanilla_answers_file,
                output_file=comparison_file
            )

            print(f"\n✓ 对比结果已保存到: {comparison_file}")
            print(f"  - 总样本数: {comparison_result['summary']['total_cases']}")
            print(f"  - 答案不一致样本数: {comparison_result['summary']['inconsistent_cases']}")
            print(f"  - 不一致率: {comparison_result['summary']['inconsistency_rate']:.2%}")

            # 打印 AMBER 评估结果对比表格
            if vanilla_results and deco_results:
                print("\n" + "=" * 100)
                print("AMBER 评估结果对比")
                print("=" * 100)

                # 生成式任务对比
                if 'generative' in vanilla_results and 'generative' in deco_results:
                    print("\nGenerative Task:")
                    print(f"{'Metric':<15} {'Vanilla':<15} {'Deco':<15} {'Diff':<15}")
                    print("-" * 60)
                    for metric in ['CHAIR', 'Cover', 'Hal', 'Cog']:
                        v_val = vanilla_results['generative'][metric]
                        d_val = deco_results['generative'][metric]
                        diff = d_val - v_val
                        diff_str = f"{diff:+.1f}" if diff != 0 else "0.0"
                        print(f"{metric:<15} {v_val:<15.1f} {d_val:<15.1f} {diff_str:<15}")

                # 判别式任务对比
                if 'discriminative' in vanilla_results and 'discriminative' in deco_results:
                    print("\nDiscriminative Task:")
                    print(f"{'Metric':<15} {'Vanilla':<15} {'Deco':<15} {'Diff':<15}")
                    print("-" * 60)
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                        v_val = vanilla_results['discriminative'][metric]
                        d_val = deco_results['discriminative'][metric]
                        diff = d_val - v_val
                        diff_str = f"{diff:+.1f}" if diff != 0 else "0.0"
                        print(f"{metric:<15} {v_val:<15.1f} {d_val:<15.1f} {diff_str:<15}")

                print("=" * 100)
        except Exception as e:
            print(f"⚠️  对比时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 不使用 Deco，正常处理
        if args.answers_file is None:
            args.answers_file = os.path.join(output_dir, f"amber_{query_type}_vanilla_{timestamp}.jsonl")

        # 运行评估
        eval_model(args)

    print("\n" + "=" * 80)
    print("✓ 所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
