import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def evaluate_pope(gt_files_path: str, gen_files_path: str,
                  output_errors_path: Optional[str] = None,
                  verbose: bool = True) -> Dict:
    """
    评估 POPE 数据集的结果

    Args:
        gt_files_path: 真值文件路径（JSONL 格式）
        gen_files_path: 生成答案文件路径（JSONL 格式）
        output_errors_path: 错误样本输出文件路径（可选）
        verbose: 是否打印详细信息

    Returns:
        包含评估结果的字典，包括：
        - metrics: 性能指标（precision, recall, f1, accuracy等）
        - statistics: 统计信息（TP, TN, FP, FN等）
        - error_samples: 错误样本列表
    """
    # 加载文件
    gt_files = [json.loads(q) for q in open(os.path.expanduser(gt_files_path), "r")]
    gen_files = [json.loads(q) for q in open(os.path.expanduser(gen_files_path), "r")]

    # 创建生成文件的字典，以 question_id 为键（更健壮的匹配方式）
    gen_dict = {item["question_id"]: item for item in gen_files}

    # 创建 GT 文件的字典，以 question_id 为键
    gt_dict = {item["question_id"]: item for item in gt_files}

    # 只评估生成文件中实际存在的 question_id（而不是所有 GT 问题）
    # 这样可以处理部分评测的情况（例如只评测了前100个case）
    valid_question_ids = set(gen_dict.keys()) & set(gt_dict.keys())

    if len(valid_question_ids) == 0:
        raise ValueError("No matching question_ids found between GT and generated files!")

    if verbose:
        print(f"找到 {len(valid_question_ids)} 个匹配的 question_id")
        print(f"GT 文件总问题数: {len(gt_files)}")
        print(f"生成文件总答案数: {len(gen_files)}")
        print(f"将评估 {len(valid_question_ids)} 个问题")

    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(valid_question_ids)  # 只统计实际评估的问题数
    yes_answers = 0
    missing_answers = 0

    # 保存错误样本的列表
    error_samples = []

    # compare answers - 只评估匹配的 question_id
    for idx in sorted(valid_question_ids):
        gt_line = gt_dict[idx]
        gen_line = gen_dict[idx]

        gt_answer = gt_line["label"]
        gen_answer = gen_line["text"]

        # convert to lowercase
        gt_answer_lower = gt_answer.lower()
        gen_answer_lower = gen_answer.lower()
        # strip
        gt_answer_lower = gt_answer_lower.strip()
        gen_answer_lower = gen_answer_lower.strip()

        # 判断是否正确
        is_correct = False
        error_type = None

        # pos = 'yes', neg = 'no'
        if gt_answer_lower == 'yes':
            if 'yes' in gen_answer_lower:
                true_pos += 1
                yes_answers += 1
                is_correct = True
            else:
                false_neg += 1
                error_type = "FN"  # False Negative: GT是yes但模型回答no
        elif gt_answer_lower == 'no':
            if 'no' in gen_answer_lower:
                true_neg += 1
                is_correct = True
            else:
                yes_answers += 1
                false_pos += 1
                error_type = "FP"  # False Positive: GT是no但模型回答yes
        else:
            if verbose:
                print(f'Warning: unknown gt_answer: {gt_answer}')
            unknown += 1
            error_type = "Unknown"

        # 如果是错误样本，保存详细信息
        if not is_correct and error_type:
            # 优先使用生成文件中的图像路径（更准确，可能是可视化后的路径）
            image_path = gen_line.get("image", gt_line.get("image", ""))

            error_sample = {
                "question_id": idx,
                "error_type": error_type,
                "gt_answer": gt_answer,
                "gen_answer": gen_answer,
                "question": gt_line.get("text", ""),
                "image": image_path,
                "image_gt": gt_line.get("image", ""),  # GT 文件中的原始路径
                "image_gen": gen_line.get("image", ""),  # 生成文件中的路径
                "raw_output": gen_line.get("metadata", {}).get("raw_output", ""),
                "model_id": gen_line.get("model_id", ""),
                "prompt": gen_line.get("prompt", gt_line.get("text", "")),  # 实际使用的 prompt
            }
            error_samples.append(error_sample)

    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    if total_questions == 0:
        raise ValueError("No valid questions found!")

    if (true_pos + false_pos) > 0:
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = 0.0

    if (true_pos + false_neg) > 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions

    # 构建结果字典
    results = {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "yes_proportion": yes_proportion,
            "unknown_proportion": unknown_prop
        },
        "statistics": {
            "total_questions": total_questions,
            "gt_total": len(gt_files),
            "gen_total": len(gen_files),
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg,
            "unknown": unknown
        },
        "error_samples": error_samples
    }

    # 打印结果
    if verbose:
        print("\n" + "=" * 80)
        print("POPE 评估结果")
        print("=" * 80)
        print(f'评估的问题数: {total_questions}')
        print(f'GT 文件总问题数: {len(gt_files)}')
        print(f'生成文件总答案数: {len(gen_files)}')
        print(f'\n详细统计:')
        print(f'  - True Positive (TP): {true_pos}')
        print(f'  - True Negative (TN): {true_neg}')
        print(f'  - False Positive (FP): {false_pos}')
        print(f'  - False Negative (FN): {false_neg}')
        print(f'  - Unknown: {unknown}')
        print(f'\n性能指标:')
        print(f'  - Precision: {precision:.4f}')
        print(f'  - Recall: {recall:.4f}')
        print(f'  - F1: {f1:.4f}')
        print(f'  - Accuracy: {accuracy:.4f}')
        print(f'  - Yes proportion: {yes_proportion:.4f}')
        print(f'  - Unknown proportion: {unknown_prop:.4f}')
        print("=" * 80)

    # 保存错误样本到文件
    if output_errors_path:
        error_output_file = os.path.expanduser(output_errors_path)
        os.makedirs(os.path.dirname(error_output_file) if os.path.dirname(error_output_file) else ".", exist_ok=True)

        with open(error_output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_errors": len(error_samples),
                    "false_positives": false_pos,
                    "false_negatives": false_neg,
                    "unknown": unknown
                },
                "errors": error_samples
            }, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\n✓ 错误样本已保存到: {error_output_file}")
            print(f"  - 总错误数: {len(error_samples)}")
            print(f"  - False Positives (FP): {false_pos}")
            print(f"  - False Negatives (FN): {false_neg}")
            print(f"  - Unknown: {unknown}")
    elif len(error_samples) > 0 and verbose:
        print(f"\n提示: 发现 {len(error_samples)} 个错误样本，可以使用 --output-errors 参数保存到文件")

    return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_files", type=str, default="",
                       help="真值文件路径（JSONL 格式）")
    parser.add_argument("--gen_files", type=str, default="",
                       help="生成答案文件路径（JSONL 格式）")
    parser.add_argument("--output-errors", type=str, default=None,
                       help="输出错误样本到 JSON 文件（如果不指定，则不保存）")
    args = parser.parse_args()

    if not args.gt_files or not args.gen_files:
        parser.print_help()
        print("\n错误: 必须提供 --gt_files 和 --gen_files 参数")
        exit(1)

    evaluate_pope(
        gt_files_path=args.gt_files,
        gen_files_path=args.gen_files,
        output_errors_path=args.output_errors,
        verbose=True
    )


if __name__ == "__main__":
    main()