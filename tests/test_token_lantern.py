#!/usr/bin/env python3
"""
测试验证脚本：验证 "lantern" 这个单词的 token ID 和数量

用法:
    python test_token_lantern.py [--model-path <model_path>] [--word <word>]
"""

import argparse
import os
import sys

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'MLLM_Deco' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def test_word_tokenization(tokenizer, word="lantern"):
    """
    测试单词的 tokenization

    Args:
        tokenizer: tokenizer 对象
        word: 要测试的单词（默认 "lantern"）
    """
    print(f"\n{'='*60}")
    print(f"测试单词: '{word}'")
    print(f"{'='*60}\n")

    # 方法1: 直接 tokenize 单词
    print("方法1: 直接 tokenize 单词")
    print("-" * 60)
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Token 数量: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")

    # 详细显示每个 token
    print("\n详细 Token 信息:")
    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
        decoded = tokenizer.decode([token_id])
        print(f"  [{i}] ID: {token_id:6d} | Token: {token:20s} | Decoded: '{decoded}'")

    # 方法2: 使用 tokenize 方法（返回 token 字符串）
    print("\n" + "-" * 60)
    print("方法2: 使用 tokenizer.tokenize()")
    print("-" * 60)
    token_strings = tokenizer.tokenize(word)
    print(f"Token 字符串: {token_strings}")
    print(f"Token 数量: {len(token_strings)}")

    # 方法3: 检查单词是否在词汇表中
    print("\n" + "-" * 60)
    print("方法3: 检查单词是否在词汇表中")
    print("-" * 60)
    vocab = tokenizer.get_vocab()
    if word.lower() in vocab:
        print(f"  '{word.lower()}' 在词汇表中，ID: {vocab[word.lower()]}")
    else:
        print(f"  '{word.lower()}' 不在词汇表中")

    # 检查各种变体
    variants = [word, word.lower(), word.upper(), word.capitalize()]
    for variant in variants:
        if variant in vocab:
            print(f"  '{variant}' 在词汇表中，ID: {vocab[variant]}")

    # 方法4: 在句子中 tokenize（更接近实际使用场景）
    print("\n" + "-" * 60)
    print("方法4: 在句子中 tokenize")
    print("-" * 60)
    sentence = f"A {word} is hanging on the wall."
    print(f"测试句子: '{sentence}'")
    sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    sentence_tokens = tokenizer.convert_ids_to_tokens(sentence_token_ids)

    # 找到 "lantern" 对应的 token 位置
    word_lower = word.lower()
    word_positions = []
    for i, token in enumerate(sentence_tokens):
        if word_lower in token.lower() or token.lower() in word_lower:
            word_positions.append(i)

    print(f"句子 Token 数量: {len(sentence_token_ids)}")
    print(f"句子 Tokens: {sentence_tokens}")
    if word_positions:
        print(f"\n'{word}' 在句子中的位置: {word_positions}")
        print("对应的 Token IDs:")
        for pos in word_positions:
            print(f"  位置 {pos}: ID={sentence_token_ids[pos]}, Token='{sentence_tokens[pos]}'")
    else:
        print(f"\n未找到 '{word}' 的完整匹配，可能被拆分为多个 token")
        # 尝试找到包含该单词的 token
        for i, token in enumerate(sentence_tokens):
            if any(char in token.lower() for char in word_lower):
                print(f"  位置 {i}: ID={sentence_token_ids[i]}, Token='{token}'")

    # 总结
    print("\n" + "="*60)
    print("总结:")
    print("="*60)
    print(f"单词 '{word}' 对应的 Token ID 数量: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print(f"Token 字符串: {tokens}")
    print("="*60 + "\n")

    return token_ids, tokens


def main():
    parser = argparse.ArgumentParser(
        description='测试验证单词的 token ID 和数量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试 "lantern"（使用默认模型路径）
  python test_token_lantern.py

  # 测试其他单词
  python test_token_lantern.py --word "chair"

  # 指定模型路径
  python test_token_lantern.py --model-path /path/to/model --word "lantern"
        """
    )

    parser.add_argument('--model-path', type=str, default=None,
                       help='模型路径（默认：从项目配置读取）')
    parser.add_argument('--word', type=str, default='lantern',
                       help='要测试的单词（默认：lantern）')

    args = parser.parse_args()

    # 如果没有指定模型路径，尝试从项目配置读取（和 test_chair_attention.py 保持一致）
    if args.model_path is None:
        try:
            import project as project
            args.model_path = project.llava_v15_7b_path
        except:
            print("错误: 请指定模型路径 (--model-path)")
            print("或者确保项目配置中有 llava_v15_7b_path")
            return

    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    print(f"正在加载模型和 tokenizer...")
    print(f"模型路径: {args.model_path}")

    # 禁用 torch 初始化（如果需要）
    disable_torch_init()

    # 获取模型名称
    model_name = get_model_name_from_path(args.model_path)
    print(f"模型名称: {model_name}")

    # 加载模型和 tokenizer
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, None, model_name, device='cpu'
        )
        print("✓ 模型和 tokenizer 加载完成\n")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试单词 tokenization
    test_word_tokenization(tokenizer, word=args.word)


if __name__ == '__main__':
    main()
