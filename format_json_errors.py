#!/usr/bin/env python3
"""
格式化JSON文件，每个key占一行，数组值放在同一行
"""

import json
import sys
import os


def format_json_compact_arrays(obj, indent_level=0, indent_str='  '):
    """
    格式化JSON，每个key占一行，数组值放在同一行

    Args:
        obj: 要格式化的JSON对象
        indent_level: 当前缩进级别
        indent_str: 缩进字符串（默认2个空格）

    Returns:
        格式化后的字符串
    """
    indent = indent_str * indent_level
    next_indent = indent_str * (indent_level + 1)

    if isinstance(obj, dict):
        if not obj:
            return '{}'

        lines = []
        for key, value in obj.items():
            key_str = f'"{key}"'

            if isinstance(value, (list, tuple)):
                # 数组：检查元素类型
                if all(isinstance(item, (str, int, float, bool, type(None), list, tuple)) for item in value):
                    # 如果都是简单类型或嵌套数组，放在一行（紧凑格式）
                    json_value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                    lines.append(f'{next_indent}{key_str}: {json_value}')
                else:
                    # 如果有对象类型，每个对象一行，但对象内部的数组值放在同一行
                    formatted_array = format_json_compact_arrays(value, indent_level + 1, indent_str)
                    lines.append(f'{next_indent}{key_str}: {formatted_array}')
            elif isinstance(value, dict):
                # 嵌套字典：递归处理
                formatted_value = format_json_compact_arrays(value, indent_level + 1, indent_str)
                lines.append(f'{next_indent}{key_str}: {formatted_value}')
            else:
                # 其他类型（字符串、数字、布尔值、null）：正常格式
                json_value = json.dumps(value, ensure_ascii=False)
                lines.append(f'{next_indent}{key_str}: {json_value}')

        return '{\n' + ',\n'.join(lines) + '\n' + indent + '}'

    elif isinstance(obj, (list, tuple)):
        # 列表：每个元素一行（但如果是简单值，可以紧凑）
        if not obj:
            return '[]'

        # 检查列表元素类型
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in obj):
            # 如果都是简单类型，放在一行
            json_value = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
            return json_value
        else:
            # 如果有复杂类型，每个元素一行
            lines = []
            for item in obj:
                if isinstance(item, (dict, list)):
                    formatted_item = format_json_compact_arrays(item, indent_level + 1, indent_str)
                    lines.append(f'{next_indent}{formatted_item},')
                else:
                    json_item = json.dumps(item, ensure_ascii=False)
                    lines.append(f'{next_indent}{json_item},')

            # 移除最后一个逗号
            if lines and lines[-1].endswith(','):
                lines[-1] = lines[-1][:-1]

            return '[\n' + '\n'.join(lines) + '\n' + indent + ']'

    else:
        # 其他类型：直接JSON序列化
        return json.dumps(obj, ensure_ascii=False)


def format_json_file(input_file, output_file=None):
    """
    格式化JSON文件

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径（如果为None，则覆盖原文件）
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 格式化
    formatted_json = format_json_compact_arrays(data)

    # 确定输出文件
    if output_file is None:
        output_file = input_file

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_json)

    print(f"✓ JSON文件已格式化: {output_file}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python format_json_errors.py <input_json_file> [output_json_file]")
        print("  如果未指定输出文件，将覆盖原文件")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)

    format_json_file(input_file, output_file)


if __name__ == '__main__':
    main()
