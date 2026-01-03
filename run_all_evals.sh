#!/bin/bash
# 依次运行所有评估脚本
# MME, POPE, AMBER, CHAIR

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "开始运行所有评估脚本"
echo "=========================================="
echo ""

# 运行 MME 评估
echo "=========================================="
echo "[1/4] 运行 MME 评估"
echo "=========================================="
python3 run_mme_eval.py
echo ""
echo "✓ MME 评估完成"
echo ""

# 运行 POPE 评估
echo "=========================================="
echo "[2/4] 运行 POPE 评估"
echo "=========================================="
python3 run_pope_eval.py
echo ""
echo "✓ POPE 评估完成"
echo ""

# 运行 AMBER 评估
echo "=========================================="
echo "[3/4] 运行 AMBER 评估"
echo "=========================================="
python3 run_amber_eval.py
echo ""
echo "✓ AMBER 评估完成"
echo ""

# 运行 CHAIR 评估
echo "=========================================="
echo "[4/4] 运行 CHAIR 评估"
echo "=========================================="
python3 run_chair_eval.py
echo ""
echo "✓ CHAIR 评估完成"
echo ""

echo "=========================================="
echo "所有评估脚本运行完成！"
echo "=========================================="
