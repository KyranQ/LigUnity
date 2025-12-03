#!/bin/bash

# ================= 配置区域 =================
# 请在这里确认你的 checkpoint 路径
pocket_ckpt="./ckpts/pocket_checkpoint_avg_41-50_1.pt"
protein_ckpt="./ckpts/protein_checkpoint_avg_41-50_1.pt"
support_num=0.6
# ===========================================

# 检查文件是否存在
if [ ! -f "$pocket_ckpt" ]; then
    echo "错误: 找不到 Pocket checkpoint 文件: $pocket_ckpt"
    exit 1
fi

if [ ! -f "$protein_ckpt" ]; then
    echo "错误: 找不到 Protein checkpoint 文件: $protein_ckpt"
    exit 1
fi

echo "使用 Pocket Checkpoint: $pocket_ckpt"
echo "使用 Protein Checkpoint: $protein_ckpt"
echo "Support Num: $support_num"

# 运行 6 次重复实验
for r in {1..6}; do
    echo "==========================================="
    echo "正在运行 Few-shot 第 ${r} 次重复实验 (Repeat ${r}/6)..."
    echo "==========================================="
    
    # Pocket Ranking Fine-tuning
    echo "Running Pocket Ranking Few-shot..."
    CUDA_VISIBLE_DEVICES=0 bash test_fewshot.sh FEP pocket_ranking ${support_num} "${pocket_ckpt}" "./result/pocket_ranking/FEP_fewshot/repeat_${r}"
    
    # Protein Ranking Fine-tuning
    echo "Running Protein Ranking Few-shot..."
    CUDA_VISIBLE_DEVICES=0 bash test_fewshot.sh FEP protein_ranking ${support_num} "${protein_ckpt}" "./result/protein_ranking/FEP_fewshot/repeat_${r}"
done

echo "==========================================="
echo "所有实验完成，正在汇总结果..."
echo "==========================================="

# 汇总结果
python ensemble_result.py fewshot FEP_fewshot ${support_num}

echo "完成！"
