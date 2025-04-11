import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

# 定义颜色
colors = ['#C7EAEC', '#AED9CE', '#96BAC7']

# 读取结果文件夹
results_dir = 'results'
all_results_files = [f for f in os.listdir(results_dir) if f.endswith('_6_all_results.json')]

# 存储处理后的数据
model_data = {}

# 处理每个文件
for filename in sorted(all_results_files):
    filepath = os.path.join(results_dir, filename)

    # 提取模型名称
    model_name = re.match(r'(.+)_6_all_results\.json', filename).group(1)

    with open(filepath, 'r') as f:
        data = json.load(f)

        # 获取轮次数据
        rounds_data = data.get('summary', {}).get('rounds', {})

        # 如果存在有效数据
        if rounds_data:
            # 计算每轮的平均值
            round1 = rounds_data.get('1', {})
            criminal_acc1 = round1.get('criminal_accuracy', 0)
            self_role_acc1 = round1.get('self_role_accuracy', 0)

            round2 = rounds_data.get('2', {})
            criminal_acc2 = round2.get('criminal_accuracy', 0)
            self_role_acc2 = round2.get('self_role_accuracy', 0)

            round3 = rounds_data.get('3', {})
            criminal_acc3 = round3.get('criminal_accuracy', 0)
            self_role_acc3 = round3.get('self_role_accuracy', 0)

            model_data[model_name] = {
                'round1': (criminal_acc1 + self_role_acc1) / 2,
                'round2': (criminal_acc2 + self_role_acc2) / 2,
                'round3': (criminal_acc3 + self_role_acc3) / 2
            }

# 模型名称映射字典，用于重命名模型
model_name_map = {
    'deepseek-r1': 'DeepSeek-R1',
    'gemma-2-27B': 'Gemma-2-27B',
    'llama-3.1-8B': 'Llama-3.1-8B',
    'llama-3.3-70B': 'Llama-3.3-70B',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-4o': 'GPT-4o',
    'qwen-2.5-72B': 'Qwen-2.5-72B',
    'qwq': 'QWQ-32B'
}

# 指定模型的顺序
model_order = [
    'llama-3.1-8B',
    'llama-3.3-70B',
    'gemma-2-27B',
    'qwen-2.5-72B',
    'gpt-4o-mini',
    'gpt-4o',
    'o3-mini',
    'qwq',  
    'deepseek-r1'
]

# 重组数据以匹配新的顺序和名称
models = [m for m in model_order if m in model_data]
display_names = [model_name_map.get(m, m) for m in models]

round1_values = [model_data[m]['round1'] for m in models]
round2_values = [model_data[m]['round2'] for m in models]
round3_values = [model_data[m]['round3'] for m in models]

# 设置图形风格
plt.figure(figsize=(14, 7))
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

# 设置柱状图的位置
x = np.arange(len(models))
width = 0.23  # 柱子宽度

# 创建三个柱状图，前两轮降低透明度
linewidth = 1.5  # 边框线条宽度
alpha_early_rounds1 = 0.5  # 前两轮的透明度
alpha_early_rounds2 = 0.7  # 前两轮的透明度
bars1 = plt.bar(x - width, round1_values, width, label='Round 1', color=colors[0], 
                edgecolor='black', linewidth=linewidth, alpha=alpha_early_rounds1)
bars2 = plt.bar(x, round2_values, width, label='Round 2', color=colors[1], 
                edgecolor='black', linewidth=linewidth, alpha=alpha_early_rounds2)
bars3 = plt.bar(x + width, round3_values, width, label='Round 3', color=colors[2], 
                edgecolor='black', linewidth=linewidth)  # 第三轮保持完全不透明

# 保留左侧的y轴标签
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

# 设置x轴标签为模型显示名称
plt.xticks(x, display_names, rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# 设置y轴范围，使图表更紧凑
y_max = max(max(round1_values), max(round2_values), max(round3_values))
plt.ylim(0, min(100, y_max * 1.15))  # 设置上限为最大值的1.15倍或100，取小值

# 添加网格线以便于阅读，但降低可见度
plt.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# 将图例放在图表完全外部的顶部，横向排列
plt.legend(bbox_to_anchor=(0.5, 1.12), loc='upper center', ncol=3, fontsize=12, 
           frameon=True, edgecolor='black')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.8)  # 为图例留出更多空间

# 保存图像
plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

print("图表已保存为 model_accuracy_comparison.png")