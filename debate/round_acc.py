import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 定义颜色
colors = ['#FFCCCB', '#FF9797', '#AF3240']

# 定义文件夹和文件
results_dirs = ['results', 'results_2', 'results_3']
summary_filename = 'summary_debate.json'

# 存储处理后的数据
model_data = {}

# 阶段名称
stage_names = ['Paper Informantion', 'With Reviews', 'After Rebuttal']

# 读取和处理每个阶段的数据
for stage_idx, results_dir in enumerate(results_dirs):
    summary_path = os.path.join(results_dir, summary_filename)

    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            stage_data = json.load(f)

            # 处理每个模型的数据
            for model_name, model_results in stage_data.items():
                if model_name not in model_data:
                    model_data[model_name] = {
                        'stage1': 0,
                        'stage2': 0,
                        'stage3': 0,
                    }

                # 获取精确度数据 (使用accuracy)
                accuracy = model_results.get('accuracy', 0)
                model_data[model_name][f'stage{stage_idx+1}'] = accuracy
    else:
        print(f"警告: 找不到文件 {summary_path}")

# 模型名称映射字典，用于重命名模型
model_name_map = {
    'deepseek-r1': 'DeepSeek-R1',
    'gemma-2-9b': 'Gemma-2-9B',
    'gemma-2-27B': 'Gemma-2-27B',
    'llama-3.1-8B': 'Llama-3.1-8B', 
    'llama-3.3-70B': 'Llama-3.3-70B',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-4o': 'GPT-4o',
    'qwen-2.5-72B': 'Qwen-2.5-72B',
    'qwq': 'QWQ-32B',
    'o1':'o1',
    'gemeni-2.5': 'Gemeni-2.5-Pro'
}

# 指定模型的顺序
model_order = [
    'llama-3.1-8B',
    'llama-3.3-70B', 
    'qwen-2.5-72B',
    'gpt-4o-mini',
    'gpt-4o',
    'o3-mini',
    'qwq',
    'deepseek-r1',
    'o1',
    'gemeni-2.5'
]

# 重组数据以匹配新的顺序和名称
models = [m for m in model_order if m in model_data]
display_names = [model_name_map.get(m, m) for m in models]

stage1_values = [model_data[m]['stage1'] for m in models]
stage2_values = [model_data[m]['stage2'] for m in models]
stage3_values = [model_data[m]['stage3'] for m in models]

# 设置图形风格
plt.figure(figsize=(14, 7))
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

# 设置柱状图的位置
x = np.arange(len(models))
width = 0.23  # 柱子宽度

# 创建三个柱状图，前两轮降低透明度
linewidth = 1.5  # 边框线条宽度
alpha_stages = [0.7, 0.7, 1.0]  # 三个阶段的透明度

bars1 = plt.bar(x - width, stage1_values, width, label=stage_names[0], color=colors[0], 
                edgecolor='black', linewidth=linewidth, alpha=alpha_stages[0])
bars2 = plt.bar(x, stage2_values, width, label=stage_names[1], color=colors[1], 
                edgecolor='black', linewidth=linewidth, alpha=alpha_stages[1])
bars3 = plt.bar(x + width, stage3_values, width, label=stage_names[2], color=colors[2], 
                edgecolor='black', linewidth=linewidth, alpha=alpha_stages[2])

# 添加标题和标签
#plt.title('Model Accuracy in Paper Acceptance Prediction', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

# 设置x轴标签为模型显示名称
plt.xticks(x, display_names, rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# 设置y轴范围，使图表更紧凑
y_max = max(max(stage1_values), max(stage2_values), max(stage3_values))
plt.ylim(0, min(100, y_max * 1.15))  # 设置上限为最大值的1.15倍或100，取小值

# 添加网格线以便于阅读，但降低可见度
plt.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# 将图例放在图表完全外部的顶部，横向排列
plt.legend(bbox_to_anchor=(0.5, 1.12), loc='upper center', ncol=3, fontsize=12, 
           frameon=True, edgecolor='black')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 为图例和标题留出更多空间

# 保存图像
plt.savefig('multiturn_debate.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

print("图表已保存为 multiturn_debate.png")