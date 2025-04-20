import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# 增大全局字体
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})
plt.rcParams['axes.linewidth'] = 1.6

# 初始化角色及其颜色，使用新的配色方案
roles = ['Investigator', 'Criminal', 'Rumormonger', 'Lunatic']
role_colors = {
    'Investigator': '#325373',
    'Criminal': '#4F72A8',
    'Rumormonger': '#82A5C9',
    'Lunatic': '#CFDCEA'
}

# 定义模型展示名称 —— 顺序很重要
model_display_names = {
    'llama-3.1-8B': 'Llama-3.1-8B',
    'llama-3.3-70B': 'Llama-3.3-70B',
    'qwen-2.5-72B': 'Qwen-2.5-72B',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-4o': 'GPT-4o',
    'o3-mini': 'o3-mini',
    'qwq': 'QwQ-32B',
    'deepseek-r1': 'DeepSeek-R1',
    'o1': 'o1',
    'Gemeni-2.5-Pro': 'Gemeni-2.5-Pro'
}
models = list(model_display_names.keys())

# 从结果文件中提取角色特定数据的函数
def extract_role_specific_data(file_path):
    role_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'summary' in data and 'role_specific' in data['summary']:
                for role in roles:
                    if role in data['summary']['role_specific'] and '3' in data['summary']['role_specific'][role]:
                        role_data[role] = {
                            'criminal_accuracy': data['summary']['role_specific'][role]['3']['criminal_accuracy'],
                            'self_role_accuracy': data['summary']['role_specific'][role]['3']['self_role_accuracy']
                        }
    return role_data

# 针对 o1 和 Gemeni-2.5-Pro 模型硬编码数据
special_model_data = {
    'o1': {
        'Investigator': {'criminal_accuracy': 96, 'self_role_accuracy': 100},
        'Criminal': {'criminal_accuracy': 68, 'self_role_accuracy': 96},
        'Rumormonger': {'criminal_accuracy': 65, 'self_role_accuracy': 73},
        'Lunatic': {'criminal_accuracy': 83, 'self_role_accuracy': 87}
    },
    'Gemeni-2.5-Pro': {
        'Investigator': {'criminal_accuracy': 96, 'self_role_accuracy': 100},
        'Criminal': {'criminal_accuracy': 84, 'self_role_accuracy': 98},
        'Rumormonger': {'criminal_accuracy': 80, 'self_role_accuracy': 85},
        'Lunatic': {'criminal_accuracy': 88, 'self_role_accuracy': 91}
    }
}

# 收集所有模型的数据
results = {}
for model in models:
    if model in special_model_data:
        results[model] = special_model_data[model]
    else:
        file_path = f"blood/results/{model}_6_all_results.json"
        results[model] = extract_role_specific_data(file_path)

# 创建绘图窗口
fig, ax_bar = plt.subplots(figsize=(20, 8))
ax_line = ax_bar.twinx()

# 设置柱状图宽度
bar_width = 0.15
index = np.arange(len(models))

# 用来存放图例元素
legend_elements = []

# 针对每个角色分别绘图
for i, role in enumerate(roles):
    criminal_accuracy_data = []
    self_role_accuracy_data = []

    for model in models:
        model_data = results.get(model, {})
        role_data = model_data.get(role, {'criminal_accuracy': 0, 'self_role_accuracy': 0})
        criminal_accuracy_data.append(role_data.get('criminal_accuracy', 0))
        self_role_accuracy_data.append(role_data.get('self_role_accuracy', 0))

    # 计算柱状图位置
    bar_position = index + (i - 1.5) * bar_width
    # 绘制自身角色识别准确率柱状图（左侧坐标轴）
    ax_bar.bar(bar_position, self_role_accuracy_data, bar_width, 
               color=role_colors[role], edgecolor='black', linewidth=1.5, alpha=0.6)

    # 绘制犯罪角色识别准确率的折线图（右侧坐标轴）
    line = ax_line.plot(index, criminal_accuracy_data, marker='o', linestyle='-', 
                 color=role_colors[role], linewidth=4, markersize=8,
                 markeredgecolor='black', markeredgewidth=0.8
                 )
    
    # 添加黑色描边效果
    line[0].set_path_effects([path_effects.Stroke(linewidth=5, foreground='black'),
                           path_effects.Normal()])

    # 为每个角色添加图例元素（一次添加柱状图和折线图图例）
    legend_elements.append(Patch(facecolor=role_colors[role], edgecolor='black', alpha=0.6,
                                 label=f'{role} (Self-Role)'))
    legend_elements.append(Line2D([0], [0], color=role_colors[role], marker='o', linestyle='-',
                                  linewidth=4, markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                  path_effects=[path_effects.Stroke(linewidth=5, foreground='black'),
                                               path_effects.Normal()],
                                  label=f'{role} (Criminal)'))

# 设置y轴标签（保留y轴标签以便说明数值含义）
ax_bar.set_ylabel('Self-Role Identification Accuracy (%)', fontsize=26)
ax_line.set_ylabel('Criminal Identification Accuracy (%)', fontsize=26)

# 设置x轴刻度
ax_bar.set_xticks(index)
ax_bar.set_xticklabels([model_display_names[model] for model in models], rotation=20, ha='center', fontsize=18)

# 设置y轴范围
ax_bar.set_ylim(0, 103)
ax_line.set_ylim(0, 103)

# 添加网格线
ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# 将图例放置在图上方，并设置半透明背景
ax_bar.legend(handles=legend_elements, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
             framealpha=0.8, fontsize=17)

plt.tight_layout()
plt.savefig('role_accuracy.png', dpi=300, bbox_inches='tight')
print("Role-specific accuracy chart saved as 'role_accuracy.png'")