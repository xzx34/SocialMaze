import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# 增大全局字体
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})
plt.rcParams['axes.linewidth'] = 1.6

# Initialize data structures
scenarios = ['original', 'lunatic',  'rumormonger','all']
scenario_names = {
    'original': 'Original Task',
    'lunatic': 'Lunatic Task',
    'rumormonger': 'Rumormonger Task',
    'all': 'Full Task'
}
metrics = ['criminal_accuracy', 'self_role_accuracy']
metric_names = {
    'criminal_accuracy': 'Criminal Identification Accuracy',
    'self_role_accuracy': 'Self-Role Identification Accuracy'
}

# 更新配色方案，使用同样风格的颜色
scenario_colors = {
    'original': '#2E594D',
    'lunatic': '#44886F',
    'rumormonger': '#87C2A9',
    'all': '#D9EEE3'
}

# Define model display names - ORDER MATTERS HERE
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

# Use the order of models directly from model_display_names
models = list(model_display_names.keys())

results = {}

# Function to load data from individual result files
def load_model_data(model, scenario):
    file_path = f"blood/results/{model}_6_{scenario}_results.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if 'summary' in data and 'rounds' in data['summary'] and '3' in data['summary']['rounds']:
            return {
                'criminal_accuracy': data['summary']['rounds']['3']['criminal_accuracy'],
                'self_role_accuracy': data['summary']['rounds']['3']['self_role_accuracy']
            }
    return {
        'criminal_accuracy': 0,
        'self_role_accuracy': 0
    }

for model in models:
    results[model] = {}
    for scenario in scenarios:
        # For o1 and Gemeni-2.5-Pro, use the directly specified values
        if model == 'o1':
            results[model] = {
                'original': {'criminal_accuracy': 100, 'self_role_accuracy': 100},
                'lunatic': {'criminal_accuracy': 94, 'self_role_accuracy': 96},
                'rumormonger': {'criminal_accuracy': 80, 'self_role_accuracy': 82},
                'all': {'criminal_accuracy': 77, 'self_role_accuracy': 75}
            }
            continue
        elif model == 'Gemeni-2.5-Pro':
            results[model] = {
                'original': {'criminal_accuracy': 100, 'self_role_accuracy': 100},
                'lunatic': {'criminal_accuracy': 97, 'self_role_accuracy': 97},
                'rumormonger': {'criminal_accuracy': 93, 'self_role_accuracy': 95},
                'all': {'criminal_accuracy': 83, 'self_role_accuracy': 87}
            }
            continue
        # For other models, load data from files as usual
        results[model][scenario] = load_model_data(model, scenario)

# Prepare visualization data
model_display_list = [model_display_names[model] for model in models]
num_models = len(models)
num_scenarios = len(scenarios)

# 创建绘图窗口，减小高度
fig, ax_bar = plt.subplots(figsize=(20, 8))
ax_line = ax_bar.twinx()

# Set width of bars
bar_width = 0.15
index = np.arange(num_models)

# Bar positions adjustment for each scenario
bar_positions = {}
for i, scenario in enumerate(scenarios):
    bar_positions[scenario] = index + (i - 1.5) * bar_width

# 用来存放图例元素
legend_elements = []

# Plot bar charts for self-role accuracy
for i, scenario in enumerate(scenarios):
    scenario_data = []
    for model in models:
        if model in results and scenario in results[model]:
            scenario_data.append(results[model][scenario]['self_role_accuracy'])
        else:
            scenario_data.append(0)
    
    bar = ax_bar.bar(bar_positions[scenario], scenario_data, bar_width, 
                 color=scenario_colors[scenario], edgecolor='black', linewidth=1.5, alpha=0.6)

# Plot lines for criminal accuracy
for i, scenario in enumerate(scenarios):
    scenario_data = []
    for model in models:
        if model in results and scenario in results[model]:
            scenario_data.append(results[model][scenario]['criminal_accuracy'])
        else:
            scenario_data.append(0)
    
    # 绘制犯罪角色识别准确率的折线图
    line = ax_line.plot(index, scenario_data, marker='o', linestyle='-', 
                  color=scenario_colors[scenario], linewidth=4, markersize=8,
                  markeredgecolor='black', markeredgewidth=0.8)
    
    # 添加黑色描边效果
    line[0].set_path_effects([path_effects.Stroke(linewidth=5, foreground='black'),
                           path_effects.Normal()])
    
    # 为每个场景添加图例元素
    legend_elements.append(Patch(facecolor=scenario_colors[scenario], edgecolor='black', alpha=0.6,
                                label=f'{scenario_names[scenario]} (Self-Role)'))
    legend_elements.append(Line2D([0], [0], color=scenario_colors[scenario], marker='o', linestyle='-',
                                linewidth=4, markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                path_effects=[path_effects.Stroke(linewidth=5, foreground='black'),
                                            path_effects.Normal()],
                                label=f'{scenario_names[scenario]} (Criminal)'))

# 设置y轴标签
ax_bar.set_ylabel('Self-Role Identification Accuracy (%)', fontsize=26)
ax_line.set_ylabel('Criminal Identification Accuracy (%)', fontsize=26)

# 设置x轴刻度
ax_bar.set_xticks(index)
ax_bar.set_xticklabels(model_display_list, rotation=20, ha='center', fontsize=18)

# 设置y轴范围，使左右y轴对齐
ax_bar.set_ylim(25, 103)
ax_line.set_ylim(25, 103)
ax_bar.set_yticks(np.arange(25, 103, 10))
ax_line.set_yticks(np.arange(25, 103, 10))

# 添加网格线
ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# 将图例放置在图上方，并设置半透明背景
ax_bar.legend(handles=legend_elements, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
             framealpha=0.8, fontsize=17)


plt.tight_layout()
plt.savefig('task_accuracy.png', dpi=300, bbox_inches='tight')
print("Combined chart saved as 'task_accuracy.png'")