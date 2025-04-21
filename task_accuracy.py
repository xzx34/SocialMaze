import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# 全局字体设置对齐前一段代码
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16})
plt.rcParams['axes.linewidth'] = 1.6

# 配置
scenarios = ['original', 'lunatic', 'rumormonger', 'all']
scenario_names = {
    'original': 'Original Task',
    'lunatic': 'Lunatic Task',
    'rumormonger': 'Rumormonger Task',
    'all': 'Full Task'
}
metrics = ['criminal_accuracy', 'self_role_accuracy']
scenario_colors = {
    'original': '#210F37',
    'lunatic': '#4F1C51',
    'rumormonger': '#A55B4B',
    'all': '#DCA06D'
}
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

# 加载数据
results = {}
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
    return {'criminal_accuracy': 0, 'self_role_accuracy': 0}

for model in models:
    if model == 'o1':
        results[model] = {
            'original': {'criminal_accuracy': 100, 'self_role_accuracy': 100},
            'lunatic': {'criminal_accuracy': 94, 'self_role_accuracy': 96},
            'rumormonger': {'criminal_accuracy': 80, 'self_role_accuracy': 82},
            'all': {'criminal_accuracy': 77, 'self_role_accuracy': 75}
        }
    elif model == 'Gemeni-2.5-Pro':
        results[model] = {
            'original': {'criminal_accuracy': 100, 'self_role_accuracy': 100},
            'lunatic': {'criminal_accuracy': 97, 'self_role_accuracy': 97},
            'rumormonger': {'criminal_accuracy': 93, 'self_role_accuracy': 95},
            'all': {'criminal_accuracy': 83, 'self_role_accuracy': 87}
        }
    else:
        results[model] = {}
        for scenario in scenarios:
            results[model][scenario] = load_model_data(model, scenario)

# 绘图
fig, ax_bar = plt.subplots(figsize=(17, 5))
ax_line = ax_bar.twinx()

bar_width = 0.15
index = np.arange(len(models))
legend_elements = []

for i, scenario in enumerate(scenarios):
    bar_pos = index + (i - 1.5) * bar_width
    self_data = [results[model][scenario]['self_role_accuracy'] for model in models]
    crim_data = [results[model][scenario]['criminal_accuracy'] for model in models]

    # 柱状图
    ax_bar.bar(bar_pos, self_data, bar_width, 
               color=scenario_colors[scenario], edgecolor='black', linewidth=1.5, alpha=0.6)

    # 折线图
    line = ax_line.plot(index, crim_data, marker='o', linestyle='-', 
                        color=scenario_colors[scenario], linewidth=4, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.8)
    line[0].set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='black'),
        path_effects.Normal()
    ])

    # 图例元素
    legend_elements.append(Patch(facecolor=scenario_colors[scenario], edgecolor='black', alpha=0.6,
                                 label=f'{scenario_names[scenario]} (Self-Role)'))
    legend_elements.append(Line2D([0], [0], color=scenario_colors[scenario], marker='o', linestyle='-',
                                  linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                  path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                                path_effects.Normal()],
                                  label=f'{scenario_names[scenario]} (Criminal)'))

# 坐标轴设置
ax_bar.set_ylabel('Self-Role Identification Acc. (%)', fontsize=14)
ax_line.set_ylabel('Criminal Identification Acc. (%)', fontsize=14)

ax_bar.set_xticks(index)
ax_bar.set_xticklabels([model_display_names[model] for model in models], rotation=20, ha='center', fontsize=14)

ax_bar.set_ylim(25, 103)
ax_line.set_ylim(25, 103)

ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# 图例设置
ax_bar.legend(handles=legend_elements, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.3), 
              framealpha=0.8, fontsize=14)

plt.tight_layout()
plt.savefig('task_accuracy.pdf', dpi=300, bbox_inches='tight')
print("Combined chart saved as 'task_accuracy.pdf'")
