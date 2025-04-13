import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

# Create combined figure
plt.figure(figsize=(20, 7))
fig, ax = plt.subplots(figsize=(20, 7))

# Set width of bars
bar_width = 0.15
index = np.arange(num_models)

# Create color gradients for bar charts and lines - ordered from light to dark
line_colors = cm.Reds([0.3, 0.5, 0.7, 0.9])  # Increasing red intensity for bars (now for self-role)
bar_colors= cm.Blues([0.3, 0.5, 0.7, 0.9])  # Increasing blue intensity for lines (now for criminal)

# Bar positions adjustment for each scenario
bar_positions = {}
for i, scenario in enumerate(scenarios):
    bar_positions[scenario] = index + (i - 1.5) * bar_width

bars = []
lines = []
line_markers = ['o', 's', 'D', '^']

# Create second y-axis for line plots
ax2 = ax.twinx()

# Plot bar charts for self-role accuracy (previously was criminal accuracy)
for i, scenario in enumerate(scenarios):
    scenario_data = []
    for model in models:
        if model in results and scenario in results[model]:
            scenario_data.append(results[model][scenario]['self_role_accuracy'])
        else:
            scenario_data.append(0)
    
    bar = ax.bar(bar_positions[scenario], scenario_data, bar_width, 
                 color=bar_colors[i], label=f'{scenario_names[scenario]} (Self-Role)', 
                 edgecolor='black', linewidth=0.5, alpha=0.8)
    
    bars.append(bar)

# Plot lines for criminal accuracy (previously was self-role accuracy)
for i, scenario in enumerate(scenarios):
    scenario_data = []
    for model in models:
        if model in results and scenario in results[model]:
            scenario_data.append(results[model][scenario]['criminal_accuracy'])
        else:
            scenario_data.append(0)
    
    # Use a different position set for lines (center of the model group)
    line_pos = index
    line = ax2.plot(line_pos, scenario_data, marker=line_markers[i], linestyle='-', 
                    color=line_colors[i], linewidth=2, markersize=8,
                    label=f'{scenario_names[scenario]} (Criminal)')
    
    lines.extend(line)

# Set title and labels
ax.set_title('Model Performance Comparison In Hidden Role Deduction', fontsize=18, fontweight='bold')
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Self-Role Identification Accuracy (%)', fontsize=14)
ax2.set_ylabel('Criminal Identification Accuracy (%)', fontsize=14)

# Set x-ticks and labels
ax.set_xticks(index)
ax.set_xticklabels(model_display_list, rotation=45, ha='right', fontsize=12)

# Set y-limits
ax.set_ylim(0, 105)
ax2.set_ylim(0, 105)

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 创建清晰的图例，确保柱状图颜色有明确标注
legend_elements = []

# 柱状图图例（带颜色说明）- 现在是Self-Role
for i, scenario in enumerate(scenarios):
    legend_elements.append(Patch(facecolor=bar_colors[i], edgecolor='black', 
                               label=f'{scenario_names[scenario]} (Self-Role)', alpha=0.8))

# 折线图图例 - 现在是Criminal
for i, scenario in enumerate(scenarios):
    legend_elements.append(Line2D([0], [0], color=line_colors[i], marker=line_markers[i],
                               label=f'{scenario_names[scenario]} (Criminal)', 
                               markersize=8, linestyle='-', linewidth=2))

# 添加组合图例
fig.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0.1, 1, 0.96])
plt.savefig('task_accuracy.png', dpi=300, bbox_inches='tight')
print("Combined chart saved as 'task_accuracy.png'") 