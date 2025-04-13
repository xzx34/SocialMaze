import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Initialize data structures
roles = ['Investigator', 'Criminal', 'Rumormonger', 'Lunatic']
role_colors = {
    'Investigator': 'forestgreen',
    'Criminal': 'darkred',
    'Rumormonger': 'darkorange',
    'Lunatic': 'darkblue'
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

# Function to load data from individual result files
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

# Hard-code data for o1 and Gemeni-2.5-Pro models
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

# Collect data for all models
results = {}
for model in models:
    if model in special_model_data:
        results[model] = special_model_data[model]
    else:
        file_path = f"blood/results/{model}_6_all_results.json"
        results[model] = extract_role_specific_data(file_path)

# Create figure
plt.figure(figsize=(20, 10))
fig, ax_bar = plt.subplots(figsize=(20, 10))
ax_line = ax_bar.twinx()

# Set width of bars
bar_width = 0.15
index = np.arange(len(models))

# Create arrays for legend
bar_handles = []
line_handles = []

# Plot for each role
for i, role in enumerate(roles):
    # Prepare data arrays
    criminal_accuracy_data = []
    self_role_accuracy_data = []
    
    for model in models:
        model_data = results.get(model, {})
        role_data = model_data.get(role, {'criminal_accuracy': 0, 'self_role_accuracy': 0})
        criminal_accuracy_data.append(role_data.get('criminal_accuracy', 0))
        self_role_accuracy_data.append(role_data.get('self_role_accuracy', 0))
    
    # Plot bars for self-role accuracy on left axis
    bar_position = index + (i - 1.5) * bar_width
    bars = ax_bar.bar(bar_position, self_role_accuracy_data, bar_width, 
                     color=role_colors[role], edgecolor='black', alpha=0.6)
    bar_handles.append(bars)
    
    # Plot lines for criminal accuracy on right axis
    line = ax_line.plot(index, criminal_accuracy_data, marker='o', linestyle='-', 
                       color=role_colors[role], linewidth=2, markersize=8)
    line_handles.append(line[0])

# Set labels and title
ax_bar.set_title('Role-Specific Performance in Full Task', fontsize=18, fontweight='bold')
ax_bar.set_xlabel('Models', fontsize=14)
ax_bar.set_ylabel('Self-Role Identification Accuracy (%)', fontsize=14)
ax_line.set_ylabel('Criminal Identification Accuracy (%)', fontsize=14)

# Set x-ticks and labels
ax_bar.set_xticks(index)
ax_bar.set_xticklabels([model_display_names[model] for model in models], rotation=45, ha='right', fontsize=12)

# Set y-limits
ax_bar.set_ylim(0, 105)
ax_line.set_ylim(0, 105)

# Add grid
ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

# Create legend with custom handles
legend_elements = []
for i, role in enumerate(roles):
    # Add bar element
    legend_elements.append(Patch(facecolor=role_colors[role], edgecolor='black', alpha=0.6,
                                label=f'{role} (Self-Role)'))
    
    # Add line element
    legend_elements.append(Line2D([0], [0], color=role_colors[role], marker='o', linestyle='-',
                                linewidth=2, markersize=8, label=f'{role} (Criminal)'))

# Add legend
fig.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0.1, 1, 0.96])
plt.savefig('role_accuracy.png', dpi=300, bbox_inches='tight')
print("Role-specific accuracy chart saved as 'role_accuracy.png'") 