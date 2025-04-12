import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define task folders
task_folders = ['blood', 'debate', 'relation', 'review', 'spy', 'user']

# Define long and short chain models
long_chain_models = ['deepseek-r1', 'qwq']

# Store all task data
all_task_data = {}

# Read model_length_stats.csv from each task folder
for task in task_folders:
    file_path = os.path.join(task, 'model_length_stats.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            all_task_data[task] = df
            print(f"Successfully read data from {task}")
        except Exception as e:
            print(f"Error reading {task} data: {e}")
    else:
        print(f"File not found: {file_path}")

# Calculate average accuracy and length for long/short chain models for each task
task_results = {}

for task, df in all_task_data.items():
    # Filter long/short chain models
    long_chain_df = df[df['model'].isin(long_chain_models)]
    short_chain_df = df[~df['model'].isin(long_chain_models)]

    # Calculate averages
    if not long_chain_df.empty:
        long_chain_avg_accuracy = long_chain_df['avg_accuracy'].mean()
        long_chain_avg_length = long_chain_df['avg_response_length'].mean()
    else:
        long_chain_avg_accuracy = 0
        long_chain_avg_length = 0

    if not short_chain_df.empty:
        short_chain_avg_accuracy = short_chain_df['avg_accuracy'].mean()
        short_chain_avg_length = short_chain_df['avg_response_length'].mean()
    else:
        short_chain_avg_accuracy = 0
        short_chain_avg_length = 0

    # Store results
    task_results[task] = {
        'long_chain_accuracy': long_chain_avg_accuracy,
        'short_chain_accuracy': short_chain_avg_accuracy,
        'long_chain_length': long_chain_avg_length,
        'short_chain_length': short_chain_avg_length,
        'length_ratio': long_chain_avg_length / short_chain_avg_length if short_chain_avg_length > 0 else 0
    }

# Sort tasks by length ratio (descending) to reflect "deep thinking" vs "shallow thinking"
sorted_tasks = sorted(task_results.keys(), key=lambda x: task_results[x]['length_ratio'], reverse=True)

# Extract sorted data
long_chain_accuracies = [task_results[task]['long_chain_accuracy'] for task in sorted_tasks]
short_chain_accuracies = [task_results[task]['short_chain_accuracy'] for task in sorted_tasks]
long_chain_lengths = [task_results[task]['long_chain_length'] for task in sorted_tasks]
short_chain_lengths = [task_results[task]['short_chain_length'] for task in sorted_tasks]

# Task name mapping for better readability
task_name_map = {
    'blood': 'Hidden Role',
    'debate': 'Review Decision',
    'relation': 'Social Graph',
    'review': 'Rating Estimation',
    'spy': 'Word Spy',
    'user': 'User Profile Inference'
}

display_tasks = [task_name_map.get(task, task) for task in sorted_tasks]

# Set up light color palette and marker styles
colors = ['#C7EAEC', '#AED9CE']
line_colors = ['#2E86C1', '#28B463']  # Slightly darker for better line visibility
marker_styles = ['o', 's']
line_width = 2.5
marker_size = 9

# 计算长度比
length_ratios = [task_results[task]['length_ratio'] for task in sorted_tasks]

# 创建组合图表（准确率折线图 + 长度比柱状图）
plt.figure(figsize=(14, 8))
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

fig, ax1 = plt.subplots(figsize=(14, 8))

# 绘制准确率折线图（左Y轴）- 移除阴影填充
x = np.arange(len(sorted_tasks))
line1 = ax1.plot(x, long_chain_accuracies, marker=marker_styles[0], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Long-Chain Models (Accuracy)', color=line_colors[0])
line2 = ax1.plot(x, short_chain_accuracies, marker=marker_styles[1], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Short-Chain Models (Accuracy)', color=line_colors[1])

# 设置左Y轴（准确率）
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, min(100, max(max(long_chain_accuracies), max(short_chain_accuracies)) * 1.15))
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# 创建右Y轴用于长度比柱状图 - 修改颜色并添加黑色边框
ax2 = ax1.twinx()
bar_width = 0.4
bar_color = '#AED9CE'  # 更改柱状图颜色
bars = ax2.bar(x, length_ratios, bar_width, alpha=0.7, color=bar_color, 
              label='Length Ratio (Long/Short)', 
              edgecolor='black', linewidth=2)  # 添加粗黑色边框

# 移除柱状图数值标签

# 设置右Y轴（长度比）
ax2.set_ylabel('Length Ratio (Long/Short)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(length_ratios) * 2.5)  # 为标签留出空间

# 设置X轴
ax1.set_xticks(x)
ax1.set_xticklabels(display_tasks, fontsize=14)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)

plt.tight_layout()
plt.savefig('combined_accuracy_length_ratio.png', dpi=300, bbox_inches='tight')
plt.close()

print("Combined chart generated successfully!")