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
length_ratios = [task_results[task]['length_ratio'] for task in sorted_tasks]

# Task name mapping for better readability
task_name_map = {
    'blood': 'Role\nDeduction',
    'debate': 'Review\nDecision',
    'relation': 'Graph\nAnalysis',
    'review': 'Rating\nEstimation',
    'spy': 'Find\nthe Spy',
    'user': 'User Profile\nInference'
}

display_tasks = [task_name_map.get(task, task) for task in sorted_tasks]

# 设置更优雅的配色方案
line_colors = ['#47659D', '#8B8CC2']  
bar_color = '#B8A7CD'  
marker_styles = ['o', 's']
line_width = 3
marker_size = 10

# 创建图表
plt.figure(figsize=(14, 9))  # 增加高度以适应两行任务名
fig, ax1 = plt.subplots(figsize=(14, 9))

# 设置全局字体大小
plt.rcParams.update({'font.size': 18})  # 增加基础字体大小
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5  # 稍微增加轴线宽度

# 定义x坐标
x = np.arange(len(sorted_tasks))
bar_width = 0.6

# 创建右Y轴用于长度比柱状图
ax2 = ax1.twinx()

# 先绘制柱状图，并增加更高的透明度
bars = ax2.bar(x, length_ratios, bar_width, color=bar_color,
              label='Output Length Ratio (Long/Short)', 
              edgecolor='black', linewidth=2,
              alpha=0.8)  # 增加透明度使折线更清晰可见

# 确保折线图的标记点更大更突出
line1 = ax1.plot(x, long_chain_accuracies, marker=marker_styles[0], linestyle='-', linewidth=line_width+1, 
        markersize=marker_size+2, label='Long-Chain Models (Accuracy)', color=line_colors[0])
line2 = ax1.plot(x, short_chain_accuracies, marker=marker_styles[1], linestyle='-', linewidth=line_width+1, 
        markersize=marker_size+2, label='Short-Chain Models (Accuracy)', color=line_colors[1])

# 添加折线图的标记点边框，使它们更加突出
for i, (y1, y2) in enumerate(zip(long_chain_accuracies, short_chain_accuracies)):
    ax1.plot(x[i], y1, 'o', markersize=marker_size+2, markerfacecolor=line_colors[0], 
             markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(x[i], y2, 's', markersize=marker_size+2, markerfacecolor=line_colors[1], 
             markeredgecolor='white', markeredgewidth=1.5)

# 设置左Y轴（准确率）
ax1.set_ylabel('Accuracy (%)', fontsize=26, fontweight='bold')  # 增大字体
ax1.set_ylim(0, min(100, max(max(long_chain_accuracies), max(short_chain_accuracies)) * 1.15))
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
ax1.tick_params(axis='y', labelsize=20)  # 增大刻度标签

# 设置右Y轴（长度比）
ax2.set_ylabel('Output Length Ratio (Long/Short)', fontsize=26, fontweight='bold')  # 增大字体
ax2.set_ylim(0, max(length_ratios) * 2.4)
ax2.tick_params(axis='y', labelsize=22)  # 增大刻度标签

# 设置X轴 - 不旋转，并增加间距适应两行文本
ax1.set_xticks(x)
ax1.set_xticklabels(display_tasks, fontsize=22, rotation=0, ha='center')  # 不旋转，居中对齐
ax1.tick_params(axis='x', labelsize=22, pad=10)  # 增大刻度标签并增加底部间距

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=22, framealpha=1)  # 增大图例字体

# 添加网格线以便更好地阅读数据
ax1.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=2.0)  # 增加额外的填充以确保标签不被裁剪
plt.savefig('deep_reason.png', dpi=300, bbox_inches='tight')
plt.close()

print("Optimized chart generated successfully!")