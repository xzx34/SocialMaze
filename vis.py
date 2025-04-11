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

# Create Figure 1: Accuracy Comparison (Line Chart)
plt.figure(figsize=(12, 7))
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines with markers
x = np.arange(len(sorted_tasks))
ax.plot(x, long_chain_accuracies, marker=marker_styles[0], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Long-Chain Models', color=line_colors[0])
ax.plot(x, short_chain_accuracies, marker=marker_styles[1], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Short-Chain Models', color=line_colors[1])

# Fill areas under lines with light colors
ax.fill_between(x, long_chain_accuracies, alpha=0.2, color=colors[0])
ax.fill_between(x, short_chain_accuracies, alpha=0.2, color=colors[1])

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_tasks, fontsize=11)  # 增大x轴标签字体
ax.legend(fontsize=14, loc='upper right')  # 增大图例标签字体

# Add light grid lines only on y-axis
ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Set y-axis limits
y_max = max(max(long_chain_accuracies), max(short_chain_accuracies))
ax.set_ylim(0, min(100, y_max * 1.15))

plt.tight_layout()
plt.savefig('accuracy_comparison_line.png', dpi=300, bbox_inches='tight')
plt.close()

# Create Figure 2: Length Comparison (Line Chart)
plt.figure(figsize=(12, 7))
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines with markers
ax.plot(x, long_chain_lengths, marker=marker_styles[0], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Long-Chain Models', color=line_colors[0])
ax.plot(x, short_chain_lengths, marker=marker_styles[1], linestyle='-', linewidth=line_width, 
        markersize=marker_size, label='Short-Chain Models', color=line_colors[1])

# Fill areas under lines with light colors
ax.fill_between(x, long_chain_lengths, alpha=0.2, color=colors[0])
ax.fill_between(x, short_chain_lengths, alpha=0.2, color=colors[1])

ax.set_ylabel('Response Length (chars)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(display_tasks, fontsize=11)  # 增大x轴标签字体
ax.legend(fontsize=14, loc='upper right')  # 增大图例标签字体

# Add light grid lines only on y-axis
ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Set y-axis limits
y_max = max(max(long_chain_lengths), max(short_chain_lengths))
ax.set_ylim(0, y_max * 1.15)

plt.tight_layout()
plt.savefig('length_comparison_line.png', dpi=300, bbox_inches='tight')
plt.close()

print("Line charts generated successfully!")