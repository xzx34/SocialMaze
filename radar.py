import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Style and Font ---
plt.style.use('seaborn-v0_8-colorblind')
# Optional: Font settings if needed
# try:
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
# except:
#     print("SimHei font not found, using default.")
#     pass

# 设置全局字体大小
plt.rcParams['font.size'] = 20

# --- Data Definition (Same as before) ---
tasks = [
    'Hidden Role\nDeduction', # Top label
    'Find the\nSpy',
    'Rating\nEstimation',
    'Social Graph\nAnalysis', # Bottom-left label
    'Review Decision\nPrediction', # Bottom-right label
    'User Profile\nInference'
]

model_data = {
    'Llama-3.1-8B':    [2.0, 37.2, 57.2, 28.2, 62.0, 60.2],
    'Llama-3.3-70B':   [9.0, 60.0, 74.8, 81.0, 72.2, 78.6],
    'Phi-4':           [6.2, 45.2, 60.4, 40.6, 61.4, 62.4],
    'Qwen-2.5-72B':    [5.6, 48.9, 72.2, 80.6, 65.8, 68.0],
    'QwQ-32B':         [59.4, 50.2, 74.4, 95.0, 79.6, 72.2],
    'GPT-4o-mini':     [4.6, 61.2, 75.8, 53.0, 85.0, 74.4],
    'GPT-4o':          [8.2, 69.2, 76.0, 83.2, 90.2, 79.2],
    'o3-mini':         [22.2, 74.0, 71.2, 99.0, 78.6, 71.4],
    'o1':              [50.8, 78.4, 76.2, 99.2, 78.2, 77.0],
    'DeepSeek-R1':     [85.6, 70.2, 71.0, 98.6, 82.0, 74.6],
    'Gemeni-2.5-Pro':  [90.2, 76.6, 73.6, 100.0, 77.6, 73.0]
}

# --- Model Selection (Your chosen models) ---
models_to_plot = [
    'GPT-4o',
    'QwQ-32B',
    'Llama-3.3-70B',
    'DeepSeek-R1',
    'Gemeni-2.5-Pro'
]

# --- Plotting Code with Enhancements ---
num_vars = len(tasks)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Close the loop

# --- ADJUST FIGURE SIZE FOR WIDER ASPECT RATIO ---
fig, ax = plt.subplots(figsize=(16, 14), subplot_kw=dict(polar=True)) # 增加整体图表尺寸

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Helper function
def add_to_radar(model_name, data, color=None, marker='o'):
    values = data + data[:1]
    ax.plot(angles, values,
            color=color,
            linewidth=1.8,
            linestyle='solid',
            label=model_name,
            marker=marker,
            markersize=5)
    ax.fill(angles, values, color=color, alpha=0.2)

# Plot selected models
for i, model_name in enumerate(models_to_plot):
    if model_name in model_data:
        data = model_data[model_name]
        add_to_radar(model_name, data, color=colors[i % len(colors)])
    else:
        print(f"Warning: Model '{model_name}' not found in data. Skipping.")

# --- Customize the plot appearance ---
ax.set_ylim(0, 100) # Keep Y limits standard for percentage

# Set axis labels (task names)
ax.set_xticks(angles[:-1])
# --- INCREASED FONT SIZE FURTHER ---
ax.set_xticklabels(tasks, size=20) # 增加任务标签字体大小

# --- INCREASE PADDING TO TASK LABELS AGAIN ---
pad_value = 55 # 增加内边距以适应更大的字体
ax.tick_params(axis='x', pad=pad_value)

# Set Y-axis labels (percentage values)
ax.set_yticks(np.arange(0, 101, 25))
ax.set_yticklabels([f"{i}%" for i in np.arange(0, 101, 25)], size=14, color='grey') # 增加百分比标签字体大小

ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.8)
ax.spines['polar'].set_visible(True)

ax.set_rlabel_position(180) # Position percentage labels

# Legend - Adjust position if needed due to wider figure
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.9), # 调整图例位置
          fontsize=16, frameon=True, shadow=True) # 增加图例字体大小

# --- ADJUST OVERALL PADDING ---
plt.tight_layout(pad=9.0) # 增加内边距以适应更大的元素

# --- SAVE FIGURE AS PDF ---
output_filename_pdf = "socialmaze_model_comparison.pdf"
output_filename_png = "socialmaze_model_comparison.png"
plt.savefig(output_filename_pdf, format='pdf', bbox_inches='tight', dpi=300)

# 同时保存PNG格式便于查看
plt.savefig(output_filename_png, format='png', bbox_inches='tight', dpi=300)

print(f"雷达图已保存为 {output_filename_pdf} 和 {output_filename_png}")

# Optionally, still display the plot if running interactively
# plt.show()