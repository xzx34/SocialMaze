import json
import os
from pathlib import Path

# 根目录
base_dir = Path("hugging_face")
processed_data_dir = base_dir / "processed_data"
output_path = base_dir / "sft_data.json"

# 确保输出目录存在
os.makedirs(base_dir, exist_ok=True)

# SFT数据列表
sft_data = []

# 只处理包含easy.json的文件
target_file = processed_data_dir / "socialmaze_easy.json"
print(f"处理文件: {target_file}")

if not target_file.exists():
    print(f"错误：文件 {target_file} 不存在")
    exit(1)

# 加载原始数据
with open(target_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 限制提取的数据条数
max_items = 2000
data = data[:max_items] if len(data) > max_items else data

# 遍历每个数据项
count = 0
for item in data:
    # 只有当prompt和reasoning_process都不为空时才添加
    if item.get("prompt") and item.get("reasoning_process"):
        # 创建SFT数据项，使用新格式
        sft_item = {
            "instruction": item.get("system_prompt", "") + item.get("prompt", ""),
            "input": "",
            "output": item.get("reasoning_process", "")
        }
        
        sft_data.append(sft_item)
        count += 1

print(f"从{target_file}中提取了{count}条SFT数据")

# 保存SFT数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print(f"已生成SFT数据，共 {len(sft_data)} 条，保存到 {output_path}") 