import json
import os
import glob
from pathlib import Path

# 根目录
base_dir = Path("hugging_face")
result_dir = base_dir / "result"
dpo_dir = base_dir / "dpo"  # 添加dpo目录
processed_data_path = base_dir / "processed_data" / "socialmaze_easy.json"
output_path = base_dir / "dpo_phi_4.json"

# 加载socialmaze_easy.json数据
with open(processed_data_path, 'r', encoding='utf-8') as f:
    socialmaze_data = json.load(f)

# 用于存储DPO数据的列表
dpo_data = []

# 获取所有模型文件夹（从两个目录）
model_dirs = ['llama-3.1-8B']
dirs_to_check = [result_dir, dpo_dir]  # 需要检查的两个目录

# 遍历每个目录
for dir_path in dirs_to_check:
    if not dir_path.exists():
        print(f"目录 {dir_path} 不存在，跳过")
        continue
    
    print(f"处理目录 {dir_path}...")
    
    # 遍历每个模型文件夹
    for model_name in model_dirs:
        model_dir = dir_path / model_name
        responses_path = model_dir / "responses_easy.json"
        
        if not model_dir.exists() or not responses_path.exists():
            print(f"跳过 {model_dir}，目录不存在或找不到 responses_easy.json")
            continue
        
        print(f"处理 {model_dir}...")
        
        # 加载模型的响应数据
        with open(responses_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        
        # 遍历响应数据
        for response_item in responses:
            # 检查是否both_correct为false
            if not response_item.get("both_correct", True):
                sample_id = response_item["sample_id"]
                
                # 确保sample_id在有效范围内
                if sample_id < 0 or sample_id >= len(socialmaze_data):
                    print(f"跳过 sample_id {sample_id}，超出socialmaze_easy.json范围")
                    continue
                
                socialmaze_item = socialmaze_data[sample_id]
                
                # 创建DPO数据项
                dpo_item = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": socialmaze_item.get("system_prompt", "") + socialmaze_item.get("prompt", "")
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": socialmaze_item.get("reasoning_process", ""),
                        "model": "Algorithm"
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": response_item.get("response", ""),
                        "model": model_name
                    }
                }
                
                dpo_data.append(dpo_item)
        
        print(f"完成 {model_dir}，找到 {len([r for r in responses if not r.get('both_correct', True)])} 个响应")

# 保存DPO数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(f"已生成DPO数据，共 {len(dpo_data)} 条，保存到 {output_path}") 