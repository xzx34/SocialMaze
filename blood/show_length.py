import os
import json
import csv
import re

def extract_model_name(filename):
    """从文件名中提取模型名称"""
    match = re.match(r'(.+)_6_all_results\.json', filename)
    if match:
        return match.group(1)
    return None

def calculate_average_length(data, round_num=3):
    """计算特定轮次的平均回答长度"""
    total_length = 0
    count = 0
    
    for scenario in data["detailed_results"]:
        for round_data in scenario.get("rounds", []):
            if round_data.get("round") == 3 or round_data.get("round") == 2 or round_data.get("round") == 1:
                response = round_data.get("response", "")
                # 移除<think>...</think>标记内的内容
                # response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                total_length += len(response)
                count += 1   
    
    return total_length / count * 3 if count > 0 else 0

def get_round_accuracy(data, round_num=3):
    """获取特定轮次的准确率"""
    summary = data.get("summary", {})
    rounds = summary.get("rounds", {})
    round_data = rounds.get(str(round_num), {})
    
    criminal_accuracy = round_data.get("criminal_accuracy", 0)
    self_role_accuracy = round_data.get("self_role_accuracy", 0)
    
    return criminal_accuracy, self_role_accuracy

def main():
    results_dir = "results"
    output_file = "model_round3_stats.csv"
    
    # 收集结果
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_6_all_results.json"):
            model_name = extract_model_name(filename)
            if model_name:
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    avg_length = calculate_average_length(data, 3)
                    criminal_acc, self_role_acc = get_round_accuracy(data, 3)
                    
                    results.append({
                        "model": model_name,
                        "avg_response_length": avg_length,
                        "criminal_accuracy": criminal_acc,
                        "self_role_accuracy": self_role_acc
                    })
                    
                    print(f"处理完成: {model_name}")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
    
    # 写入CSV文件
    if results:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "avg_response_length", "criminal_accuracy", "self_role_accuracy"])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"结果已保存到 {output_file}")
    else:
        print("没有找到可以处理的文件")

if __name__ == "__main__":
    main()