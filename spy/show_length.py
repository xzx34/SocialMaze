import os
import json
import csv
import re

def extract_model_name(filename):
    """从文件名中提取模型名称"""
    match = re.match(r'(.+)_results\.json', filename)
    if match:
        return match.group(1)
    return None

def calculate_average_length(data):
    """计算平均回答长度"""
    total_length = 0
    count = 0

    for result in data.get("results", []):
        response = result.get("response", "")
        total_length += len(response)
        count += 1

    return total_length / count if count > 0 else 0

def get_accuracy(data):
    """获取准确率数据"""
    return data.get("accuracy", 0)

def main():
    results_dir = "results"
    exclude_models = []  # 如果需要排除模型，可以在这里添加

    all_results = []
    model_length_stats = {}

    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json") and not filename.startswith("summary"):
            model_name = extract_model_name(filename)

            # 跳过排除的模型
            if model_name in exclude_models:
                continue

            if model_name:
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    avg_length = calculate_average_length(data)
                    accuracy = get_accuracy(data)

                    # 添加到所有结果列表
                    all_results.append({
                        "model": model_name,
                        "avg_response_length": avg_length,
                        "accuracy": accuracy
                    })

                    # 累计长度和准确率数据用于计算平均值
                    if model_name not in model_length_stats:
                        model_length_stats[model_name] = {
                            "model": model_name,
                            "avg_response_length": avg_length,
                            "avg_accuracy": accuracy
                        }

                    print(f"处理完成: {model_name}")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    # 写入汇总CSV文件（按模型）
    summary_output = "spy_accuracy_summary.csv"

    if all_results:
        with open(summary_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "avg_response_length", "accuracy"])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"汇总结果已保存到 {summary_output}")

    # 写入模型长度统计CSV文件（本例中与summary相同，因为每个模型只有一个文件）
    model_stats_output = "model_length_stats.csv"

    if model_length_stats:
        with open(model_stats_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "avg_response_length", "avg_accuracy"])
            writer.writeheader()
            writer.writerows(model_length_stats.values())

        print(f"模型长度统计已保存到 {model_stats_output}")

if __name__ == "__main__":
    main()