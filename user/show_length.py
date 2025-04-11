import os
import json
import csv
import re

def extract_model_and_task(filename):
    """从文件名中提取模型名称和任务类型"""
    pattern = r'(.+)_(entity|persona)_results\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def calculate_average_length(data):
    """计算平均回答长度"""
    total_length = 0
    count = 0

    # 尝试获取结果列表
    results = data.get("results", [])
    if not results and isinstance(data, dict) and "summary" in data:
        # 如果结果在其他位置，尝试获取
        results = data.get("detailed_results", [])

    for result in results:
        response = result.get("response", "")
        total_length += len(response)
        count += 1

    return total_length / count if count > 0 else 0

def get_accuracy(data):
    """获取准确率数据"""
    # 如果accuracy被包裹在summary中
    if isinstance(data, dict) and "summary" in data:
        summary = data.get("summary", {})
        # 尝试获取overall_accuracy，如果不存在则尝试其他可能的字段
        accuracy = summary.get("overall_accuracy", None)
        if accuracy is None:
            accuracy = summary.get("accuracy", 0)
    else:
        accuracy = data.get("accuracy", 0)

    return accuracy

def main():
    results_dir = "results"
    exclude_models = []  # 如果需要排除模型，可以在这里添加

    all_results = []
    model_length_stats = {}

    for filename in os.listdir(results_dir):
        if (filename.endswith("_entity_results.json") or 
            filename.endswith("_persona_results.json")) and not filename.startswith("entity_summary") and not filename.startswith("persona_summary"):

            model_name, task_type = extract_model_and_task(filename)

            # 跳过排除的模型
            if model_name in exclude_models:
                continue

            if model_name and task_type:
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    avg_length = calculate_average_length(data)
                    accuracy = get_accuracy(data)

                    # 添加到所有结果列表
                    all_results.append({
                        "task": task_type,
                        "model": model_name,
                        "avg_response_length": avg_length,
                        "accuracy": accuracy
                    })

                    # 为每个模型累计数据
                    if model_name not in model_length_stats:
                        model_length_stats[model_name] = {
                            "model": model_name,
                            "total_length": 0,
                            "total_accuracy": 0,
                            "count": 0
                        }

                    model_length_stats[model_name]["total_length"] += avg_length
                    model_length_stats[model_name]["total_accuracy"] += accuracy
                    model_length_stats[model_name]["count"] += 1

                    print(f"处理完成: {task_type} - {model_name}")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    # 计算每个模型的平均长度和平均准确率
    for model_name in model_length_stats:
        stats = model_length_stats[model_name]
        if stats["count"] > 0:
            stats["avg_response_length"] = stats["total_length"] / stats["count"]
            stats["avg_accuracy"] = stats["total_accuracy"] / stats["count"]
        else:
            stats["avg_response_length"] = 0
            stats["avg_accuracy"] = 0

        # 删除中间计算字段
        del stats["total_length"]
        del stats["total_accuracy"]
        del stats["count"]

    # 写入汇总CSV文件（按任务和模型）
    summary_output = "user_accuracy_summary.csv"

    if all_results:
        with open(summary_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["task", "model", "avg_response_length", "accuracy"])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"汇总结果已保存到 {summary_output}")

    # 写入模型长度统计CSV文件
    model_stats_output = "model_length_stats.csv"

    if model_length_stats:
        with open(model_stats_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "avg_response_length", "avg_accuracy"])
            writer.writeheader()
            writer.writerows(model_length_stats.values())

        print(f"模型长度统计已保存到 {model_stats_output}")

if __name__ == "__main__":
    main()