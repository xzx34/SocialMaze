import os
import json
import csv
import re

def extract_model_task_difficulty(filename):
    """从文件名中提取模型名称、任务类型和难度"""
    pattern = r'(.+)_(easy|hard)_(cluster|count|group|reasoning)_results\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(3), match.group(2)
    return None, None, None

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
        # 尝试获取accuracy字段
        accuracy = summary.get("accuracy", 0)
    else:
        accuracy = data.get("accuracy", 0)

    return accuracy

def main():
    results_dir = "results"
    exclude_models = []  # 如果需要排除模型，可以在这里添加

    all_results = []
    model_length_stats = {}
    model_task_difficulty_stats = {}  # 存储模型-任务-难度组合的统计信息

    for filename in os.listdir(results_dir):
        if "_results.json" in filename and not filename.startswith("relation_summary"):
            model_name, task_type, difficulty = extract_model_task_difficulty(filename)

            # 跳过排除的模型
            if model_name in exclude_models:
                continue

            if model_name and task_type and difficulty:
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    avg_length = calculate_average_length(data)
                    accuracy = get_accuracy(data)

                    # 创建完整任务标识符
                    full_task = f"{difficulty}_{task_type}"

                    # 添加到所有结果列表
                    all_results.append({
                        "model": model_name,
                        "task": task_type,
                        "difficulty": difficulty,
                        "full_task": full_task,
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

                    # 存储模型-任务-难度组合的统计信息
                    key = f"{model_name}_{task_type}_{difficulty}"
                    model_task_difficulty_stats[key] = {
                        "model": model_name,
                        "task": task_type,
                        "difficulty": difficulty,
                        "full_task": full_task,
                        "avg_response_length": avg_length,
                        "accuracy": accuracy
                    }

                    print(f"处理完成: {difficulty} {task_type} - {model_name}")

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

    # 写入详细汇总CSV文件
    summary_output = "relation_accuracy_summary.csv"

    if all_results:
        with open(summary_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "task", "difficulty", "full_task", 
                "avg_response_length", "accuracy"
            ])
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

    # 另外生成一个按任务类型和难度分组的汇总表
    task_summary_output = "relation_task_summary.csv"

    # 计算每个任务类型和难度级别的平均值
    task_difficulty_stats = {}
    for key, data in model_task_difficulty_stats.items():
        task = data["task"]
        difficulty = data["difficulty"]
        full_task = data["full_task"]

        task_key = f"{task}_{difficulty}"
        if task_key not in task_difficulty_stats:
            task_difficulty_stats[task_key] = {
                "task": task,
                "difficulty": difficulty,
                "full_task": full_task,
                "total_length": 0,
                "total_accuracy": 0,
                "count": 0
            }

        task_difficulty_stats[task_key]["total_length"] += data["avg_response_length"]
        task_difficulty_stats[task_key]["total_accuracy"] += data["accuracy"]
        task_difficulty_stats[task_key]["count"] += 1

    # 计算平均值
    for key in task_difficulty_stats:
        stats = task_difficulty_stats[key]
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

    # 写入任务汇总文件
    if task_difficulty_stats:
        with open(task_summary_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "task", "difficulty", "full_task", 
                "avg_response_length", "avg_accuracy"
            ])
            writer.writeheader()
            writer.writerows(task_difficulty_stats.values())

        print(f"任务汇总结果已保存到 {task_summary_output}")

if __name__ == "__main__":
    main()