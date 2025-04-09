import json
import os
import re
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import sys
import tiktoken  # 导入tiktoken用于计算token数量
import concurrent.futures  # 添加并发模块

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string.
    
    Args:
        text: The text string to count tokens for
        model: The model name to use for tokenization
        
    Returns:
        The number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # 默认使用cl100k_base编码
    
    return len(encoding.encode(text))

def extract_decision_prediction(response):
    """Extract the predicted decision from model response.
    
    Args:
        response: The model's response text
        
    Returns:
        A string 'Accept' or 'Reject', or 'Unknown' if extraction failed
    """
    # Pattern to match decision expressions
    decision_patterns = [
        r"Final Decision: (Accept|Reject)"
    ]
    
    # Check for patterns
    for pattern in decision_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            decision = match.group(1).lower()
            if decision in ['accept', 'accepted']:
                return 'Accept'
            elif decision in ['reject', 'rejected']:
                return 'Reject'
    
    # If no match found with specific patterns, look for any mentions of accept/reject
    if re.search(r'\b(accept|accepted)\b', response.lower()) and not re.search(r'\b(reject|rejected)\b', response.lower()):
        return 'Accept'
    elif re.search(r'\b(reject|rejected)\b', response.lower()) and not re.search(r'\b(accept|accepted)\b', response.lower()):
        return 'Reject'
    
    return 'Unknown'  # Return Unknown if no valid decision found

def normalize_decision(decision):
    """Normalize the decision string to Accept or Reject.
    
    Args:
        decision: The decision string from the dataset
        
    Returns:
        'Accept' or 'Reject' or 'Unknown'
    """
    if not decision or decision == 'Decision not available' or decision == 'Unknown':
        return 'Unknown'
    
    decision = decision.lower()
    
    if any(term in decision for term in ['accept', 'spotlight', 'poster', 'oral']):
        return 'Accept'
    elif any(term in decision for term in ['reject', 'desk-reject', 'declined']):
        return 'Reject'
    else:
        return 'Unknown'

def generate_system_prompt():
    """Generate system prompt for the paper evaluation task."""
    return """You are an expert reviewer for a prestigious academic conference. Your task is to evaluate a research paper and determine whether it should be accepted or rejected for publication.

Important context:
- You have access to the paper's title, abstract, reviewer comments, and author responses
- The paper should be judged by the standards of a top-tier conference
"""

def generate_user_prompt(debate_item, max_tokens=8000):
    """Generate user prompt containing the paper information, reviews, and rebuttals.
    
    Args:
        debate_item: The debate item containing paper information
        max_tokens: The maximum number of tokens allowed in the prompt
        
    Returns:
        A string with the formatted prompt
    """
    # 基础提示信息
    base_prompt = """Please analyze the following research paper and determine whether it should be accepted or rejected for publication at a top-tier conference.

"""
    
    # 添加论文信息部分
    paper_info = ""
    if len(debate_item["statements"]) >= 1 and "round1" in debate_item["statements"][0]:
        paper_info = "## Paper Information\n\n"
        paper_info += debate_item["statements"][0]["round1"] + "\n\n"
    
    # 添加审稿人评论部分
    reviewer_comments = ""
    if len(debate_item["statements"]) >= 2 and "round2" in debate_item["statements"][1]:
        reviewer_comments = "## Reviewer Comments\n\n"
        reviewer_comments += debate_item["statements"][1]["round2"] + "\n\n"
    
    # 添加作者回复部分
    author_response = ""
    if len(debate_item["statements"]) >= 3 and "round3" in debate_item["statements"][2]:
        author_response = "## Author Response\n\n"
        author_response += debate_item["statements"][2]["round3"] + "\n\n"
    
    # 结束提示信息
    final_instruction = """
Based on all the information provided, carefully analyze whether this paper should be accepted or rejected for publication.

First, provide your detailed reasoning. Then, You must conclude with your final decision in exactly this format:
Final Decision: [Accept/Reject]
"""

    # 计算各部分token数并判断是否需要截断
    current_prompt = base_prompt + paper_info + reviewer_comments + author_response + final_instruction
    current_tokens = count_tokens(current_prompt)
    
    # 如果总长度超过限制，截断作者回复部分
    if current_tokens > max_tokens and len(author_response) > 0:
        # 计算需要保留的token数量
        tokens_to_keep = max_tokens - count_tokens(base_prompt + paper_info + reviewer_comments + final_instruction)
        tokens_to_keep = max(0, tokens_to_keep)  # 确保不为负数
        
        if tokens_to_keep < 100:  # 如果空间太小，直接删除整个作者回复
            author_response = ""
            truncation_note = "## Author Response\n\n[Author response truncated due to length constraints]\n\n"
        else:
            # 截断作者回复的内容
            encoding = tiktoken.get_encoding("cl100k_base")
            full_response = debate_item["statements"][2]["round3"]
            truncated_response = encoding.decode(encoding.encode(full_response)[:tokens_to_keep])
            
            # 确保在合理的位置截断（如句子结尾）
            last_period = truncated_response.rfind('.')
            if last_period > 0 and last_period > 0.8 * len(truncated_response):
                truncated_response = truncated_response[:last_period+1]
            
            author_response = "## Author Response\n\n" + truncated_response + "\n\n[Response truncated due to length constraints]\n\n"
    
    # 组合最终提示
    prompt = base_prompt + paper_info + reviewer_comments + author_response + final_instruction
    
    prompt=base_prompt + paper_info + reviewer_comments + final_instruction
    return prompt

def process_paper(debate_item, model, system_prompt):
    """处理单篇论文并返回结果（用于并行处理）
    
    Args:
        debate_item: 包含论文信息的数据项
        model: 使用的模型名称
        system_prompt: 系统提示
        
    Returns:
        包含评估结果的字典
    """
    paper_id = debate_item["id"]
    true_decision = normalize_decision(debate_item["decision"])
    
    # 跳过未知决策的论文
    if true_decision == 'Unknown':
        return None
    
    # 生成用户提示
    user_prompt = generate_user_prompt(debate_item)
    
    # 获取模型响应
    response = get_chat_response(
        model=model,
        system_message=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.1
    )
    
    # 提取预测结果
    prediction = extract_decision_prediction(response)
    
    # 检查未知预测
    if prediction == 'Unknown':
        # 默认使用最常见的决策（通常是Reject）
        prediction = 'Reject'
        print(f"Warning: Unknown prediction for paper {paper_id}, defaulting to 'Reject'")
    
    is_correct = prediction == true_decision
    
    # 存储结果
    result = {
        "paper_id": paper_id,
        "source": debate_item.get("source", "Unknown"),
        "true_decision": true_decision,
        "prediction": prediction,
        "correct": is_correct,
        "response": response
    }
    
    return result

def evaluate_model(model, dataset_path, num_papers=None, output_file=None, max_workers=None):
    """
    Evaluate model performance on the paper acceptance decision task
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_papers: number of papers to evaluate (None for all)
        output_file: optional file to save results
        max_workers: maximum number of concurrent workers (None for default)
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample papers to evaluate
    if num_papers is not None and num_papers < len(dataset):
        dataset = dataset[:num_papers]
    
    results = []
    correct_predictions = 0
    total_valid_papers = 0
    
    # Create confusion matrix (true_decision, predicted_decision)
    # 0: Reject, 1: Accept
    confusion_matrix = np.zeros((2, 2), dtype=int)
    decision_mapping = {"Reject": 0, "Accept": 1}
    
    # 生成系统提示（所有请求共用）
    system_prompt = generate_system_prompt()
    
    # 使用进程池并行处理论文
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_paper = {
            executor.submit(process_paper, debate_item, model, system_prompt): debate_item 
            for debate_item in dataset
        }
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_paper), 
                          total=len(future_to_paper),
                          desc=f"Evaluating {model}"):
            result = future.result()
            
            # 跳过无效结果（未知决策的论文）
            if result is None:
                continue
                
            results.append(result)
            total_valid_papers += 1
            
            # 更新正确预测计数
            if result["correct"]:
                correct_predictions += 1
            
            # 更新混淆矩阵
            true_decision = result["true_decision"]
            prediction = result["prediction"]
            
            if true_decision in decision_mapping and prediction in decision_mapping:
                true_idx = decision_mapping[true_decision]
                pred_idx = decision_mapping[prediction]
                confusion_matrix[true_idx][pred_idx] += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_valid_papers * 100 if total_valid_papers > 0 else 0
    
    # Calculate statistics per class
    stats = {}
    for decision in ["Reject", "Accept"]:
        if decision in decision_mapping:
            idx = decision_mapping[decision]
            true_positives = confusion_matrix[idx][idx]
            false_negatives = sum(confusion_matrix[idx]) - true_positives
            false_positives = sum(confusion_matrix[:, idx]) - true_positives
            
            total = sum(confusion_matrix[idx])
            if total > 0:
                recall = true_positives / total * 100
            else:
                recall = 0
                
            total_predicted = sum(confusion_matrix[:, idx])
            if total_predicted > 0:
                precision = true_positives / total_predicted * 100
            else:
                precision = 0
                
            stats[decision] = {
                "total": int(total),
                "recall": recall,
                "precision": precision
            }
    
    # Create summary
    summary = {
        "model": model,
        "total_papers": total_valid_papers,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "stats": stats
    }
    
    # Save results if output file specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        full_results = {
            "summary": summary,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model performance on paper acceptance decisions')
    # llama-3.1-8B, gemma-2-9b, gemma-2-27B,llama-3.3-70B,qwen-2.5-72B,qwq,deepseek-r1
    parser.add_argument('--models', type=str, nargs='+', default=['llama-3.1-8B','gemma-2-9b','gemma-2-27B','llama-3.3-70B','qwen-2.5-72B','qwq','deepseek-r1','gpt-4o-mini','gpt-4o','o3-mini'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--dataset', type=str, default='data/debate.json', 
                        help='Path to debate dataset')
    parser.add_argument('--num_papers', type=int, default=100, 
                        help='Number of papers to evaluate (default: all)')
    parser.add_argument('--force_reevaluate', action='store_true',
                        help='Force re-evaluation of models even if they exist in summary')
    parser.add_argument('--max_workers', type=int, default=25,
                        help='Maximum number of concurrent workers (default: auto)')
    
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_arguments()
    
    # Ensure results directory exists
    results_dir = Path("results")
    if not results_dir.exists():
        os.makedirs(results_dir)
    
    # Get dataset basename for result files
    dataset_basename = os.path.basename(args.dataset).split('.')[0]
    
    # Create a summary dict to track all model results
    summary_file = results_dir / f"summary_{dataset_basename}.json"
    
    all_results = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                print(f"Loaded existing summary file with {len(all_results)} models.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing summary file. Starting with empty results.")
    
    # Evaluate each model
    for model in args.models:
        if model in all_results and not args.force_reevaluate:
            print(f"\nModel {model} already evaluated. Skipping.")
            continue
            
        print(f"\nEvaluating model: {model}")
        
        # Create unique output filename for this model
        output_file = results_dir / f"{model}_{dataset_basename}_results.json"
        
        # Run evaluation
        model_results = evaluate_model(model, args.dataset, args.num_papers, output_file, args.max_workers)
        
        # Add to results summary
        all_results[model] = {
            "accuracy": model_results["accuracy"],
            "correct": model_results["correct_predictions"],
            "total": model_results["total_papers"],
            "stats": model_results["stats"]
        }
        
        # Print results
        print(f"\nResults for {model}:")
        print(f"Accuracy: {model_results['accuracy']:.2f}%")
        print(f"Correct: {model_results['correct_predictions']} / {model_results['total_papers']}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("             | Predicted Reject | Predicted Accept |")
        print("-------------|------------------|------------------|")
        print(f"True Reject  | {model_results['confusion_matrix'][0][0]:18d} | {model_results['confusion_matrix'][0][1]:18d} |")
        print(f"True Accept  | {model_results['confusion_matrix'][1][0]:18d} | {model_results['confusion_matrix'][1][1]:18d} |")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary of all models saved to {summary_file}")

if __name__ == "__main__":
    main()
