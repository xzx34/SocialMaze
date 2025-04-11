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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

def extract_answer(response, question_type):
    """Extract the predicted answer from model response based on question type.
    
    Args:
        response: The model's response text
        question_type: Type of question (reasoning, group, cluster, count)
        
    Returns:
        Extracted answer or 'Unknown' if extraction failed
    """
    # More flexible pattern to match "Final Answer: " format with potential markdown and whitespace
    pattern = r"\*?\*?Final\s*Answer\s*:\s*(.+?)(?:\*?\*?)?(?:$|\n)"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if not match:
        return "Unknown"
    
    answer = match.group(1).strip()
    
    # Process answer based on question type
    if question_type == "reasoning":
        # Yes/No answer
        if answer.lower() in ["yes", "no"]:
            return answer.capitalize()
        return "Unknown"
    
    elif question_type == "group":
        # List of people or "No one"
        return answer
    
    elif question_type == "cluster":
        # Number of groups - should be a number
        if answer.isdigit():
            return answer
        # Try to extract a number if embedded in text
        number_match = re.search(r"\d+", answer)
        if number_match:
            return number_match.group(0)
        return "Unknown"
    
    elif question_type == "count":
        # Format: "X pairs have good relationships, Y pairs have bad relationships"
        pairs_match = re.search(r"(\d+)\s+pairs\s+have\s+good\s+relationships,\s+(\d+)\s+pairs\s+have\s+bad\s+relationships", answer)
        if pairs_match:
            return f"{pairs_match.group(1)} pairs have good relationships, {pairs_match.group(2)} pairs have bad relationships"
        return "Unknown"
    
    return "Unknown"

def is_correct_answer(prediction, true_answer, question_type):
    """Check if the prediction matches the true answer.
    
    Args:
        prediction: Extracted prediction from model
        true_answer: Ground truth answer
        question_type: Type of question
        
    Returns:
        Boolean indicating if the answer is correct
    """
    if prediction == "Unknown":
        return False
    
    if question_type == "reasoning":
        # Simple Yes/No comparison
        return prediction.lower() == true_answer.lower()
    
    elif question_type == "group":
        # Handle "No one" case
        if true_answer == "No one" and prediction.lower() in ["no one", "none"]:
            return True
        
        # Compare lists of names
        true_names = set(name.strip() for name in true_answer.split(","))
        pred_names = set(name.strip() for name in prediction.split(","))
        return true_names == pred_names
    
    elif question_type == "cluster":
        # Compare numbers
        return prediction.strip() == true_answer.strip()
    
    elif question_type == "count":
        # Compare count values
        true_good_count = re.search(r"(\d+)\s+pairs\s+have\s+good", true_answer)
        true_bad_count = re.search(r"(\d+)\s+pairs\s+have\s+bad", true_answer)
        
        pred_good_count = re.search(r"(\d+)\s+pairs\s+have\s+good", prediction)
        pred_bad_count = re.search(r"(\d+)\s+pairs\s+have\s+bad", prediction)
        
        if not (true_good_count and true_bad_count and pred_good_count and pred_bad_count):
            return False
        
        return (true_good_count.group(1) == pred_good_count.group(1) and
                true_bad_count.group(1) == pred_bad_count.group(1))
    
    return False

def evaluate_single_sample(args):
    """Evaluate a single sample
    
    Args:
        args: tuple containing (model, sample, idx, question_type)
        
    Returns:
        Dictionary with the evaluation result
    """
    model, sample, idx, question_type = args
    
    # Skip invalid samples
    if "user_prompt" not in sample or "answer" not in sample:
        return None
    
    # Extract true answer
    true_answer = sample["answer"]
    if "Final Answer: " in true_answer:
        true_answer = true_answer.replace("Final Answer: ", "")
    
    # Extract system_prompt from the sample data
    system_prompt = sample["system_prompt"]
    user_prompt = sample["user_prompt"]
    
    try:
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract prediction
        prediction = extract_answer(response, question_type)
        
        # Check if prediction is correct
        is_correct = is_correct_answer(prediction, true_answer, question_type)
        
        # Store result
        result = {
            "sample_id": idx,
            "difficulty": sample.get("difficulty", "unknown"),
            "question_type": question_type,
            "true_answer": true_answer,
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        
        return result
    except Exception as e:
        # Handle errors
        return {
            "sample_id": idx,
            "error": str(e),
            "correct": False
        }

def evaluate_model(model, difficulty, question_type, dataset_path, num_samples=None, output_file=None, concurrency=1):
    """
    Evaluate model performance on a specific relation task type and difficulty
    
    Args:
        model: model name
        difficulty: difficulty level ('easy' or 'hard')
        question_type: type of question (reasoning, group, cluster, count)
        dataset_path: path to dataset json file
        num_samples: number of samples to evaluate (None for all)
        output_file: optional file to save results
        concurrency: number of concurrent evaluations to run
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample papers to evaluate
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset[:num_samples]
    
    results = []
    correct_predictions = 0
    total_valid_samples = 0
    
    # Prepare tasks for parallel execution
    tasks = [(model, sample, idx, question_type) for idx, sample in enumerate(dataset)]
    
    # Use ThreadPoolExecutor for parallel evaluation
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks and track with tqdm
        futures = {executor.submit(evaluate_single_sample, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model} on {difficulty} {question_type}"):
            result = future.result()
            if result is None:
                continue
                
            total_valid_samples += 1
            
            if result.get("correct", False):
                correct_predictions += 1
                
            results.append(result)
            
            # Small delay between tasks to avoid overloading API
            time.sleep(0.1)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_valid_samples * 100 if total_valid_samples > 0 else 0
    
    # Create summary
    summary = {
        "model": model,
        "difficulty": difficulty,
        "question_type": question_type,
        "total_samples": total_valid_samples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy
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
    parser = argparse.ArgumentParser(description='Evaluate model performance on relation tasks')
    parser.add_argument('--models', type=str, nargs='+', default=['gemma-2-9B'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--difficulties', type=str, nargs='+', default=['easy', 'hard'],
                        help='Difficulty levels to evaluate')
    parser.add_argument('--question_types', type=str, nargs='+', 
                        default=['reasoning', 'group', 'cluster', 'count'],
                        help='Question types to evaluate')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--force_reevaluate', action='store_true',
                        help='Force re-evaluation of models even if they exist in summary')
    parser.add_argument('--concurrency', type=int, default=25,
                        help='Number of concurrent evaluations to run (default: 4)')
    
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_arguments()
    
    # Ensure results directory exists
    results_dir = Path("results")
    if not results_dir.exists():
        os.makedirs(results_dir)
    
    # Create a summary dict to track all model results
    summary_file = results_dir / "relation_summary.json"
    
    all_results = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                print(f"Loaded existing summary file with data for {len(all_results)} models.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing summary file. Starting with empty results.")
    
    # Evaluate each model on each difficulty and question type
    for model in args.models:
        if model not in all_results:
            all_results[model] = {}
            
        print(f"\nEvaluating model: {model}")
        
        for difficulty in args.difficulties:
            if difficulty not in all_results[model]:
                all_results[model][difficulty] = {}
                
            for question_type in args.question_types:
                # Check if we need to evaluate this configuration
                if (question_type in all_results[model][difficulty] and 
                    not args.force_reevaluate):
                    print(f"\nModel {model} already evaluated on {difficulty} {question_type}. Skipping.")
                    continue
                    
                print(f"\nEvaluating {model} on {difficulty} {question_type}")
                
                # Create dataset path for this configuration
                dataset_path = f"data/{difficulty}/relation_{question_type}.json"
                
                # Create unique output filename for this evaluation
                output_file = results_dir / f"{model}_{difficulty}_{question_type}_results.json"
                
                # Run evaluation with concurrency
                try:
                    model_results = evaluate_model(
                        model=model,
                        difficulty=difficulty,
                        question_type=question_type,
                        dataset_path=dataset_path,
                        num_samples=args.num_samples,
                        output_file=output_file,
                        concurrency=args.concurrency  # Pass concurrency parameter
                    )
                    
                    # Add to results summary
                    all_results[model][difficulty][question_type] = {
                        "accuracy": model_results["accuracy"],
                        "correct": model_results["correct_predictions"],
                        "total": model_results["total_samples"]
                    }
                    
                    # Print results
                    print(f"\nResults for {model} on {difficulty} {question_type}:")
                    print(f"Accuracy: {model_results['accuracy']:.2f}%")
                    print(f"Correct: {model_results['correct_predictions']} / {model_results['total_samples']}")
                
                except Exception as e:
                    print(f"Error evaluating {model} on {difficulty} {question_type}: {e}")
                
                # Save updated summary after each evaluation
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate final summary table
    print("\n===== FINAL SUMMARY =====")
    print("\nAccuracy by model, difficulty, and question type:")
    
    # Header
    header = "Model"
    for difficulty in args.difficulties:
        for question_type in args.question_types:
            header += f" | {difficulty}_{question_type}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for model in args.models:
        row = model
        for difficulty in args.difficulties:
            if difficulty in all_results.get(model, {}):
                for question_type in args.question_types:
                    if question_type in all_results[model][difficulty]:
                        accuracy = all_results[model][difficulty][question_type]["accuracy"]
                        row += f" | {accuracy:.2f}%"
                    else:
                        row += " | N/A"
            else:
                for _ in args.question_types:
                    row += " | N/A"
        print(row)
    
    print(f"\nDetailed summary saved to {summary_file}")

if __name__ == "__main__":
    main()
