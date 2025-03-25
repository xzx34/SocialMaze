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
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

def extract_rating_prediction(response):
    """Extract the predicted rating from model response.
    
    Args:
        response: The model's response text
        
    Returns:
        An integer rating between 1-5, or 0 if extraction failed
    """
    
    # Pattern to match "Final Rating: X" where X is a digit, with optional markdown bold markers
    pattern = r"\*{0,2}Final Rating:\*{0,2}\s*(\d+)"
    
    # Check for pattern
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        try:
            rating = int(match.group(1))
            # Ensure rating is between 1-5
            return max(1, min(5, rating))
        except ValueError:
            return 0
    
    return 0  # Return 0 if no valid rating found

def generate_system_prompt():
    """Generate system prompt for the review evaluation task."""
    return """You are an AI assistant tasked with evaluating product quality based on customer reviews.

Your job is to analyze multiple reviews for a product and determine the most likely TRUE quality of the product on a scale of 1-5 stars.

Important context:
- Most reviewers are genuine customers providing honest feedback
- A few reviewers may be professional positive reviewers who tend to exaggerate good qualities
- A few reviewers may be malicious negative reviewers who focus primarily on flaws
- The TRUE quality is what a typical, unbiased consumer would rate the product
"""

def generate_user_prompt(scenario):
    """Generate user prompt containing ONLY the reviews to evaluate.
    
    Args:
        scenario: The scenario data containing product info and reviews
        
    Returns:
        A string with the formatted prompt
    """
    prompt = """Please analyze the following product reviews and determine the true quality of the product on a scale of 1-5 stars.

Reviews:
"""
    
    # Add each review
    for review in scenario["formatted_reviews"]:
        reviewer_id = review["reviewer_id"]
        text = review["text"]
        prompt += f"\nReviewer #{reviewer_id}: \"{text}\"\n"
    
    prompt += """
Analyze these reviews carefully. What is the TRUE quality of this product on a scale of 1-5 stars?  The rating should be rounded to the nearest whole number (e.g., 4.6 rounds to 5, 3.1 rounds to 3).
First explain your reasoning, then provide your final rating in exactly this format:
Final Rating: X
Where X is a number from 1 to 5.
"""
    
    return prompt

def evaluate_model(model, dataset_path, num_scenarios=None, output_file=None, max_workers=10):
    """
    Evaluate model performance on the review rating inference task
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (None for all)
        output_file: optional file to save results
        max_workers: maximum number of concurrent workers
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample scenarios to evaluate
    if num_scenarios is not None and num_scenarios < len(dataset):
        dataset = dataset[:num_scenarios]
    
    total_scenarios = len(dataset)
    
    # Create confusion matrix (true_rating, predicted_rating)
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    # Define a worker function to process a single scenario
    def process_scenario(scenario):
        scenario_id = scenario["scenario_id"]
        true_rating = scenario["true_rating"]
        
        # Generate the prompts
        system_prompt = generate_system_prompt()
        user_prompt = generate_user_prompt(scenario)
        
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract prediction
        prediction = extract_rating_prediction(response)
        
        is_correct = prediction == true_rating
        
        # Store result
        result = {
            "scenario_id": scenario_id,
            "true_rating": true_rating,
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        
        # Optional: Add delay to avoid rate limits
        time.sleep(0.5)
        
        return result
    
    # Use ThreadPoolExecutor for concurrent evaluation
    results = []
    correct_predictions = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_scenario = {executor.submit(process_scenario, scenario): scenario 
                              for scenario in dataset}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_scenario), 
                          total=len(future_to_scenario), 
                          desc=f"Evaluating {model}"):
            result = future.result()
            results.append(result)
            
            # Update correct count
            if result["correct"]:
                correct_predictions += 1
            
            # Update confusion matrix
            prediction = result["prediction"]
            true_rating = result["true_rating"]
            if 1 <= prediction <= 5 and 1 <= true_rating <= 5:
                confusion_matrix[true_rating-1][prediction-1] += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_scenarios * 100 if total_scenarios > 0 else 0
    
    # Calculate distribution for each true rating
    rating_distributions = {}
    for true_rating in range(1, 6):
        true_idx = true_rating - 1
        total = np.sum(confusion_matrix[true_idx])
        
        if total > 0:
            distribution = {}
            for pred_rating in range(1, 6):
                pred_idx = pred_rating - 1
                count = confusion_matrix[true_idx][pred_idx]
                percentage = (count / total) * 100 if total > 0 else 0
                distribution[pred_rating] = percentage
            
            rating_distributions[true_rating] = {
                "total": int(total),
                "distribution": distribution
            }
    
    # Create summary
    summary = {
        "model": model,
        "total_scenarios": total_scenarios,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "rating_distributions": rating_distributions
    }
    
    # Save results if output file specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Check if results file already exists
        if os.path.exists(output_file):
            # Read existing results
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                
            # Update with new scenarios
            existing_results_dict = {r["scenario_id"]: r for r in existing_results.get("results", [])}
            for result in results:
                existing_results_dict[result["scenario_id"]] = result
                
            # Recalculate confusion matrix and stats
            updated_results = list(existing_results_dict.values())
            updated_confusion_matrix = np.zeros((5, 5), dtype=int)
            updated_correct = 0
            
            for r in updated_results:
                if 1 <= r["prediction"] <= 5 and 1 <= r["true_rating"] <= 5:
                    updated_confusion_matrix[r["true_rating"]-1][r["prediction"]-1] += 1
                if r["correct"]:
                    updated_correct += 1
            
            updated_total = len(updated_results)
            updated_accuracy = updated_correct / updated_total * 100 if updated_total > 0 else 0
            
            # Recalculate rating distributions
            updated_rating_distributions = {}
            for true_rating in range(1, 6):
                true_idx = true_rating - 1
                total = np.sum(updated_confusion_matrix[true_idx])
                
                if total > 0:
                    distribution = {}
                    for pred_rating in range(1, 6):
                        pred_idx = pred_rating - 1
                        count = updated_confusion_matrix[true_idx][pred_idx]
                        percentage = (count / total) * 100 if total > 0 else 0
                        distribution[pred_rating] = percentage
                    
                    updated_rating_distributions[true_rating] = {
                        "total": int(total),
                        "distribution": distribution
                    }
            
            # Update summary with new data
            summary = {
                "model": model,
                "total_scenarios": updated_total,
                "correct_predictions": updated_correct,
                "accuracy": updated_accuracy,
                "confusion_matrix": updated_confusion_matrix.tolist(),
                "rating_distributions": updated_rating_distributions,
                "results": updated_results
            }
        else:
            # Add results to summary for new files
            summary["results"] = results
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model performance on e-commerce review inference')
    #llama-3.1-8B gemma-2-9B gemma-2-27B llama-3.3-70B qwen-2.5-72B qwq deepseek-r1
    parser.add_argument('--models', type=str, nargs='+', default=['llama-3.1-8B', 'gemma-2-9B', 'gemma-2-27B', 'llama-3.3-70B', 'qwen-2.5-72B', 'qwq', 'deepseek-r1'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--dataset', type=str, default='data/review_amazon.json', 
                        help='Path to dataset')
    parser.add_argument('--num_scenarios', type=int, default=100, 
                        help='Number of scenarios to evaluate (default: all)')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of concurrent workers (default: 10)')
    
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
    
    # Load existing summary if it exists
    summary_file = results_dir / f"summary_{dataset_basename}.json"
    existing_summary = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or not found, start with empty summary
            existing_summary = {}
    
    # Evaluate each model
    results_summary = existing_summary.copy()
    for model in args.models:
        # Create unique output filename for this model
        output_file = results_dir / f"{model}_{dataset_basename}_results.json"
        
        # Run evaluation with concurrent processing
        model_results = evaluate_model(model, args.dataset, args.num_scenarios, output_file, args.max_workers)
        
        # Add or update model results in summary
        results_summary[model] = {
            "accuracy": model_results["accuracy"],
            "correct": model_results["correct_predictions"],
            "total": model_results["total_scenarios"],
            "rating_distributions": model_results["rating_distributions"]
        }
    
    # Save updated summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()