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
    # First look for the exact "Final Rating: X" format
    final_rating_match = re.search(r"final rating:\s*(\d+)", response.lower())
    if final_rating_match:
        try:
            rating = int(final_rating_match.group(1))
            # Ensure rating is between 1-5
            return max(1, min(5, rating))
        except ValueError:
            pass
    
    # Pattern to match rating expressions - fallback options
    rating_patterns = [
        r"Final Rating: (\d+)"
    ]
    
    # Check for patterns
    for pattern in rating_patterns:
        match = re.search(pattern, response.lower())
        if match:
            try:
                rating = int(match.group(1))
                # Ensure rating is between 1-5
                return max(1, min(5, rating))
            except ValueError:
                continue
    
    # If no match found with the specific patterns, look for any isolated numbers 1-5
    # This is a fallback method
    isolated_numbers = re.findall(r'(?<!\d)([1-5])(?!\d)', response)
    if isolated_numbers:
        return int(isolated_numbers[0])
    
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
Analyze these reviews carefully. What is the TRUE quality of this product on a scale of 1-5 stars?
First explain your reasoning, then provide your final rating in exactly this format:
Final Rating: X
Where X is a number from 1 to 5.
"""
    
    return prompt

def evaluate_model(model, dataset_path, num_scenarios=None, output_file=None):
    """
    Evaluate model performance on the review rating inference task
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (None for all)
        output_file: optional file to save results
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample scenarios to evaluate
    if num_scenarios is not None and num_scenarios < len(dataset):
        dataset = dataset[:num_scenarios]
    
    results = []
    correct_predictions = 0
    total_scenarios = len(dataset)
    
    # Create confusion matrix (true_rating, predicted_rating)
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    # Evaluate each scenario
    for scenario in tqdm(dataset, desc=f"Evaluating {model}"):
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
        if prediction == 0:
            print(f"Warning: Could not extract rating for scenario {scenario_id}. Using default of 3.")
            prediction = 3  # Default to middle rating if extraction fails
        
        is_correct = prediction == true_rating
        
        if is_correct:
            correct_predictions += 1
        
        # Update confusion matrix (subtract 1 because indices are 0-based)
        if 1 <= prediction <= 5 and 1 <= true_rating <= 5:
            confusion_matrix[true_rating-1][prediction-1] += 1
        
        # Store result
        result = {
            "scenario_id": scenario_id,
            "true_rating": true_rating,
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        results.append(result)
        
        # Optional: Add delay to avoid rate limits
        time.sleep(0.5)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_scenarios * 100 if total_scenarios > 0 else 0
    
    # Calculate per-rating accuracy and distribution
    per_rating_metrics = {}
    for true_rating in range(1, 6):
        # Get indices for this true rating (zero-indexed)
        true_idx = true_rating - 1
        
        # Count total instances of this true rating
        total = np.sum(confusion_matrix[true_idx])
        
        if total > 0:
            # Calculate accuracy for this rating
            correct = confusion_matrix[true_idx][true_idx]
            rating_accuracy = (correct / total) * 100
            
            # Calculate distribution of predictions for this true rating
            distribution = {}
            for pred_rating in range(1, 6):
                pred_idx = pred_rating - 1
                count = confusion_matrix[true_idx][pred_idx]
                percentage = (count / total) * 100 if total > 0 else 0
                distribution[pred_rating] = {
                    "count": int(count),
                    "percentage": percentage
                }
            
            per_rating_metrics[true_rating] = {
                "total": int(total),
                "correct": int(correct),
                "accuracy": rating_accuracy,
                "distribution": distribution
            }
    
    # Summary
    summary = {
        "model": model,
        "total_scenarios": total_scenarios,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "per_rating_metrics": per_rating_metrics,
        "results": results
    }
    
    # Print summary
    print(f"\nModel: {model}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_scenarios})")
    
    # Print confusion matrix
    print("\nConfusion Matrix (rows: true rating, columns: predicted rating):")
    headers = [""] + [f"Pred {i}" for i in range(1, 6)]
    table_data = []
    for i in range(5):
        row = [f"True {i+1}"] + [confusion_matrix[i][j] for j in range(5)]
        table_data.append(row)
    
    # Print per-rating metrics
    print("\nPer-Rating Metrics:")
    for rating, metrics in per_rating_metrics.items():
        print(f"\nTrue Rating: {rating}")
        print(f"  Total samples: {metrics['total']}")
        print(f"  Correctly predicted: {metrics['correct']} ({metrics['accuracy']:.2f}%)")
        print("  Prediction distribution:")
        for pred_rating, dist in metrics['distribution'].items():
            print(f"    Predicted as {pred_rating}: {dist['count']} ({dist['percentage']:.2f}%)")
    
    # Save results if output file specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model performance on e-commerce review inference')
    parser.add_argument('--models', type=str, nargs='+', default=['gpt-4o-mini','llama-3.3-70B'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--dataset', type=str, default='data/review_dataset_eval.json', 
                        help='Path to dataset')
    parser.add_argument('--num_scenarios', type=int, default=None, 
                        help='Number of scenarios to evaluate (default: all)')
    
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
    
    # Evaluate each model
    results_summary = {}
    for model in args.models:
        # Create unique output filename for this model
        output_file = results_dir / f"{model}_{dataset_basename}_results.json"
        
        # Run evaluation
        model_results = evaluate_model(model, args.dataset, args.num_scenarios, output_file)
        
        # Add to summary
        results_summary[model] = {
            "accuracy": model_results["accuracy"],
            "correct": model_results["correct_predictions"],
            "total": model_results["total_scenarios"],
            "per_rating_accuracy": {
                rating: metrics["accuracy"] 
                for rating, metrics in model_results["per_rating_metrics"].items()
            }
        }
    
    # Save overall summary
    summary_file = results_dir / f"summary_{dataset_basename}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nOverall evaluation summary:")
    for model, stats in results_summary.items():
        print(f"{model}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        
        print("  Per-rating accuracy:")
        for rating, accuracy in stats["per_rating_accuracy"].items():
            print(f"    Rating {rating}: {accuracy:.2f}%")
            
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()