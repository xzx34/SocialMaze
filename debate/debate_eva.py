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

def generate_user_prompt(debate_item):
    """Generate user prompt containing the paper information, reviews, and rebuttals.
    
    Args:
        debate_item: The debate item containing paper information
        
    Returns:
        A string with the formatted prompt
    """
    prompt = """Please analyze the following research paper and determine whether it should be accepted or rejected for publication at a top-tier conference.

"""
    
    # Add Round 1: Paper title, keywords, and abstract
    if len(debate_item["statements"]) >= 1 and "round1" in debate_item["statements"][0]:
        prompt += "## Paper Information\n\n"
        prompt += debate_item["statements"][0]["round1"] + "\n\n"
    
    # Add Round 2: Reviewer comments
    if len(debate_item["statements"]) >= 2 and "round2" in debate_item["statements"][1]:
        prompt += "## Reviewer Comments\n\n"
        prompt += debate_item["statements"][1]["round2"] + "\n\n"
    
    # Add Round 3: Author rebuttals
    if len(debate_item["statements"]) >= 3 and "round3" in debate_item["statements"][2]:
        prompt += "## Author Response\n\n"
        prompt += debate_item["statements"][2]["round3"] + "\n\n"
    
    prompt += """
Based on all the information provided, carefully analyze whether this paper should be accepted or rejected for publication.

First, provide your detailed reasoning. Then, conclude with your final decision in exactly this format:
Final Decision: [Accept/Reject]
"""
    
    return prompt

def evaluate_model(model, dataset_path, num_papers=None, output_file=None):
    """
    Evaluate model performance on the paper acceptance decision task
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_papers: number of papers to evaluate (None for all)
        output_file: optional file to save results
    
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
    
    # Evaluate each paper
    for debate_item in tqdm(dataset, desc=f"Evaluating {model}"):
        paper_id = debate_item["id"]
        true_decision = normalize_decision(debate_item["decision"])
        
        # Skip papers with unknown decisions
        if true_decision == 'Unknown':
            continue
        
        total_valid_papers += 1
        
        # Generate the prompts
        system_prompt = generate_system_prompt()
        user_prompt = generate_user_prompt(debate_item)
        
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract prediction
        prediction = extract_decision_prediction(response)
        
        # Check for unknown predictions
        if prediction == 'Unknown':
            # Default to the most common decision (usually Reject)
            prediction = 'Reject'
            print(f"Warning: Unknown prediction for paper {paper_id}, defaulting to 'Reject'")
        
        is_correct = prediction == true_decision
        
        if is_correct:
            correct_predictions += 1
        
        # Update confusion matrix
        if true_decision in decision_mapping and prediction in decision_mapping:
            true_idx = decision_mapping[true_decision]
            pred_idx = decision_mapping[prediction]
            confusion_matrix[true_idx][pred_idx] += 1
        
        # Store result
        result = {
            "paper_id": paper_id,
            "source": debate_item.get("source", "Unknown"),
            "true_decision": true_decision,
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        results.append(result)
        
        # Optional: Add delay to avoid rate limits
        time.sleep(0.5)
    
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
    parser.add_argument('--models', type=str, nargs='+', default=['gemma-2-9B','gemma-2-27B','qwen-2.5-72B','llama-3.3-70B','qwq'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--dataset', type=str, default='data/debate.json', 
                        help='Path to debate dataset')
    parser.add_argument('--num_papers', type=int, default=100, 
                        help='Number of papers to evaluate (default: all)')
    parser.add_argument('--force_reevaluate', action='store_true',
                        help='Force re-evaluation of models even if they exist in summary')
    
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
        model_results = evaluate_model(model, args.dataset, args.num_papers, output_file)
        
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
