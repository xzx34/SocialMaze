import json
import os
import re
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import sys
import concurrent.futures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

def extract_spy_prediction(response):
    """Extract spy prediction from model response"""
    patterns = [
        r"Final Spy: Player (\d)",
        r"Finial Spy: Player (\d)",  # Handle common typo
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, response, re.IGNORECASE)
        if matches:
            return matches.group(1)
    
    # Try to find just a number if no pattern matches
    number_only = re.search(r"(?:^|\D)([1-4])(?:\D|$)", response)
    if number_only:
        return number_only.group(1)
    
    return 0

def generate_system_prompt(scenario=None):
    """Generate the system prompt for the model
    
    Args:
        scenario: Optional scenario data to include player's word
    """
    base_prompt = (
        "You are a skilled player in a word description game. "
        "Your task is to identify which player is the 'spy' based on their descriptions.\n\n"
        "Game Rules:\n"
        "1. There are 4 players in the game.\n"
        "2. 3 players received the same word (normal players).\n"
        "3. 1 player received a different but related word (the spy).\n"
        "4. Each player describes their word without saying it directly.\n"
        "5. You need to determine who is the spy based on these descriptions.\n\n"
    )
    
    # Add information about being Player 1 and their word if scenario is provided
    player1_word = scenario["player_words"]["1"]
    base_prompt += f"You are Player 1, and your word is: \"{player1_word}\".\n\n"
    
    base_prompt += "Analyze the descriptions carefully. Look for subtle differences that might reveal who has a different word."
    
    return base_prompt

def generate_user_prompt(scenario):
    """Generate a user prompt based on the scenario statements across all available rounds.
    
    Args:
        scenario: The scenario data containing statements for each round
        
    Returns:
        A formatted prompt string
    """
    prompt = ""
    
    # Get the number of rounds from the scenario
    num_rounds = len(scenario.get("statements", []))
    
    # Generate prompt for each round
    for round_idx in range(num_rounds):
        round_num = round_idx + 1
        prompt += f"Round {round_num}:\n"
        
        # Add all player statements for this round
        for statement in scenario["statements"][round_idx]["statements"]:
            prompt += f"{statement['statement']}\n"
        
        # Add a newline between rounds (except for the last round)
        if round_idx < num_rounds - 1:
            prompt += "\n"
    
    prompt += "\nBased on these descriptions, which player is the spy (Player 1, 2, 3, or 4)?\n"
    prompt += "Explain your reasoning step by step, then provide your final answer in the format: 'Final Spy: Player X'"
    
    return prompt

def evaluate_model(model, dataset_path, num_scenarios=None, output_file=None, max_workers=4):
    """
    Evaluate model performance on the Word Spy dataset
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (None for all)
        output_file: optional file to save results
        max_workers: maximum number of concurrent workers for scenario evaluation
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
    
    def process_scenario(scenario):
        scenario_id = scenario["scenario_id"]
        spy_player = scenario["spy_player"]
        
        # Generate the system and user prompts
        system_prompt = generate_system_prompt(scenario)
        user_prompt = generate_user_prompt(scenario)
        
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract prediction
        prediction = extract_spy_prediction(response)
        is_correct = prediction == spy_player
        
        # Store result
        result = {
            "scenario_id": scenario_id,
            "spy_player": spy_player,  # Ground truth
            "prediction": prediction,
            "correct": is_correct,
            "response": response
        }
        
        # Optional: Add delay to avoid rate limits
        time.sleep(0.5)
        
        return result, is_correct
    
    # Process scenarios with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        futures = []
        for scenario in dataset:
            futures.append(executor.submit(process_scenario, scenario))
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {model}"):
            result, is_correct = future.result()
            results.append(result)
            if is_correct:
                correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_scenarios * 100 if total_scenarios > 0 else 0
    
    # Summary
    summary = {
        "model": model,
        "total_scenarios": total_scenarios,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "results": results
    }
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    return summary

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate model performance on Word Spy game')
    parser.add_argument('--models', type=str, nargs='+', default=['gpt-4o-mini'],
                        help='Models to evaluate')
    parser.add_argument('--dataset', type=str, default='data/fts_dataset_eval.json', 
                        help='Path to dataset')
    parser.add_argument('--num_scenarios', type=int, default=1000, 
                        help='Number of scenarios to evaluate')
    parser.add_argument('--model_workers', type=int, default=2,
                        help='Maximum number of concurrent workers for model evaluation')
    parser.add_argument('--scenario_workers', type=int, default=25,
                        help='Maximum number of concurrent workers for scenario evaluation')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    results_dir = Path("results")
    if not results_dir.exists():
        os.makedirs(results_dir)
    
    # Get dataset basename for result files
    dataset_basename = os.path.basename(args.dataset).split('.')[0]
    
    # Evaluate models concurrently
    results_summary = {}
    
    def evaluate_model_wrapper(model):
        # Create unique output filename for this model
        output_file = results_dir / f"{model}_results.json"
        
        # Run evaluation
        model_results = evaluate_model(model, args.dataset, args.num_scenarios, output_file, args.scenario_workers)
        
        print(f"\nModel: {model}")
        print(f"Accuracy: {model_results['accuracy']:.2f}% ({model_results['correct_predictions']}/{model_results['total_scenarios']})")
        
        return model, model_results
    
    # Use ThreadPoolExecutor for concurrent execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.model_workers) as executor:
        # Submit all model evaluation tasks
        future_to_model = {executor.submit(evaluate_model_wrapper, model): model for model in args.models}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model, model_results = future.result()
            # Add to summary
            results_summary[model] = {
                "accuracy": model_results["accuracy"],
                "correct": model_results["correct_predictions"],
                "total": model_results["total_scenarios"]
            }
    
    # Save overall summary
    summary_file = results_dir / f"summary_{dataset_basename}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nOverall evaluation summary:")
    for model, stats in results_summary.items():
        print(f"{model}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()