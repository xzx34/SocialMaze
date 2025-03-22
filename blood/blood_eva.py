import re
import json
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import sys
import time
sys.path.append("..") # Add parent directory to path
from utils.tool import get_chat_response

load_dotenv()

def extract_criminal_prediction(response):
    """Extract criminal prediction from model response"""
    patterns = [
        r"Final Criminal Is Player (\d+)"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, response, re.IGNORECASE)
        if matches:
            return matches.group(1)
    
    return 'None'

def extract_self_role_prediction(response):
    """Extract the model's prediction of its own role"""
    # Look for "My Role Is" or similar patterns
    patterns = [
        r"My Role Is (\w+)"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, response, re.IGNORECASE)
        if matches:
            role = matches.group(1).strip().title()
            # Normalize role names
            if "Investigat" in role:
                return "Investigator"
            elif "Criminal" in role or "Killer" in role or "Murderer" in role:
                return "Criminal"
            elif "Rumor" in role:
                return "Rumormonger"
            elif "Luna" in role:
                return "Lunatic"
            elif "Unknown" in role:
                return "Unknown"
    
    return "Unknown"

def evaluate_model(model, scenarios, dataset_name=None, output_file=None):
    """
    Evaluate model performance on the Blood Game scenarios focusing only on criminal 
    prediction accuracy and self-role prediction accuracy
    
    Args:
        model: model name
        scenarios: list of scenario data to evaluate
        dataset_name: optional name of the dataset
        output_file: optional file to save results
    """
    results = []
    
    # Track accuracy by round across all scenarios
    round_metrics = {1: [], 2: [], 3: []}
    
    # Track metrics by role of the player
    role_metrics = {
        "Investigator": {1: [], 2: [], 3: []},
        "Criminal": {1: [], 2: [], 3: []},
        "Rumormonger": {1: [], 2: [], 3: []},
        "Lunatic": {1: [], 2: [], 3: []},
        "Unknown": {1: [], 2: [], 3: []}  # Add Unknown role tracking
    }
    
    for scenario in tqdm(scenarios, desc=f"Evaluating {model}"):
        scenario_id = scenario['scenario_id']
        ground_truth = scenario['ground_truth']
        statements = scenario['statements']
        
        # Choose player 1's perspective for consistency
        player_id = "1"
        player_role = ground_truth["player1_role"]  # Get the actual role of player 1
        system_prompt = scenario['prompts'][player_id] if 'prompts' in scenario else system_prompts.get(scenario_id)
        
        # Get the true criminal
        true_criminal = ground_truth["criminal"]
        
        # Setup conversation history
        conversation = []
        round_results = []
        
        # Evaluate each round
        for round_idx, round_data in enumerate(statements):
            round_num = round_data["round"]
            round_statements = round_data["statements"]
            
            # Format the statements for this round
            statements_text = f"\nRound {round_num} statements:\n"
            for stmt in round_statements:
                statements_text += f"{stmt['statement']}\n"
            
            # Add this round's statements to conversation
            conversation.append({
                "role": "user", 
                "content": statements_text
            })
            
            # Get model response
            response = get_chat_response(
                model=model,
                system_message=system_prompt,
                messages=conversation,
                temperature=0.001
            )
            
            # Add model response to conversation
            conversation.append({
                "role": "assistant",
                "content": response
            })
            
            # Extract predictions
            criminal_prediction = extract_criminal_prediction(response)
            self_role_prediction = extract_self_role_prediction(response)
            
            # Calculate accuracy metrics
            criminal_correct = int(criminal_prediction == true_criminal)
            self_role_correct = int(self_role_prediction == player_role)
            
            # Save round results
            round_result = {
                "round": round_num,
                "player_role": player_role,
                "criminal_prediction": criminal_prediction,
                "true_criminal": true_criminal, 
                "criminal_correct": criminal_correct,
                "self_role_prediction": self_role_prediction,
                "self_role_correct": self_role_correct,
                "response": response
            }
            
            round_results.append(round_result)
            round_metrics[round_num].append(round_result)
            
            # Also track by player's role
            if player_role in role_metrics:
                role_metrics[player_role][round_num].append(round_result)
        
        # Save scenario results
        scenario_result = {
            "scenario_id": scenario_id,
            "dataset_type": scenario.get('dataset_type', ''),
            "ground_truth": ground_truth,
            "player_role": player_role,
            "rounds": round_results
        }
        
        results.append(scenario_result)
    
    # Calculate aggregate accuracy per round
    round_summaries = {}
    for round_num, metrics in round_metrics.items():
        if not metrics:
            continue
            
        criminal_accuracy = sum(m["criminal_correct"] for m in metrics) / len(metrics) * 100
        self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100
        
        round_summaries[round_num] = {
            "criminal_accuracy": criminal_accuracy,
            "self_role_accuracy": self_role_accuracy,
            "num_scenarios": len(metrics)
        }
    
    # Calculate role-specific performance
    role_summaries = {}
    for role, rounds in role_metrics.items():
        role_summaries[role] = {}
        
        for round_num, metrics in rounds.items():
            if not metrics:  # Skip if no data for this role in this round
                continue
                
            criminal_accuracy = sum(m["criminal_correct"] for m in metrics) / len(metrics) * 100
            self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100
            
            role_summaries[role][round_num] = {
                "criminal_accuracy": criminal_accuracy,
                "self_role_accuracy": self_role_accuracy,
                "num_scenarios": len(metrics)
            }
    
    # Create summary
    summary = {
        "model": model,
        "dataset": dataset_name,
        "num_scenarios": len(results),
        "rounds": round_summaries,
        "role_specific": role_summaries
    }
    
    # Save results if output file is provided
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_data = {
            "summary": summary,
            "detailed_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    return summary, results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on Blood Game scenarios')
    parser.add_argument('--models', nargs='+', default=['llama-3.3-70B', 'gemma-3-27B', 'qwen-2.5-72B','qwq'], 
                        help='Models to evaluate')
    parser.add_argument('--dataset_types', nargs='+', default=['all'],
                        help='Types of datasets to evaluate (original, rumormonger, lunatic, all)')
    parser.add_argument('--player_counts', type=int, nargs='+', default=[6],
                        help='Number of players in each game')
    parser.add_argument('--num_scenarios', type=int, default=2,
                        help='Number of scenarios to evaluate per dataset')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Custom directory for saving results (default: blood/results)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set the models to evaluate
    models = args.models
    
    # Dataset types to evaluate
    dataset_types = args.dataset_types
    player_counts = args.player_counts
    
    # Use specified results directory or create default
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(results_dir)}")
    
    # Data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    all_results = {}
    
    # First, load existing overall summary if it exists
    overall_summary_path = os.path.join(results_dir, "overall_summary.json")
    if os.path.exists(overall_summary_path):
        try:
            with open(overall_summary_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                print(f"Loaded existing results from {overall_summary_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            all_results = {}
    
    for model in models:
        model_results = all_results.get(model, {})
        
        for player_count in player_counts:
            for dataset_type in dataset_types:
                dataset_path = os.path.join(data_dir, f"blood_{player_count}_{dataset_type}.json")
                
                if not os.path.exists(dataset_path):
                    print(f"Warning: {dataset_path} does not exist. Skipping.")
                    continue
                
                # Load the full dataset
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    full_dataset = json.load(f)

                # Main results file
                output_file = os.path.join(results_dir, f"{model}_{player_count}_{dataset_type}_results.json")
                detailed_results = None
                
                # Check if existing results file exists
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            detailed_results = existing_data.get("detailed_results", [])
                            print(f"Loaded existing detailed results for {model} on {os.path.basename(dataset_path)}")
                            print(f"Found {len(detailed_results)} existing scenario results")
                    except Exception as e:
                        print(f"Error loading existing detailed results: {e}")
                        # Create a backup of the problematic file
                        if os.path.exists(output_file):
                            backup_file = output_file + f".backup_{int(time.time())}"
                            try:
                                import shutil
                                shutil.copy2(output_file, backup_file)
                                print(f"Created backup of problematic file: {backup_file}")
                            except Exception as backup_err:
                                print(f"Failed to create backup: {backup_err}")
                
                print(f"Evaluating {model} on {os.path.basename(dataset_path)}")
                
                # Determine how many new scenarios to evaluate
                num_existing = len(detailed_results) if detailed_results else 0
                num_new = max(0, args.num_scenarios - num_existing)
                
                if num_new <= 0:
                    print(f"Already have {num_existing} results, which meets or exceeds the requested {args.num_scenarios}.")
                    print("Skipping evaluation. To force re-evaluation, use a higher --num_scenarios value.")
                    continue
                
                print(f"Will evaluate {num_new} new scenarios to reach target of {args.num_scenarios}")
                
                # Get a slice of the dataset that excludes already evaluated scenarios
                if detailed_results:
                    # Get IDs of scenarios already evaluated
                    existing_ids = {result['scenario_id'] for result in detailed_results}
                    
                    # Find scenarios in the dataset that haven't been evaluated yet
                    new_scenarios = [s for s in full_dataset if s['scenario_id'] not in existing_ids]
                    
                    if len(new_scenarios) < num_new:
                        print(f"Warning: Only {len(new_scenarios)} new scenarios available")
                        num_new = len(new_scenarios)
                    
                    # Take only the number of new scenarios needed
                    eval_scenarios = new_scenarios[:num_new]
                else:
                    # No existing results, just take the first num_new scenarios
                    eval_scenarios = full_dataset[:num_new]
                
                # Only evaluate if there are new scenarios to process
                if eval_scenarios:
                    summary, new_results = evaluate_model(
                        model=model, 
                        scenarios=eval_scenarios,
                        output_file=None  # Don't save directly, we'll handle it ourselves
                    )
                    
                    # If we have existing results, merge with new ones
                    if detailed_results:
                        # Merge the new results with existing ones
                        combined_results = detailed_results + new_results
                        print(f"Combined {len(detailed_results)} existing and {len(new_results)} new results")
                        
                        # Recalculate summary based on all results
                        all_round_metrics = {1: [], 2: [], 3: []}
                        all_role_metrics = {
                            "Investigator": {1: [], 2: [], 3: []},
                            "Criminal": {1: [], 2: [], 3: []},
                            "Rumormonger": {1: [], 2: [], 3: []},
                            "Lunatic": {1: [], 2: [], 3: []},
                            "Unknown": {1: [], 2: [], 3: []}
                        }
                        
                        # Collect metrics from all results
                        for result in combined_results:
                            for round_result in result['rounds']:
                                round_num = round_result['round']
                                all_round_metrics[round_num].append(round_result)
                                
                                player_role = round_result['player_role']
                                if player_role in all_role_metrics:
                                    all_role_metrics[player_role][round_num].append(round_result)
                        
                        # Recalculate summary
                        updated_summary = {
                            "model": model,
                            "dataset": os.path.basename(dataset_path),
                            "num_scenarios": len(combined_results),
                            "rounds": {},
                            "role_specific": {}
                        }
                        
                        # Update round metrics
                        for round_num, metrics in all_round_metrics.items():
                            if not metrics:
                                continue
                                
                            criminal_accuracy = sum(m["criminal_correct"] for m in metrics) / len(metrics) * 100
                            self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100
                            
                            updated_summary["rounds"][str(round_num)] = {
                                "criminal_accuracy": criminal_accuracy,
                                "self_role_accuracy": self_role_accuracy,
                                "num_scenarios": len(metrics)
                            }
                        
                        # Update role-specific metrics
                        for role, rounds in all_role_metrics.items():
                            updated_summary["role_specific"][role] = {}
                            
                            for round_num, metrics in rounds.items():
                                if not metrics:
                                    continue
                                    
                                criminal_accuracy = sum(m["criminal_correct"] for m in metrics) / len(metrics) * 100
                                self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100
                                
                                updated_summary["role_specific"][role][str(round_num)] = {
                                    "criminal_accuracy": criminal_accuracy,
                                    "self_role_accuracy": self_role_accuracy,
                                    "num_scenarios": len(metrics)
                                }
                        
                        # Save updated results
                        output_data = {
                            "summary": updated_summary,
                            "detailed_results": combined_results
                        }
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=2)
                        
                        model_results[f"{player_count}_{dataset_type}"] = updated_summary
                    else:
                        # No existing results, save new results as is
                        output_data = {
                            "summary": summary,
                            "detailed_results": new_results
                        }
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=2)
                        
                        model_results[f"{player_count}_{dataset_type}"] = summary
                else:
                    print(f"No new scenarios to evaluate for {model} on {os.path.basename(dataset_path)}")
        
        all_results[model] = model_results
    
    # Print overall summary with clear round separation
    print("\n===== EVALUATION SUMMARY =====\n")
    
    for model, model_results in all_results.items():
        print(f"Model: {model}")
        
        # Overall performance by round
        print("\n  === Overall Performance by Round ===")
        for config, summary in model_results.items():
            player_count, dataset_type = config.split('_')
            print(f"\n  Config: {player_count}p {dataset_type}")
            print(f"  {'Round':<10} {'Criminal Acc':<15} {'Self-Role Acc':<15}")
            print(f"  {'-'*45}")
            
            for round_num, round_summary in sorted(summary["rounds"].items()):
                print(f"  {round_num:<10} {round_summary['criminal_accuracy']:>5.1f}%{'':<9} "
                      f"{round_summary['self_role_accuracy']:>5.1f}%{'':<9}")
        
        # Performance by role
        print("\n  === Performance by Player Role ===")
        
        for config, summary in model_results.items():
            player_count, dataset_type = config.split('_')
            print(f"\n  Config: {player_count}p {dataset_type}")
            
            if "role_specific" not in summary:
                print("  No role-specific data available")
                continue
                
            for role, role_data in summary["role_specific"].items():
                if not role_data:  # Skip if no data for this role
                    continue
                    
                print(f"\n  Role: {role}")
                print(f"  {'Round':<10} {'Criminal Acc':<15} {'Self-Role Acc':<15}")
                print(f"  {'-'*45}")
                
                for round_num, round_summary in sorted(role_data.items()):
                    if "num_scenarios" in round_summary and round_summary["num_scenarios"] > 0:
                        print(f"  {round_num:<10} {round_summary['criminal_accuracy']:>5.1f}%{'':<9} "
                              f"{round_summary['self_role_accuracy']:>5.1f}%{'':<9}")
        
        print("\n" + "-"*80 + "\n")
    
    # Save overall summary
    with open(os.path.join(results_dir, "overall_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()