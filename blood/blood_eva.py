import re
import json
import os
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import sys
sys.path.append("..") # Add parent directory to path
from utils.tool import get_chat_response

load_dotenv()

def extract_criminal_prediction(response):
    """Extract criminal prediction from model response"""
    patterns = [
        r"Final Criminal Is Player (\d+)",
        r"Final Criminal is Player (\d+)",
        r"Final Criminal is player (\d+)"
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, response, re.IGNORECASE)
        if matches:
            return matches.group(1)
    
    return 'None'

def extract_role_predictions(response, num_players):
    """Extract role predictions for all players from model response"""
    role_predictions = {}
    
    # Different patterns to match role assignments in the response
    for player_id in range(1, num_players + 1):
        player_str = str(player_id)
        patterns = [
            rf"Final Player {player_str}:\s*(\w+)",
            rf"Final Player {player_str} - (\w+)"
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                role = matches.group(1).strip().title()
                # Normalize role names
                if "Investigat" in role:
                    role = "Investigator"
                elif "Criminal" in role or "Killer" in role or "Murderer" in role:
                    role = "Criminal"
                elif "Rumor" in role:
                    role = "Rumormonger"
                elif "Luna" in role:
                    role = "Lunatic"
                    
                role_predictions[player_str] = role
                break
    
    return role_predictions

def extract_self_role_prediction(response, player_id):
    """Extract the model's prediction of its own role"""
    # Look for "My Role Is" or similar patterns
    patterns = [
        r"My Role Is (\w+)",
        r"My Role is (\w+)",
        r"My role is (\w+)"
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

def evaluate_model(model, dataset_path, num_scenarios, output_file=None):
    """
    Evaluate model performance on the MetaSkeptic dataset focusing only on criminal 
    prediction accuracy and self-role prediction accuracy
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate 
        output_file: optional file to save results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Sample scenarios to evaluate
    if num_scenarios < len(dataset):
        evaluated_scenarios = dataset[:num_scenarios]
    else:
        evaluated_scenarios = dataset
    
    results = []
    
    # Track accuracy by round across all scenarios
    round_metrics = {1: [], 2: [], 3: []}
    
    # Track metrics by role of the player
    role_metrics = {
        "Investigator": {1: [], 2: [], 3: []},
        "Criminal": {1: [], 2: [], 3: []},
        "Rumormonger": {1: [], 2: [], 3: []},
        "Lunatic": {1: [], 2: [], 3: []}
    }
    
    for scenario in tqdm(evaluated_scenarios, desc=f"Evaluating {model}"):
        scenario_id = scenario['scenario_id']
        ground_truth_roles = scenario['ground_truth_roles']
        num_players = len(ground_truth_roles)
        statements = scenario['statements']
        
        # Choose player 1's perspective for consistency
        player_id = "1"
        player_role = ground_truth_roles[player_id]  # Get the actual role of player 1
        system_prompt = scenario['prompts'][player_id]
        
        # Find the true criminal from ground truth
        true_criminal = None
        for pid, role in ground_truth_roles.items():
            if role == "Criminal":
                true_criminal = pid
                break
        
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
            self_role_prediction = extract_self_role_prediction(response, player_id)
            
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
            "ground_truth_roles": ground_truth_roles,
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
        "dataset": os.path.basename(dataset_path),
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
    parser.add_argument('--models', nargs='+', default=['gpt-4o-mini'], 
                        help='Models to evaluate')
    parser.add_argument('--dataset_types', nargs='+', default=['all'],
                        help='Types of datasets to evaluate (original, rumormonger, lunatic, all)')
    parser.add_argument('--player_counts', type=int, nargs='+', default=[6],
                        help='Number of players in each game')
    parser.add_argument('--num_scenarios', type=int, default=1,
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

                output_file = os.path.join(results_dir, f"{model}_{player_count}_{dataset_type}_results.json")
                detailed_results = None
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            detailed_results = existing_data.get("detailed_results", [])
                            print(f"Loaded existing detailed results for {model} on {os.path.basename(dataset_path)}")
                    except Exception as e:
                        print(f"Error loading existing detailed results: {e}")
                
                print(f"Evaluating {model} on {os.path.basename(dataset_path)}")
                
                summary, new_results = evaluate_model(
                    model=model, 
                    dataset_path=dataset_path,
                    num_scenarios=args.num_scenarios,
                    output_file=None  # Don't save directly, we'll handle it ourselves
                )
                
                # If we have existing results, merge with new ones
                if detailed_results:
                    # Create a lookup of existing scenario IDs
                    existing_ids = {result['scenario_id'] for result in detailed_results}
                    
                    # Add only new scenarios that don't already exist
                    for result in new_results:
                        if result['scenario_id'] not in existing_ids:
                            detailed_results.append(result)
                            existing_ids.add(result['scenario_id'])
                    
                    # Save merged results
                    output_data = {
                        "summary": summary,
                        "detailed_results": detailed_results
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2)
                else:
                    # Save new results as is
                    output_data = {
                        "summary": summary,
                        "detailed_results": new_results
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2)
                
                model_results[f"{player_count}_{dataset_type}"] = summary
        
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