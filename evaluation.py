import re
import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()

model_dict = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o1-mini': 'o1-mini-2024-09-12',
    'chatgpt-4o-latest': 'chatgpt-4o-latest',
    'llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'gemma-2-27B': 'google/gemma-2-27b-it',
    'gemma-2-9B': 'google/gemma-2-9b-it',
    'qwen-2.5-72B': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen-2.5-32B': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen-2.5-14B': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen-2.5-7B': 'Qwen/Qwen2.5-7B-Instruct',
    'yi-lightning': 'yi-lightning',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'qwq':'Qwen/QwQ-32B',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek-r1-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-r1-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
}


def get_chat_response(model, system_message, messages, temperature=0.001):
    """
    Get response from a model with multi-turn conversation history
    
    Args:
        model: model name
        system_message: system message
        messages: list of message dicts with 'role' and 'content'
        temperature: sampling temperature
    
    Returns:
        model response as string
    """
    if model in ['claude-3.5-sonnet']:
        client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Format messages for Anthropic
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({
                'role': msg['role'], 
                'content': [{'type': 'text', 'text': msg['content']}]
            })
            
        message = client.messages.create(
            model=model_dict[model],
            max_tokens=2048,
            temperature=temperature,
            system=system_message,
            messages=anthropic_messages
        )
        return message.content[0].text
    
    # Handle DeepInfra models
    if model in ['deepseek-v3', 'deepseek-r1', 'deepseek-r1-32B', 'deepseek-r1-70B', 'qwq', 'llama-3.3-70B', 'llama-3.1-70B', 'llama-3.1-8B', 'qwen-2.5-72B', 'gemma-2-27B']:
        client = OpenAI(
            api_key=os.getenv('DEEPINFRA_API_KEY'),
            base_url=os.getenv('DEEPINFRA_BASE_URL')
        )
        
        # Format messages with system message for DeepInfra
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)
        
        full_response = ''
        chat_completion = client.chat.completions.create(
            model=model_dict[model],
            temperature=temperature,
            messages=formatted_messages,
            stream=True,
        )

        for event in chat_completion:
            if event.choices[0].finish_reason:
                break 
            else:
                content = event.choices[0].delta.content or ""
                full_response += content

        return full_response
    
    # Handle Yi Lightning
    elif model == 'yi-lightning':
        client = OpenAI(
            api_key=os.getenv('YI_API_KEY'),
            base_url=os.getenv('YI_BASE_URL')
        )
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=model_dict[model],
            messages=formatted_messages,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    # Handle OpenAI models
    else:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=model_dict[model],
            messages=formatted_messages,
            temperature=temperature,
        )

        return response.choices[0].message.content


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


def extract_self_role_prediction(response, player_id, num_players):
    """Extract the model's prediction of its own role (player_id)"""
    patterns = [
        rf"Final Player {player_id}:\s*(\w+)",
        rf"Final Player {player_id} - (\w+)",
        rf"Player {player_id}:\s*(\w+)",
        rf"Player {player_id} - (\w+)",
        rf"Player {player_id} is (?:an? )?(\w+)"
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
    
    return None


def calculate_scores(criminal_prediction, role_predictions, ground_truth_roles):
    """
    Calculate accuracy scores using the custom scoring system:
    - 30 points for correctly identifying the criminal
    - 70 points distributed evenly among other correct role identifications
    
    Returns a tuple of (score, criminal_correct, roles_correct, total_roles)
    """
    # Find the true criminal from ground truth
    true_criminal = None
    for player_id, role in ground_truth_roles.items():
        if role == "Criminal":
            true_criminal = player_id
            break
    
    criminal_correct = 0
    criminal_score = 0
    if criminal_prediction == true_criminal:
        criminal_correct = 1
        criminal_score = 30
    
    # Calculate role score for non-criminal players (70 points distributed evenly)
    roles_correct = 0
    non_criminal_count = len(ground_truth_roles) - 1  # Exclude criminal
    points_per_role = 70 / non_criminal_count if non_criminal_count > 0 else 0
    
    role_score = 0
    for player_id, true_role in ground_truth_roles.items():
        if player_id == true_criminal:
            continue  # Skip the criminal, already counted
            
        if player_id in role_predictions and role_predictions[player_id] == true_role:
            roles_correct += 1
            role_score += points_per_role
    
    # Total score (0-100)
    total_score = criminal_score + role_score
    
    return total_score, criminal_correct, roles_correct, non_criminal_count


def calculate_scores_with_self_role(criminal_prediction, role_predictions, ground_truth_roles, player_id):
    """
    Calculate accuracy scores including self-role identification
    
    Returns a tuple of (
        total_score, 
        criminal_correct, 
        roles_correct, 
        non_criminal_count,
        self_role_correct,
        self_role_prediction
    )
    """
    # Find the true criminal from ground truth
    true_criminal = None
    for pid, role in ground_truth_roles.items():
        if role == "Criminal":
            true_criminal = pid
            break
    
    criminal_correct = 0
    criminal_score = 0
    if criminal_prediction == true_criminal:
        criminal_correct = 1
        criminal_score = 30
    
    # Extract self role prediction and check if correct
    self_role_prediction = role_predictions.get(player_id)
    true_self_role = ground_truth_roles.get(player_id)
    self_role_correct = 0
    if self_role_prediction == true_self_role:
        self_role_correct = 1
    
    # Calculate role score for non-criminal players (70 points distributed evenly)
    roles_correct = 0
    non_criminal_count = len(ground_truth_roles) - 1  # Exclude criminal
    points_per_role = 70 / non_criminal_count if non_criminal_count > 0 else 0
    
    role_score = 0
    for pid, true_role in ground_truth_roles.items():
        if pid == true_criminal:
            continue  # Skip the criminal, already counted
            
        if pid in role_predictions and role_predictions[pid] == true_role:
            roles_correct += 1
            role_score += points_per_role
    
    # Total score (0-100)
    total_score = criminal_score + role_score
    
    return total_score, criminal_correct, roles_correct, non_criminal_count, self_role_correct, self_role_prediction


def evaluate_model(model, dataset_path, num_scenarios=5, output_file=None):
    """
    Evaluate model performance on the MetaSkeptic dataset with per-round tracking
    and role-specific performance tracking
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (default: 5)
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
    
    # Track scores by round across all scenarios
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
            role_predictions = extract_role_predictions(response, num_players)
            
            # Calculate scores with new scoring system
            total_score, criminal_correct, roles_correct, non_criminal_count, self_role_correct, self_role_prediction = calculate_scores_with_self_role(
                criminal_prediction, role_predictions, ground_truth_roles, player_id
            )
            
            # Save round results
            round_result = {
                "round": round_num,
                "player_role": player_role,
                "criminal_prediction": criminal_prediction,
                "criminal_correct": criminal_correct,
                "criminal_score": 30 if criminal_correct else 0,
                "role_predictions": role_predictions,
                "roles_correct": roles_correct,
                "non_criminal_count": non_criminal_count,
                "role_score": total_score - (30 if criminal_correct else 0),
                "total_score": total_score,
                "self_role_correct": self_role_correct,
                "self_role_prediction": self_role_prediction,
                "response": response
            }
            
            round_results.append(round_result)
            round_metrics[round_num].append(round_result)
            
            # Also track by player's role
            if player_role in role_metrics:
                role_metrics[player_role][round_num].append(round_result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Save scenario results
        scenario_result = {
            "scenario_id": scenario_id,
            "dataset_type": scenario.get('dataset_type', ''),
            "ground_truth_roles": ground_truth_roles,
            "player_role": player_role,
            "rounds": round_results
        }
        
        results.append(scenario_result)
    
    # Calculate aggregate scores per round
    round_summaries = {}
    for round_num, metrics in round_metrics.items():
        criminal_accuracy = sum(m["criminal_correct"] for m in metrics) / len(metrics) * 100 if metrics else 0
        avg_total_score = sum(m["total_score"] for m in metrics) / len(metrics) if metrics else 0
        avg_criminal_score = sum(m["criminal_score"] for m in metrics) / len(metrics) if metrics else 0
        avg_role_score = sum(m["role_score"] for m in metrics) / len(metrics) if metrics else 0
        self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100 if metrics else 0
        
        round_summaries[round_num] = {
            "criminal_accuracy": criminal_accuracy,
            "avg_total_score": avg_total_score,
            "avg_criminal_score": avg_criminal_score,
            "avg_role_score": avg_role_score,
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
            avg_total_score = sum(m["total_score"] for m in metrics) / len(metrics)
            avg_criminal_score = sum(m["criminal_score"] for m in metrics) / len(metrics) 
            avg_role_score = sum(m["role_score"] for m in metrics) / len(metrics)
            self_role_accuracy = sum(m["self_role_correct"] for m in metrics) / len(metrics) * 100
            
            role_summaries[role][round_num] = {
                "criminal_accuracy": criminal_accuracy,
                "avg_total_score": avg_total_score,
                "avg_criminal_score": avg_criminal_score,
                "avg_role_score": avg_role_score,
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
        output_data = {
            "summary": summary,
            "detailed_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    return summary, results


def main():
    # Set the models to evaluate
    models = ['gpt-4o-mini','llama-3.3-70B','deepseek-r1-32B']
    
    # Dataset types to evaluate
    dataset_types = ['all']
    player_counts = [6]
    all_results = {}
    
    for model in models:
        model_results = {}
        
        for player_count in player_counts:
            for dataset_type in dataset_types:
                dataset_path = f"d:/github/MetaSkeptic/metaskeptic_{player_count}_{dataset_type}.json"
                
                if not os.path.exists(dataset_path):
                    print(f"Warning: {dataset_path} does not exist. Skipping.")
                    continue
                
                output_file = f"d:/github/MetaSkeptic/results/{model}_{player_count}_{dataset_type}_results.json"
                
                # Create results directory if it doesn't exist
                os.makedirs("d:/github/MetaSkeptic/results/", exist_ok=True)
                
                print(f"Evaluating {model} on {dataset_path}")
                
                summary, _ = evaluate_model(
                    model=model, 
                    dataset_path=dataset_path,
                    num_scenarios=17,  # Adjust as needed
                    output_file=output_file
                )
                
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
            print(f"  {'Round':<10} {'Criminal Acc':<15} {'Self-Role Acc':<15} {'Total Score':<15}")
            print(f"  {'-'*60}")
            
            for round_num, round_summary in sorted(summary["rounds"].items()):
                print(f"  {round_num:<10} {round_summary['criminal_accuracy']:>5.1f}%{'':<9} "
                      f"{round_summary['self_role_accuracy']:>5.1f}%{'':<9} "
                      f"{round_summary['avg_total_score']:>5.1f}/100{'':<5}")
        
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
                print(f"  {'Round':<10} {'Criminal Acc':<15} {'Self-Role Acc':<15} {'Total Score':<15}")
                print(f"  {'-'*60}")
                
                for round_num, round_summary in sorted(role_data.items()):
                    if "num_scenarios" in round_summary and round_summary["num_scenarios"] > 0:
                        print(f"  {round_num:<10} {round_summary['criminal_accuracy']:>5.1f}%{'':<9} "
                              f"{round_summary['self_role_accuracy']:>5.1f}%{'':<9} "
                              f"{round_summary['avg_total_score']:>5.1f}/100{'':<5}")
        
        print("\n" + "-"*80 + "\n")
    
    # Save overall summary
    with open("d:/github/MetaSkeptic/results/overall_summary.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()