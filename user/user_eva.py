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

def extract_demographic_prediction(response):
    """Extract the predicted age group and gender from model response.
    
    Args:
        response: The model's response text
        
    Returns:
        A tuple (age_group, gender) with the extracted predictions
    """
    age_patterns = [
        r"age group:\s*(18-34|35-54|55\+)",
        r"age group\s*:\s*(18-34|35-54|55\+)"
    ]
    
    gender_patterns = [
        r"gender:\s*(male|female|non-binary)",
        r"gender\s*:\s*(male|female|non-binary)"
    ]
    
    age_group = None
    for pattern in age_patterns:
        match = re.search(pattern, response.lower())
        if match:
            age_value = match.group(1).strip()
            age_group = age_value
            if age_value == "55+":
                age_group = "55+"
            break
    
    gender = None
    for pattern in gender_patterns:
        match = re.search(pattern, response.lower())
        if match:
            gender_value = match.group(1).strip()
            if gender_value == "male":
                gender = "Male"
            elif gender_value == "female":
                gender = "Female"
            elif gender_value == "non-binary":
                gender = "Non-binary"
            break
    
    if age_group is None:
        if re.search(r'\b18-34\b|\byoung\b|\byouth\b', response.lower()):
            age_group = "18-34"
        elif re.search(r'\b35-54\b|\bmiddle\b|\bmiddle-aged\b', response.lower()):
            age_group = "35-54"
        elif re.search(r'\b55\+\b|\bolderly\b|\bolder\b|\bsenior\b', response.lower()):
            age_group = "55+"
    
    if gender is None:
        if re.search(r'\b(?<!fe)male\b', response.lower()):  # male但前面不是fe
            gender = "Male"
        elif re.search(r'\bfemale\b|\bwoman\b|\bwomen\b', response.lower()):
            gender = "Female"
        elif re.search(r'\bnon-binary\b|\bnonbinary\b|\bnon binary\b', response.lower()):
            gender = "Non-binary"
    
    return age_group, gender

def generate_system_prompt_entity():
    """Generate system prompt for the user entity detection task."""
    return """Your job is to analyze multiple reviews for a product and determine the most likely demographics of the primary user group who wrote these reviews.

Focus on identifying the MAJORITY demographic group based on text patterns and content of the reviews.
"""

def generate_system_prompt_persona():
    """Generate system prompt for the individual user persona detection task."""
    return """Your job is to analyze multiple reviews written by the same person and determine their likely age group and gender based on their writing style, interests, and perspectives.

Focus on identifying the most likely demographic profile from the text patterns, interests, and perspectives in the comments.
"""

def generate_user_prompt_entity(scenario):
    """Generate user prompt for entity detection task containing product reviews.
    
    Args:
        scenario: The scenario data containing product info and reviews
        
    Returns:
        A string with the formatted prompt
    """
    prompt = f"""Reviews:
"""
    
    # Add each review
    for comment_data in scenario["comments"]:
        comment = comment_data["comment"]
        prompt += f"\n\"{comment}\"\n"
    
    prompt += """
Analyze these reviews carefully. What is the primary demographic group writing these reviews?

First explain your reasoning, then provide your final demographic classification in exactly this format:
Age Group: [18-34 OR 35-54 OR 55+]
Gender: [Male OR Female OR Non-binary]
"""
    
    return prompt

def generate_user_prompt_persona(profile_group):
    """Generate user prompt for persona detection task containing user reviews.
    
    Args:
        profile_group: The profile group data containing user info and comments
        
    Returns:
        A string with the formatted prompt
    """
    prompt = """Reviews:
"""
    
    # Add each review
    for comment_data in profile_group["comments"]:
        product = comment_data["product"]
        comment = comment_data["comment"]
        prompt += f"\nOn {product}: \"{comment}\"\n"
    
    prompt += """
Analyze these reviews carefully. What are the likely demographic characteristics of this user?

First explain your reasoning, then provide your final demographic classification in exactly this format:
Age Group: [18-34 OR 35-54 OR 55+]
Gender: [Male OR Female OR Non-binary]
"""
    
    return prompt

def evaluate_single_scenario_entity(model, scenario, scenario_id, true_age_group, true_gender):
    """Evaluate a single scenario for entity detection - helper function for parallel processing"""
    try:
        # Generate the prompts
        system_prompt = generate_system_prompt_entity()
        user_prompt = generate_user_prompt_entity(scenario)
        
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract predictions
        pred_age, pred_gender = extract_demographic_prediction(response)
        
        # Check correctness
        is_age_correct = pred_age == true_age_group
        is_gender_correct = pred_gender == true_gender
        
        # Store result
        result = {
            "scenario_id": scenario_id,
            "product_name": scenario["product_name"],
            "true_age_group": true_age_group,
            "true_gender": true_gender,
            "pred_age_group": pred_age,
            "pred_gender": pred_gender,
            "age_correct": is_age_correct,
            "gender_correct": is_gender_correct,
            "response": response
        }
        
        print(f"Completed evaluation for scenario {scenario_id}")
        return result
    except Exception as e:
        print(f"Error evaluating scenario {scenario_id}: {e}")
        return None

def evaluate_single_profile_persona(model, group, group_id, true_age_group, true_gender):
    """Evaluate a single profile group for persona detection - helper function for parallel processing"""
    try:
        # Generate the prompts
        system_prompt = generate_system_prompt_persona()
        user_prompt = generate_user_prompt_persona(group)
        
        # Get model response
        response = get_chat_response(
            model=model,
            system_message=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        
        # Extract predictions
        pred_age, pred_gender = extract_demographic_prediction(response)
        
        # Check correctness
        is_age_correct = pred_age == true_age_group
        is_gender_correct = pred_gender == true_gender
        
        # Store result
        result = {
            "group_id": group_id,
            "true_age_group": true_age_group,
            "true_gender": true_gender,
            "pred_age_group": pred_age,
            "pred_gender": pred_gender,
            "age_correct": is_age_correct,
            "gender_correct": is_gender_correct,
            "response": response
        }
        
        print(f"Completed evaluation for profile group {group_id}")
        return result
    except Exception as e:
        print(f"Error evaluating profile group {group_id}: {e}")
        return None

def evaluate_model_entity(model, dataset_path, num_scenarios=None, output_file=None, max_workers=5):
    """
    Evaluate model performance on the user entity detection task with parallel processing
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (None for all)
        output_file: optional file to save results
        max_workers: maximum number of concurrent evaluation threads
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Extract scenarios
    scenarios = list(dataset.get("scenarios", {}).values())
    scenario_ids = list(dataset.get("scenarios", {}).keys())
    
    # Sample scenarios to evaluate
    if num_scenarios is not None and num_scenarios < len(scenarios):
        scenarios = scenarios[:num_scenarios]
        scenario_ids = scenario_ids[:num_scenarios]
    
    total_scenarios = len(scenarios)
    print(f"Starting evaluation of {total_scenarios} scenarios using {model} model with {max_workers} workers")
    
    results = []
    age_correct = 0
    gender_correct = 0
    
    # Confusion matrix for age and gender
    age_confusion = {
        "18-34": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0},
        "35-54": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0},
        "55+": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0}
    }
    
    gender_confusion = {
        "Male": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0},
        "Female": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0},
        "Non-binary": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0}
    }
    
    # Use thread pool for concurrent evaluation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create evaluation tasks
        future_to_scenario = {}
        for i, (scenario, scenario_id) in enumerate(zip(scenarios, scenario_ids)):
            true_age_group = scenario["primary_user_group"]["primary_age_group"]
            true_gender = scenario["primary_user_group"]["primary_gender"]
            
            future = executor.submit(
                evaluate_single_scenario_entity,
                model,
                scenario,
                scenario_id,
                true_age_group,
                true_gender
            )
            future_to_scenario[future] = (scenario, scenario_id, i)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_scenario), total=len(future_to_scenario), desc=f"Evaluating {model} on entity detection"):
            scenario, scenario_id, idx = future_to_scenario[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # Update counters and confusion matrices
                    if result["age_correct"]:
                        age_correct += 1
                    if result["gender_correct"]:
                        gender_correct += 1
                    
                    pred_age = result["pred_age_group"]
                    pred_gender = result["pred_gender"]
                    true_age_group = result["true_age_group"]
                    true_gender = result["true_gender"]
                    
                    # Update confusion matrices
                    if pred_age:
                        age_confusion[true_age_group][pred_age] += 1
                    else:
                        age_confusion[true_age_group]["Unknown"] += 1
                        
                    if pred_gender:
                        gender_confusion[true_gender][pred_gender] += 1
                    else:
                        gender_confusion[true_gender]["Unknown"] += 1
            except Exception as e:
                print(f"Error processing result for scenario {scenario_id}: {e}")
    
    # Calculate accuracy
    age_accuracy = age_correct / total_scenarios * 100 if total_scenarios > 0 else 0
    gender_accuracy = gender_correct / total_scenarios * 100 if total_scenarios > 0 else 0
    overall_accuracy = (age_correct + gender_correct) / (total_scenarios * 2) * 100 if total_scenarios > 0 else 0
    
    # Create summary
    summary = {
        "model": model,
        "total_scenarios": total_scenarios,
        "age_correct": age_correct,
        "gender_correct": gender_correct,
        "age_accuracy": age_accuracy,
        "gender_accuracy": gender_accuracy,
        "overall_accuracy": overall_accuracy,
        "age_confusion": age_confusion,
        "gender_confusion": gender_confusion
    }
    
    # Save results if output file specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save detailed results and summary
        full_results = {
            "summary": summary,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    return summary

def evaluate_model_persona(model, dataset_path, num_scenarios=None, output_file=None, max_workers=5):
    """
    Evaluate model performance on the user persona detection task with parallel processing
    
    Args:
        model: model name
        dataset_path: path to dataset json file
        num_scenarios: number of scenarios to evaluate (None for all)
        output_file: optional file to save results
        max_workers: maximum number of concurrent evaluation threads
    
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Extract profile groups
    profile_groups = dataset.get("profile_groups", [])
    
    # Sample profile groups to evaluate
    if num_scenarios is not None and num_scenarios < len(profile_groups):
        profile_groups = profile_groups[:num_scenarios]
    
    total_groups = len(profile_groups)
    print(f"Starting evaluation of {total_groups} profile groups using {model} model with {max_workers} workers")
    
    results = []
    age_correct = 0
    gender_correct = 0
    
    # Confusion matrix for age and gender
    age_confusion = {
        "18-34": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0},
        "35-54": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0},
        "55+": {"18-34": 0, "35-54": 0, "55+": 0, "Unknown": 0}
    }
    
    gender_confusion = {
        "Male": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0},
        "Female": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0},
        "Non-binary": {"Male": 0, "Female": 0, "Non-binary": 0, "Unknown": 0}
    }
    
    # Use thread pool for concurrent evaluation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create evaluation tasks
        future_to_group = {}
        for group in profile_groups:
            group_id = group["group_id"]
            true_age_group = group["demographics"]["age_group"]
            true_gender = group["demographics"]["gender"]
            
            future = executor.submit(
                evaluate_single_profile_persona,
                model,
                group,
                group_id,
                true_age_group,
                true_gender
            )
            future_to_group[future] = (group, group_id)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_group), total=len(future_to_group), desc=f"Evaluating {model} on persona detection"):
            group, group_id = future_to_group[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # Update counters and confusion matrices
                    if result["age_correct"]:
                        age_correct += 1
                    if result["gender_correct"]:
                        gender_correct += 1
                    
                    pred_age = result["pred_age_group"]
                    pred_gender = result["pred_gender"]
                    true_age_group = result["true_age_group"]
                    true_gender = result["true_gender"]
                    
                    # Update confusion matrices
                    if pred_age:
                        age_confusion[true_age_group][pred_age] += 1
                    else:
                        age_confusion[true_age_group]["Unknown"] += 1
                        
                    if pred_gender:
                        gender_confusion[true_gender][pred_gender] += 1
                    else:
                        gender_confusion[true_gender]["Unknown"] += 1
            except Exception as e:
                print(f"Error processing result for profile group {group_id}: {e}")
    
    # Calculate accuracy
    age_accuracy = age_correct / total_groups * 100 if total_groups > 0 else 0
    gender_accuracy = gender_correct / total_groups * 100 if total_groups > 0 else 0
    overall_accuracy = (age_correct + gender_correct) / (total_groups * 2) * 100 if total_groups > 0 else 0
    
    # Create summary
    summary = {
        "model": model,
        "total_groups": total_groups,
        "age_correct": age_correct,
        "gender_correct": gender_correct,
        "age_accuracy": age_accuracy,
        "gender_accuracy": gender_accuracy,
        "overall_accuracy": overall_accuracy,
        "age_confusion": age_confusion,
        "gender_confusion": gender_confusion
    }
    
    # Save results if output file specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save detailed results and summary
        full_results = {
            "summary": summary,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model performance on user demographic prediction')
    parser.add_argument('--models', type=str, nargs='+', default=['gemma-2-9B', 'gemma-3-27B', 'qwen-2.5-72B', 'gpt-4o-mini','llama-3.3-70B','qwq'],
                        help='Models to evaluate (can provide multiple)')
    parser.add_argument('--entity_dataset', type=str, default='data/user_entity.json', 
                        help='Path to entity dataset')
    parser.add_argument('--persona_dataset', type=str, default='data/user_persona.json', 
                        help='Path to persona dataset')
    parser.add_argument('--num_scenarios', type=int, default=None, 
                        help='Number of scenarios to evaluate (default: all)')
    parser.add_argument('--task', type=str, choices=['entity', 'persona', 'both'], default='both',
                        help='Which evaluation task to run')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of concurrent workers for parallel processing')
    
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_arguments()
    
    # Ensure results directory exists
    results_dir = Path("results")
    if not results_dir.exists():
        os.makedirs(results_dir)
    
    # Entity detection task
    if args.task in ['entity', 'both']:
        entity_summary = {}
        # 读取已有的summary文件（如果存在）
        entity_summary_path = results_dir / "entity_summary.json"
        if entity_summary_path.exists():
            try:
                with open(entity_summary_path, 'r', encoding='utf-8') as f:
                    entity_summary = json.load(f)
                print(f"已加载现有的entity_summary文件，包含{len(entity_summary)}个模型的结果")
            except Exception as e:
                print(f"读取entity_summary文件时出错: {e}")
                entity_summary = {}
        
        for model in args.models:
            print(f"\nEvaluating {model} on entity detection task...")
            
            # Create unique output filename for this model
            output_file = results_dir / f"{model}_entity_results.json"
            
            # Run evaluation
            model_results = evaluate_model_entity(
                model, 
                args.entity_dataset, 
                args.num_scenarios, 
                output_file,
                args.max_workers
            )
            
            # Add model results to summary
            entity_summary[model] = {
                "age_accuracy": model_results["age_accuracy"],
                "gender_accuracy": model_results["gender_accuracy"],
                "overall_accuracy": model_results["overall_accuracy"]
            }
            
            print(f"Age Group Accuracy: {model_results['age_accuracy']:.2f}%")
            print(f"Gender Accuracy: {model_results['gender_accuracy']:.2f}%")
            print(f"Overall Accuracy: {model_results['overall_accuracy']:.2f}%")
        
        # Save entity summary
        with open(entity_summary_path, 'w', encoding='utf-8') as f:
            json.dump(entity_summary, f, indent=2, ensure_ascii=False)
        print(f"更新后的entity_summary文件已保存，共包含{len(entity_summary)}个模型的结果")
    
    # Persona detection task
    if args.task in ['persona', 'both']:
        persona_summary = {}
        # 读取已有的summary文件（如果存在）
        persona_summary_path = results_dir / "persona_summary.json"
        if persona_summary_path.exists():
            try:
                with open(persona_summary_path, 'r', encoding='utf-8') as f:
                    persona_summary = json.load(f)
                print(f"已加载现有的persona_summary文件，包含{len(persona_summary)}个模型的结果")
            except Exception as e:
                print(f"读取persona_summary文件时出错: {e}")
                persona_summary = {}
        
        for model in args.models:
            print(f"\nEvaluating {model} on persona detection task...")
            
            # Create unique output filename for this model
            output_file = results_dir / f"{model}_persona_results.json"
            
            # Run evaluation
            model_results = evaluate_model_persona(
                model, 
                args.persona_dataset, 
                args.num_scenarios, 
                output_file,
                args.max_workers
            )
            
            # Add model results to summary
            persona_summary[model] = {
                "age_accuracy": model_results["age_accuracy"],
                "gender_accuracy": model_results["gender_accuracy"],
                "overall_accuracy": model_results["overall_accuracy"]
            }
            
            print(f"Age Group Accuracy: {model_results['age_accuracy']:.2f}%")
            print(f"Gender Accuracy: {model_results['gender_accuracy']:.2f}%")
            print(f"Overall Accuracy: {model_results['overall_accuracy']:.2f}%")
        
        # Save persona summary
        with open(persona_summary_path, 'w', encoding='utf-8') as f:
            json.dump(persona_summary, f, indent=2, ensure_ascii=False)
        print(f"更新后的persona_summary文件已保存，共包含{len(persona_summary)}个模型的结果")

if __name__ == "__main__":
    main()
