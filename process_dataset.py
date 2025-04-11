import json
import os

def process_data(file_path):
    """
    Process the blood dataset into HuggingFace format
    
    Args:
        file_path: Path to the blood_6_all.json file
    
    Returns:
        List of formatted data entries
    """
    # Load the data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for scenario in data:
        # Extract base information
        scenario_id = scenario['scenario_id']
        system_prompt = scenario['prompts']['1']
        ground_truth = scenario['ground_truth']
        statements = scenario['statements']
        reasoning_process = scenario['solution_analysis']['reasoning_process']
        
        # Extract role information
        player_id = "1"
        player_role = scenario['roles'][player_id]
        perceived_role = 'Criminal' if player_role == 'Lunatic' or player_role == 'Criminal' else 'Investigator'
        
        # Format rounds information
        rounds_info = {}
        for round_data in statements:
            round_num = round_data["round"]
            round_statements = round_data["statements"]
            
            # Format statements for this round
            round_text = f"Round {round_num} statements:\n"
            for stmt in round_statements:
                round_text += f"{stmt['statement']}\n"
            
            rounds_info[f"Round {round_num}"] = round_text.strip()
        
        # Create the reminder text as in blood_eva.py
        reminder_text = f"""Remember the following rules:
- Rumormongers believe they are Investigators (they are told they are Investigators), but they provide statements that may be incorrect.
- Lunatics believe they are Criminals (they are told they are Criminals) and make statements that may be truthful or false.
You are Player {player_id}, and you are told that you are {perceived_role}.

Your Task:
After considering the statements from each round, you must provide your judgment in the following format:

Final Judgment:
Final Criminal Is Player [Criminal Player Number].
My Role Is [Player 1's Role or "Unknown"].
"""
        
        # Create the full prompt
        prompt = reminder_text + "\n\n"
        
        # Add statements from all three rounds
        for round_num in range(1, 4):
            round_key = f"Round {round_num}"
            if round_key in rounds_info:
                prompt += rounds_info[round_key] + "\n\n"
        
        # Format the answer
        answer = f"Final Criminal Is Player {ground_truth['criminal']}.\nMy Role Is {ground_truth['player1_role']}."
        
        # Create the formatted entry
        entry = {
            "task": "Hidden Role Deduction",
            "system_prompt": system_prompt,
            "prompt": prompt.strip(),
            "Answer": answer,
            "reasoning_process": reasoning_process,
            "Round 1": rounds_info.get("Round 1", ""),
            "Round 2": rounds_info.get("Round 2", ""),
            "Round 3": rounds_info.get("Round 3", "")
        }
        
        formatted_data.append(entry)
    
    return formatted_data

def save_to_jsonl(data, output_path):
    """
    Save the processed data to a JSONL file for HuggingFace
    
    Args:
        data: List of formatted data entries
        output_path: Path to save the JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # Input file path
    input_file = 'blood/data/blood_6_all.json'
    
    # Output file path
    output_dir = 'blood/huggingface_data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'blood_6_all_hf.jsonl')
    
    # Process the data
    formatted_data = process_data(input_file)
    
    # Save the processed data
    save_to_jsonl(formatted_data, output_file)
    
    print(f"Successfully processed {len(formatted_data)} scenarios to {output_file}")

if __name__ == "__main__":
    main() 