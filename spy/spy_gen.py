import json
import random
import os
import time
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

# Word pairs for the game - normal word and its related spy word
WORD_PAIRS = [
    ("apple", "orange"),
    ("dog", "cat"),
    ("coffee", "tea"),
    ("China", "Japan"),
    ("book", "magazine"),
    ("piano", "guitar"),
    ("sun", "moon"),
    ("ocean", "lake"),
    ("mountain", "hill"),
    ("soccer", "basketball"),
    ("car", "bicycle"),
    ("cake", "cookie"),
    ("chair", "table"),
    ("pen", "pencil"),
    ("phone", "computer"),
    ("jacket", "sweater"),
    ("river", "stream"),
    ("star", "planet"),
    ("butter", "cheese"),
    ("bird", "insect"),
    ("ring", "bracelet"),
    ("hotel", "motel"),
    ("candle", "lamp"),
    ("forest", "jungle"),
    ("autumn", "spring"),
    ("train", "bus"),
    ("potato", "carrot"),
    ("hammer", "screwdriver"),
    ("shoe", "boot"),
    ("camera", "binoculars"),
    ("sword", "knife"),
    ("snow", "rain"),
    ("bridge", "tunnel"),
    ("drum", "flute"),
    ("king", "queen"),
    ("diamond", "pearl"),
    ("pizza", "pasta"),
    ("theater", "cinema"),
    ("tiger", "lion"),
    ("bread", "rice"),
    ("wallet", "purse"),
    ("castle", "palace"),
    ("gold", "silver"),
    ("beer", "wine"),
    ("vampire", "zombie"),
    ("hat", "cap"),
    ("glove", "mitten"),
    ("sofa", "couch"),
    ("rose", "tulip"),
    ("novel", "poem"),
    ("fork", "spoon")
]


def generate_scenario(scenario_id, player1_is_spy=None, num_rounds=2):
    """Generate a single game scenario with 4 players.
    
    Args:
        scenario_id: Unique identifier for the scenario
        player1_is_spy: If specified, forces player 1 to be spy or not
        num_rounds: Number of rounds of descriptions
    
    Returns:
        Dict containing the scenario data
    """
    # Choose a random word pair
    normal_word, spy_word = random.choice(WORD_PAIRS)
    
    # Determine who is the spy
    if player1_is_spy is None:
        spy_player = str(random.randint(1, 4))
    else:
        spy_player = "1" if player1_is_spy else str(random.randint(2, 4))
    
    # Assign words to players
    player_words = {}
    for i in range(1, 5):
        player_id = str(i)
        player_words[player_id] = spy_word if player_id == spy_player else normal_word
    
    # Create prompts for each player
    prompts = {}
    for player_id, word in player_words.items():
        prompts[player_id] = generate_player_prompt(player_id, word, num_rounds)
    
    # Data structure to store the scenario
    scenario = {
        "scenario_id": scenario_id,
        "normal_word": normal_word,
        "spy_word": spy_word,
        "spy_player": spy_player,
        "player_words": player_words,
        "prompts": prompts,
        "num_rounds": num_rounds
    }
    
    return scenario

def generate_player_prompt(player_id, word, num_rounds=2):
    """Generate a prompt for a player to describe their word.
    
    Args:
        player_id: Player's identifier
        word: The word assigned to the player
        num_rounds: Number of rounds of descriptions (default: 2)
    
    Returns:
        A string containing the prompt for the player
    """
    rounds_format = "\n".join([f"Round {i+1} description: [your description from a different angle]" for i in range(num_rounds)])
    
    prompt = f"""You are playing a word description game with 4 players including yourself. You are Player {player_id}.

Game Rules:
1. In this game, 3 players receive the same word (normal word), and 1 player receives a different but related word (spy word).
2. Each player must describe their word without saying the actual word.
3. If you're a normal player, your goal is to identify the spy.
4. If you're the spy, your goal is to blend in and avoid detection.

Your word is: "{word}"


Format your response as:
{rounds_format}

Remember:
- Don't use the actual word in your descriptions
- In each round, describe your word from a different perspective or focus on different aspects
- Balance between being clear enough for teammates but not too obvious
- Each description should be 1 sentence only
"""
    return prompt

def generate_dataset(n_scenarios, num_rounds=2):
    """Generate a dataset with a specified number of scenarios.
    
    Args:
        n_scenarios: Number of scenarios to generate
        num_rounds: Number of rounds of descriptions
    
    Returns:
        A list of scenario dictionaries
    """
    dataset = []
    
    # Ensure equal distribution of player 1 being spy vs. not spy
    spy_count = n_scenarios // 2
    non_spy_count = n_scenarios - spy_count
    
    for i in range(1, spy_count + 1):
        scenario = generate_scenario(f"spy_{i}", player1_is_spy=True, num_rounds=num_rounds)
        dataset.append(scenario)
    
    for i in range(1, non_spy_count + 1):
        scenario = generate_scenario(f"normal_{i}", player1_is_spy=False, num_rounds=num_rounds)
        dataset.append(scenario)
    
    random.shuffle(dataset)
    
    return dataset

def get_description_from_api(prompt, player_id, num_rounds=2, model_mapping=None):
    """
    Get descriptions from API using different models for each player.
    
    Args:
        prompt: The prompt to send to the API
        player_id: Player's identifier (1-4)
        num_rounds: Number of rounds of descriptions
        model_mapping: Optional dictionary mapping player IDs to model names
    
    Returns:
        A dictionary with round descriptions
    """
    # Default model mapping if none provided
    if model_mapping is None:
        model_mapping = {
            "1": "gpt-4o-mini",
            "2": "llama-3.3-70B",
            "3": "gemma-2-27B", 
            "4": "qwen-2.5-72B"
        }
    
    # Use the appropriate model based on player_id
    model = model_mapping[player_id]
    print(f"Getting descriptions for Player {player_id} using model: {model}")
    
    # Call the API using tool.get_chat_response
    response = get_chat_response(
        model=model,
        system_message="You are helping a player describe a word in a word guessing game.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    # Extract the descriptions from the response for each round
    descriptions = {}
    for round_num in range(1, num_rounds + 1):
        # Pattern to match each round's description, handling the last round differently
        if round_num < num_rounds:
            pattern = f"Round {round_num} description:\\s*(.*?)(?=Round {round_num+1}|$)"
        else:
            pattern = f"Round {round_num} description:\\s*(.*?)(?=$)"
        
        round_match = re.search(pattern, response, re.DOTALL)
        descriptions[f"round{round_num}"] = round_match.group(1).strip() if round_match else "No description provided."
    
    # Add full response for debugging
    descriptions["full_response"] = response
    
    # Add a small delay to avoid rate limiting
    time.sleep(1)
    
    return descriptions

def format_scenario_for_eval(scenario, include_descriptions=False, model_mapping=None):
    """Format a scenario into a format suitable for evaluation.
    
    Args:
        scenario: The scenario data
        include_descriptions: Whether to include API-generated descriptions
        model_mapping: Optional dictionary mapping player IDs to model names
    
    Returns:
        A dict with the scenario formatted for evaluation
    """
    normal_word = scenario["normal_word"]
    spy_word = scenario["spy_word"]
    spy_player = scenario["spy_player"]
    num_rounds = scenario["num_rounds"]
    
    players = ["1", "2", "3", "4"]
    descriptions = {}
    
    if include_descriptions:
        for player_id in players:
            prompt = scenario["prompts"][player_id]
            descriptions[player_id] = get_description_from_api(prompt, player_id, num_rounds, model_mapping)
    
    eval_scenario = {
        "scenario_id": scenario["scenario_id"],
        "normal_word": normal_word,
        "spy_word": spy_word,
        "spy_player": spy_player,
        "player_words": scenario["player_words"],
        "descriptions": descriptions if include_descriptions else {}
    }
    
    return eval_scenario

def generate_statements_from_descriptions(scenario):
    """
    Convert player descriptions into formatted statements for the game.
    
    Args:
        scenario: The scenario with descriptions
        
    Returns:
        A list of rounds, each containing statements from all players
    """
    statements = []
    num_rounds = scenario["num_rounds"]
    
    for round_num in range(1, num_rounds + 1):
        round_statements = {
            "round": round_num,
            "statements": []
        }
        
        # Add statements for each player
        for player_id in ["1", "2", "3", "4"]:
            if player_id in scenario["descriptions"]:
                descriptions = scenario["descriptions"][player_id]
                
                # Add round statement
                round_statement = {
                    "player": player_id,
                    "statement": f"Player {player_id}: {descriptions[f'round{round_num}']}"
                }
                round_statements["statements"].append(round_statement)
        
        statements.append(round_statements)
    
    return statements

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate dataset for the Spy game')
    parser.add_argument('--n_scenarios', type=int, default=10, 
                        help='Number of scenarios to generate')
    parser.add_argument('--num_rounds', type=int, default=3,
                        help='Number of rounds of descriptions (default: 2)')
    parser.add_argument('--output_name', type=str, default='spy_dataset',
                        help='Name of the output file (without extension)')
    parser.add_argument('--models', type=str, nargs=4, 
                        default=['gpt-4o-mini', 'llama-3.3-70B', 'gemma-2-27B', 'qwen-2.5-72B'],
                        help='Models to use for each player (4 models required)')
    return parser.parse_args()

def main():
    """Generate and save the dataset."""
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    if not data_dir.exists():
        os.makedirs(data_dir)
    
    # Set up model mapping from arguments
    model_mapping = {
        "1": args.models[0],
        "2": args.models[1],
        "3": args.models[2],
        "4": args.models[3]
    }
    
    # Generate dataset
    n_scenarios = args.n_scenarios
    num_rounds = args.num_rounds
    
    # Modify generate_dataset to include num_rounds
    dataset = []
    
    # Ensure equal distribution of player 1 being spy vs. not spy
    spy_count = n_scenarios // 2
    non_spy_count = n_scenarios - spy_count
    
    for i in range(1, spy_count + 1):
        scenario = generate_scenario(f"spy_{i}", player1_is_spy=True, num_rounds=num_rounds)
        dataset.append(scenario)
    
    for i in range(1, non_spy_count + 1):
        scenario = generate_scenario(f"normal_{i}", player1_is_spy=False, num_rounds=num_rounds)
        dataset.append(scenario)
    
    random.shuffle(dataset)
    
    print(f"Generated {len(dataset)} scenarios with {num_rounds} rounds each")
    
    # Save raw dataset
    raw_output_path = data_dir / f"{args.output_name}_raw.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Raw dataset saved to {raw_output_path}")
    
    # Create evaluation dataset with API-generated descriptions
    eval_dataset = []
    for i, scenario in enumerate(dataset):
        print(f"Processing scenario {i+1}/{len(dataset)}")
        
        # Update format_scenario_for_eval to handle num_rounds
        # Extract from scenario for use in get_description_from_api
        scenario_num_rounds = scenario.get("num_rounds", 2)
        
        eval_scenario = {
            "scenario_id": scenario["scenario_id"],
            "normal_word": scenario["normal_word"],
            "spy_word": scenario["spy_word"],
            "spy_player": scenario["spy_player"],
            "player_words": scenario["player_words"],
            "num_rounds": scenario_num_rounds,
            "descriptions": {}
        }
        
        # Get descriptions from API
        if True:  # include_descriptions=True
            for player_id in ["1", "2", "3", "4"]:
                prompt = scenario["prompts"][player_id]
                eval_scenario["descriptions"][player_id] = get_description_from_api(
                    prompt, player_id, scenario_num_rounds, model_mapping
                )
        
        # Add formatted statements to the scenario
        eval_scenario["statements"] = generate_statements_from_descriptions(eval_scenario)
        
        eval_dataset.append(eval_scenario)
        
        # Save progress after each scenario to avoid losing data if the script crashes
        temp_output_path = data_dir / f"{args.output_name}_progress.json"
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
    
    # Save final evaluation dataset
    eval_output_path = data_dir / f"{args.output_name}_eval.json"
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
    print(f"Evaluation dataset with {num_rounds} rounds of descriptions saved to {eval_output_path}")

if __name__ == "__main__":
    main()