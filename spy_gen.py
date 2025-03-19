import json
import random
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from tool import get_chat_response

load_dotenv()

# Word pairs for the game - normal word and its related spy word
# WORD_PAIRS = [
#     ("apple", "orange"),
#     ("dog", "cat"),
#     ("coffee", "tea"),
#     ("book", "magazine"),
#     ("piano", "guitar"),
#     ("sun", "moon"),
#     ("ocean", "lake"),
#     ("mountain", "hill"),
#     ("soccer", "basketball"),
#     ("car", "bicycle"),
#     ("cake", "cookie"),
#     ("chair", "table"),
#     ("pen", "pencil"),
#     ("phone", "computer"),
#     ("jacket", "sweater"),
#     ("river", "stream"),
#     ("star", "planet"),
#     ("butter", "cheese"),
#     ("bird", "insect"),
#     ("ring", "bracelet"),
#     ("hotel", "motel"),
#     ("candle", "lamp"),
#     ("forest", "jungle"),
#     ("autumn", "spring"),
#     ("train", "bus"),
#     ("potato", "carrot"),
#     ("hammer", "screwdriver"),
#     ("shoe", "boot"),
#     ("camera", "binoculars"),
#     ("sword", "knife"),
#     ("snow", "rain"),
#     ("bridge", "tunnel"),
#     ("drum", "flute"),
#     ("king", "queen"),
#     ("diamond", "pearl"),
#     ("pizza", "pasta"),
#     ("theater", "cinema"),
#     ("tiger", "lion"),
#     ("bread", "rice"),
#     ("wallet", "purse"),
#     ("castle", "palace"),
#     ("gold", "silver"),
#     ("beer", "wine"),
#     ("vampire", "zombie"),
#     ("hat", "cap"),
#     ("glove", "mitten"),
#     ("sofa", "couch"),
#     ("rose", "tulip"),
#     ("novel", "poem"),
#     ("fork", "spoon")
# ]
WORD_PAIRS = [
    ("China", "Japan")
]


def generate_scenario(scenario_id, player1_is_spy=None):
    """Generate a single game scenario with 4 players.
    
    Args:
        scenario_id: Unique identifier for the scenario
        player1_is_spy: If specified, forces player 1 to be spy or not
    
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
        prompts[player_id] = generate_player_prompt(player_id, word)
    
    # Data structure to store the scenario
    scenario = {
        "scenario_id": scenario_id,
        "normal_word": normal_word,
        "spy_word": spy_word,
        "spy_player": spy_player,
        "player_words": player_words,
        "prompts": prompts
    }
    
    return scenario

def generate_player_prompt(player_id, word):
    """Generate a prompt for a player to describe their word.
    
    Args:
        player_id: Player's identifier
        word: The word assigned to the player
    
    Returns:
        A string containing the prompt for the player
    """
    prompt = f"""You are playing a word description game with 4 players including yourself. You are Player {player_id}.

Game Rules:
1. In this game, 3 players receive the same word (normal word), and 1 player receives a different but related word (spy word).
2. Each player must describe their word without saying the actual word.
3. If you're a normal player, your goal is to identify the spy.
4. If you're the spy, your goal is to blend in and avoid detection.

Your word is: "{word}"

Please provide two different descriptions of your word:
1. A straightforward but subtle description that doesn't directly reveal your word
2. A more creative or metaphorical description that still relates to your word

Format your response as:
Round 1 description: [your first description]
Round 2 description: [your second description]

Remember:
- Don't use the actual word in your descriptions
- Balance between being clear enough for teammates but not too obvious
- Each description should be 1 sentences only
"""
    return prompt

def generate_dataset(n_scenarios):
    """Generate a dataset with a specified number of scenarios.
    
    Args:
        n_scenarios: Number of scenarios to generate
    
    Returns:
        A list of scenario dictionaries
    """
    dataset = []
    
    # Ensure equal distribution of player 1 being spy vs. not spy
    spy_count = n_scenarios // 2
    non_spy_count = n_scenarios - spy_count
    
    for i in range(1, spy_count + 1):
        scenario = generate_scenario(f"spy_{i}", player1_is_spy=True)
        dataset.append(scenario)
    
    for i in range(1, non_spy_count + 1):
        scenario = generate_scenario(f"normal_{i}", player1_is_spy=False)
        dataset.append(scenario)
    
    random.shuffle(dataset)
    
    return dataset

def get_description_from_api(prompt, player_id):
    """
    Get descriptions from API using different models for each player.
    
    Args:
        prompt: The prompt to send to the API
        player_id: Player's identifier (1-4)
        mock_descriptions: Whether to use mock descriptions (for testing)
    
    Returns:
        A dictionary with round 1 and round 2 descriptions
    """
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
    
    # Extract the descriptions from the response using regex
    round1_match = re.search(r"Round 1 description:\s*(.*?)(?=Round 2|$)", response, re.DOTALL)
    round2_match = re.search(r"Round 2 description:\s*(.*?)(?=$)", response, re.DOTALL)
    
    round1 = round1_match.group(1).strip() if round1_match else "No description provided."
    round2 = round2_match.group(1).strip() if round2_match else "No description provided."
    
    # Add a small delay to avoid rate limiting
    time.sleep(1)
    
    return {
        "round1": round1,
        "round2": round2,
        "full_response": response  # Store the full response for debugging
    }

def format_scenario_for_eval(scenario, include_descriptions=False):
    """Format a scenario into a format suitable for evaluation.
    
    Args:
        scenario: The scenario data
        include_descriptions: Whether to include API-generated descriptions
    
    Returns:
        A dict with the scenario formatted for evaluation
    """
    normal_word = scenario["normal_word"]
    spy_word = scenario["spy_word"]
    spy_player = scenario["spy_player"]
    
    players = ["1", "2", "3", "4"]
    descriptions = {}
    
    if include_descriptions:
        for player_id in players:
            prompt = scenario["prompts"][player_id]
            descriptions[player_id] = get_description_from_api(prompt, player_id, mock_descriptions=False)
    
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
    
    # Create statements for round 1
    round1_statements = {
        "round": 1,
        "statements": []
    }
    
    # Create statements for round 2
    round2_statements = {
        "round": 2,
        "statements": []
    }
    
    # Add statements for each player
    for player_id in ["1", "2", "3", "4"]:
        if player_id in scenario["descriptions"]:
            descriptions = scenario["descriptions"][player_id]
            
            # Add round 1 statement
            round1_statement = {
                "player": player_id,
                "statement": f"Player {player_id}: {descriptions['round1']}"
            }
            round1_statements["statements"].append(round1_statement)
            
            # Add round 2 statement
            round2_statement = {
                "player": player_id,
                "statement": f"Player {player_id}: {descriptions['round2']}"
            }
            round2_statements["statements"].append(round2_statement)
    
    statements.append(round1_statements)
    statements.append(round2_statements)
    
    return statements

def main(output_dir=None):
    """Generate and save the dataset."""
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir)
    
    # Generate dataset
    n_scenarios = 1  # Reduced for API usage
    dataset = generate_dataset(n_scenarios)
    print(f"Generated {len(dataset)} scenarios")
    
    # Save raw dataset
    raw_output_path = output_dir / "word_spy_raw.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Raw dataset saved to {raw_output_path}")
    
    # Create evaluation dataset with API-generated descriptions
    eval_dataset = []
    for i, scenario in enumerate(dataset):
        print(f"Processing scenario {i+1}/{len(dataset)}")
        eval_scenario = format_scenario_for_eval(scenario, include_descriptions=True)
        
        # Add formatted statements to the scenario
        eval_scenario["statements"] = generate_statements_from_descriptions(eval_scenario)
        
        eval_dataset.append(eval_scenario)
        
        # Save progress after each scenario to avoid losing data if the script crashes
        temp_output_path = output_dir / "word_spy_eval_progress.json"
        with open(temp_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
    
    # Save final evaluation dataset
    eval_output_path = output_dir / "word_spy_eval.json"
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2, ensure_ascii=False)
    print(f"Evaluation dataset with descriptions saved to {eval_output_path}")

if __name__ == "__main__":
    # Uncomment to use a specific directory for outputs
    # main("word_spy_data")
    main()