import json
import random
from itertools import combinations

def assign_roles(number=6, dataset_type='original'):
    roles_config = {
        "6-original": {"Criminal": 1, "Investigator": 5},
        "6-rumormonger": {"Criminal": 1, "Investigator": 4, "Rumormonger": 1},
        "6-lunatic": {"Criminal": 1, "Investigator": 4, "Lunatic": 1},
        "6-all": {"Criminal": 1, "Investigator": 3, "Rumormonger": 1, "Lunatic": 1},
        '10-original': {"Criminal": 1, "Investigator": 9},
        '10-rumormonger': {"Criminal": 1, "Investigator": 7, "Rumormonger": 2},
        '10-lunatic': {"Criminal": 1, "Investigator": 7, "Lunatic": 2},
        '10-all': {"Criminal": 1, "Investigator": 5, "Rumormonger": 2, "Lunatic": 2}
    }

    role_counts = roles_config[f"{number}-{dataset_type}"]
    
    # Get all possible roles in this game configuration
    possible_roles = list(role_counts.keys())
    
    # Randomly select a role for player 1 with equal probability among available roles
    player1_role = random.choice(possible_roles)
    
    # Decrease the count of the selected role
    role_counts[player1_role] -= 1
    
    # Remove roles with zero count
    roles_list = []
    for role, count in role_counts.items():
        roles_list.extend([role] * count)
    
    # Shuffle the remaining roles
    random.shuffle(roles_list)
    
    # Assign player 1 first, then the rest
    player_roles = {"1": player1_role}
    
    # Assign the remaining roles to players 2 through number
    for i in range(1, number):
        player_roles[str(i+1)] = roles_list[i-1]
    
    return player_roles

def generate_statement(player_id, player_role, all_players, criminal_id):
    target_player = random.choice(all_players)
    while (target_player == player_id): 
        target_player = random.choice(all_players)

    is_criminal = target_player == criminal_id
    statement_type = random.choice(["is", "is not"])

    if player_role == "Investigator":
        # Investigator always tells the truth
        statement_truth = "is" if is_criminal else "is not"
        statement_target = statement_truth 
    elif player_role == "Criminal" or player_role == "Lunatic":
        # Criminal tells lies, but sometimes tells the truth to confuse others
        statement_target = "is"
        if random.random() < 0.5: 
            statement_target = "is not"
    elif player_role == "Rumormonger":
        # Rumormonger thinks they are helping, but actually not
        statement_target = statement_type

    statement = f"Player {player_id} says Player {target_player} {statement_target} the criminal."
    return {
        "text": statement,
        "target_player": target_player,
        "statement_type": statement_target
    }

def generate_prompts(player_roles, number, dataset_type):
    prompts = {}
    
    role_counts = {
        "6-original": {"Investigator": 5, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        "6-rumormonger": {"Investigator": 4, "Criminal": 1, "Rumormonger": 1, "Lunatic": 0},
        "6-lunatic": {"Investigator": 4, "Criminal": 1, "Rumormonger": 0, "Lunatic": 1},
        "6-all": {"Investigator": 3, "Criminal": 1, "Rumormonger": 1, "Lunatic": 1},
        '10-original': {"Investigator": 9, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        '10-rumormonger': {"Investigator": 7, "Criminal": 1, "Rumormonger": 2, "Lunatic": 0},
        '10-lunatic': {"Investigator": 7, "Criminal": 1, "Rumormonger": 0, "Lunatic": 2},
        '10-all': {"Investigator": 5, "Criminal": 1, "Rumormonger": 2, "Lunatic": 2}
    }
    
    counts = role_counts[f"{number}-{dataset_type}"]
    
    for player_id, role in player_roles.items():
        # Determine the perceived role that will be shown to the player
        perceived_role = role
        if role == "Rumormonger":
            perceived_role = "Investigator"  # Rumormongers think they are Investigators
        elif role == "Lunatic":
            perceived_role = "Criminal"      # Lunatics think they are Criminals
        
        base_prompt = f"""You are an expert in social reasoning. You are participating in a social deduction game with a total of {number} players including yourself. Here are the game rules:

Game Roles:
- There are {counts["Investigator"]} Investigator(s), {counts["Criminal"]} Criminal(s), {counts["Rumormonger"]} Rumormonger(s), and {counts["Lunatic"]} Lunatic(s) in this game.

Game Mechanics:
- There are 3 rounds in this game.
- In each round, every player will make one statement about another player, claiming whether that player is the criminal or not.
- Investigators always provide truthful information about whether someone is a Criminal.
- Criminals make statements about other players that may be truthful or false to confuse others.
- Rumormongers believe they are Investigators (they are told they are Investigators), but they provide statements that may be incorrect.
- Lunatics believe they are Criminals (they are told they are Criminals) and make statements that may be truthful or false.
- Only Investigators provide consistently truthful information. Other roles may provide misleading or incorrect information.

You are Player {player_id}, and your role is {perceived_role}.
"""
        analysis_task = """
After each round of statements, you need to analyze all the statements so far and deduce who you think is the criminal and what is the role of each player.

Your Task:
After considering the statements from each round, please provide your judgment in the following format:

Final Judgment:
Final Criminal Is Player [Criminal Player Number].
The roles of players 1-{} are:
""".format(number)

        for i in range(1, number + 1):
            analysis_task += f"Final Player {i}: [Role]\n"
        
        analysis_task += """

Please provide this Final Judgment after each round of statements, updating your analysis as you receive more information. Even if you're uncertain, you must provide your best guess for the criminal and each player's role. Think step-by-step to reach your judgment, and then commit to your most likely conclusion based on available information.
"""

        prompt = base_prompt + analysis_task
        prompts[player_id] = prompt
    
    return prompts


def generate_dataset(number, dataset_type, n_scenarios_per_type):
    dataset = []
    solvable_scenarios = 0
    
    while solvable_scenarios < n_scenarios_per_type:
        scenario_id = f"{dataset_type}_{solvable_scenarios+1}"
        player_roles = assign_roles(number, dataset_type)
        criminal_id = [player for player, role in player_roles.items() if role == "Criminal"][0] 
        statements = []
        all_players = list(player_roles.keys())

        for round_num in range(1, 4): # 3 rounds
            round_statements = []
            for player_id in all_players:
                # Generate statement with structured information
                statement_info = generate_statement(player_id, player_roles[player_id], all_players, criminal_id)
                # Keep both structured and text versions
                round_statements.append({
                    "player": player_id, 
                    "statement": statement_info["text"],
                    "target_player": statement_info["target_player"],
                    "statement_type": statement_info["statement_type"]
                })
            statements.append({"round": round_num, "statements": round_statements})

        prompts = generate_prompts(player_roles, number, dataset_type)
        prompts = {k: v for k, v in prompts.items() if k== '1'}

        scenario_data = {
            "scenario_id": scenario_id,
            "dataset_type": dataset_type,
            "roles": player_roles,
            "statements": statements,
            "ground_truth_roles": player_roles,
            "prompts": prompts 
        }
        
        # Check if this scenario has a unique solution
        solution_analysis = check_unique_solution(scenario_data)
        scenario_data["solution_analysis"] = solution_analysis
        
        # Only include scenarios with a unique solution
        if solution_analysis["unique_solution"]:
            dataset.append(scenario_data)
            solvable_scenarios += 1
            
            # Print progress
            if solvable_scenarios % 10 == 0:
                print(f"Generated {solvable_scenarios}/{n_scenarios_per_type} solvable scenarios for {dataset_type} with {number} players")
    
    return dataset

def check_unique_solution(scenario_data):
    """
    Check if there's a unique solution for identifying the criminal from player 1's perspective.
    
    Returns:
        dict: Contains 'unique_solution' (bool), 'possible_criminals' (list), and analysis details
    """
    player_roles = scenario_data["roles"]
    all_players = list(player_roles.keys())
    statements = scenario_data["statements"]
    number = len(all_players)
    dataset_type = scenario_data["dataset_type"]
    
    # Get role counts for this scenario
    role_counts = {
        "6-original": {"Investigator": 5, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        "6-rumormonger": {"Investigator": 4, "Criminal": 1, "Rumormonger": 1, "Lunatic": 0},
        "6-lunatic": {"Investigator": 4, "Criminal": 1, "Rumormonger": 0, "Lunatic": 1},
        "6-all": {"Investigator": 3, "Criminal": 1, "Rumormonger": 1, "Lunatic": 1},
        '10-original': {"Investigator": 9, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        '10-rumormonger': {"Investigator": 7, "Criminal": 1, "Rumormonger": 2, "Lunatic": 0},
        '10-lunatic': {"Investigator": 7, "Criminal": 1, "Rumormonger": 0, "Lunatic": 2},
        '10-all': {"Investigator": 5, "Criminal": 1, "Rumormonger": 2, "Lunatic": 2}
    }
    
    counts = role_counts[f"{number}-{dataset_type}"]
    investigator_count = counts["Investigator"]
    
    # Player 1's actual role
    player1_role = player_roles["1"]
    possible_criminals = set()
    
    # Determine which perspectives player 1 might have based on their actual role
    perspective_roles = []
    
    if player1_role == "Investigator" or player1_role == "Rumormonger":
        perspective_roles.append("Investigator")
        perspective_roles.append("Rumormonger")
        #我被告知是Investigator,但我也有可能是Rumormonger

        for perspective in perspective_roles:
            # Find all possible combinations of investigators
            potential_investigators = []
            
            if perspective == "Investigator":
                potential_investigators.append("1")
                remaining_investigator_count = investigator_count - 1
                #假如说我是Investigator,那么还剩下remaining_investigator_count个Investigator
            elif perspective == "Rumormonger":
                remaining_investigator_count = investigator_count
                #假如说我是Rumormonger,那么还剩下remaining_investigator_count个Investigator
            
            # Add all other players who could be investigators
            # Only players who have consistent statements can be potential investigators
            for player_id in all_players:
                #分析一下除了我之外的玩家,还有谁可以做Investigator
                if player_id != "1":
                    can_be_investigator = True
                    criminal_accusations = set()  
                    not_criminal_claims = set()  
                    for round_data in statements:
                        for statement in round_data["statements"]:
                            if statement["player"] == player_id:
                                target = statement["target_player"]
                                claim = statement["statement_type"]
                                
                                if target == "1" and claim == "is":
                                    #player_id说我是criminal,但我不可能是criminal,它一定不是Investigator
                                    can_be_investigator = False
                                    break
                                
                                if claim == "is":
                                    criminal_accusations.add(target)
                                elif claim == "is not":
                                    not_criminal_claims.add(target)
                                
                                if any(p in not_criminal_claims for p in criminal_accusations):
                                    #player_id说p是criminal但又说p不是,自相矛盾,它一定不是Investigator
                                    can_be_investigator = False
                                    break
                                

                                if len(criminal_accusations) > 1:
                                    can_be_investigator = False
                                    #player_id说过不止一个人是criminal,它一定不是Investigator
                                    break
                    
                        if not can_be_investigator:
                            break
                    
                    if can_be_investigator:
                        potential_investigators.append(player_id)
                    
            #所以最后可能做Investigator的人有:potential_investigators
            for investigators in combinations(potential_investigators[1:], remaining_investigator_count):
                if perspective == "Investigator":
                    investigator_set = {"1"}.union(set(investigators))
                    #当我是Investigator时,我一定是Investigator
                else:
                    investigator_set = set(investigators)
                    #当我是Rumormonger时,我一定不是Investigator
                is_consistent = True
                
                #假设真正的的investigator是investigators

                criminal_candidates = set(all_players) - investigator_set - {"1"}
                #那么可能是criminal的人有:criminal_candidates
                #我们回忆下所有investigators的statement
                for round_data in statements:
                    for statement in round_data["statements"]:
                        player = statement["player"]
                        target = statement["target_player"]
                        claim = statement["statement_type"]
                        
                        if player in investigator_set:
                            if claim == "is":
                                # player说过target是criminal
                                if target not in criminal_candidates:
                                    is_consistent = False
                                    #但target不在可能是criminal的人里,这是矛盾的
                                    break
                                else:
                                    criminal_candidates = {target}
                                    # 所以可能是criminal的人只有target
                            
                            elif claim == "is not":
                                # player说过target不是criminal
                                if target in criminal_candidates:
                                    criminal_candidates.remove(target)
                                    #但target在可能是criminal的人里,所以我们从中去掉target
                                    
                                if not criminal_candidates:
                                    is_consistent = False
                                    #如果可能是criminal的人为空,这是矛盾的
                                    break
                    
                    if not is_consistent:
                        # 所以这种情况不可能发生
                        break
                
                if is_consistent and len(criminal_candidates) == 1:
                    # 最终我们发现,有且仅有xxx可能是criminal
                    possible_criminals.update(criminal_candidates)

    # Handle Criminal and Lunatic perspectives
    if player1_role == "Criminal" or player1_role == "Lunatic":
        perspective_roles.append("Criminal")
        perspective_roles.append("Lunatic")
        #我被告知是Criminal,但我也有可能是Lunatic

        for perspective in perspective_roles:
            potential_investigators = []
            remaining_investigator_count = investigator_count

            for player_id in all_players:
                #分析一下除了我之外的玩家,还有谁可以做Investigator
                if player_id != "1":
                    can_be_investigator = True
                    criminal_accusations = set()  
                    not_criminal_claims = set()  
                    for round_data in statements:
                        for statement in round_data["statements"]:
                            if statement["player"] == player_id:
                                target = statement["target_player"]
                                claim = statement["statement_type"]
                                
                                if target == "1" and claim == "is" and perspective == "Lunatic":
                                    #player_id说我是criminal,但我是Lunatic,它一定不是Investigator
                                    can_be_investigator = False
                                    break
                                if target == "1" and claim == "is not" and perspective == "Criminal":
                                    #player_id说我不是criminal,但我是Criminal,它一定不是Investigator
                                    can_be_investigator = False
                                    break
                                if claim == "is":
                                    criminal_accusations.add(target)
                                elif claim == "is not":
                                    not_criminal_claims.add(target)
                                
                                if any(p in not_criminal_claims for p in criminal_accusations):
                                    #player_id说p是criminal但又说p不是,自相矛盾,它一定不是Investigator
                                    can_be_investigator = False
                                    break
                                

                                if len(criminal_accusations) > 1:
                                    can_be_investigator = False
                                    #player_id说过不止一个人是criminal,它一定不是Investigator
                                    break
                    
                        if not can_be_investigator:
                            break
                    
                    if can_be_investigator:
                        potential_investigators.append(player_id)
                
                #所以最后可能做Investigator的人有:potential_investigators
                for investigators in combinations(potential_investigators, remaining_investigator_count):
                    investigator_set = set(investigators)
                    #假设真正的investigator是investigators
                    
                    is_consistent = True
                   
                    if perspective == "Criminal":
                        criminal_candidates = {"1"}
                        #当我是Criminal时,我一定是唯一的criminal
                    elif perspective == "Lunatic":
                        criminal_candidates = set(all_players) - {"1"}
                        #当我是Lunatic时,除了我之外,其他人都有可能是criminal
                    
                        #我们回忆下所有investigators的statement
                    for round_data in statements:
                        for statement in round_data["statements"]:
                            player = statement["player"]
                            target = statement["target_player"]
                            claim = statement["statement_type"]
                            
                            if player in investigator_set:
                                if claim == "is":
                                    # player说过target是criminal
                                    if target not in criminal_candidates:
                                        is_consistent = False
                                        #但target不在可能是criminal的人里,这是矛盾的
                                        break
                                    else:
                                        criminal_candidates = {target}
                                        # 所以可能是criminal的人只有target
                                
                                elif claim == "is not":
                                    # player说过target不是criminal
                                    if target in criminal_candidates:
                                        criminal_candidates.remove(target)
                                        #但target在可能是criminal的人里,所以我们从中去掉target
                                        
                                    if not criminal_candidates:
                                        is_consistent = False
                                        #如果可能是criminal的人为空,这是矛盾的
                                        break
                        
                        if not is_consistent:
                            # 所以这种情况不可能发生
                            break
                    
                    if is_consistent and len(criminal_candidates) == 1:
                        # 最终我们发现,有且仅有xxx可能是criminal
                        possible_criminals.update(criminal_candidates)
        
    return {
        "unique_solution": len(possible_criminals) == 1,
        "possible_criminals": list(possible_criminals),
        "player1_role": player1_role
    }

if __name__ == "__main__":
    dataset_types = ["original", "rumormonger", "lunatic", "all"] 
    # Fix 2: Rename the variable to avoid name conflict
    player_counts = [6, 10]
    n_scenarios_per_type = 202  

    for dataset_type in dataset_types:
        # Fix 3: Use player_count instead of number
        for player_count in player_counts:
            dataset = generate_dataset(player_count, dataset_type, n_scenarios_per_type)
            # Fix 4: Include player count in output filename for clarity
            output_filename = f"metaskeptic_{player_count}_{dataset_type}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False) 
            print(f"Dataset '{dataset_type}' with {player_count} players saved to '{output_filename}'")

    print("All datasets generated successfully!")