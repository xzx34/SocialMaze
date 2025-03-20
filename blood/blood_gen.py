import json
import random
import os
import argparse
from itertools import combinations

def assign_roles(number=6, dataset_type='original', player1_role=None):
    """
    Assign roles to players with optional specification of player 1's role.
    
    Args:
        number: Total number of players
        dataset_type: Type of dataset/game configuration
        player1_role: If specified, forces player 1 to have this role
    """
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

    role_counts = roles_config[f"{number}-{dataset_type}"].copy()
    
    # Get all possible roles in this game configuration
    possible_roles = list(role_counts.keys())
    
    # If player1_role is specified and valid, use it
    if player1_role and player1_role in possible_roles and role_counts[player1_role] > 0:
        player1_role = player1_role
    else:
        # Otherwise randomly select a role for player 1 with equal probability among available roles
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
My Role Is [Player 1's Role or "Unknown"].

Please provide this Final Judgment after each round of statements, updating your analysis as you receive more information. Think step-by-step to reach your judgment, and then commit to your most likely conclusion based on available information. If you cannot determine your role with sufficient confidence, state your role as "Unknown".
"""

        prompt = base_prompt + analysis_task
        prompts[player_id] = prompt
    
    return prompts


def generate_dataset(number, dataset_type, n_scenarios_per_type):
    """
    Generate a dataset with equal distribution of player 1's roles.
    """
    dataset = []
    solvable_scenarios = 0
    
    # Determine how many scenarios we need per player 1 role
    possible_roles = []
    role_counts = {
        "6-original": {"Investigator": 5, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        "6-rumormonger": {"Investigator": 4, "Criminal": 1, "Rumormonger": 1, "Lunatic": 0},
        "6-lunatic": {"Investigator": 4, "Criminal": 1, "Rumormonger": 0, "Lunatic": 1},
        "6-all": {"Investigator": 3, "Criminal": 1, "Rumormonger": 1, "Lunatic": 1},
        '10-original': {"Investigator": 9, "Criminal": 1, "Rumormonger": 0, "Lunatic": 0},
        '10-rumormonger': {"Investigator": 7, "Criminal": 1, "Rumormonger": 2, "Lunatic": 0},
        '10-lunatic': {"Investigator": 7, "Criminal": 1, "Rumormonger": 0, "Lunatic": 2},
        '10-all': {"Investigator": 5, "Criminal": 1, "Rumormonger": 2, "Lunatic": 2}
    }[f"{number}-{dataset_type}"]
    
    for role, count in role_counts.items():
        if count > 0:
            possible_roles.append(role)
    
    # Calculate scenarios per role
    scenarios_per_role = n_scenarios_per_type // len(possible_roles)
    extra_scenarios = n_scenarios_per_type % len(possible_roles)
    
    # Keep track of generated scenarios per role
    role_scenario_counts = {role: 0 for role in possible_roles}
    role_scenario_targets = {role: scenarios_per_role + (1 if i < extra_scenarios else 0) 
                            for i, role in enumerate(possible_roles)}
    
    # Generate scenarios until we have enough for each role
    attempts = 0
    max_attempts = n_scenarios_per_type * 100  # Safeguard against infinite loops
    
    while solvable_scenarios < n_scenarios_per_type and attempts < max_attempts:
        attempts += 1
        
        # Determine which role to generate next
        available_roles = [role for role in possible_roles 
                          if role_scenario_counts[role] < role_scenario_targets[role]]
        
        if not available_roles:
            break
        
        target_role = random.choice(available_roles)
        
        # Generate scenario with the target role
        scenario_id = f"{dataset_type}_{solvable_scenarios+1}"
        player_roles = assign_roles(number, dataset_type, player1_role=target_role)
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
        # Only keep player 1's prompt
        prompts = {k: v for k, v in prompts.items() if k == '1'}

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
            role_scenario_counts[player_roles["1"]] += 1
            
            # Print progress
            if solvable_scenarios % 10 == 0 or solvable_scenarios == n_scenarios_per_type:
                print(f"Generated {solvable_scenarios}/{n_scenarios_per_type} solvable scenarios for {dataset_type} with {number} players")
                print(f"Role distribution: {role_scenario_counts}")
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts ({max_attempts}) for {dataset_type} with {number} players")
        print(f"Only generated {solvable_scenarios}/{n_scenarios_per_type} scenarios")
    
    return dataset

def check_unique_solution(scenario_data):
    """
    Check if there's a unique solution for identifying the criminal from player 1's perspective.
    """
    # Note: This function is quite large so I'm keeping it as is
    # Your existing check_unique_solution implementation
    player_roles = scenario_data["roles"]
    all_players = list(player_roles.keys())
    statements = scenario_data["statements"]
    number = len(all_players)
    dataset_type = scenario_data["dataset_type"]
    reasoning_process=""
    return_reasoning_process=''
    investigator_reasoning=""
    investigator_possible=False
    rumormonger_reasoning=""
    rumormonger_possible=False
    lunatic_reasoning=""
    lunatic_possible=False
    criminal_reasoning=""
    criminal_possible=False
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

        return_reasoning_process+="我被告知是Investigator,但我也有可能是Rumormonger,我需要讨论这两种情况的可能性\n"
        for perspective in perspective_roles:
            reasoning_process=""
            potential_investigators = []
            
            if perspective == "Investigator":
                potential_investigators.append("1")
                remaining_investigator_count = investigator_count - 1
                reasoning_process+=f"考虑我是Investigator的情况,那么还剩下{remaining_investigator_count}个Investigator\n"
            elif perspective == "Rumormonger":
                remaining_investigator_count = investigator_count
                #考虑我是Rumormonger的情况,那么还剩下remaining_investigator_count个Investigator
                reasoning_process+=f"考虑我是Rumormonger的情况,那么还剩下{remaining_investigator_count}个Investigator\n"

            reasoning_process+="这种情况下，有哪些玩家可以做Investigator呢？\n"
            for player_id in all_players:
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
                                    reasoning_process+=f"player{player_id}说我是criminal,但我不可能是criminal,,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                                
                                if claim == "is":
                                    criminal_accusations.add(target)
                                elif claim == "is not":
                                    not_criminal_claims.add(target)
                                
                                if any(p in not_criminal_claims for p in criminal_accusations):
                                    #player_id说p是criminal但又说p不是,自相矛盾,它一定不是Investigator
                                    p=next(iter(criminal_accusations.intersection(not_criminal_claims)))

                                    reasoning_process+=f"player{player_id}说{p}是criminal但又说{p}不是,自相矛盾,,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                                

                                if len(criminal_accusations) > 1:
                                    can_be_investigator = False
                                    #player_id说过不止一个人是criminal,它一定不是Investigator
                                    reasoning_process+=f"player{player_id}说过不止一个人是criminal,排除player{player_id}\n"
                                    break
                    
                        if not can_be_investigator:
                            break
                    
                    if can_be_investigator:
                        potential_investigators.append(player_id)
                        reasoning_process+=f"player{player_id}的发言没有矛盾，可以做Investigator\n"
                    
            #所以最后可能做Investigator的人有:potential_investigators
            reasoning_process+=f"所以最后可能做Investigator的玩家有:{potential_investigators}\n"
            if len(potential_investigators) < remaining_investigator_count:
                reasoning_process+=f"可能做Investigator的人数不足{remaining_investigator_count}个,这种情况不可能成立\n"
                if perspective == "Investigator":
                    investigator_reasoning=reasoning_process+"我一定是Rumormonger\n"
                else:
                    rumormonger_reasoning=reasoning_process+"我一定是Investigator\n"
                continue
            reasoning_process+="现在我们需要考虑所有可能的Investigator组合\n"
            for investigators in combinations(potential_investigators[1:], remaining_investigator_count):
                if perspective == "Investigator":
                    investigator_set = {"1"}.union(set(investigators))
                    #当我是Investigator时,我一定是Investigator
                    reasoning_process+=f"我是Investigator，我一定会占据组合里的一个位置\n"
                else:
                    investigator_set = set(investigators)
                    #当我是Rumormonger时,我一定不是Investigator
                    reasoning_process+=f"我是Rumormonger，我一定不会占据组合里的一个位置\n"
                is_consistent = True
                
                criminal_candidates = set(all_players) - investigator_set - {"1"}
                reasoning_process+=f"考虑investigators为{investigator_set}的情况\n"
                reasoning_process+=f"那么可能能做criminal的集合为{criminal_candidates}\n"
                reasoning_process+="我们回忆下investigators的statement\n"
                for round_data in statements:
                    for statement in round_data["statements"]:
                        player = statement["player"]
                        target = statement["target_player"]
                        claim = statement["statement_type"]
                        
                        if player in investigator_set:
                            if claim == "is":
                                if target not in criminal_candidates:
                                    is_consistent = False
                                    reasoning_process+=f"player{player}说过{target}是criminal\n"
                                    reasoning_process+=f"但{target}不在可能是criminal的人里,这是矛盾的\n"
                                    break
                                else:
                                    criminal_candidates = {target}
                                    reasoning_process+=f"player{player}说过{target}是criminal\n"
                                    reasoning_process+=f"所以可能是criminal的人只有{target}\n"
                            
                            elif claim == "is not":
                                if target in criminal_candidates:
                                    criminal_candidates.remove(target)
                                    reasoning_process+=f"player{player}说过{target}不是criminal\n"
                                    reasoning_process+=f"但{target}在可能是criminal的人里,所以我们从中去掉{target}\n"
                                    reasoning_process+=f"现在可能是criminal的人有{criminal_candidates}\n"
                                if not criminal_candidates:
                                    is_consistent = False
                                    reasoning_process+=f"此时可能是criminal的人为空,这是矛盾的\n"
                                    break
                    
                    if not is_consistent:
                        reasoning_process+=f"所以这种investigators的组合不可能成立,继续考虑其它组合\n"
                        break
                
                if is_consistent and len(criminal_candidates) == 1:
                    if perspective == "Investigator" and investigator_reasoning=='':
                        investigator_reasoning=reasoning_process+f"综上讨论，当前组合有可能成立,我是Investigator的可能性存在,此时有且仅有{criminal_candidates}可能是criminal\n"
                        investigator_possible=True
                    
                    elif perspective == "Rumormonger" and rumormonger_reasoning=='':
                        rumormonger_reasoning=reasoning_process+f"综上讨论，当前组合有可能成立,我是Rumormonger的可能性存在,此时有且仅有{criminal_candidates}可能是criminal\n"
                        rumormonger_possible=True
                    possible_criminals.update(criminal_candidates)
            if investigator_possible==False and perspective=="Investigator":
                investigator_reasoning=reasoning_process+"我已经讨论了所有的组合，我是Investigator的情况不可能成立,我一定是Rumormonger\n"
            
            if rumormonger_possible==False and perspective=="Rumormonger":
                rumormonger_reasoning=reasoning_process+"我已经讨论了所有的组合，我是Rumormonger的情况不可能成立,我一定是Investigator\n"
        # Determine the criminal from possible criminals
        criminal_player = list(possible_criminals)[0] if len(possible_criminals) == 1 else ""
        
        # Determine player1's role based on analysis
        player1_determined_role = ""
        if investigator_possible and rumormonger_possible:
            player1_determined_role = "Unknown"
        else:
            player1_determined_role = player1_role
            
        return_reasoning_process += investigator_reasoning + rumormonger_reasoning + f"""Final Judgment:
Final Criminal Is Player {criminal_player}.
My Role Is {player1_determined_role}."""

    # Handle Criminal and Lunatic perspectives
    if player1_role == "Criminal" or player1_role == "Lunatic":
        perspective_roles.append("Criminal")
        perspective_roles.append("Lunatic")

        return_reasoning_process+="我被告知是Criminal,但我也有可能是Lunatic,我需要讨论这两种情况的可能性\n"
        for perspective in perspective_roles:
            reasoning_process=""

            potential_investigators = []
            remaining_investigator_count = investigator_count
            reasoning_process+=f"考虑我是{perspective}的情况,那么还剩下{remaining_investigator_count}个Investigator\n"
            reasoning_process+="这种情况下，有哪些玩家可以做Investigator呢？\n"
            for player_id in all_players:
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
                                    reasoning_process+=f"player{player_id}说我是criminal,但我不可能是criminal,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                                if target == "1" and claim == "is not" and perspective == "Criminal":
                                    reasoning_process+=f"player{player_id}说我不是criminal,但我是criminal,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                                if claim == "is":
                                    criminal_accusations.add(target)
                                elif claim == "is not":
                                    not_criminal_claims.add(target)
                                
                                if any(p in not_criminal_claims for p in criminal_accusations):
                                    p=next(iter(criminal_accusations.intersection(not_criminal_claims)))
                                    reasoning_process+=f"player{player_id}说{p}是criminal但又说{p}不是,自相矛盾,,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                                
                                if len(criminal_accusations) > 1:
                                    reasoning_process+=f"player{player_id}说过不止一个人是criminal,排除player{player_id}\n"
                                    can_be_investigator = False
                                    break
                    
                        if not can_be_investigator:
                            break
                    
                    if can_be_investigator:
                        potential_investigators.append(player_id)
                        reasoning_process+=f"player{player_id}的发言没有矛盾，可以做Investigator\n"
                
            reasoning_process+=f"所以最后可能做Investigator的玩家有:{potential_investigators}\n"
            if len(potential_investigators) < remaining_investigator_count:
                reasoning_process+=f"可能做Investigator的人数不足{remaining_investigator_count}个,这种情况不可能成立\n"
                if perspective == "Criminal":
                    criminal_reasoning=reasoning_process+"我一定是Lunatic\n"
                else:
                    lunatic_reasoning=reasoning_process+"我一定是Criminal\n"
                continue
            reasoning_process+="现在我们需要考虑所有可能的Investigator组合\n"
            for investigators in combinations(potential_investigators, remaining_investigator_count):
                investigator_set = set(investigators)
                
                is_consistent = True
                reasoning_process+=f"考虑investigators为{investigator_set}的情况\n"
                if perspective == "Criminal":
                    criminal_candidates = {"1"}
                    reasoning_process+=f"我是Criminal,"
                elif perspective == "Lunatic":
                    criminal_candidates = set(all_players) - {"1"}
                    reasoning_process+=f"我是Lunatic,"
                reasoning_process+=f"那么可能能做criminal的集合为{criminal_candidates}\n"
                reasoning_process+="我们回忆下investigators的statement\n"
                for round_data in statements:
                    for statement in round_data["statements"]:
                        player = statement["player"]
                        target = statement["target_player"]
                        claim = statement["statement_type"]
                        
                        if player in investigator_set:
                            if claim == "is":
                                if target not in criminal_candidates:
                                    reasoning_process+=f"player{player}说过{target}是criminal\n"
                                    reasoning_process+=f"但{target}不在可能是criminal的人里,这是矛盾的\n"
                                    is_consistent = False
                                    break
                                else:
                                    reasoning_process+=f"player{player}说过{target}是criminal\n"
                                    reasoning_process+=f"所以可能是criminal的人只有{target}\n"
                                    criminal_candidates = {target}
                            
                            elif claim == "is not":
                                if target in criminal_candidates:
                                    reasoning_process+=f"player{player}说过{target}不是criminal\n"
                                    criminal_candidates.remove(target)
                                    reasoning_process+=f"但{target}在可能是criminal的人里,所以我们从中去掉{target}\n"
                                    reasoning_process+=f"现在可能是criminal的人有{criminal_candidates}\n"
                                    
                                if not criminal_candidates:
                                    reasoning_process+=f"此时可能是criminal的人为空,这是矛盾的\n"
                                    is_consistent = False
                                    break
                    
                    if not is_consistent:
                        reasoning_process+=f"所以这种investigators的组合不可能成立,继续考虑其它组合\n"
                        break
                    
                if is_consistent and len(criminal_candidates) == 1:
                    if perspective == "Criminal" and criminal_reasoning=='':
                        criminal_reasoning=reasoning_process+f"综上讨论，当前组合有可能成立,我是Criminal的可能性存在,此时有且仅有{criminal_candidates}可能是Criminal\n"
                        criminal_possible=True
                    
                    elif perspective == "Lunatic" and lunatic_reasoning=='':
                        lunatic_reasoning=reasoning_process+f"综上讨论，当前组合有可能成立,我是Lunatic的可能性存在,此时有且仅有{criminal_candidates}可能是criminal\n"
                        lunatic_possible=True
                    possible_criminals.update(criminal_candidates)
            if criminal_possible==False and perspective=="Criminal":
                criminal_reasoning=reasoning_process+"我已经讨论了所有的组合，我是Criminal的情况不可能成立,我一定是Lunatic\n"
            if lunatic_possible==False and perspective=="Lunatic":
                lunatic_reasoning=reasoning_process+"我已经讨论了所有的组合，我是Lunatic的情况不可能成立,我一定是Criminal\n"
        criminal_player = list(possible_criminals)[0] if len(possible_criminals) == 1 else ""
        
        # Determine player1's role based on analysis
        player1_determined_role = ""
        if criminal_possible and lunatic_possible:
            player1_determined_role = "Unknown"
        else:
            player1_determined_role = player1_role
            
        return_reasoning_process += criminal_reasoning + lunatic_reasoning + f"""Final Judgment:
Final Criminal Is Player {criminal_player}.
My Role Is {player1_determined_role}."""
        
    return {
        "unique_solution": len(possible_criminals) == 1,
        "possible_criminals": list(possible_criminals),
        "player1_role": player1_role,
        "reasoning_process": return_reasoning_process
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Blood Game dataset')
    parser.add_argument('--dataset_types', nargs='+', default=["original", "rumormonger", "lunatic", "all"],
                        help='Types of datasets to generate')
    parser.add_argument('--player_counts', type=int, nargs='+', default=[6, 10],
                        help='Number of players in each game')
    parser.add_argument('--n_scenarios_per_type', type=int, default=52,
                        help='Number of scenarios to generate for each dataset type')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for saving datasets')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_types = args.dataset_types
    player_counts = args.player_counts
    n_scenarios_per_type = args.n_scenarios_per_type
    
    # Use specified output directory or create default data directory
    if args.output_dir:
        data_dir = args.output_dir
    else:
        # Create data directory inside blood folder
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data will be saved to: {os.path.abspath(data_dir)}")

    for dataset_type in dataset_types:
        for player_count in player_counts:
            print(f"Generating dataset '{dataset_type}' with {player_count} players...")
            dataset = generate_dataset(player_count, dataset_type, n_scenarios_per_type)
            
            # Save to the data directory with explicit filename
            output_filename = os.path.join(data_dir, f"blood_{player_count}_{dataset_type}.json")
            
            # Ensure we're writing to the correct path
            print(f"Saving to: {os.path.abspath(output_filename)}")
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            
            print(f"Dataset '{dataset_type}' with {player_count} players saved successfully")

    print("All datasets generated successfully!")