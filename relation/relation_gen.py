import random
import json
from typing import List, Dict, Set, Tuple
import networkx as nx
from itertools import combinations
import os

class RelationGraph:
    def __init__(self, num_people: int):
        self.num_people = num_people
        self.people = [chr(65 + i) for i in range(num_people)]  # Use A, B, C... as names
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.people)
        self.relations = {}  # Store relation information
        
    def generate_relations(self):
        """Generate relations graph with transitivity property"""
        # First randomly decide how many good relation groups
        num_groups = random.randint(2, 4)
        
        # Randomly assign people to different groups
        remaining_people = self.people.copy()
        groups = []
        
        # Ensure each group has at least 2 people
        while len(remaining_people) >= 2 and len(groups) < num_groups:
            # Calculate the maximum possible size for the current group
            max_group_size = min(len(remaining_people), 5)  # Limit max group size to 5
            group_size = random.randint(2, max_group_size)
            
            # If remaining people are not enough, adjust group size
            if group_size > len(remaining_people):
                group_size = len(remaining_people)
            
            # Ensure at least 2 people
            if group_size >= 2:
                group = random.sample(remaining_people, group_size)
                groups.append(group)
                remaining_people = [p for p in remaining_people if p not in group]
            else:
                break
        
        # If there are remaining people, assign them to existing groups
        if remaining_people:
            for person in remaining_people:
                # Randomly choose a group to join
                if groups:
                    target_group = random.choice(groups)
                    target_group.append(person)
        
        # Generate "good" relations within each group (complete graph)
        for group in groups:
            # For transitivity, create a complete graph within each group
            for person1, person2 in combinations(group, 2):
                self.graph.add_edge(person1, person2, relation='good')
                self.relations[(person1, person2)] = 'good'
                self.relations[(person2, person1)] = 'good'
        
        # Add "bad" relations between different groups
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Randomly select one person from each group, add a "bad" relation edge
                person1 = random.choice(groups[i])
                person2 = random.choice(groups[j])
                self.graph.add_edge(person1, person2, relation='bad')
                self.relations[(person1, person2)] = 'bad'
                self.relations[(person2, person1)] = 'bad'
    
    def get_relation(self, person1: str, person2: str) -> str:
        """Get relation between two people"""
        if person1 == person2:
            return 'good'
        return self.relations.get((person1, person2), 'bad')
    
    def get_good_friends(self, person: str) -> List[str]:
        """Get list of people with good relations to a person"""
        friends = [p for p in self.people if p != person and self.get_relation(person, p) == 'good']
        return sorted(friends)  # Return sorted list for consistent output
    
    def get_groups(self) -> List[List[str]]:
        """Get all good relation groups (connected components)"""
        # Create a subgraph with only good relations
        good_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['relation'] == 'good']
        good_graph = nx.Graph()
        good_graph.add_nodes_from(self.people)
        good_graph.add_edges_from(good_edges)
        
        # Find connected components
        components = list(nx.connected_components(good_graph))
        return [list(component) for component in components]
    
    def count_relations(self) -> Tuple[int, int]:
        """Count the number of good and bad relation pairs"""
        good_count = 0
        bad_count = 0
        for person1, person2 in combinations(self.people, 2):
            if self.get_relation(person1, person2) == 'good':
                good_count += 1
            else:
                bad_count += 1
        return good_count, bad_count
    
    def generate_spanning_trees(self):
        """Generate a spanning tree for each good relation group"""
        groups = self.get_groups()
        trees = []
        
        for group in groups:
            if len(group) > 1:
                # Create a subgraph for this group
                subgraph = self.graph.subgraph(group)
                # Generate a spanning tree
                tree = nx.minimum_spanning_tree(subgraph)
                trees.append(tree)
                
        return trees

def generate_relation_data(num_people: int = None, difficulty: str = None) -> Dict:
    """Generate relation data"""
    if num_people is None:
        if difficulty == "easy":
            num_people = random.randint(8, 10)
        elif difficulty == "hard":
            num_people = random.randint(14, 16)
    
    # Generate relation graph
    graph = RelationGraph(num_people)
    graph.generate_relations()
    
    # Generate natural language descriptions
    descriptions = []
    
    # Generate spanning trees for each group to ensure we can infer the entire graph
    trees = graph.generate_spanning_trees()
    
    # For each tree, add good relation descriptions
    for tree in trees:
        for u, v in tree.edges():
            descriptions.append(f"{u} and {v} have a good relationship")
    
    # Add some bad relation edges
    bad_edges = [(u, v) for u, v, d in graph.graph.edges(data=True) if d['relation'] == 'bad']
    
    # Make sure we include at least one bad relation between each pair of groups
    groups = graph.get_groups()
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            added = False
            for person1 in groups[i]:
                if added:
                    break
                for person2 in groups[j]:
                    if graph.get_relation(person1, person2) == 'bad':
                        descriptions.append(f"{person1} and {person2} have a bad relationship")
                        added = True
                        break
    
    # Add some more random bad relations
    remaining_bad_edges = [edge for edge in bad_edges if f"{edge[0]} and {edge[1]} have a bad relationship" not in descriptions]
    for edge in random.sample(remaining_bad_edges, min(2, len(remaining_bad_edges))):
        descriptions.append(f"{edge[0]} and {edge[1]} have a bad relationship")
    
    # Randomly shuffle the descriptions
    random.shuffle(descriptions)
    descriptions_text = "\n".join(descriptions)
    
    # Generate questions for each type
    questions = []
    
    # 1. Relation reasoning question
    person1, person2 = random.sample(graph.people, 2)
    reasoning_q = f"Do {person1} and {person2} have a good relationship?"
    reasoning_a = "Yes" if graph.get_relation(person1, person2) == 'good' else "No"
    
    # 2. Relation group classification question
    person = random.choice(graph.people)
    good_friends = graph.get_good_friends(person)
    group_q = f"Who has a good relationship with {person}?"
    group_a = ", ".join(good_friends) if good_friends else "No one"
    
    # 3. Cluster judgment question
    groups = graph.get_groups()
    cluster_q = "How many groups of people are there?"
    cluster_a = str(len(groups))
    
    # 4. Relation count analysis question
    good_count, bad_count = graph.count_relations()
    count_q = "How many pairs of people have good/bad relationships?"
    count_a = f"{good_count} pairs have good relationships, {bad_count} pairs have bad relationships"
    
    return {
        "descriptions": descriptions_text,
        "reasoning_question": reasoning_q,
        "reasoning_answer": reasoning_a,
        "group_question": group_q,
        "group_answer": group_a,
        "cluster_question": cluster_q,
        "cluster_answer": cluster_a,
        "count_question": count_q,
        "count_answer": count_a,
        "num_people": num_people
    }

def load_template(template_name):
    """Load a template file and return its content"""
    with open(template_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_template(template, data):
    """Apply data to a template by replacing placeholders"""
    result = {}
    for key, value in template.items():
        if isinstance(value, str):
            # Replace placeholders in the string
            result[key] = value.replace('{{descriptions}}', data['descriptions'])
            result[key] = result[key].replace('{{question}}', data['question'])
            
            # Add people count information
            if 'num_people' in data:
                num_people = data['num_people']
                result[key] = result[key].replace('{{num_people}}', str(num_people))
                # Calculate the last person label (A + num_people - 1)
                last_person = chr(ord('A') + num_people - 1)
                result[key] = result[key].replace('{{last_person}}', last_person)
            
            if key == 'answer':
                result[key] = data['answer']
        else:
            result[key] = value
    return result

def create_template_files():
    """Create the template JSON files if they don't exist"""
    os.makedirs('relation', exist_ok=True)
    
    # Create a consistent system prompt for all templates
    common_system_prompt = """You are analyzing relationships between people. In this context:

1. Relationships are either 'good' or 'bad'.
2. Relationships have transitive properties:
   - If A has a good relationship with B, and B has a good relationship with C, then A has a good relationship with C.
   - If A has a bad relationship with B, and A has a good relationship with C, then B and C must have a bad relationship.
3. A 'group' is defined as a set of people where every person has a good relationship with every other person in the set.
4. People are in the same group if and only if they have good relationships with each other (directly or through transitivity).
5. Groups are completely separate - if someone from one group has a bad relationship with someone from another group, then everyone from the first group has a bad relationship with everyone from the second group.

Base your analysis strictly on the information provided and these rules."""
    
    # Template for relation_reasoning.json
    reasoning_template = {
        "system_prompt": common_system_prompt,
        "user_prompt": "There are {{num_people}} people in total, labeled from A to {{last_person}}.\n\nBased on the relationship information below, answer the following question with 'Yes' or 'No' only.\n\n{{descriptions}}\n\nQuestion: {{question}}\n\nProvide your answer in the format: Final Answer: Yes/No",
        "answer": "Final Answer: {{answer}}"
    }
    
    # Template for relation_group.json
    group_template = {
        "system_prompt": common_system_prompt,
        "user_prompt": "There are {{num_people}} people in total, labeled from A to {{last_person}}.\n\nBased on the relationship information below, list all people who have a good relationship with the person in the question. List names in alphabetical order, separated by commas. If there are no people with a good relationship to the specified person, answer 'No one'.\n\n{{descriptions}}\n\nQuestion: {{question}}\n\nProvide your answer in the format: Final Answer: list of people or 'No one'",
        "answer": "Final Answer: {{answer}}"
    }
    
    # Template for relation_cluster.json
    cluster_template = {
        "system_prompt": common_system_prompt,
        "user_prompt": "There are {{num_people}} people in total, labeled from A to {{last_person}}.\n\nBased on the relationship information below, determine how many distinct groups of people exist. A group is a set of people where everyone has a good relationship with everyone else in the group. Answer with just a number.\n\n{{descriptions}}\n\nQuestion: {{question}}\n\nProvide your answer in the format: Final Answer: number",
        "answer": "Final Answer: {{answer}}"
    }
    
    # Template for relation_count.json
    count_template = {
        "system_prompt": common_system_prompt,
        "user_prompt": "There are {{num_people}} people in total, labeled from A to {{last_person}}.\n\nBased on the relationship information below, count the total number of pairs of people who have good relationships and the total number of pairs who have bad relationships. Answer in the specified format.\n\n{{descriptions}}\n\nQuestion: {{question}}\n\nProvide your answer in the format: Final Answer: X pairs have good relationships, Y pairs have bad relationships",
        "answer": "Final Answer: {{answer}}"
    }
    
    # Save templates to files
    templates = {
        'relation/relation_reasoning.json': reasoning_template,
        'relation/relation_group.json': group_template,
        'relation/relation_cluster.json': cluster_template,
        'relation/relation_count.json': count_template
    }
    
    for file_path, template in templates.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
    
    print("Template files created successfully")

def generate_dataset_files():
    """Generate dataset files for each question type"""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/easy', exist_ok=True)
    os.makedirs('data/hard', exist_ok=True)
    
    # Make sure template files exist
    create_template_files()
    
    # Load templates
    reasoning_template = load_template('relation/relation_reasoning.json')
    group_template = load_template('relation/relation_group.json')
    cluster_template = load_template('relation/relation_cluster.json')
    count_template = load_template('relation/relation_count.json')
    
    # Generate data samples
    num_samples = 100
    
    # Generate data for different difficulty levels
    for difficulty in ["easy", "hard"]:
        print(f"Generating {difficulty} difficulty samples...")
        
        # Create separate data files for each question type
        reasoning_data = []
        group_data = []
        cluster_data = []
        count_data = []
        
        for i in range(num_samples):
            print(f"Generating {difficulty} sample {i+1}/{num_samples}")
            sample = generate_relation_data(difficulty=difficulty)
            
            # 1. Relation reasoning question
            reasoning_data.append(apply_template(reasoning_template, {
                "descriptions": sample["descriptions"],
                "question": sample["reasoning_question"],
                "answer": sample["reasoning_answer"],
                "num_people": sample["num_people"]
            }))
            
            # 2. Relation group classification
            group_data.append(apply_template(group_template, {
                "descriptions": sample["descriptions"],
                "question": sample["group_question"],
                "answer": sample["group_answer"],
                "num_people": sample["num_people"]
            }))
            
            # 3. Cluster judgment
            cluster_data.append(apply_template(cluster_template, {
                "descriptions": sample["descriptions"],
                "question": sample["cluster_question"],
                "answer": sample["cluster_answer"],
                "num_people": sample["num_people"]
            }))
            
            # 4. Relation count analysis
            count_data.append(apply_template(count_template, {
                "descriptions": sample["descriptions"],
                "question": sample["count_question"],
                "answer": sample["count_answer"],
                "num_people": sample["num_people"]
            }))
        
        # Save each dataset to a file in the appropriate difficulty folder
        with open(f'data/{difficulty}/relation_reasoning.json', 'w', encoding='utf-8') as f:
            json.dump(reasoning_data, f, ensure_ascii=False, indent=2)
        
        with open(f'data/{difficulty}/relation_group.json', 'w', encoding='utf-8') as f:
            json.dump(group_data, f, ensure_ascii=False, indent=2)
        
        with open(f'data/{difficulty}/relation_cluster.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, ensure_ascii=False, indent=2)
        
        with open(f'data/{difficulty}/relation_count.json', 'w', encoding='utf-8') as f:
            json.dump(count_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully generated {num_samples} samples for each question type and difficulty level")

def main():
    # Generate the dataset files
    generate_dataset_files()

if __name__ == "__main__":
    main()
