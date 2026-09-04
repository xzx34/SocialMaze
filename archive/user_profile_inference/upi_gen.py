import random
import json
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
import sys
import concurrent.futures
sys.path.append("..") # Add parent directory to path
from utils.tool import get_chat_response
from collections import defaultdict

# Simplified user persona definition
def create_user_personas(num_personas: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Create a simplified list of user personas with only age group and gender.
    Ensures there is a clear primary user group.
    """
    age_groups = ["18-34",'35-54','55+']
    
    # Randomly determine which group will be the primary user group
    primary_group_index = random.randint(0, 1)
    primary_age_group = age_groups[primary_group_index]
    
    # Create weighted distribution with primary group having clear majority (80-90%)
    primary_weight = random.uniform(0.8, 0.9)  # Increased from 0.7-0.8 to 0.8-0.9
    secondary_weight = 1.0 - primary_weight
    
    weights = [secondary_weight] * len(age_groups)  # Initialize with secondary weight for all groups
    weights[primary_group_index] = primary_weight   # Set primary group weight
    
    # Randomly select a primary gender
    genders = ["Male", "Female"]
    primary_gender = random.choice(genders)
    genders = ["Male", "Female", "Non-binary"]
    # Create weighted gender distribution with primary gender having clear majority
    gender_weights = [0.05] * len(genders)  # Reduced from 0.1 to 0.05
    primary_gender_index = genders.index(primary_gender)
    gender_weights[primary_gender_index] = 0.8  # Increased from 0.7 to 0.8
    
    # Distribute remaining 20% among other genders
    remaining_gender_weight = 0.2  # Reduced from 0.3 to 0.2
    for i in range(len(gender_weights)):
        if i != primary_gender_index:
            gender_weights[i] = remaining_gender_weight / (len(gender_weights) - 1)
    
    print(f"Primary User Group: {primary_age_group} {primary_gender}s")
    
    # Available models
    model_options = ['gpt-4o-mini', 'llama-3.3-70B', 'gemma-3-27B', 'qwen-2.5-72B']
    
    # Generate personas based on the weights, but enforce minimum threshold
    personas = []
    
    # Function to check if the distribution is satisfactory
    def is_distribution_valid(personas_list: List[Dict[str, Any]]) -> bool:
        if len(personas_list) < 3:  # Not enough data to make a judgment
            return True
            
        age_count = {}
        gender_count = {}
        for p in personas_list:
            age_count[p["age_group"]] = age_count.get(p["age_group"], 0) + 1
            gender_count[p["gender"]] = gender_count.get(p["gender"], 0) + 1
            
        # Check if primary age group has at least 70% representation
        primary_age_percentage = age_count.get(primary_age_group, 0) / len(personas_list)
        if primary_age_percentage < 0.7:
            return False
            
        # Check if primary gender has at least 60% representation
        primary_gender_percentage = gender_count.get(primary_gender, 0) / len(personas_list)
        if primary_gender_percentage < 0.6:
            return False
            
        return True
    
    # Keep generating personas until we get a valid distribution
    max_attempts = 5
    for attempt in range(max_attempts):
        # Reset personas
        personas = []
        
        for i in range(num_personas):
            # Sample age group based on distribution
            age_group = random.choices(
                age_groups,
                weights=weights,
                k=1
            )[0]
            
            # Sample gender based on distribution
            gender = random.choices(
                genders,
                weights=gender_weights,
                k=1
            )[0]
            
            persona = {
                "id": i + 1,
                "age_group": age_group,
                "gender": gender,
                "model": random.choice(model_options)
            }
            
            personas.append(persona)
        
        # Check if the distribution is valid
        if is_distribution_valid(personas):
            break
        else:
            print(f"Attempt {attempt+1}: Generated distribution doesn't have a clear majority. Retrying...")
    
    # If we couldn't generate a valid distribution, force the primary groups to be majority
    if not is_distribution_valid(personas):
        print("Forcing clear majority distribution...")
        # Calculate how many personas should be primary
        primary_age_count = int(num_personas * 0.75)
        primary_gender_count = int(num_personas * 0.7)
        
        # Reset personas
        personas = []
        
        for i in range(num_personas):
            if i < primary_age_count:
                age_group = primary_age_group
            else:
                age_group = age_groups[1 - primary_group_index]  # The other age group
                
            if i < primary_gender_count:
                gender = primary_gender
            else:
                # Randomly select from the other genders
                other_genders = [g for g in genders if g != primary_gender]
                gender = random.choice(other_genders)
            
            persona = {
                "id": i + 1,
                "age_group": age_group,
                "gender": gender,
                "model": random.choice(model_options)
            }
            
            personas.append(persona)
    
    return personas, {"primary_age_group": primary_age_group, "primary_gender": primary_gender}

def generate_comment_prompt(persona: Dict[str, Any], content_type: str, content_subject: str) -> str:
    """
    Generate a prompt for the LLM that emphasizes persona characteristics.
    
    Args:
        persona: The user persona dictionary
        content_type: Type of content
        content_subject: Subject of the content to comment on
        
    Returns:
        A string prompt for the LLM
    """
    
    age_group = persona['age_group']
    gender = persona['gender']
    
    
    system_prompt = f"""
    You are a {age_group} {gender} user providing a comment about this {content_type}: "{content_subject}"
    
    Write a realistic, authentic comment/review about the {content_subject}. Your comment should:
    1. Be 2-4 sentences long
    2. Sound like a genuine user, not a marketing expert or professional reviewer
    3. Potentially include common writing patterns of your age group
    
    Write only the comment text with no additional explanations or framing.
    """
    
    return system_prompt

def generate_comment_for_persona(persona, content_type, subject, temperature=0.7):
    """Helper function to generate comments for a single persona, used for concurrent processing"""
    try:
        # Generate system message
        system_message = generate_comment_prompt(persona, content_type, subject)
        
        # Create message list
        messages = [
            {"role": "user", "content": f"Please comment on this {content_type}: {subject}"}
        ]
        
        # Get comment text
        comment_text = get_chat_response(
            model=persona["model"],
            system_message=system_message,
            messages=messages,
            temperature=temperature
        ).strip()
        
        print(f"Generated comment for persona {persona['id']} on '{subject}'")
        return {
            "persona": persona,
            "comment": comment_text
        }
    except Exception as e:
        print(f"Error generating comment for persona {persona['id']} on '{subject}': {e}")
        return None

def generate_dataset(
    num_personas: int,
    content_type: str,
    content_subjects: List[str],
    temperature: float = 0.7,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Generate dataset, each scenario with its own user personas.
    Collect comments grouped by demographic features for subsequent user profile dataset.
    
    Args:
        num_personas: Number of user personas to generate per scenario
        content_type: Content type
        content_subjects: List of content topics
        temperature: LLM temperature parameter
        max_workers: Maximum number of concurrent worker threads
        
    Returns:
        Dictionary containing scenarios and comments grouped by demographic features
    """
    scenarios_data = {}
    # Store all comments by six major demographic groups
    # Keys are (age_group, gender), values are lists of comments for that demographic group
    all_comments_by_demographic = defaultdict(list)
    
    for subject in content_subjects:
        print(f"Generating comments for {content_type}: {subject}")
        
        # Generate a new set of user personas for each scenario
        personas, primary_user_group = create_user_personas(num_personas)
        print(f"Created new user distribution for scenario: {subject}")
        analyze_demographics(personas)
        
        scenario_comments = []
        
        # Use thread pool for concurrent comment generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create task list
            future_to_persona = {
                executor.submit(generate_comment_for_persona, persona, content_type, subject, temperature): persona
                for persona in personas
            }
            
            # Process results
            for future in concurrent.futures.as_completed(future_to_persona):
                persona = future_to_persona[future]
                try:
                    comment_data = future.result()
                    if comment_data:
                        scenario_comments.append(comment_data)
                        
                        # Store comments grouped by demographic features (age group + gender) for the second dataset
                        demographic_key = (persona["age_group"], persona["gender"])
                        all_comments_by_demographic[demographic_key].append({
                            "comment": comment_data["comment"],
                            "subject": subject
                        })
                except Exception as e:
                    print(f"Exception occurred while processing result for persona {persona['id']}: {e}")
        
        # Store scenario data, including comments and primary user group information
        scenarios_data[subject] = {
            "primary_user_group": primary_user_group,
            "comments": scenario_comments
        }
    
    return scenarios_data, all_comments_by_demographic

def save_product_dataset(dataset: Dict[str, Any], output_dir: str, content_type: str):
    """
    Save product review dataset as JSON file, using numbers instead of product names as scenario identifiers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reorganize data structure, using scenario ID instead of product name as key
    reorganized_scenarios = {}
    for idx, (product_name, scenario_data) in enumerate(dataset.items(), 1):
        scenario_key = f"scenario_{idx}"
        reorganized_scenarios[scenario_key] = {
            "product_name": product_name,  # Move product name inside
            "primary_user_group": scenario_data["primary_user_group"],
            "comments": scenario_data["comments"]
        }
    
    final_dataset = {
        "content_type": content_type,
        "scenarios": reorganized_scenarios
    }
    
    output_file = os.path.join(output_dir, "user_entity.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Product review dataset saved to {output_file}")

def generate_user_profile_dataset(all_comments_by_demographic, n_scenarios, profiling_users, output_dir="data"):
    """
    Generate user profile dataset.
    
    For each scenario (n_scenarios total):
    1. Randomly select a demographic group
    2. Randomly sample profiling_users comments from that group
    3. Compose these comments into a data sample
    
    Args:
        all_comments_by_demographic: Dictionary of comments grouped by demographic features
        n_scenarios: Number of scenarios to create
        profiling_users: Number of comments to include per scenario
        output_dir: Output directory
    """
    # Filter demographic groups that have enough comments
    valid_demographics = {k: v for k, v in all_comments_by_demographic.items() if len(v) >= profiling_users}
    
    if not valid_demographics:
        print("Error: No demographic groups have enough comments to create user profiles")
        return
    
    # Get list of available demographic groups
    available_demographics = list(valid_demographics.keys())
    
    # If there are fewer available demographic groups than requested scenarios, issue a warning
    if len(available_demographics) < n_scenarios:
        print(f"Warning: Requested to create {n_scenarios} scenarios, but only {len(available_demographics)} different demographic groups are available.")
        print("Some demographic groups will be used multiple times.")
    
    profile_groups = []
    
    # Create a data sample for each scenario
    for i in range(n_scenarios):
        # Randomly select a demographic group (can select the same group multiple times)
        selected_demographic = random.choice(available_demographics)
        age_group, gender = selected_demographic
        
        # Ensure the group has enough comments
        available_comments = valid_demographics[selected_demographic]
        if len(available_comments) < profiling_users:
            print(f"Warning: Demographic group {age_group} {gender} only has {len(available_comments)} comments, but {profiling_users} are needed.")
            # If there aren't enough comments, use all available comments
            selected_comments = available_comments
        else:
            # Otherwise, randomly select the specified number of comments
            selected_comments = random.sample(available_comments, profiling_users)
        
        # Create a scenario
        profile_group = {
            "group_id": i + 1,
            "demographics": {
                "age_group": age_group,
                "gender": gender
            },
            "comments": [
                {
                    "product": comment["subject"],
                    "comment": comment["comment"]
                } for comment in selected_comments
            ]
        }
        
        profile_groups.append(profile_group)
    
    # Save the dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "user_persona.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "description": "Dataset for inferring user demographic features from user comments",
            "profile_groups": profile_groups
        }, f, indent=2, ensure_ascii=False)
    
    print(f"User profile dataset saved to {output_file}")
    print(f"Created {len(profile_groups)} scenarios, each representing a sample of a demographic group")
    
    # Track usage of each demographic group
    demographic_usage = {}
    for group in profile_groups:
        key = (group["demographics"]["age_group"], group["demographics"]["gender"])
        demographic_usage[key] = demographic_usage.get(key, 0) + 1
    
    print("\nDemographic group usage:")
    for demo, count in demographic_usage.items():
        age_group, gender = demo
        print(f"  {age_group} {gender}: Selected {count} times")

def analyze_demographics(personas: List[Dict[str, Any]]):
    """
    Analyze and print demographics of the generated personas.
    
    Args:
        personas: List of user personas
    """
    total = len(personas)
    
    # Count by age group
    age_groups = {}
    for p in personas:
        age_group = p["age_group"]
        age_groups[age_group] = age_groups.get(age_group, 0) + 1
    
    # Count by gender
    genders = {}
    for p in personas:
        gender = p["gender"]
        genders[gender] = genders.get(gender, 0) + 1
    
    print("\nUser Demographics Analysis:")
    print("==========================")
    print("Age Groups:")
    for group, count in age_groups.items():
        print(f"  {group}: {count} users ({count/total*100:.1f}%)")
    
    print("\nGender Distribution:")
    for gender, count in genders.items():
        print(f"  {gender}: {count} users ({count/total*100:.1f}%)")
    
    # Find the primary user group (highest percentage)
    primary_age = max(age_groups, key=age_groups.get)
    primary_gender = max(genders, key=genders.get)
    
    print(f"\nPrimary User Group: {primary_age} {primary_gender}s")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate simulated user comments dataset")
    parser.add_argument("--num_personas", type=int, default=10, help="Number of user personas to generate per scenario")
    parser.add_argument("--content_type", type=str, default="product", help="Type of content (product, video, news, etc.)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output data")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for LLM responses")
    parser.add_argument("--n_scenarios", type=int, default=10, help="Number of scenarios/objects to include in the dataset")
    parser.add_argument("--profiling_users", type=int, default=4, help="Number of comments per group in profiling dataset")
    args = parser.parse_args()
    
    # Expanded content subjects
    all_content_subjects = [
        # Electronics
        "iPhone 15 Pro Max", "Samsung Galaxy S24", "Google Pixel 8 Pro", "OnePlus 12",
        "iPad Pro 2023", "Samsung Galaxy Tab S9", "Amazon Fire HD 10", "Lenovo Tab P12 Pro",
        "MacBook Air M2", "Dell XPS 13", "HP Spectre x360", "Lenovo ThinkPad X1 Carbon",
        "Sony WH-1000XM5 Headphones", "Bose QuietComfort Earbuds", "Apple AirPods Pro 2", "Sennheiser Momentum 4",
        "Nintendo Switch OLED", "PlayStation 5", "Xbox Series X", "Steam Deck",
        "Kindle Paperwhite", "Kobo Libra 2", "reMarkable 2 Tablet", "Onyx Boox Note Air 2",
        
        # Home Appliances
        "Dyson V12 Vacuum", "Roomba j7+", "Shark Navigator Lift-Away", "Miele Complete C3",
        "Instant Pot Duo Crisp", "Ninja Foodi", "KitchenAid Stand Mixer", "Vitamix E310 Blender",
        "Samsung Family Hub Refrigerator", "LG InstaView Refrigerator", "Whirlpool Top-Freezer", "GE Profile Smart Refrigerator",
        "IKEA HEMNES Dresser", "West Elm Mid-Century Bed", "La-Z-Boy Recliner", "Herman Miller Aeron Chair",
        
        # Transportation
        "Tesla Model Y", "Toyota RAV4 Hybrid", "Honda Civic", "Ford F-150 Lightning",
        "VanMoof S5 E-Bike", "Rad Power RadRunner", "Brompton Folding Bike", "Trek Domane SL6",
        
        # Clothing & Accessories
        "Nike Air Zoom Pegasus 39", "Adidas Ultraboost 23", "Hoka Clifton 9", "New Balance 990v6",
        "Lululemon Align Leggings", "Nike Dri-FIT Running Shorts", "Patagonia Better Sweater", "The North Face Thermoball Jacket",
        "Apple Watch Series 9", "Garmin Fenix 7", "Fitbit Charge 6", "Samsung Galaxy Watch 6",
        
        # Software & Services
        "ChatGPT Plus Subscription", "Microsoft 365 Family Plan", "Adobe Creative Cloud", "Notion Premium",
        "Netflix Premium Plan", "Disney+ Bundle", "Spotify Premium", "YouTube Premium",
        "Amazon Prime Membership", "Costco Gold Star Membership", "Sam's Club Plus", "Walmart+ Membership",
        
        # Entertainment
        "The Last of Us TV Show", "Succession", "House of the Dragon", "Ted Lasso",
        "Oppenheimer Movie", "Barbie Movie", "Dune: Part Two", "Poor Things",
        "Taylor Swift's 'The Tortured Poets Department'", "Beyoncé's 'Cowboy Carter'", "Billie Eilish's 'Hit Me Hard and Soft'", 
        "Call of Duty: Modern Warfare 3", "Baldur's Gate 3", "The Legend of Zelda: Tears of the Kingdom", "Starfield",
        
        # Food & Drinks
        "Starbucks Pumpkin Spice Latte", "McDonald's McSpicy", "Chipotle Burrito Bowl", "Shake Shack ShackBurger",
        "Coca-Cola Zero Sugar", "LaCroix Sparkling Water", "Liquid Death Mountain Water", "Athletic Brewing Non-Alcoholic Beer",
        
        # Health & Beauty
        "Dyson Airwrap", "Theragun Elite", "Olaplex Hair Treatment", "Cerave Moisturizing Cream",
        "Peloton Bike+", "Hydrow Rowing Machine", "WHOOP 4.0 Fitness Tracker", "Theragun Mini",
        
        # Travel
        "Airbnb Plus Stays", "Marriott Bonvoy Program", "TSA PreCheck Membership", "Away The Carry-On Suitcase",
        
        # Books & Literature
        "Atomic Habits by James Clear", "Fourth Wing by Rebecca Yarros", "The Housemaid by Freida McFadden", "Iron Flame by Rebecca Yarros"

        # Game
        "The Witcher 3: Wild Hunt", "Elden Ring", "Horizon Zero Dawn", "The Last of Us","Genshin Impact", "Honkai: Star Rail", "White Album 2", "Steins;Gate"
    ]
    
    # Select n_scenarios content subjects
    if args.n_scenarios <= len(all_content_subjects):
        content_subjects = random.sample(all_content_subjects, args.n_scenarios)
    else:
        print(f"Warning: Requested {args.n_scenarios} scenarios but only {len(all_content_subjects)} are available.")
        content_subjects = all_content_subjects
    
    # Generate primary dataset
    print(f"\nGenerating comments for {len(content_subjects)} {args.content_type} items...")
    dataset, all_comments_by_demographic = generate_dataset(
        num_personas=args.num_personas,
        content_type=args.content_type,
        content_subjects=content_subjects,
        temperature=args.temperature
    )
    
    # Save primary dataset
    save_product_dataset(dataset, args.output_dir, args.content_type)
    
    # Generate and save user profiling dataset
    print("\nGenerating user profiling dataset...")
    generate_user_profile_dataset(
        all_comments_by_demographic=all_comments_by_demographic,
        n_scenarios=args.n_scenarios,
        profiling_users=args.profiling_users,
        output_dir=args.output_dir
    )
    
    # Print statistics
    total_scenarios = len(content_subjects)
    total_comments = total_scenarios * args.num_personas
    print(f"\nGenerated {total_comments} comments across {total_scenarios} {args.content_type} scenarios")
    print(f"Data saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
