import json
import random
import os
import time
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
import sys
import concurrent.futures
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import get_chat_response

load_dotenv()

# Product categories and example products
PRODUCT_CATEGORIES = [
    {
        "category": "Electronics",
        "products": [
            "Wireless Earbuds", "Smartphone", "Laptop", "Smart Watch", "Bluetooth Speaker",
            "Tablet", "Digital Camera", "Fitness Tracker", "Wireless Charger", "Power Bank"
        ]
    },
    {
        "category": "Clothing",
        "products": [
            "Jeans", "T-shirt", "Dress", "Running Shoes", "Winter Jacket",
            "Sweater", "Yoga Pants", "Formal Shirt", "Sunglasses", "Backpack"
        ]
    },
    {
        "category": "Home & Kitchen",
        "products": [
            "Coffee Maker", "Blender", "Air Fryer", "Robot Vacuum", "Bed Sheets",
            "Cookware Set", "Toaster", "Microwave Oven", "Kitchen Knife Set", "Food Storage Containers"
        ]
    },
    {
        "category": "Beauty & Personal Care",
        "products": [
            "Face Moisturizer", "Shampoo", "Electric Toothbrush", "Perfume", "Hair Dryer",
            "Face Wash", "Makeup Set", "Beard Trimmer", "Sunscreen", "Facial Serum"
        ]
    },
    {
        "category": "Books",
        "products": [
            "Fiction Novel", "Self-Help Book", "Cookbook", "Biography", "History Book",
            "Children's Book", "Science Fiction", "Business Book", "Travel Guide", "Mystery Novel"
        ]
    }
]

# Reviewer roles without bias values
REVIEWER_ROLES = {
    "1": {
        "type": "genuine_customer",
        "persona": "A straightforward, honest customer who provides balanced feedback based on the true product quality"
    },
    "2": {
        "type": "genuine_customer",
        "persona": "A detail-oriented customer who focuses on product functionality while being honest about the true quality"
    },
    "3": {
        "type": "genuine_customer",
        "persona": "A practical customer who evaluates products based on value for money in relation to the true quality"
    },
    "4": {
        "type": "professional_positive",
        "persona": "A professional reviewer who knows the true quality but tends to emphasize positive aspects and downplay negatives"
    },
    "5": {
        "type": "malicious_negative",
        "persona": "A critic who knows the true quality but tends to focus on flaws and negative aspects while minimizing positives"
    }
}

def generate_product_scenario(scenario_id):
    """Generate a single product review scenario.
    
    Args:
        scenario_id: Unique identifier for the scenario
    
    Returns:
        Dict containing the scenario data
    """
    # Choose a random category and product
    category_data = random.choice(PRODUCT_CATEGORIES)
    category = category_data["category"]
    product = random.choice(category_data["products"])
    
    # Generate a true quality rating (1-5)
    true_rating = random.randint(1, 5)
    
    # Create a price appropriate to the category (just for more realistic prompts)
    base_prices = {
        "Electronics": (50, 1000),
        "Clothing": (20, 200),
        "Home & Kitchen": (30, 300),
        "Beauty & Personal Care": (10, 100),
        "Books": (10, 50)
    }
    min_price, max_price = base_prices.get(category, (20, 200))
    price = round(random.uniform(min_price, max_price), 2)
    
    # Create a brand name (just for more realistic prompts)
    brands = [
        "TechWave", "NatureEssence", "UrbanStyle", "HomeComfort", "LuxeLife",
        "EcoFriendly", "PrimePick", "QualityPlus", "ValueChoice", "PremiumSelect"
    ]
    brand = random.choice(brands)
    
    # Generate a product description
    description = f"{brand} {product} - A quality {category.lower()} product"
    
    # Assign ratings to each reviewer based on their role
    reviewer_ratings = {}
    for reviewer_id, role_info in REVIEWER_ROLES.items():
        reviewer_type = role_info["type"]
        
        if reviewer_type == "genuine_customer":
            # Genuine customers give ratings close to the true rating
            final_rating = max(1, min(5, true_rating + random.uniform(-0.5, 0.5)))
        elif reviewer_type == "professional_positive":
            # Professional positive reviewers artificially inflate ratings
            if true_rating < 5:
                # Inflate low to medium ratings more significantly
                inflation = max(1, (5 - true_rating) / 2)
                final_rating = min(5, true_rating + inflation + random.uniform(0, 0.5))
            else:
                # Already 5 stars, can't go higher
                final_rating = 5.0
        elif reviewer_type == "malicious_negative":
            # Malicious negative reviewers artificially deflate ratings
            if true_rating > 1:
                # Deflate medium to high ratings more significantly
                deflation = max(1, true_rating / 2)
                final_rating = max(1, true_rating - deflation - random.uniform(0, 0.5))
            else:
                # Already 1 star, can't go lower
                final_rating = 1.0
        
        reviewer_ratings[reviewer_id] = round(final_rating)
    
    # Create prompts for each reviewer
    prompts = {}
    for reviewer_id, role_info in REVIEWER_ROLES.items():
        rating = reviewer_ratings[reviewer_id]
        prompts[reviewer_id] = generate_reviewer_prompt(
            reviewer_id, 
            role_info["persona"],
            role_info["type"],
            product,
            category,
            brand,
            price,
            description,
            rating,
            true_rating
        )
    
    # Data structure to store the scenario
    scenario = {
        "scenario_id": scenario_id,
        "product": product,
        "category": category,
        "brand": brand,
        "price": price,
        "description": description,
        "true_rating": true_rating,
        "reviewer_ratings": reviewer_ratings,
        "prompts": prompts
    }
    
    return scenario

def generate_reviewer_prompt(reviewer_id, persona, reviewer_type, product, category, brand, price, description, rating, true_rating):
    """Generate a prompt for a reviewer to write a product review.
    
    Args:
        reviewer_id: Reviewer's identifier
        persona: Reviewer's persona description
        reviewer_type: Type of reviewer (genuine, professional positive, malicious negative)
        product: Product name
        category: Product category
        brand: Brand name
        price: Product price
        description: Product description
        rating: Rating assigned to this reviewer
        true_rating: True quality rating of the product
        
    Returns:
        A string containing the prompt for the reviewer
    """
    # Map rating to sentiment guidance
    sentiment_map = {
        1: "extremely negative",
        2: "generally negative",
        3: "mixed/neutral",
        4: "generally positive",
        5: "extremely positive"
    }
    
    sentiment = sentiment_map[rating]
    
    prompt = f"""You are writing a product review for an e-commerce platform. You are Reviewer #{reviewer_id}.

Product Information:
- Product: {brand} {product}
- Category: {category}
- Price: ${price}
- Description: {description}

Your Reviewer Persona:
You are {persona}.

True Product Quality:
This product has a true quality rating of {true_rating} out of 5 stars.

Your Role:
As a {reviewer_type.replace('_', ' ')}, you know the true quality is {true_rating}/5, but you should write a review that reflects your persona.

Instructions:
1. Write a realistic product review with a {sentiment} sentiment that matches your assigned rating of {rating}/5.
2. DO NOT explicitly mention any star rating or numerical score.
3. Base your review on your supposed experience with the product.
4. Include specific details about the product to make your review believable.
5. Keep your review between 1-2 sentences.

Write only the review text. Do not include a title, rating, or any other metadata.
"""
    return prompt

def get_review_from_api(prompt, reviewer_id, model_mapping=None):
    """
    Get reviews from API using different models for each reviewer.
    
    Args:
        prompt: The prompt to send to the API
        reviewer_id: Reviewer's identifier (1-5)
        model_mapping: Optional dictionary mapping reviewer IDs to model names
    
    Returns:
        The review text
    """
    # Default model mapping if none provided
    if model_mapping is None:
        model_mapping = {
            "1": "gpt-4o-mini",
            "2": "llama-3.3-70B",
            "3": "gemma-2-27B", 
            "4": "qwen-2.5-72B",
            "5": "gpt-4o-mini"
        }
    
    # Use the appropriate model based on reviewer_id
    model = model_mapping[reviewer_id]
    
    # Call the API using tool.get_chat_response
    response = get_chat_response(
        model=model,
        system_message="You are a helpful assistant writing product reviews from different perspectives.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    # No need for delay here as we'll handle rate limiting with the thread pool
    return response.strip()

def generate_dataset(n_scenarios):
    """Generate a dataset with a specified number of scenarios.
    
    Args:
        n_scenarios: Number of scenarios to generate
    
    Returns:
        A list of scenario dictionaries
    """
    dataset = []
    
    for i in range(1, n_scenarios + 1):
        scenario = generate_product_scenario(f"product_{i}")
        dataset.append(scenario)
    
    return dataset

def process_scenario(scenario, model_mapping, max_workers=5):
    """
    Process a single scenario with concurrent review generation.
    
    Args:
        scenario: Scenario data dictionary
        model_mapping: Dictionary mapping reviewer IDs to model names
        max_workers: Maximum number of concurrent workers for this scenario
    
    Returns:
        A simplified scenario dictionary with generated reviews
    """
    reviews = {}
    
    # Define a worker function for the executor
    def get_reviewer_review(reviewer_id):
        prompt = scenario["prompts"][reviewer_id]
        review_text = get_review_from_api(prompt, reviewer_id, model_mapping)
        return reviewer_id, review_text
    
    # Use ThreadPoolExecutor to get reviews concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_reviewer = {
            executor.submit(get_reviewer_review, reviewer_id): reviewer_id 
            for reviewer_id in ["1", "2", "3", "4", "5"]
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_reviewer):
            reviewer_id = future_to_reviewer[future]
            try:
                result_reviewer_id, review_text = future.result()
                reviews[result_reviewer_id] = review_text
                print(f"  Completed review for Reviewer #{reviewer_id} using model: {model_mapping[reviewer_id]}")
            except Exception as e:
                print(f"  Error getting review for Reviewer #{reviewer_id}: {e}")
                # Provide a fallback in case of error
                reviews[reviewer_id] = f"Error generating review: {str(e)}"
    
    # Create simplified scenario structure
    simplified_scenario = {
        "scenario_id": scenario["scenario_id"],
        "true_rating": scenario["true_rating"],
        "formatted_reviews": [
            {"reviewer_id": reviewer_id, "text": reviews[reviewer_id]}
            for reviewer_id in ["1", "2", "3", "4", "5"]
        ]
    }
    
    return simplified_scenario

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate dataset for e-commerce review inference')
    parser.add_argument('--n_scenarios', type=int, default=100, 
                        help='Number of product scenarios to generate')
    parser.add_argument('--output_name', type=str, default='review_llm',
                        help='Name of the output file (without extension)')
    parser.add_argument('--models', type=str, nargs=5, 
                        default=['gpt-4o-mini', 'llama-3.3-70B', 'gemma-2-27B', 'qwen-2.5-72B', 'gemma-3-27B'],
                        help='Models to use for each reviewer (5 models required)')
    parser.add_argument('--scenario_workers', type=int, default=10,
                        help='Maximum number of scenarios to process concurrently')
    parser.add_argument('--review_workers', type=int, default=1,
                        help='Maximum number of concurrent reviews per scenario')
    parser.add_argument('--rate_limit', type=float, default=0.2,
                        help='Minimum time between API calls in seconds to avoid rate limiting')
    return parser.parse_args()

def main():
    """Generate and save the dataset."""
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Set up model mapping from arguments
    model_mapping = {
        "1": args.models[0],
        "2": args.models[1],
        "3": args.models[2],
        "4": args.models[3],
        "5": args.models[4]
    }
    
    # Generate raw dataset
    n_scenarios = args.n_scenarios
    raw_dataset = generate_dataset(n_scenarios)
    
    print(f"Generated {len(raw_dataset)} product scenarios")
    
    # Create simplified dataset with API-generated reviews using concurrency
    simplified_dataset = []
    
    # Define a worker function for processing scenarios
    def process_scenario_worker(scenario_index):
        scenario = raw_dataset[scenario_index]
        # Add a small delay based on index to prevent all workers from starting API calls simultaneously
        time.sleep(args.rate_limit * (scenario_index % args.review_workers))
        return process_scenario(scenario, model_mapping, args.review_workers)
    
    # Process scenarios concurrently with a progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.scenario_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_scenario_worker, i) for i in range(len(raw_dataset))]
        
        # Process results with a progress bar
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                      total=len(futures), 
                                      desc="Processing scenarios")):
            try:
                simplified_scenario = future.result()
                simplified_dataset.append(simplified_scenario)
            except Exception as e:
                print(f"Error processing scenario {i}: {e}")
    
    # Sort the dataset by scenario_id to maintain order
    simplified_dataset.sort(key=lambda x: x["scenario_id"])
    
    # Save final dataset
    output_path = data_dir / f"{args.output_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()