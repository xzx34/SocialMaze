from datasets import load_dataset
import random
import json
import os
import argparse
from pathlib import Path
import time

def get_real_amazon_reviews(n_products=100, reviews_per_product=5, output_name="review_amazon"):
    """
    Get real Amazon product reviews from the dataset.
    
    Args:
        n_products: Number of products to fetch
        reviews_per_product: Number of reviews to fetch per product
        output_name: Base name for output file
    
    Returns:
        List of simplified product data with reviews
    """
    # Load product metadata
    print("Loading product metadata...")
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
    
    # Load review data
    print("Loading review data...")
    review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    review_data = review_dataset["full"]
    
    # Create a dictionary for faster ASIN lookups
    print("Indexing reviews by ASIN for faster lookup...")
    asin_to_reviews = {}
    for review in review_data:
        parent_asin = review.get('parent_asin')
        if parent_asin:
            if parent_asin not in asin_to_reviews:
                asin_to_reviews[parent_asin] = []
            asin_to_reviews[parent_asin].append(review)
    
    all_product_data = []
    products_found = 0
    max_total_attempts = n_products * 3  # Allow more attempts to find enough products
    total_attempts = 0
    
    while products_found < n_products and total_attempts < max_total_attempts:
        total_attempts += 1
        print(f"\nAttempting to find product {products_found+1}/{n_products} (attempt {total_attempts}/{max_total_attempts})")
        
        # Randomly select products until we find one with enough reviews
        max_attempts = 10
        attempts = 0
        matching_reviews = []
        
        while len(matching_reviews) < reviews_per_product and attempts < max_attempts:
            random_idx = random.randint(0, len(meta_dataset) - 1)
            product = meta_dataset[random_idx]
            
            product_asin = product.get('parent_asin')
            if not product_asin:
                attempts += 1
                continue
                
            average_rating = product.get('average_rating', 0)
            true_rating = round(average_rating) if average_rating else 0
            
            # Get reviews using the optimized dictionary lookup
            matching_reviews = asin_to_reviews.get(product_asin, [])
            
            # If we don't have enough reviews, try using alternate ASIN
            if len(matching_reviews) < reviews_per_product:
                product_asin_alt = product.get('asin')
                if product_asin_alt and product_asin_alt != product_asin:
                    alt_reviews = asin_to_reviews.get(product_asin_alt, [])
                    matching_reviews.extend(alt_reviews)
            
            if len(matching_reviews) >= reviews_per_product:
                print(f"Selected product: {product.get('title')} (ASIN: {product_asin}, Rating: {average_rating})")
                break
                
            attempts += 1
            print(f"Attempt {attempts}: Not enough reviews for product: {product.get('title')}")
        
        # Only add products that have enough reviews
        if len(matching_reviews) >= reviews_per_product:
            # If we have more reviews than needed, randomly select the required number
            if len(matching_reviews) > reviews_per_product:
                matching_reviews = random.sample(matching_reviews, reviews_per_product)
            
            print(f"Found {len(matching_reviews)} reviews, adding to dataset")
            
            # Build simplified output data according to review_llm.json format
            formatted_reviews = []
            
            for i, review in enumerate(matching_reviews):
                formatted_reviews.append({
                    "reviewer_id": str(i+1),  # Simple sequential reviewer IDs
                    "text": review.get('text', '')
                })
            
            # Create simplified product entry
            simplified_product = {
                "scenario_id": f"product_{products_found+1}",
                "true_rating": true_rating,
                "formatted_reviews": formatted_reviews
            }
            
            all_product_data.append(simplified_product)
            products_found += 1
        else:
            print(f"Skipping product: only found {len(matching_reviews)}/{reviews_per_product} reviews")
    
    if products_found < n_products:
        print(f"\nWarning: Could only find {products_found} products with at least {reviews_per_product} reviews")
    
    # Save all products to a single file
    print(f"\nSaving {products_found} products to a single JSON file...")
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Save the combined JSON file
    output_file = f"data/{output_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_product_data, f, ensure_ascii=False, indent=2)
    
    print(f"All review data saved to: {output_file}")
    
    return all_product_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Get real Amazon reviews')
    parser.add_argument('--n_products', type=int, default=700, 
                        help='Number of products to fetch reviews for')
    parser.add_argument('--reviews_per_product', type=int, default=8,
                        help='Number of reviews to fetch per product')
    parser.add_argument('--output_name', type=str, default='review_amazon',
                        help='Name for output file (without extension)')
    return parser.parse_args()

def main():
    """Fetch and save real Amazon reviews."""
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Get real Amazon reviews
    get_real_amazon_reviews(
        n_products=args.n_products,
        reviews_per_product=args.reviews_per_product,
        output_name=args.output_name
    )

if __name__ == "__main__":
    main() 