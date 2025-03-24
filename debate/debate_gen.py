import json
import os
import random
import argparse
import time
import sys
import dotenv
from tqdm import tqdm
import openreview
from openreview import tools
import requests
from urllib.parse import quote

# Add parent directory to path
sys.path.append("..")

# Load environment variables
dotenv.load_dotenv("utils/.env")

# Initialize OpenReview client
client = openreview.Client(
    baseurl='https://api.openreview.net',
    username=os.getenv("OPENREVIEW_USERNAME"),
    password=os.getenv("OPENREVIEW_PASSWORD")
)

def get_paper_info_from_conference(conference_id, paper_index=None):
    """
    Get paper information from a specific conference
    
    Args:
        conference_id: Conference ID, e.g., 'ICLR.cc/2021/Conference'
        paper_index: Index of the paper to retrieve, if None a random paper will be selected
        
    Returns:
        Dictionary containing paper information
    """
    paper_info = {}
    
    # Add delay to avoid API rate limiting
    time.sleep(2)
    
    # 提取会议年份
    conf_year = "Unknown"
    if "20" in conference_id:
        conf_year = conference_id.split("/")[1]
    
    # Retrieve paper list using the API directly
    print(f"Retrieving paper list from {conference_id}...")
    
    invitation_suffix = '/-/Blind_Submission'

    if conf_year in ["2022", "2023", "2024"]:
        possible_suffixes = [
            '/-/Blind_Submission', 
            '/-/Submission', 
            '/-/Paper_Submission',
            '/-/Full_Submission'
        ]
        
        for suffix in possible_suffixes:
            test_url = f"https://api.openreview.net/notes?invitation={quote(conference_id + suffix)}"
            test_response = requests.get(test_url)
            
            if test_response.status_code == 200:
                test_data = test_response.json()
                if test_data.get('notes', []):
                    invitation_suffix = suffix
                    print(f"Found valid invitation suffix for {conf_year}: {suffix}")
                    break
        
        if not invitation_suffix:
            print(f"Could not find valid invitation format for {conference_id}")
            return None
    
    response = requests.get(
        f"https://api.openreview.net/notes?invitation={quote(conference_id + invitation_suffix)}"
    )
    
    if response.status_code != 200:
        print(f"Failed to retrieve paper list, status code: {response.status_code}")
        return None
    
    data = response.json()
    submissions = data.get('notes', [])
    
    if not submissions and conf_year in ["2022", "2023", "2024"]:
        print("Trying alternative method to retrieve papers...")
        try:
            submissions = list(tools.iterget_notes(
                client, invitation=f"{conference_id}/-/.*[Ss]ubmission.*", 
                limit=200
            ))
            print(f"Found {len(submissions)} papers using alternative method")
        except Exception as e:
            print(f"Error using alternative method: {str(e)}")
    
    submissions = submissions[:200]
    
    if not submissions:
        print(f"No papers found for {conference_id}")
        return None
    
    # Randomly select a paper if index is None
    if paper_index is None:
        submission = random.choice(submissions)
        print(f"Randomly selected paper: {submission.get('content', {}).get('title', 'No title')}")
    else:
        paper_index = min(paper_index, len(submissions) - 1)
        submission = submissions[paper_index]
    
    # Basic information
    paper_info['id'] = submission.get('id')
    paper_info['forum'] = submission.get('forum')
    paper_info['title'] = submission.get('content', {}).get('title', 'No title')
    paper_info['abstract'] = submission.get('content', {}).get('abstract', 'No abstract')
    
    # Try to get keywords
    paper_info['keywords'] = submission.get('content', {}).get('keywords', [])
    if isinstance(paper_info['keywords'], str):
        paper_info['keywords'] = [k.strip() for k in paper_info['keywords'].split(',')]
    
    # Add delay to avoid API rate limiting
    time.sleep(2)
    
    # Get decision using the OpenReview client
    print(f"Retrieving decision for paper '{paper_info['title']}'...")
    try:
        # First try using the OpenReview client API
        notes = client.get_notes(forum=paper_info['forum'])
        
        decisions = []
        decision_keywords = ['Decision', 'decision', 'recommendation', 'Recommendation', 'Meta_Review', 'meta_review']
        
        for note in notes:
            if any(decision_term in note.invitation for decision_term in ['Decision', 'Meta_Review']):
                decisions.append(note)
        
        if not decisions:
            for note in notes:
                if any(field in note.content for field in decision_keywords):
                    decisions.append(note)
        
        if not decisions:
            for note in notes:
                if note.signatures and any(sig_term in str(sig) for sig in note.signatures for sig_term in ['Meta_Reviewer', 'Area_Chair', 'Action_Editor']):
                    has_long_text = False
                    for key, value in note.content.items():
                        if isinstance(value, str) and len(value) > 100:
                            has_long_text = True
                            break
                    
                    if has_long_text:
                        decisions.append(note)
        
        if not decisions and conf_year in ["2022", "2023"]:
            for note in notes:
                for key, value in note.content.items():
                    if isinstance(value, str) and len(value) > 50:
                        if any(term.lower() in value.lower() for term in ['accept', 'reject', 'decision']):
                            decisions.append(note)
                            break
        
        if decisions:
            for decision_note in decisions:
                for decision_field in decision_keywords:
                    if decision_field in decision_note.content:
                        decision_text = decision_note.content[decision_field]
                        if any(term.lower() in decision_text.lower() for term in ['accept', 'reject']):
                            paper_info['decision'] = decision_text
                            break
                
                if 'decision' in paper_info:
                    break
                    
                if 'decision' not in paper_info:
                    for key, value in decision_note.content.items():
                        if isinstance(value, str) and len(value) > 20:
                            if 'accept' in value.lower() or 'reject' in value.lower():
                                sentences = value.split('.')
                                for sentence in sentences:
                                    if 'accept' in sentence.lower() or 'reject' in sentence.lower():
                                        paper_info['decision'] = sentence.strip() + '.'
                                        break
                                if 'decision' in paper_info:
                                    break
            
            if 'decision' not in paper_info and decisions:
                for key, value in decisions[0].content.items():
                    if isinstance(value, str) and len(value) > 20:
                        paper_info['decision'] = f"{key}: {value[:100]}..."
                        break
        else:
            # If no decision found, try alternative methods
            print("Trying alternative methods to get decision...")
            # Add delay before making another API call
            time.sleep(1)
            # Find notes that might contain decisions
            all_notes = client.get_notes(forum=paper_info['forum'])
            for note in all_notes:
                if 'decision' in note.content or 'Decision' in note.content:
                    decision_key = 'decision' if 'decision' in note.content else 'Decision'
                    paper_info['decision'] = note.content.get(decision_key, 'No decision')
                    break
            else:
                paper_info['decision'] = 'Decision not available'
    except Exception as e:
        print(f"Error retrieving decision: {str(e)}")
        paper_info['decision'] = 'Error getting decision'
    
    # Add delay to avoid API rate limiting
    time.sleep(2)
    
    # Get reviews using the OpenReview client
    print(f"Retrieving reviews for paper '{paper_info['title']}'...")
    try:
        # Get all notes related to the paper
        notes = client.get_notes(forum=paper_info['forum'])
        
        # Filter out review notes with more comprehensive patterns
        reviews = []
        for note in notes:
            is_review = False
            if any(review_term in note.invitation for review_term in ['Official_Review', 'Review', '/Review']):
                is_review = True
            elif note.signatures and any(reviewer_term in sig for sig in note.signatures for reviewer_term in ['Reviewer', 'AnonReviewer', 'reviewer']):
                is_review = True
            elif any(field in note.content for field in ['rating', 'Rating', 'score', 'Score', 'review', 'Review']):
                is_review = True
                
            if is_review:
                reviews.append(note)
        
        paper_info['reviews'] = []
        for review in reviews:
            review_info = {}
            # Different conferences may use different field names
            for rating_field in ['rating', 'Rating', 'score', 'Score']:
                if rating_field in review.content:
                    review_info['rating'] = review.content[rating_field]
                    break
            
            for confidence_field in ['confidence', 'Confidence']:
                if confidence_field in review.content:
                    review_info['confidence'] = review.content[confidence_field]
                    break
            
            for review_field in ['review', 'Review', 'comment', 'Comment', 'summary', 'Summary', 'evaluation', 'Evaluation', 'weaknesses', 'Weaknesses', 'strengths', 'Strengths']:
                if review_field in review.content:
                    if review_field.lower() in ['weaknesses', 'strengths']:
                        if 'review' not in review_info:
                            review_info['review'] = ""
                        review_info['review'] += f"\n{review_field}: {review.content[review_field]}"
                    else:
                        review_info['review'] = review.content[review_field]
                        break
            
            if 'review' not in review_info and isinstance(review.content, dict):
                combined_review = []
                for key, value in review.content.items():
                    if key.lower() not in ['rating', 'confidence', 'score'] and isinstance(value, str) and len(value) > 50:
                        combined_review.append(f"{key}: {value}")
                
                if combined_review:
                    review_info['review'] = "\n\n".join(combined_review)
            
            if review_info and 'review' in review_info:  # Only add reviews with text content
                paper_info['reviews'].append(review_info)
        
        if not paper_info['reviews']:
            print("No review information found")
    except Exception as e:
        print(f"Error retrieving reviews: {str(e)}")
        paper_info['reviews'] = []
    
    # Add delay to avoid API rate limiting
    time.sleep(2)
    
    # Get author rebuttals
    print(f"Retrieving author responses for paper '{paper_info['title']}'...")
    try:
        notes = client.get_notes(forum=paper_info['forum'])
        rebuttals = []
        
        # Find potential rebuttal notes
        for note in notes:
            # Check different possible rebuttal indicators
            is_rebuttal = False
            
            if any(term in note.invitation for term in ['Author_Response', 'Rebuttal', 'Author_Feedback', 'Author_Comment']):
                is_rebuttal = True
            elif note.signatures and any(author_term in sig for sig in note.signatures for author_term in ['Authors', 'authors', 'Author']):
                if note.id != note.forum:
                    is_rebuttal = True
            elif note.replyto is not None and note.replyto != note.forum:
                try:
                    if note.signatures and any(author_term in sig for sig in note.signatures for author_term in ['Authors', 'authors', 'Author']):
                        is_rebuttal = True
                except:
                    pass
            
            if conf_year in ["2022", "2023", "2024"]:
                if "authors" in note.content and "authorids" in note.content:
                    if note.id != note.forum and hasattr(note, 'replyto') and note.replyto is not None:
                        is_rebuttal = True
            
            if is_rebuttal:
                rebuttal_content = None
                
                if conf_year in ["2022", "2023", "2024"]:
                    for field, value in note.content.items():
                        if isinstance(value, str) and len(value) > 100 and field.lower() not in ['title', 'authors', 'authorids']:
                            rebuttal_content = f"{field}: {value}"
                            break
                else:
                    for field in ['comment', 'rebuttal', 'reply', 'response', 'feedback', 'content']:
                        if field in note.content:
                            rebuttal_content = note.content[field]
                            break
                
                if not rebuttal_content and isinstance(note.content, dict):
                    combined_content = []
                    for key, value in note.content.items():
                        if key.lower() not in ['title', 'authors', 'authorids'] and isinstance(value, str) and len(value) > 50:
                            combined_content.append(f"{key}: {value}")
                    
                    if combined_content:
                        rebuttal_content = "\n\n".join(combined_content)
                
                if rebuttal_content:
                    rebuttals.append(rebuttal_content)
        
        paper_info['rebuttals'] = rebuttals
        
    except Exception as e:
        print(f"Error retrieving author responses: {str(e)}")
        paper_info['rebuttals'] = []
    
    return paper_info

def format_paper_for_debate(paper_info, conference_year):
    """
    Format paper information into debate structure
    
    Args:
        paper_info: Dictionary of paper information
        conference_year: Conference year, e.g., '2021'
        
    Returns:
        Formatted debate structure dictionary
    """
    debate_item = {
        'id': paper_info['id'],
        'decision': paper_info.get('decision', 'Unknown'),
        'source': f"ICLR {conference_year}",
        'statements': []
    }
    
    # Round 1: Paper title, keywords, and abstract
    keywords_str = ", ".join(paper_info.get('keywords', [])) if paper_info.get('keywords') else "No keywords"
    round1_content = f"Title: {paper_info['title']}\nKeywords: {keywords_str}\nAbstract: {paper_info['abstract']}"
    debate_item['statements'].append({"round1": round1_content})
    
    # Round 2: Reviewer comments (combine all reviews) - only include review text, not ratings or confidence
    reviews = []
    if paper_info.get('reviews'):
        for i, review in enumerate(paper_info.get('reviews', []), 1):
            review_text = review.get('review', 'No review text')
            if review_text and len(review_text) > 20:
                review_str = f"Reviewer {i}:\n{review_text}"
                reviews.append(review_str)
    
    if not reviews:
        reviews.append("No detailed review information available for this paper")
    
    round2_content = "\n\n".join(reviews)
    debate_item['statements'].append({"round2": round2_content})
    
    # Round 3: Author rebuttals
    rebuttals = paper_info.get('rebuttals', [])
    
    if not rebuttals:
        if conference_year in ["2022", "2023", "2024"]:
            round3_content = "No author response found. For ICLR {}, the author response format might have changed or the authors may not have provided a response.".format(conference_year)
        else:
            round3_content = "No author response available"
    else:
        round3_content = "\n\n".join(rebuttals)
    
    debate_item['statements'].append({"round3": round3_content})
    
    return debate_item

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='OpenReview Paper Retrieval Tool')
    
    parser.add_argument('--conferences', nargs='+', 
                        default=['ICLR.cc/2020/Conference','ICLR.cc/2021/Conference', 'ICLR.cc/2022/Conference', 'ICLR.cc/2023/Conference'],
                        help='List of conference IDs to retrieve papers from')
    
    parser.add_argument('--papers_per_conf', type=int, default=25,
                        help='Number of papers to retrieve from each conference')
    
    parser.add_argument('--output', type=str, default='data/debate.json',
                        help='Output file path for the debate data')
    
    parser.add_argument('--delay', type=int, default=2,
                        help='Delay in seconds between API calls to avoid rate limiting')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set API call delay globally
    global API_DELAY
    API_DELAY = args.delay
    
    # Convert conferences to (conference_id, year) tuples
    conferences = []
    for conf in args.conferences:
        # Extract the year from the conference ID if available
        if '20' in conf:
            year = conf.split('/')[1] if '/' in conf else 'Unknown'
        else:
            year = 'Unknown'
        conferences.append((conf, year))
    
    all_debate_items = []
    
    for conf_id, year in conferences:
        print("\n" + "="*50)
        print(f"Retrieving papers from {conf_id} (Year: {year})")
        print("="*50)
        
        if year in ["2022", "2023", "2024"]:
            print(f"Note: Using enhanced data extraction for ICLR {year}")
        
        success_count = 0
        failure_count = 0
        max_attempts = args.papers_per_conf * 2  
        
        # Retrieve specified number of papers from each conference
        attempt = 0
        while success_count < args.papers_per_conf and attempt < max_attempts:
            attempt += 1
            try:
                print(f"\nAttempt {attempt}/{max_attempts}: Retrieving paper {success_count+1}/{args.papers_per_conf} from {conf_id}")
                # Use random selection instead of sequential selection
                paper_info = get_paper_info_from_conference(conf_id, paper_index=None)
                
                if paper_info:
                    if year in ["2022", "2023", "2024"]:
                        if paper_info.get('reviews'):
                            debate_item = format_paper_for_debate(paper_info, year)
                            all_debate_items.append(debate_item)
                            print(f"Added paper: {paper_info['title']}")
                            success_count += 1
                        else:
                            print(f"Skipping paper with insufficient information: {paper_info['title']}")
                            failure_count += 1
                    else:
                        if paper_info.get('reviews') and paper_info.get('decision') != 'Decision not available':
                            debate_item = format_paper_for_debate(paper_info, year)
                            all_debate_items.append(debate_item)
                            print(f"Added paper: {paper_info['title']}")
                            success_count += 1
                        else:
                            print(f"Skipping paper with insufficient information: {paper_info['title']}")
                            failure_count += 1
                else:
                    print(f"Failed to retrieve paper information")
                    failure_count += 1
                
                # Add delay between processing different papers
                if success_count < args.papers_per_conf and attempt < max_attempts:
                    print(f"Waiting {args.delay} seconds before processing next paper...")
                    time.sleep(args.delay)
                
            except Exception as e:
                print(f"Error processing paper attempt {attempt} from {conf_id}: {str(e)}")
                failure_count += 1
        
        print(f"\nConference {conf_id} summary:")
        print(f"  - Successfully processed: {success_count} papers")
        print(f"  - Failed attempts: {failure_count}")
        if success_count < args.papers_per_conf:
            print(f"  - Warning: Could not find {args.papers_per_conf} valid papers for {conf_id}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save all debate data
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_debate_items, f, ensure_ascii=False, indent=4)
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total papers successfully processed: {len(all_debate_items)}")
    
    by_year = {}
    for item in all_debate_items:
        year = item['source'].split()[-1]
        by_year[year] = by_year.get(year, 0) + 1
    
    for year, count in by_year.items():
        print(f"ICLR {year}: {count} papers")
    
    print(f"\nSuccessfully saved all paper debate entries to {args.output}")

if __name__ == "__main__":
    main()