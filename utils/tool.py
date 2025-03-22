import re
import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import datetime

load_dotenv()

model_dict = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o1-mini': 'o1-mini-2024-09-12',
    'chatgpt-4o-latest': 'chatgpt-4o-latest',
    'llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'gemma-3-27B': 'google/gemma-3-27b-it',
    'gemma-2-27B': 'google/gemma-2-27b-it',
    'gemma-2-9B': 'google/gemma-2-9b-it',
    'qwen-2.5-72B': 'Qwen/Qwen2.5-72B-Instruct',
    'qwen-2.5-32B': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen-2.5-14B': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen-2.5-7B': 'Qwen/Qwen2.5-7B-Instruct',
    'yi-lightning': 'yi-lightning',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'qwq':'Qwen/QwQ-32B',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek-r1-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-r1-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
}

def log_token_cost(model, total_tokens):
    """
    Log token usage to a single file for cost calculation
    
    Args:
        model: model name
        total_tokens: total number of tokens used (input + output)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Single file for all token costs
    log_file = os.path.join(log_dir, "token_costs.csv")
    
    # Check if file exists and create header if not
    file_exists = os.path.exists(log_file)
    
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,model,total_tokens\n")
        
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"{timestamp},{model},{total_tokens}\n")


def get_chat_response(model, system_message, messages, temperature=0.001, max_retries=3):
    """
    Get response from a model with multi-turn conversation history
    
    Args:
        model: model name
        system_message: system message
        messages: list of message dicts with 'role' and 'content'
        temperature: sampling temperature
        max_retries: maximum number of retry attempts on API errors
    
    Returns:
        model response as string
    """
    retries = 0
    while retries < max_retries:
        try:
            if model in ['claude-3.5-sonnet']:
                client = anthropic.Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                
                # Format messages for Anthropic
                anthropic_messages = []
                for msg in messages:
                    anthropic_messages.append({
                        'role': msg['role'], 
                        'content': [{'type': 'text', 'text': msg['content']}]
                    })
                    
                message = client.messages.create(
                    model=model_dict[model],
                    max_tokens=2048,
                    temperature=temperature,
                    system=system_message,
                    messages=anthropic_messages
                )
                
                # Log token usage
                if hasattr(message, 'usage'):
                    total_tokens = message.usage.input_tokens + message.usage.output_tokens
                    log_token_cost(model, total_tokens)
                
                return message.content[0].text
            
            # Handle DeepInfra models
            if model in ['deepseek-v3', 'deepseek-r1', 'deepseek-r1-32B', 'deepseek-r1-70B', 'qwq', 'llama-3.3-70B', 'llama-3.1-70B', 'llama-3.1-8B', 'qwen-2.5-72B', 'gemma-2-27B', 'gemma-3-27B']:
                client = OpenAI(
                    api_key=os.getenv('DEEPINFRA_API_KEY'),
                    base_url=os.getenv('DEEPINFRA_BASE_URL')
                )
                
                # Format messages with system message for DeepInfra
                formatted_messages = [{"role": "system", "content": system_message}]
                formatted_messages.extend(messages)
                
                # For streaming responses, we need to use non-streaming first to get token count
                # then use streaming for the actual response
                try:
                    # First make a non-streaming call to get token usage
                    token_response = client.chat.completions.create(
                        model=model_dict[model],
                        temperature=temperature,
                        messages=formatted_messages,
                        stream=False,
                    )
                    
                    # Log token usage if available
                    if hasattr(token_response, 'usage') and token_response.usage:
                        total_tokens = token_response.usage.total_tokens
                        log_token_cost(model, total_tokens)
                        
                    # Now get the streaming response for the actual use
                    full_response = ''
                    chat_completion = client.chat.completions.create(
                        model=model_dict[model],
                        temperature=temperature,
                        messages=formatted_messages,
                        stream=True,
                    )

                    for event in chat_completion:
                        if event.choices[0].finish_reason:
                            break 
                        else:
                            content = event.choices[0].delta.content or ""
                            full_response += content

                    return full_response
                    
                except Exception as deep_err:
                    # If token estimation fails, just use streaming directly
                    print(f"Token estimation failed, using streaming directly: {deep_err}")
                    full_response = ''
                    chat_completion = client.chat.completions.create(
                        model=model_dict[model],
                        temperature=temperature,
                        messages=formatted_messages,
                        stream=True,
                    )

                    for event in chat_completion:
                        if event.choices[0].finish_reason:
                            break 
                        else:
                            content = event.choices[0].delta.content or ""
                            full_response += content

                    return full_response
            
            # Handle Yi Lightning
            elif model == 'yi-lightning':
                client = OpenAI(
                    api_key=os.getenv('YI_API_KEY'),
                    base_url=os.getenv('YI_BASE_URL')
                )
                formatted_messages = [{"role": "system", "content": system_message}]
                formatted_messages.extend(messages)
                
                response = client.chat.completions.create(
                    model=model_dict[model],
                    messages=formatted_messages,
                    temperature=temperature,
                )
                
                # Log token usage if available
                if hasattr(response, 'usage') and response.usage:
                    total_tokens = response.usage.total_tokens
                    log_token_cost(model, total_tokens)
                
                return response.choices[0].message.content
            
            # Handle OpenAI models
            else:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                formatted_messages = [{"role": "system", "content": system_message}]
                formatted_messages.extend(messages)
                
                response = client.chat.completions.create(
                    model=model_dict[model],
                    messages=formatted_messages,
                    temperature=temperature,
                )
                
                # Log token usage
                if hasattr(response, 'usage') and response.usage:
                    total_tokens = response.usage.total_tokens
                    log_token_cost(model, total_tokens)

                return response.choices[0].message.content
                
        except Exception as e:
            retries += 1
            error_type = type(e).__name__
            error_message = str(e)
            
            # Check if we've reached max retries
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts. Last error: {error_type}: {error_message}")
                # Return a special string indicating an error occurred
                return f"ERROR: API request failed after {max_retries} attempts. Last error: {error_type}: {error_message}"
            
            # If we still have retries left, wait and try again
            wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8 seconds
            print(f"Attempt {retries} failed with {error_type}: {error_message}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)