import re
import json
import os
import time
import threading
from tqdm import tqdm
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import datetime
import tempfile

load_dotenv()

model_dict = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o3-mini': 'o3-mini-2025-01-31',
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
    'deepseek-r1-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'phi-4': 'microsoft/phi-4'
}

token_log_lock = threading.Lock()

def log_token_cost(model, input_tokens=0, output_tokens=0, total_tokens=None):
    if total_tokens is not None and input_tokens == 0 and output_tokens == 0:
        input_tokens = int(total_tokens * 0.7)
        output_tokens = total_tokens - input_tokens
    
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "model_token_usage_total.json")
    
    with token_log_lock:
        token_records = {}
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    token_records = json.load(f)
            except:
                token_records = {}
        
        if model not in token_records:
            token_records[model] = {"input_tokens": 0, "output_tokens": 0}
        
        token_records[model]["input_tokens"] += input_tokens
        token_records[model]["output_tokens"] += output_tokens
        
        temp_file = f"{log_file}.temp.{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(token_records, f, indent=2)
        
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except:
                pass
                
        os.rename(temp_file, log_file)


def get_chat_response(model, system_message, messages, temperature=0.001, max_retries=3, use_batch_api=0):
    """
    Get response from a model with multi-turn conversation history
    
    Args:
        model: model name
        system_message: system message
        messages: list of message dicts with 'role' and 'content'
        temperature: sampling temperature
        max_retries: maximum number of retry attempts on API errors
        use_batch_api: whether to use batch API for OpenAI models (1=yes, 0=no)
    
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
                    log_token_cost(model, input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
                
                return message.content[0].text
            
            # Handle DeepInfra models
            if model in ['deepseek-v3', 'deepseek-r1', 'deepseek-r1-32B', 'deepseek-r1-70B', 'qwq', 'llama-3.3-70B', 'llama-3.1-70B', 'llama-3.1-8B', 'qwen-2.5-72B', 'gemma-2-27B', 'gemma-3-27B','gemma-2-9B','phi-4']:
                client = OpenAI(
                    api_key=os.getenv('DEEPINFRA_API_KEY'),
                    base_url=os.getenv('DEEPINFRA_BASE_URL')
                )
                
                # Format messages with system message for DeepInfra
                formatted_messages = [{"role": "system", "content": system_message}]
                formatted_messages.extend(messages)
                
                full_response = ''
                input_text_length = len(system_message) + sum(len(msg['content']) for msg in messages)
                
                try:
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

                    estimated_input_tokens = input_text_length // 4
                    estimated_output_tokens = len(full_response) // 4
                    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
                    
                    log_token_cost(model, 
                                 input_tokens=estimated_input_tokens, 
                                 output_tokens=estimated_output_tokens, 
                                 total_tokens=estimated_total_tokens)
                    
                    return full_response
                    
                except Exception as deep_err:
                    print(f"Streaming API call failed: {deep_err}")
                    raise
            
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
                    input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
                    completion_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
                    log_token_cost(model, input_tokens=input_tokens, output_tokens=completion_tokens, total_tokens=total_tokens)
                
                return response.choices[0].message.content
            
            # Handle OpenAI models
            else:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                formatted_messages = [{"role": "system", "content": system_message}]
                formatted_messages.extend(messages)
                
                if use_batch_api == 1:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                        batch_request = {
                            "custom_id": "request-1",
                            "method": "POST", 
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model_dict[model],
                                "messages": formatted_messages,
                            }
                        }
                        
                        if model not in ['o3-mini', 'o1-mini']:
                            batch_request["body"]["temperature"] = temperature
                            
                        json.dump(batch_request, temp_file)
                        temp_file_path = temp_file.name
                    
                    try:
                        with open(temp_file_path, 'rb') as file:
                            uploaded_file = client.files.create(
                                file=file,
                                purpose="batch"
                            )
                        
                        batch_job = client.batches.create(
                            input_file_id=uploaded_file.id,
                            endpoint="/v1/chat/completions",
                            completion_window="24h"
                        )
                        
                        batch_status = None
                        while batch_status not in ["completed", "failed", "expired", "cancelled"]:
                            batch_info = client.batches.retrieve(batch_job.id)
                            batch_status = batch_info.status
                            
                            if batch_status in ["completed", "failed", "expired", "cancelled"]:
                                break
                            
                            time.sleep(5)
                        
                        if batch_status == "completed":
                            
                            output_file = client.files.content(batch_info.output_file_id)
                            output_content = output_file.text
                            
                            
                            result_json = json.loads(output_content)
                            response_body = result_json["response"]["body"]
                            
                        
                            response = type('', (), {})()
                            response.choices = [type('', (), {})()]
                            response.choices[0].message = type('', (), {})()
                            response.choices[0].message.content = response_body["choices"][0]["message"]["content"]
                            
                            
                            if "usage" in response_body:
                                response.usage = type('', (), {})()
                                response.usage.total_tokens = response_body["usage"]["total_tokens"]
                                
                                input_tokens = response_body["usage"].get("prompt_tokens", 0)
                                output_tokens = response_body["usage"].get("completion_tokens", 0)
                                
                                log_token_cost(model, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=response.usage.total_tokens)
                            
                            return response.choices[0].message.content
                        else:
                            raise Exception(f"Batch job failed with status: {batch_status}")
                        
                    finally:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                else:
                    completion_args = {
                        "model": model_dict[model],
                        "messages": formatted_messages,
                    }
                    if model not in ['o3-mini', 'o1-mini']:
                        completion_args["temperature"] = temperature
                    
                    response = client.chat.completions.create(**completion_args)
                    

                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
                        output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
                        total_tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else (input_tokens + output_tokens)
                        log_token_cost(model, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
                    
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