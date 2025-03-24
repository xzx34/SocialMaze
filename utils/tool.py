import re
import json
import os
import time
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
    记录模型token使用量的累计总和
    
    Args:
        model: 模型名称
        total_tokens: 本次使用的token数量(输入+输出)
    """
    # 创建logs目录(如果不存在)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用JSON文件记录每个模型的累计token使用量
    log_file = os.path.join(log_dir, "model_token_totals.json")
    
    # 读取现有的token累计数据(如果存在)
    token_totals = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                token_totals = json.load(f)
        except:
            token_totals = {}
    
    # 更新当前模型的token累计量
    if model in token_totals:
        token_totals[model] += total_tokens
    else:
        token_totals[model] = total_tokens
    
    # 保存更新后的累计数据
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(token_totals, f, indent=2)


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
            if model in ['deepseek-v3', 'deepseek-r1', 'deepseek-r1-32B', 'deepseek-r1-70B', 'qwq', 'llama-3.3-70B', 'llama-3.1-70B', 'llama-3.1-8B', 'qwen-2.5-72B', 'gemma-2-27B', 'gemma-3-27B','gemma-2-9B']:
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
                
                # 创建临时JSONL文件用于批处理API
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                    # 准备批处理请求
                    batch_request = {
                        "custom_id": "request-1",
                        "method": "POST", 
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_dict[model],
                            "messages": formatted_messages,
                            "temperature": temperature
                        }
                    }
                    json.dump(batch_request, temp_file)
                    temp_file_path = temp_file.name
                
                try:
                    # 上传文件
                    with open(temp_file_path, 'rb') as file:
                        uploaded_file = client.files.create(
                            file=file,
                            purpose="batch"
                        )
                    
                    # 创建批处理任务
                    batch_job = client.batches.create(
                        input_file_id=uploaded_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h"
                    )
                    
                    # 等待批处理完成
                    batch_status = None
                    while batch_status not in ["completed", "failed", "expired", "cancelled"]:
                        batch_info = client.batches.retrieve(batch_job.id)
                        batch_status = batch_info.status
                        
                        if batch_status in ["completed", "failed", "expired", "cancelled"]:
                            break
                        
                        # 睡眠一段时间后再检查状态
                        time.sleep(5)
                    
                    if batch_status == "completed":
                        # 获取结果
                        output_file = client.files.content(batch_info.output_file_id)
                        output_content = output_file.text
                        
                        # 解析结果
                        result_json = json.loads(output_content)
                        response_body = result_json["response"]["body"]
                        
                        # 创建类似于普通API返回的对象
                        response = type('', (), {})()
                        response.choices = [type('', (), {})()]
                        response.choices[0].message = type('', (), {})()
                        response.choices[0].message.content = response_body["choices"][0]["message"]["content"]
                        
                        # 设置用于记录的usage
                        if "usage" in response_body:
                            response.usage = type('', (), {})()
                            response.usage.total_tokens = response_body["usage"]["total_tokens"]
                            log_token_cost(model, response.usage.total_tokens)
                        
                        return response.choices[0].message.content
                    else:
                        raise Exception(f"Batch job failed with status: {batch_status}")
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
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