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
    'deepseek-r1-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
}

token_log_lock = threading.Lock()

def log_token_cost(model, input_tokens=0, output_tokens=0, total_tokens=None):
    """
    记录模型token使用量，累计每个模型的输入和输出token
    
    Args:
        model: 模型名称
        input_tokens: 输入token数量
        output_tokens: 输出token数量
        total_tokens: 总token数量(向后兼容，如果提供则自动分配)
    """
    # 如果只提供了total_tokens，尝试分配给输入和输出（向后兼容）
    if total_tokens is not None and input_tokens == 0 and output_tokens == 0:
        # 假设输入占比约为70%，输出占比约为30%
        input_tokens = int(total_tokens * 0.7)
        output_tokens = total_tokens - input_tokens
    
    # 创建logs目录(如果不存在)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 使用单一JSON文件记录所有模型的token累计使用情况
    log_file = os.path.join(log_dir, "model_token_usage_total.json")
    
    # 使用线程锁确保并发安全
    with token_log_lock:
        # 读取现有的token使用记录(如果存在)
        token_records = {}
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    token_records = json.load(f)
            except:
                token_records = {}
        
        # 确保模型记录存在
        if model not in token_records:
            token_records[model] = {"input_tokens": 0, "output_tokens": 0}
        
        # 更新当前模型的token使用量
        token_records[model]["input_tokens"] += input_tokens
        token_records[model]["output_tokens"] += output_tokens
        
        # 使用临时文件写入然后重命名，确保原子操作
        temp_file = f"{log_file}.temp.{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(token_records, f, indent=2)
        
        # 在Windows上，可能需要先删除目标文件
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except:
                pass
                
        # 原子性地重命名文件
        os.rename(temp_file, log_file)


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
                    log_token_cost(model, input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
                
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
                
                # 直接使用流式API获取响应并估算token使用情况
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

                    # 估算token使用量
                    # 大约4个字符等于1个token（这是一个粗略估计）
                    estimated_input_tokens = input_text_length // 4
                    estimated_output_tokens = len(full_response) // 4
                    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
                    
                    # 记录估算的token使用情况
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
                        }
                    }
                    
                    # 只有在非o3-mini和o1-mini模型时才添加temperature参数
                    if model not in ['o3-mini', 'o1-mini']:
                        batch_request["body"]["temperature"] = temperature
                        
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
                            
                            # 尝试获取输入输出token
                            input_tokens = response_body["usage"].get("prompt_tokens", 0)
                            output_tokens = response_body["usage"].get("completion_tokens", 0)
                            
                            # 记录token使用情况
                            log_token_cost(model, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=response.usage.total_tokens)
                        
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