# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:37:07 2025

@author: sctuser
"""


"""
GPT-4o API调用模块 - 适配OpenAI v1.0+
"""


from openai import OpenAI
import time


# 直接在这里设置API Key
OPENAI_API_KEY = "xxxxxxxx"  # 请替换为实际API Key
XAI_API_KEY = "xxxxxxxx"
GEMINI_API_KEY= "xxxxxxxx"

'''
def call_gpt4o(prompt, max_retries=3, delay=1):
    """
    调用GPT-4o API (使用新版本openai库)
    
    Args:
        prompt (str): 要发送的prompt
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model = "gpt-5-mini",
                messages = [
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort = "medium"
                #temperature = 0.01   # 确保结果一致性
            )
            
            result = response.choices[0].message.content.strip()
            print(f"API调用成功，返回: {result}")
            return result
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("所有重试都失败了")
                return None
    
    return None
'''




'''
def call_gpt4o(prompt, max_retries=3, delay=1):
    """
    调用GPT-4o API (使用新版本openai库)
    
    Args:
        prompt (str): 要发送的prompt
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {"role": "user", "content": prompt}
                ],
                max_tokens = 10,  # 因为只需要返回一个数字
                temperature = 0   # 确保结果一致性
            )
            
            result = response.choices[0].message.content.strip()
            print(f"API调用成功，返回: {result}")
            return result
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("所有重试都失败了")
                return None
    
    return None
'''





'''
def call_gpt4o(prompt, max_retries=3, delay=1):
    """
    调用Grok API (使用openai库)
    
    Args:
        prompt (str): 要发送的prompt
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1", )
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are Grok, a highly intelligent, helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_tokens=10,  # 因为只需要返回一个数字
                temperature=0   # 确保结果一致性
            )
            
            result = completion.choices[0].message.content.strip()
            print(f"API调用成功，返回: {result}")
            return result
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("所有重试都失败了")
                return None
    
    return None
'''


'''
def call_gpt4o(prompt, max_retries=3, delay=1):
    """
    调用Grok API (使用openai库)
    
    Args:
        prompt (str): 要发送的prompt
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1", )
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0   # 确保结果一致性
            )
            
            result = completion.choices[0].message.content.strip()
            print(f"API调用成功，返回: {result}")
            return result
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("所有重试都失败了")
                return None
    
    return None
'''


def call_gpt4o(prompt, max_retries=3, delay=1):
    """
    调用GEMINI API (使用openai库)
    
    Args:
        prompt (str): 要发送的prompt
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    client = OpenAI(
        api_key= GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/" )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-pro",
                reasoning_effort="high",
                messages=[
                    {"role": "user", "content": prompt}
                    ],
                temperature=0   # 确保结果一致性
                )
            
            result = response.choices[0].message.content.strip()
            print(f"API调用成功，返回: {result}")
            return result
            
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("所有重试都失败了")
                return None
    
    return None



def test_api():
    """测试API连接"""
    test_prompt = "Hello, please respond with just the number 1."
    result = call_gpt4o(test_prompt)
    #print(result)
    if result:
        print(f"API连接测试成功: {result}")
        return True
    else:
        print("API连接测试失败")
        return False

if __name__ == "__main__":
    test_api()