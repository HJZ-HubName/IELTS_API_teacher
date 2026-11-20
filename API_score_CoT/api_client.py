# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 07:12:39 2025

@author: peace
"""


"""
LLM API调用模块 - CoT版本
"""


from openai import OpenAI
import time


# 直接在这里设置API Key
OPENAI_API_KEY = "xxxxxxxx"
XAI_API_KEY = "xxxxxxxx"
GEMINI_API_KEY= "xxxxxxxx"


LLM_NAME = "gemini-2.5-pro"

def call_API(prompt, mode="scoring", max_retries=3, delay=1):
    """
    调用LLM API - 支持CoT两步调用
    
    Args:
        prompt (str): 要发送的prompt
        mode (str): 调用模式 - "rationale"(生成理由) 或 "scoring"(生成分数)
        max_retries (int): 最大重试次数
        delay (float): 重试间隔（秒）
        
    Returns:
        str: API返回的结果，如果失败返回None
    """
    # 创建客户端实例
    #client = OpenAI(api_key = OPENAI_API_KEY)
    #client = OpenAI(api_key = XAI_API_KEY, base_url="https://api.x.ai/v1", )
    client = OpenAI(api_key = GEMINI_API_KEY, 
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" )
    
    # 根据模式设置不同参数
    if mode == "rationale":
        #max_tokens = 400
        temperature = 0.7
        model_name = LLM_NAME
    else:  # scoring mode
        #max_tokens = 5
        temperature = 0
        model_name = LLM_NAME
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                #max_tokens=max_tokens,
                reasoning_effort="high",
                temperature=temperature
            )
            
            result = response.choices[0].message.content.strip()
            print(f"API调用成功 ({mode}模式)，返回: {result[:50]}...")
            return result
            
        except Exception as e:
            print(f"API调用失败 ({mode}模式, 尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"{mode}模式：所有重试都失败了")
                return None
    
    return None


def test_api():
    """测试API连接 - 测试两种模式"""
    print("测试rationale模式...")
    test_prompt_rationale = "Please analyze why the number 1 is important in mathematics. Give a brief explanation."
    result1 = call_API(test_prompt_rationale, mode="rationale")
    
    if result1:
        print(f"Rationale模式测试成功: {result1[:100]}...")
    else:
        print("Rationale模式测试失败")
        return False
    
    print("\n测试scoring模式...")
    test_prompt_scoring = "Please respond with just the number 1."
    result2 = call_API(test_prompt_scoring, mode="scoring")
    
    if result2:
        print(f"Scoring模式测试成功: {result2}")
        return True
    else:
        print("Scoring模式测试失败")
        return False


if __name__ == "__main__":
    test_api()