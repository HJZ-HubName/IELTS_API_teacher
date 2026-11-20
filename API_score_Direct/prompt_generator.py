# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:13:52 2025

@author: sctuser
"""


"""
Prompt生成器 - 独立处理版本
"""


import pandas as pd
from datetime import datetime
from config import (
    DIMENSIONAL_PROMPT_TEMPLATE, 
    OVERALL_PROMPT_TEMPLATE, 
    DIMENSIONS, 
    BAND_DESCRIPTORS,
    API_delay   )
from data_loader import load_ielts_dataset, get_essay_data
from api_client import call_gpt4o, test_api
import time
import json

def generate_dimensional_prompt(dimension, question, essay):
    """生成单个维度的评分prompt"""
    band_descriptors = BAND_DESCRIPTORS[dimension]
    
    return DIMENSIONAL_PROMPT_TEMPLATE.format(
        dimension=dimension,
        band_descriptors=band_descriptors,
        question=question,
        essay=essay
    )

def generate_overall_prompt(question, essay):
    """生成overall评分prompt"""
    return OVERALL_PROMPT_TEMPLATE.format(
        question=question,
        essay=essay
    )

def process_single_essay(essay_data):
    """
    处理单篇文章，生成5个prompt并分别调用API
    
    Args:
        essay_data (dict): 文章数据
        
    Returns:
        dict: 包含所有评分结果的字典
    """
    essay_id = essay_data['id']
    question = essay_data['question']
    essay = essay_data['essay']
    human_score = essay_data.get('human_score')
    
    print(f"\n开始处理文章 {essay_id}")
    print(f"题目: {question[:50]}...")
    
    results = {
        'essay_id': essay_id,
        'question': question,
        'essay': essay,
        'human_score': human_score,
        'ai_scores': {},
        'prompts': {}
    }
    
    # 处理4个维度
    for i, dimension in enumerate(DIMENSIONS, 1):
        print(f"\n--- 处理维度 {i}/4: {dimension} ---")
        
        # 生成prompt
        prompt = generate_dimensional_prompt(dimension, question, essay)
        #results['prompts'][dimension] = prompt
        
        print("调用API评分...")
        # 调用API
        ai_score = call_gpt4o(prompt)
        # 添加延迟，避免请求过于频繁
        time.sleep(API_delay)  # 延迟2秒
        
        results['ai_scores'][dimension] = ai_score
        print(f"{dimension}: {ai_score}")
    
    # 处理Overall评分
    print("\n--- 处理维度 5/5: Overall ---")
    
    # 生成overall prompt
    overall_prompt = generate_overall_prompt(question, essay)
    #results['prompts']['Overall'] = overall_prompt
    
    print("调用API评分...")
    # 调用API
    overall_score = call_gpt4o(overall_prompt)
    # 在Overall评分后也添加延迟
    time.sleep(API_delay)
    
    results['ai_scores']['Overall'] = overall_score
    print(f"Overall: {overall_score}")
    print(f"人工评分: {human_score}")
    
    return results

def save_results(all_results, filename=None):
    """
    保存结果到文件
    
    Args:
        all_results (list): 所有结果列表
        filename (str): 保存文件名，如果为None则自动生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ielts_scoring_results_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, 
                      default=str)  # 添加这个参数，将无法序列化的对象转为字符串
        print(f"结果已保存到: {filename}")
    except Exception as e:
        print(f"保存失败: {e}")

def create_results_summary(all_results):
    """
    创建结果汇总表
    
    Args:
        all_results (list): 所有结果列表
        
    Returns:
        pandas.DataFrame: 汇总表
    """
    summary_data = []
    
    for result in all_results:
        row = {
            'essay_id': result['essay_id'],
            'human_score': result['human_score']
        }
        
        # 添加AI评分
        for dimension in DIMENSIONS:
            row[f'ai_{dimension.replace(" ", "_").lower()}'] = result['ai_scores'].get(dimension)
        
        row['ai_overall'] = result['ai_scores'].get('Overall')
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def main():
    """主函数：处理所有文章"""
    # 测试API连接
    print("测试API连接...")
    if not test_api():
        print("API连接失败，请检查API Key")
        return
    
    # 加载数据集
    df = load_ielts_dataset()
    if df is None:
        return
    
    all_results = []
    
    # 处理每篇文章
    for i in range(len(df)):
        time.sleep(API_delay/3)
        essay_data = get_essay_data(df, i)
        
        try:
            # 处理单篇文章（生成5个prompt，分别调用API）
            result = process_single_essay(essay_data)
            all_results.append(result)
            
            print(f"文章 {essay_data['id']} 处理完成")
            
        except Exception as e:
            print(f"处理文章 {essay_data['id']} 时出错: {e}")
            continue
    
    # 保存结果
    save_results(all_results)
    
    # 创建汇总表
    summary_df = create_results_summary(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"ielts_scoring_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"汇总表已保存到: {summary_filename}")
    
    print(f"\n全部完成！处理了 {len(all_results)} 篇文章")

if __name__ == "__main__":
    main()