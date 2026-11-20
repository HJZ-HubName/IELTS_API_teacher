# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 07:14:31 2025

@author: peace
"""


"""
IELTS评分系统主程序 - CoT版本
"""


import csv
from datetime import datetime
from config import API_delay
from data_loader import load_ielts_dataset, get_essay_data
from prompt_generator import process_essays_batch
from api_client import test_api



def save_scores_to_csv(scores_data, filename):
    """
    保存评分结果到CSV文件
    
    Args:
        scores_data: 评分数据列表
        filename: 输出文件名
    """
    if not scores_data:
        print("没有评分数据可保存")
        return
    
    try:
        fieldnames = [
            'essay_id',
            'human_score',
            'ai_task_response',
            'ai_coherence_and_cohesion', 
            'ai_lexical_resource',
            'ai_grammatical_range_and_accuracy',
            'ai_overall'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in scores_data:
                writer.writerow(row)
        
        print(f"评分结果已保存到: {filename}")
        
    except Exception as e:
        print(f"保存评分结果失败: {e}")


def save_rationales_to_csv(rationales_data, filename):
    """
    保存评分理由到CSV文件
    
    Args:
        rationales_data: 评分理由数据列表
        filename: 输出文件名
    """
    if not rationales_data:
        print("没有评分理由数据可保存")
        return
    
    try:
        fieldnames = [
            'essay_id',
            'human_score',
            'task_response_rationale',
            'coherence_rationale',
            'lexical_rationale', 
            'grammar_rationale',
            'overall_rationale'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rationales_data:
                writer.writerow(row)
        
        print(f"评分理由已保存到: {filename}")
        
    except Exception as e:
        print(f"保存评分理由失败: {e}")


def main():
    """主程序"""
    print("=== IELTS评分系统 - CoT版本 ===\n")
    
    # 测试API连接
    print("测试API连接...")
    if not test_api():
        print("API连接失败，程序退出")
        return
    
    print("API连接正常\n")
    
    # 使用原有的data_loader加载数据
    print("加载数据集...")
    df = load_ielts_dataset()
    
    if df is None:
        print("数据加载失败，程序退出")
        return
    
    # 准备essays数据
    essays_data = []
    for i in range(len(df)):
        essay_data = get_essay_data(df, i)
        essays_data.append(essay_data)
    
    # 确认处理
    total_essays = len(essays_data)
    total_api_calls = total_essays * 10  # 每篇文章10次API调用（5个维度x2步）
    
    print(f"即将处理 {total_essays} 篇文章")
    print(f"预计API调用次数: {total_api_calls}")
    
    # 开始处理
    print("\n开始批量处理...")
    start_time = datetime.now()
    
    scores_data, rationales_data = process_essays_batch(essays_data, API_delay)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 生成文件名（带时间戳）
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    scores_filename = f"ielts_scoring_summary_{timestamp}.csv"
    rationales_filename = f"ielts_rationale_{timestamp}.csv"
    
    # 保存结果
    save_scores_to_csv(scores_data, scores_filename)
    save_rationales_to_csv(rationales_data, rationales_filename)
    
    # 打印统计
    print("\n=== 处理完成 ===")
    print(f"处理时间: {duration}")
    print(f"成功处理: {len(scores_data)} 篇文章")
    print("输出文件:")
    print(f"  - 评分结果: {scores_filename}")
    print(f"  - 评分理由: {rationales_filename}")

if __name__ == "__main__":
    main()