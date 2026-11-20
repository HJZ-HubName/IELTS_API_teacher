# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 07:13:55 2025

@author: peace
"""


"""
IELTS评分Prompt生成器 - CoT版本
"""

from config import (
    RATIONALE_PROMPT_TEMPLATE, 
    COT_SCORING_PROMPT_TEMPLATE,
    OVERALL_RATIONALE_PROMPT_TEMPLATE,
    OVERALL_COT_SCORING_PROMPT_TEMPLATE,
    DIMENSIONS, 
    BAND_DESCRIPTORS
)
from api_client import call_API
import time


def generate_rationale_prompt(question, essay, dimension):
    """生成第一步：评分理由的prompt"""
    band_descriptors = BAND_DESCRIPTORS.get(dimension, "")
    
    return RATIONALE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        band_descriptors=band_descriptors,
        question=question,
        essay=essay
    )


def generate_cot_scoring_prompt(question, essay, rationale, dimension):
    """生成第二步：基于理由评分的prompt"""
    band_descriptors = BAND_DESCRIPTORS.get(dimension, "")
    
    return COT_SCORING_PROMPT_TEMPLATE.format(
        dimension=dimension,
        question=question,
        essay=essay,
        rationale=rationale,
        band_descriptors=band_descriptors
    )


def generate_overall_rationale_prompt(question, essay):
    """生成overall评分的理由prompt"""
    return OVERALL_RATIONALE_PROMPT_TEMPLATE.format(
        question=question,
        essay=essay
    )


def generate_overall_cot_scoring_prompt(question, essay, rationale):
    """生成overall评分的分数prompt"""
    return OVERALL_COT_SCORING_PROMPT_TEMPLATE.format(
        question=question,
        essay=essay,
        rationale=rationale
    )


def process_single_essay(essay_id, question, essay, delay=1.2):
    """
    处理单篇文章 - CoT两步法
    返回: (scores_dict, rationales_dict)
    """
    print(f"\n处理Essay {essay_id}...")
    
    scores = {}
    rationales = {}
    
    # 处理四个维度
    for dimension in DIMENSIONS:
        print(f"  评估维度: {dimension}")
        
        # 第一步：生成评分理由
        print("    第1步：生成评分理由...")
        rationale_prompt = generate_rationale_prompt(question, essay, dimension)
        rationale = call_API(rationale_prompt, mode="rationale")
        
        if rationale is None:
            print(f"    维度 {dimension} 理由生成失败")
            rationales[dimension] = "Failed to generate rationale"
            scores[dimension] = None
            continue
            
        rationales[dimension] = rationale
        time.sleep(delay)
        
        # 第二步：基于理由生成分数
        print("    第2步：基于理由生成分数...")
        scoring_prompt = generate_cot_scoring_prompt(question, essay, rationale, dimension)
        score_text = call_API(scoring_prompt, mode="scoring")
        
        if score_text is None:
            print(f"    维度 {dimension} 分数生成失败")
            scores[dimension] = None
            continue
        
        scores[dimension] = score_text.strip()
        print(f" {dimension}: {score_text.strip()}")
        
        time.sleep(delay)
    
    # 处理Overall评分
    print("  评估Overall...")
    
    # 第一步：生成overall理由
    print("    第1步：生成overall理由...")
    overall_rationale_prompt = generate_overall_rationale_prompt(question, essay)
    overall_rationale = call_API(overall_rationale_prompt, mode="rationale")
    
    if overall_rationale is None:
        print("    Overall理由生成失败")
        rationales["overall"] = "Failed to generate overall rationale"
        scores["overall"] = None
    else:
        rationales["overall"] = overall_rationale
        time.sleep(delay)
        
        # 第二步：生成overall分数
        print("    第2步：生成overall分数...")
        overall_scoring_prompt = generate_overall_cot_scoring_prompt(question, essay, overall_rationale)
        overall_score_text = call_API(overall_scoring_prompt, mode="scoring")
        
        if overall_score_text is None:
            print("    Overall分数生成失败")
            scores["overall"] = None
        else:
            scores["overall"] = overall_score_text.strip()
    
    return scores, rationales


def process_essays_batch(essays_data, delay=1.2):
    """
    批量处理文章
    
    Args:
        essays_data: 包含essay数据的列表，每个元素应有id, question, essay字段
        delay: API调用间隔
        
    Returns:
        tuple: (all_scores, all_rationales) 两个列表
    """
    all_scores = []
    all_rationales = []
    
    total_essays = len(essays_data)
    
    for i, essay_data in enumerate(essays_data, 1):
        print(f"\n=== 处理进度: {i}/{total_essays} ===")
        
        essay_id = essay_data.get('id', i)
        question = essay_data.get('question', '')
        essay = essay_data.get('essay', '')
        human_score = essay_data.get('human_score', '')
        
        if not essay.strip():
            print(f"Essay {essay_id}: 内容为空，跳过")
            continue
        
        # 处理单篇文章
        scores, rationales = process_single_essay(essay_id, question, essay, delay)
        
        # 准备分数数据
        score_row = {
            'essay_id': essay_id,
            'human_score': human_score,
            'ai_task_response': scores.get('Task Response'),
            'ai_coherence_and_cohesion': scores.get('Coherence and Cohesion'),
            'ai_lexical_resource': scores.get('Lexical Resource'),
            'ai_grammatical_range_and_accuracy': scores.get('Grammatical Range and Accuracy'),
            'ai_overall': scores.get('overall')
        }
        
        # 准备理由数据
        rationale_row = {
            'essay_id': essay_id,
            'human_score': human_score,
            'task_response_rationale': rationales.get('Task Response', ''),
            'coherence_rationale': rationales.get('Coherence and Cohesion', ''),
            'lexical_rationale': rationales.get('Lexical Resource', ''),
            'grammar_rationale': rationales.get('Grammatical Range and Accuracy', ''),
            'overall_rationale': rationales.get('overall', '')
        }
        
        all_scores.append(score_row)
        all_rationales.append(rationale_row)
        
        print(f"Essay {essay_id} 处理完成")
    
    return all_scores, all_rationales
