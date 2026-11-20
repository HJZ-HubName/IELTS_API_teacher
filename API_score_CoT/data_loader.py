# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:11:00 2025

@author: sctuser
"""


"""
数据加载器
用于读取和处理CSV数据集
"""


import os
import pandas as pd
import chardet
from config import File_Name

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR, File_Name)


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 读取前10万字节检测
    print(f"检测结果: {result}")
    return result['encoding']

def load_ielts_dataset(csv_path = file_path):
    """读取IELTS写作数据集"""
    try:
        df = pd.read_csv(csv_path, encoding='windows-1252')
        print(f"成功加载数据集，共 {len(df)} 篇文章")
        return df
    except FileNotFoundError:
        print(f"文件未找到: {csv_path}")
        return None
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None

def get_essay_data(df, index):
    """
    获取指定索引的文章数据
    
    Args:
        df: DataFrame
        index: 行索引 (0, 1, 2, ...)
        
    Returns:
        dict: 文章数据
    """
    return {
        'id': df.iloc[index]['No.'],
        'question': df.iloc[index]['Question'],
        'essay': df.iloc[index]['Essay'],
        'human_score': df.iloc[index]['Overall']
    }

if __name__ == "__main__":
    # 测试
    df = load_ielts_dataset()
    if df is not None:
        print(f"数据集shape: {df.shape}")
        print(f"第一篇文章ID: {df.iloc[0]['No.']}")