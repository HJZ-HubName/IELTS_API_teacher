# -*- coding: utf-8 -*-


"""
IELTS评分系统 - 数据处理模块
负责数据加载、验证、特征准备和数据划分
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import (
    FEATURE_COLUMNS, 
    TARGET_COLUMN, 
    AI_OVERALL_COLUMN,
    validate_file_exists,
    extract_model_name_from_path
)

class IELTSDataHandler:
    """IELTS评分数据处理器"""
    
    def __init__(self, data_path):
        """
        初始化数据处理器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.model_name = extract_model_name_from_path(data_path)
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """
        加载数据文件
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        validate_file_exists(self.data_path)
        
        try:
            # 加载数据
            data = pd.read_csv(self.data_path)
            print(f"原始数据形状: {data.shape}")
            
            # 检查空值情况
            print("\n=== 空值检查 ===")
            null_counts = data.isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                print(f"发现空值! 总计: {total_nulls} 个")
                print("各列空值统计:")
                for col, count in null_counts.items():
                    if count > 0:
                        print(f"  {col}: {count} 个空值")
                
                # 显示包含空值的行数
                rows_with_nulls = data.isnull().any(axis=1).sum()
                print(f"包含空值的行数: {rows_with_nulls} 行")
                
                # 删除包含空值的行
                print("\n清理空值...")
                data_cleaned = data.dropna()
                removed_rows = len(data) - len(data_cleaned)
                
                print(f"删除了 {removed_rows} 行包含空值的数据")
                print(f"清理后数据形状: {data_cleaned.shape}")
                
                self.data = data_cleaned
                
            else:
                self.data = data
                print("未发现空值")
            
            print(f"成功加载数据: {self.data_path}")
            print(f"数据集大小: {len(self.data)} 行 × {len(self.data.columns)} 列")
            return self.data
        except Exception as e:
            raise Exception(f"数据加载失败: {str(e)}")
    
    def validate_data(self):
        """
        验证数据质量和完整性
        
        Returns:
            dict: 数据验证报告
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        report = {}
        
        # 检查必要列是否存在
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, AI_OVERALL_COLUMN]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")
        
        # 数据质量统计
        report['total_samples'] = len(self.data)
        report['missing_values'] = self.data[required_columns].isnull().sum().to_dict()
        report['total_missing'] = self.data[required_columns].isnull().sum().sum()
        
        # 分数范围检查
        report['human_score_range'] = (
            float(self.data[TARGET_COLUMN].min()), 
            float(self.data[TARGET_COLUMN].max())
        )
        report['ai_overall_range'] = (
            float(self.data[AI_OVERALL_COLUMN].min()), 
            float(self.data[AI_OVERALL_COLUMN].max())
        )
        
        # 分数分布
        report['human_score_distribution'] = self.data[TARGET_COLUMN].value_counts().sort_index().to_dict()
        
        return report
    
    def prepare_features(self):
        """
        准备特征和目标变量
        
        Returns:
            tuple: (X, y) 特征矩阵和目标向量
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 移除缺失值（如果有的话）
        clean_data = self.data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        
        if len(clean_data) < len(self.data):
            removed_count = len(self.data) - len(clean_data)
            print(f"已移除包含缺失值的记录: {removed_count} 条")
        
        # 准备特征和目标
        X = clean_data[FEATURE_COLUMNS].copy()
        y = clean_data[TARGET_COLUMN].copy()
        
        print(f"特征维度: {X.shape}")
        print(f"目标变量维度: {y.shape}")
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集 (80/20)
        
        Args:
            test_size (float): 测试集比例，默认0.2
            random_state (int): 随机种子
            
        Returns:
            tuple: (train_data, test_data) 训练集和测试集DataFrame
        """
        
        self.train_data, self.test_data = train_test_split(
                self.data,
                test_size=test_size,
                random_state=random_state
            )
        
        print("数据划分完成:")
        
        return self.train_data, self.test_data
    
    def get_data_info(self):
        """
        获取数据基本信息
        
        Returns:
            dict: 数据信息字典
        """
        if self.data is None:
            return None
        
        return {
            'file_path': self.data_path,
            'model_name': self.model_name,
            'total_samples': len(self.data),
            'train_samples': len(self.train_data) if self.train_data is not None else None,
            'test_samples': len(self.test_data) if self.test_data is not None else None,
            'feature_columns': FEATURE_COLUMNS,
            'target_column': TARGET_COLUMN
        }

# 便捷函数
def quick_load_and_split(data_path, test_size=0.2, random_state=42):
    """
    快速加载数据并划分训练测试集
    
    Args:
        data_path (str): 数据文件路径
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (data_handler, train_data, test_data)
    """
    handler = IELTSDataHandler(data_path)
    handler.load_data()
    handler.validate_data()
    train_data, test_data = handler.split_data(test_size, random_state)
    
    return handler, train_data, test_data