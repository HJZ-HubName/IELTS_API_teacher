# -*- coding: utf-8 -*-
"""
IELTS评分系统 - 工具函数模块
提供文件命名、常量定义、评估指标等基础功能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from rounding import apply_rounding

# ==========================================
# 数据列常量
# ==========================================
FEATURE_COLUMNS = [
    'ai_task_response',
    'ai_coherence_and_cohesion', 
    'ai_lexical_resource',
    'ai_grammatical_range_and_accuracy'
]

TARGET_COLUMN = 'human_score'
AI_OVERALL_COLUMN = 'ai_overall'

# 特征名称映射(用于可视化)
FEATURE_NAME_MAPPING = {
    'ai_task_response': 'Task Response (TR)',
    'ai_coherence_and_cohesion': 'Coherence and Cohesion (CC)',
    'ai_lexical_resource': 'Lexical Resource (LR)',
    'ai_grammatical_range_and_accuracy': 'Grammatical Range and Accuracy (GRA)'
}

# 方法名称映射
METHOD_NAME_MAPPING = {
    'M1': 'Arithmetic_Mean',
    'M2': 'LightGBM',
    'M3': 'Direct_LLM',
    'M4': 'Ridge_Regression'
}

# ==========================================
# 评估配置常量(新增)
# ==========================================
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_ROUNDING_MODE = 'nearest'
DEFAULT_ROUNDING_CONFIG = {
    'M1': DEFAULT_ROUNDING_MODE,
    'M2': DEFAULT_ROUNDING_MODE,
    'M4': DEFAULT_ROUNDING_MODE
}

# 分数段划分配置(新增)
# 低分段 [0, 5], 中分段 (5, 7], 高分段 (7, 9]
SCORE_BAND_CONFIG = {
    'low':    {'label': 'Low [0, 5]',    'condition': lambda s: s <= 5},
    'medium': {'label': 'Medium (5, 7]', 'condition': lambda s: 5 < s <= 7},
    'high':   {'label': 'High (7, 9]',   'condition': lambda s: s > 7}
}

# ==========================================
# 全局变量
# ==========================================
_current_output_folder = None

# ==========================================
# 分数处理函数
# ==========================================

def round_up_to_half(score):
    """IELTS向上进分:向上舍入到最近的整数或0.5"""
    if isinstance(score, (pd.Series, np.ndarray)):
        return np.ceil(score * 2) / 2
    else:
        return np.ceil(score * 2) / 2

def round_to_half(score):
    """最近半分,中点向上,并限制在 [0, 9]。"""
    return apply_rounding(score, 'nearest')

def apply_rounding_rule(predictions, rounding_mode=DEFAULT_ROUNDING_MODE):
    """统一预测后处理;none 用于保留 Direct LLM 的原始半分输出。"""
    if rounding_mode == 'none':
        return predictions
    return apply_rounding(predictions, rounding_mode)

# ==========================================
# 评估指标函数(新增)
# ==========================================

def compute_qwk(y_true, y_pred):
    """计算 Quadratic Weighted Kappa
    
    将分数(0.5步长)×2转为整数后计算。
    预测值若为连续值,先就近舍入到0.5步长。
    
    Args:
        y_true: 真实分数
        y_pred: 预测分数
        
    Returns:
        float: QWK值,范围[-1, 1],越高越好
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    # 就近舍入到0.5步长,然后×2转为整数
    y_true_int = np.round(y_true_arr * 2).astype(int)
    y_pred_int = np.round(y_pred_arr * 2).astype(int)
    
    # 限制在合法范围 [0, 18] (即IELTS 0-9分,×2)
    y_true_int = np.clip(y_true_int, 0, 18)
    y_pred_int = np.clip(y_pred_int, 0, 18)
    
    # 用统一labels防止边缘情况下label集合不一致
    labels = list(range(0, 19))
    
    try:
        qwk = cohen_kappa_score(y_true_int, y_pred_int, 
                                weights='quadratic', labels=labels)
        return float(qwk)
    except Exception as e:
        print(f"QWK计算失败: {str(e)}")
        return None

def compute_within_05_accuracy(y_true, y_pred):
    """计算预测误差在 ±0.5 内的样本比例
    
    Returns:
        float: 范围[0, 1],越高越好
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 0.5))

def compute_within_1_accuracy(y_true, y_pred):
    """计算预测误差在 ±1.0 内的样本比例(辅助指标)"""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 1.0))

# ==========================================
# 分数段分析函数(新增)
# ==========================================

def get_score_band(score):
    """根据分数返回所属分段
    
    分段定义:
        low:    [0, 5]
        medium: (5, 7]
        high:   (7, 9]
    """
    if score <= 5:
        return 'low'
    elif score <= 7:
        return 'medium'
    else:
        return 'high'

def compute_band_metrics(y_true, y_pred):
    """按分数段计算各项指标
    
    Args:
        y_true: 真实分数(用于分段)
        y_pred: 预测分数
        
    Returns:
        dict: 每个分段的指标字典
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    results = {}
    for band_name, band_info in SCORE_BAND_CONFIG.items():
        mask = np.array([band_info['condition'](s) for s in y_true_arr])
        n = int(mask.sum())
        
        if n == 0:
            results[band_name] = {
                'label': band_info['label'],
                'n_samples': 0,
                'mae': None,
                'rmse': None,
                'within_0_5_acc': None,
                'mean_error': None
            }
        else:
            true_vals = y_true_arr[mask]
            pred_vals = y_pred_arr[mask]
            errors = pred_vals - true_vals
            
            results[band_name] = {
                'label': band_info['label'],
                'n_samples': n,
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'within_0_5_acc': float(np.mean(np.abs(errors) <= 0.5)),
                'mean_error': float(np.mean(errors))  # 系统性偏差(>0=高估,<0=低估)
            }
    return results

# ==========================================
# 输出文件夹管理
# ==========================================

def create_output_folder(input_file_path, base_folder=None):
    """创建按时间命名的输出文件夹
    
    Args:
        input_file_path: 输入文件路径
        base_folder: 父级文件夹(支持嵌套结构)
        
    Returns:
        str: 输出文件夹路径
    """
    global _current_output_folder
    
    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{filename}_{timestamp}"
    
    if base_folder:
        folder_path = os.path.join(base_folder, folder_name)
    else:
        folder_path = folder_name
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建输出文件夹: {folder_path}")
    
    _current_output_folder = folder_path
    return folder_path

def set_output_folder(folder_path):
    """直接设置全局输出文件夹(新增,用于批处理)"""
    global _current_output_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    _current_output_folder = folder_path
    return folder_path

def get_output_folder():
    """获取当前输出文件夹"""
    return _current_output_folder

def generate_file_prefix(input_file_path):
    """根据输入文件路径生成输出文件名前缀"""
    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename}_{timestamp}"

def get_output_filepath(filename):
    """生成输出文件的完整路径"""
    if _current_output_folder is None:
        return filename
    return os.path.join(_current_output_folder, filename)

# ==========================================
# LLM/Prompting识别(新增)
# ==========================================

def extract_llm_name(file_path):
    """从文件路径提取LLM名称
    
    支持格式:
    - ielts_scoring_summary_20250709_grok-3_deduped.csv -> grok-3
    - ielts_scoring_summary_20250831_GPT-4.1-mini-CoT_deduped.csv -> GPT-4.1-mini
    """
    filename = os.path.splitext(os.path.basename(file_path))[0]
    name = filename.replace('ielts_scoring_summary_', '').replace('_deduped', '')
    
    # 移除日期前缀(8位数字)
    parts = name.split('_', 1)
    if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 8:
        name = parts[1]
    
    # 移除CoT后缀
    name = name.replace('-CoT', '').replace('_CoT', '')
    
    return name

def detect_prompting_strategy(file_path):
    """根据文件路径检测prompting策略
    
    判断依据:
    1. 文件路径中是否包含 'cot_all' 文件夹
    2. 文件名中是否包含 'CoT' 标识
    
    Returns:
        str: 'CoT' 或 'Direct'
    """
    path_normalized = file_path.replace('\\', '/').lower()
    if 'cot_all' in path_normalized:
        return 'CoT'
    
    filename_lower = os.path.basename(file_path).lower()
    if '-cot' in filename_lower or '_cot' in filename_lower:
        return 'CoT'
    
    return 'Direct'

def extract_model_name_from_path(file_path):
    """从文件路径提取模型名称(向后兼容)"""
    return extract_llm_name(file_path)

# ==========================================
# 文件操作工具
# ==========================================

def validate_file_exists(file_path):
    """验证文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    return True

def ensure_directory_exists(file_path):
    """确保文件所在目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def format_duration(seconds):
    """格式化时间持续长度"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def get_file_size_mb(file_path):
    """获取文件大小(MB)"""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path) / (1024 * 1024)

def validate_data_columns(df, required_columns=None):
    """验证DataFrame是否包含必要的列"""
    if required_columns is None:
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, AI_OVERALL_COLUMN]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns

def safe_round(value, decimals=3):
    """安全的数值四舍五入,处理None和NaN"""
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None

