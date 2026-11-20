# -*- coding: utf-8 -*-
"""
IELTS评分系统 - 模型实现模块
四种评分方法,统一在 RepeatedKFold OOF 框架下评估
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from utils import (
    FEATURE_COLUMNS, TARGET_COLUMN, AI_OVERALL_COLUMN,
    apply_rounding_rule,
    compute_qwk, compute_within_05_accuracy,
    DEFAULT_N_SPLITS, DEFAULT_N_REPEATS, DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_MODE, DEFAULT_ROUNDING_CONFIG
)


# ==========================================
# 基类
# ==========================================

class BaseModel:
    """模型基类,统一定义 RepeatedKFold OOF 评估流程"""

    def __init__(self, method_code, method_name, rounding_mode=DEFAULT_ROUNDING_MODE):
        if rounding_mode not in ('nearest', 'ceil', 'none'):
            raise ValueError(f"未知的半分取整规则: {rounding_mode}")
        self.method_code = method_code
        self.method_name = method_name
        self.rounding_mode = rounding_mode
        self.method_description = ""
        self.results = {}

    # ----- 子类需实现的接口 -----
    def fit_fold(self, fold_train_data):
        """在一个 fold 的训练折上拟合(默认 no-op,适用于 M1/M3)"""
        pass

    def predict_fold(self, fold_eval_data):
        """对一个 fold 的验证折预测"""
        raise NotImplementedError

    def post_fold_hook(self, fold_idx, repeat_idx, fold_train, fold_val, fold_pred):
        """fold 结束后的钩子,用于子类记录额外信息"""
        pass

    # ----- 评估指标 -----
    def evaluate(self, y_true, y_pred):
        """计算所有指标(MAE, RMSE, MSE, Pearson r, QWK, Within-0.5)"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        try:
            corr, p_val = pearsonr(y_true, y_pred)
            if np.isnan(corr):
                corr, p_val = 0.0, 1.0
        except Exception:
            corr, p_val = 0.0, 1.0

        qwk = compute_qwk(y_true, y_pred)
        if qwk is None or np.isnan(qwk):
            qwk = 0.0
        within_05 = compute_within_05_accuracy(y_true, y_pred)

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'correlation': float(corr),
            'p_value': float(p_val),
            'qwk': float(qwk),
            'within_0_5_acc': float(within_05)
        }

    # ----- 主评估流程 -----
    def run_repeated_cv(self, data,
                        n_splits=DEFAULT_N_SPLITS,
                        n_repeats=DEFAULT_N_REPEATS,
                        random_state=DEFAULT_RANDOM_STATE):
        """统一的 RepeatedKFold OOF 评估"""
        print(f"=== {self.method_name} RepeatedKFold 评估 ===")
        print(f"配置: {n_splits}-fold × {n_repeats}-repeat (共 {n_splits * n_repeats} folds)")
        print(f"半分取整规则: {self.rounding_mode}")

        n_samples = len(data)
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                            random_state=random_state)

        # 预测矩阵: n_samples × n_repeats (每个样本被预测 n_repeats 次)
        all_predictions = np.full((n_samples, n_repeats), np.nan)
        all_raw_predictions = np.full((n_samples, n_repeats), np.nan)
        per_repeat_metrics = []

        indices = np.arange(n_samples)
        y_true_full = data[TARGET_COLUMN].values

        for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(indices)):
            repeat_idx = fold_idx // n_splits
            within_repeat = fold_idx % n_splits

            fold_train = data.iloc[train_idx].reset_index(drop=True)
            fold_val = data.iloc[val_idx].reset_index(drop=True)

            # 训练 + 预测
            self.fit_fold(fold_train)
            fold_pred = np.asarray(self.predict_fold(fold_val))
            all_raw_predictions[val_idx, repeat_idx] = fold_pred
            fold_pred = apply_rounding_rule(fold_pred, self.rounding_mode)

            all_predictions[val_idx, repeat_idx] = fold_pred

            # 子类钩子(用于记录特征重要性等)
            self.post_fold_hook(fold_idx, repeat_idx, fold_train, fold_val, fold_pred)

            # 该 repeat 的最后一个 fold 完成时,计算该 repeat 的整体指标
            if within_repeat == n_splits - 1:
                repeat_pred = all_predictions[:, repeat_idx]
                metrics = self.evaluate(y_true_full, repeat_pred)
                per_repeat_metrics.append(metrics)

                if (repeat_idx + 1) % 5 == 0 or repeat_idx == n_repeats - 1:
                    print(f"  Repeat {repeat_idx + 1}/{n_repeats}: "
                          f"MAE={metrics['mae']:.3f}, "
                          f"r={metrics['correlation']:.3f}, "
                          f"QWK={metrics['qwk']:.3f}")

        # 聚合 mean ± std
        aggregated = self._aggregate_metrics(per_repeat_metrics)

        # 整合结果
        self.results = {
            'method_code': self.method_code,
            'method_name': self.method_name,
            'method_description': self.method_description,
            'rounding_mode': self.rounding_mode,
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'random_state': random_state,
            'evaluation_size': n_samples,
            'per_repeat_metrics': per_repeat_metrics,        # 配对t检验用
            'all_predictions_matrix': all_predictions,        # n_samples × n_repeats
            'all_raw_predictions_matrix': all_raw_predictions,
            'oof_predictions_first_repeat': all_predictions[:, 0],
            'oof_actual': y_true_full,
            **aggregated
        }

        # 向后兼容字段(供 evaluation.py / visualization.py 使用)
        self.results['test_predictions'] = all_predictions[:, 0]
        self.results['test_actual'] = y_true_full
        self.results['test_size'] = n_samples
        self.results['evaluation_data_type'] = 'OOF (RepeatedKFold)'

        self._print_summary()
        return self.results

    def _aggregate_metrics(self, per_repeat_metrics):
        """从多次重复的指标聚合为 mean ± std"""
        agg = {}
        keys = ['mae', 'rmse', 'mse', 'correlation', 'qwk', 'within_0_5_acc']

        for k in keys:
            values = np.array([m[k] for m in per_repeat_metrics])
            agg[f'{k}_mean'] = float(np.mean(values))
            agg[f'{k}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            # 兼容旧字段:主指标 = mean
            agg[k] = agg[f'{k}_mean']

        agg['p_value'] = float(np.mean([m['p_value'] for m in per_repeat_metrics]))
        return agg

    def _print_summary(self):
        """打印汇总指标"""
        r = self.results
        print(f"--- {self.method_name} 汇总(mean ± std) ---")
        print(f"  MAE:            {r['mae_mean']:.3f} ± {r['mae_std']:.3f}")
        print(f"  RMSE:           {r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}")
        print(f"  Pearson r:      {r['correlation_mean']:.3f} ± {r['correlation_std']:.3f}")
        print(f"  QWK:            {r['qwk_mean']:.3f} ± {r['qwk_std']:.3f}")
        print(f"  Within-0.5 Acc: {r['within_0_5_acc_mean']:.3f} ± {r['within_0_5_acc_std']:.3f}")
        print()


# ==========================================
# M1: 算术平均
# ==========================================

class ArithmeticMeanModel(BaseModel):
    """方法一:四个子维度算术平均"""

    def __init__(self, rounding_mode=DEFAULT_ROUNDING_MODE):
        super().__init__('M1', 'Arithmetic_Mean', rounding_mode)
        self.method_description = (
            f'四个子维度分数的算术平均'
            f'(半分取整: {rounding_mode})'
        )

    def predict_fold(self, fold_eval_data):
        return fold_eval_data[FEATURE_COLUMNS].mean(axis=1).values


# ==========================================
# M3: Direct LLM
# ==========================================

class DirectLLMModel(BaseModel):
    """方法三:LLM 直接给出整体评分(0.5 步长,保留原值)"""

    def __init__(self):
        super().__init__('M3', 'Direct_LLM', rounding_mode='none')
        self.method_description = 'LLM 直接给出的整体评分(0.5 步长)'

    def predict_fold(self, fold_eval_data):
        return fold_eval_data[AI_OVERALL_COLUMN].values


# ==========================================
# M2: LightGBM
# ==========================================

class LightGBMModel(BaseModel):
    """方法二:LightGBM 聚合器(固定超参数,避免数据泄漏)"""

    # 固定超参数:沿用 LightGBM 接近默认的稳健配置
    FIXED_PARAMS = {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 5
    }

    def __init__(self, rounding_mode=DEFAULT_ROUNDING_MODE, random_state=DEFAULT_RANDOM_STATE):
        super().__init__('M2', 'LightGBM', rounding_mode)
        self.random_state = random_state
        self.method_description = (
            f'LightGBM 梯度提升树聚合器(固定超参数 + early stopping)'
            f'(半分取整: {rounding_mode})'
        )
        self.feature_importance_per_fold = []
        self._current_model = None

    def fit_fold(self, fold_train_data):
        """在外层 fold 训练折上训练(85/15 + early stopping)"""
        X = fold_train_data[FEATURE_COLUMNS].values
        y = fold_train_data[TARGET_COLUMN].values

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )

        lgb_params = {
            'objective': 'regression', 'metric': 'mae',
            'boosting_type': 'gbdt', 'verbose': -1,
            'random_state': self.random_state,
            **self.FIXED_PARAMS
        }

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)

        self._current_model = lgb.train(
            lgb_params, train_set,
            valid_sets=[val_set], num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

    def predict_fold(self, fold_eval_data):
        if self._current_model is None:
            raise ValueError("请先调用 fit_fold")
        X = fold_eval_data[FEATURE_COLUMNS].values
        return self._current_model.predict(X)

    def post_fold_hook(self, fold_idx, repeat_idx, fold_train, fold_val, fold_pred):
        """记录每个 fold 的特征重要性"""
        if self._current_model is not None:
            importance = dict(zip(
                FEATURE_COLUMNS,
                self._current_model.feature_importance(importance_type='gain')
            ))
            self.feature_importance_per_fold.append(importance)

    def run_repeated_cv(self, data,
                        n_splits=DEFAULT_N_SPLITS,
                        n_repeats=DEFAULT_N_REPEATS,
                        random_state=DEFAULT_RANDOM_STATE):
        # 直接调用父类,不再做超参数选择
        results = super().run_repeated_cv(data, n_splits, n_repeats, random_state)

        # 添加 LightGBM 特有信息
        if self.feature_importance_per_fold:
            avg_importance = {}
            for feat in FEATURE_COLUMNS:
                avg_importance[feat] = float(np.mean(
                    [fi[feat] for fi in self.feature_importance_per_fold]
                ))
            results['feature_importance'] = avg_importance

        results['fixed_params'] = self.FIXED_PARAMS
        return results

# ==========================================
# M4: Ridge
# ==========================================

class RidgeModel(BaseModel):
    """方法四:Ridge 回归

    使用 RidgeCV 在每个外层 fold 内独立选择 α(无数据泄漏)
    """

    def __init__(self, rounding_mode=DEFAULT_ROUNDING_MODE,
                 alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
        super().__init__('M4', 'Ridge_Regression', rounding_mode)
        self.alphas = list(alphas)
        self.method_description = (
            f'Ridge 回归(L2 正则化,RidgeCV 自动选 α)'
            f'(半分取整: {rounding_mode})'
        )

        self.fold_alphas = []
        self.fold_coefs = []
        self._current_model = None
        self._current_scaler = None

    def fit_fold(self, fold_train_data):
        X = fold_train_data[FEATURE_COLUMNS].values
        y = fold_train_data[TARGET_COLUMN].values

        self._current_scaler = StandardScaler()
        X_scaled = self._current_scaler.fit_transform(X)

        self._current_model = RidgeCV(alphas=self.alphas, cv=5)
        self._current_model.fit(X_scaled, y)

    def predict_fold(self, fold_eval_data):
        X = fold_eval_data[FEATURE_COLUMNS].values
        X_scaled = self._current_scaler.transform(X)
        return self._current_model.predict(X_scaled)

    def post_fold_hook(self, fold_idx, repeat_idx, fold_train, fold_val, fold_pred):
        if self._current_model is not None:
            self.fold_alphas.append(float(self._current_model.alpha_))
            self.fold_coefs.append(self._current_model.coef_.copy())

    def run_repeated_cv(self, data,
                        n_splits=DEFAULT_N_SPLITS,
                        n_repeats=DEFAULT_N_REPEATS,
                        random_state=DEFAULT_RANDOM_STATE):
        results = super().run_repeated_cv(data, n_splits, n_repeats, random_state)

        # 添加 Ridge 特有信息
        if self.fold_alphas:
            results['alpha_mean'] = float(np.mean(self.fold_alphas))
            results['alpha_std'] = float(np.std(self.fold_alphas, ddof=1))
            results['alpha_per_fold'] = self.fold_alphas

            # 平均绝对系数作为特征重要性
            avg_abs_coefs = np.mean(np.abs(np.array(self.fold_coefs)), axis=0)
            results['feature_importance'] = dict(zip(FEATURE_COLUMNS, avg_abs_coefs))
            results['avg_coefficients'] = dict(zip(
                FEATURE_COLUMNS,
                np.mean(np.array(self.fold_coefs), axis=0)
            ))

            print(f"  Alpha 选择: mean = {results['alpha_mean']:.4f}, "
                  f"std = {results['alpha_std']:.4f}")
            # 显示 alpha 分布
            from collections import Counter
            alpha_counter = Counter(self.fold_alphas)
            print(f"  Alpha 分布: {dict(alpha_counter)}")
            print()

        return results


# ==========================================
# 工厂函数与批量运行
# ==========================================

def create_model(method_code, rounding_mode=DEFAULT_ROUNDING_MODE):
    """模型工厂函数"""
    if method_code == 'ArithmeticMean':
        return ArithmeticMeanModel(rounding_mode=rounding_mode)
    elif method_code == 'LightGBM':
        return LightGBMModel(rounding_mode=rounding_mode)
    elif method_code == 'DirectLLM':
        return DirectLLMModel()
    elif method_code == 'RidgeRegression':
        return RidgeModel(rounding_mode=rounding_mode)
    else:
        raise ValueError(f"未知的方法代码: {method_code}")


def run_all_models(data, rounding_config=None,
                   n_splits=DEFAULT_N_SPLITS,
                   n_repeats=DEFAULT_N_REPEATS,
                   random_state=DEFAULT_RANDOM_STATE):
    """运行所有四种模型,统一在 RepeatedKFold OOF 框架下评估

    Args:
        data: 全部数据(不再需要预先 train/test 划分)
        rounding_config: M1/M2/M4 的半分取整模式,默认均为 nearest
        n_splits: K 折数
        n_repeats: 重复次数
        random_state: 随机种子

    Returns:
        dict: 所有模型的结果
    """
    rounding_config = {**DEFAULT_ROUNDING_CONFIG, **(rounding_config or {})}

    print("=" * 60)
    print("IELTS 写作自动评分实验 (统一 RepeatedKFold OOF 框架)")
    print(f"数据规模: {len(data)} 条")
    print(f"评估配置: {n_splits}-fold × {n_repeats}-repeat")
    print("=" * 60)

    all_results = {}

    # M1: Arithmetic Mean
    print("\n[1/4] Arithmetic Mean")
    print("-" * 50)
    m1 = ArithmeticMeanModel(rounding_mode=rounding_config['M1'])
    all_results['ArithmeticMean'] = m1.run_repeated_cv(
        data, n_splits, n_repeats, random_state
    )

    # M3: Direct LLM
    print("\n[2/4] Direct LLM")
    print("-" * 50)
    m3 = DirectLLMModel()
    all_results['DirectLLM'] = m3.run_repeated_cv(
        data, n_splits, n_repeats, random_state
    )

    # M2: LightGBM
    print("\n[3/4] LightGBM")
    print("-" * 50)
    m2 = LightGBMModel(rounding_mode=rounding_config['M2'],
                       random_state=random_state)
    all_results['LightGBM'] = m2.run_repeated_cv(
        data, n_splits, n_repeats, random_state
    )

    # M4: Ridge
    print("\n[4/4] Ridge Regression")
    print("-" * 50)
    m4 = RidgeModel(rounding_mode=rounding_config['M4'])
    all_results['RidgeRegression'] = m4.run_repeated_cv(
        data, n_splits, n_repeats, random_state
    )

    print("\n" + "=" * 60)
    print("所有模型评估完成")
    print("=" * 60)

    return all_results
