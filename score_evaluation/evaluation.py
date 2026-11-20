# -*- coding: utf-8 -*-
"""
IELTS评分系统 - 评估模块
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy import stats

from utils import (
    generate_file_prefix, ensure_directory_exists, get_output_filepath,
    compute_band_metrics, SCORE_BAND_CONFIG
)


class IELTSEvaluator:
    """IELTS 评分结果评估器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.file_prefix = generate_file_prefix(data_path)
        self.model_name = os.path.splitext(os.path.basename(data_path))[0]

        self.comparison_results = None
        self.band_results = None
        self.significance_results = None

    # ==========================================
    # 1. 主对比报告(含 mean ± std + QWK + Within-0.5)
    # ==========================================

    def generate_comparison_report(self, all_results):
        """生成方法对比报告"""
        print("=== 生成方法对比报告 ===")

        rows = []
        for method_code, r in all_results.items():
            row = {
                '方法代码': r['method_code'],
                '方法名称': r['method_name'],
                '方法描述': r['method_description'],
                'Rounding_Mode': r['rounding_mode'],
                '评估方式': r.get('evaluation_data_type', 'OOF'),
                '样本数': r['evaluation_size'],
                'CV配置': f"{r['n_splits']}×{r['n_repeats']}",

                # 数值列(便于程序读取)
                'MAE_mean': r['mae_mean'],
                'MAE_std': r['mae_std'],
                'RMSE_mean': r['rmse_mean'],
                'RMSE_std': r['rmse_std'],
                'Pearson_r_mean': r['correlation_mean'],
                'Pearson_r_std': r['correlation_std'],
                'QWK_mean': r['qwk_mean'],
                'QWK_std': r['qwk_std'],
                'Within_0.5_mean': r['within_0_5_acc_mean'],
                'Within_0.5_std': r['within_0_5_acc_std'],

                # 论文风格的 mean±std 字符串(便于直接复制)
                'MAE': f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}",
                'RMSE': f"{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}",
                'Pearson r': f"{r['correlation_mean']:.3f}±{r['correlation_std']:.3f}",
                'QWK': f"{r['qwk_mean']:.3f}±{r['qwk_std']:.3f}",
                'Within-0.5': f"{r['within_0_5_acc_mean']:.3f}±{r['within_0_5_acc_std']:.3f}",

                '输入文件': os.path.basename(self.data_path),
                '评估时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 模型特有信息
            row['LightGBM_params'] = (
                str(r['best_params']) if method_code == 'LightGBM' and 'best_params' in r else None
            )
            row['Ridge_alpha_mean'] = r.get('alpha_mean') if method_code == 'RidgeRegression' else None

            rows.append(row)

        self.comparison_results = pd.DataFrame(rows)

        # 简洁打印
        display_cols = ['方法名称', 'MAE', 'RMSE', 'Pearson r', 'QWK', 'Within-0.5']
        print("\n--- 方法性能对比 ---")
        print(self.comparison_results[display_cols].to_string(index=False))
        return self.comparison_results

    def save_comparison_report(self):
        if self.comparison_results is None:
            return None
        filename = f"evaluation_metrics_{self.file_prefix}.csv"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        self.comparison_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"评估指标已保存: {filename}")
        return filepath

    # ==========================================
    # 2. 分数段误差分析
    # ==========================================

    def generate_band_report(self, all_results):
        """按分数段(low/medium/high)统计误差"""
        print("\n=== 生成分数段误差分析 ===")

        rows = []
        for method_code, r in all_results.items():
            method_name = r['method_name']
            y_true = r['oof_actual']
            y_pred = r['oof_predictions_first_repeat']

            band_metrics = compute_band_metrics(y_true, y_pred)

            for band_name, m in band_metrics.items():
                rows.append({
                    '方法代码': method_code,
                    '方法名称': method_name,
                    'Rounding_Mode': r['rounding_mode'],
                    '分数段': m['label'],
                    '分段代码': band_name,
                    '样本数': m['n_samples'],
                    'MAE': m['mae'],
                    'RMSE': m['rmse'],
                    'Within_0.5_Acc': m['within_0_5_acc'],
                    '平均偏差(>0高估,<0低估)': m['mean_error'],
                    '输入文件': os.path.basename(self.data_path)
                })

        self.band_results = pd.DataFrame(rows)

        # 透视表展示 MAE
        try:
            pivot = self.band_results.pivot_table(
                index='方法名称', columns='分数段', values='MAE', aggfunc='first'
            )
            band_order = [SCORE_BAND_CONFIG[b]['label'] for b in ['low', 'medium', 'high']]
            pivot = pivot[[c for c in band_order if c in pivot.columns]]
            print("\n--- MAE by Score Band ---")
            print(pivot.to_string(float_format='%.3f'))
        except Exception:
            pass

        # 透视表展示 平均偏差(揭示系统性高估/低估)
        try:
            pivot_bias = self.band_results.pivot_table(
                index='方法名称', columns='分数段',
                values='平均偏差(>0高估,<0低估)', aggfunc='first'
            )
            pivot_bias = pivot_bias[[c for c in band_order if c in pivot_bias.columns]]
            print("\n--- Mean Error by Score Band (>0=高估, <0=低估) ---")
            print(pivot_bias.to_string(float_format='%+.3f'))
        except Exception:
            pass

        return self.band_results

    def save_band_report(self):
        if self.band_results is None:
            return None
        filename = f"score_band_analysis_{self.file_prefix}.csv"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        self.band_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"分数段分析已保存: {filename}")
        return filepath

    # ==========================================
    # 3. 配对显著性检验
    # ==========================================

    def generate_significance_report(self, all_results, metric='mae'):
        """对所有方法对做配对 t 检验

        说明:对确定性方法(M1/M3),per_repeat 指标恒等,
        会自动退化为常数比较(test_used='Constant')。
        """
        print(f"\n=== 配对 t 检验 (指标: {metric.upper()}) ===")

        method_codes = list(all_results.keys())
        per_repeat = {
            code: np.array([m[metric] for m in all_results[code]['per_repeat_metrics']])
            for code in method_codes
        }

        rows = []
        for i, m1 in enumerate(method_codes):
            for m2 in method_codes[i + 1:]:
                v1, v2 = per_repeat[m1], per_repeat[m2]
                diff = v1 - v2

                # 检测差异是否为常数(M1/M3 deterministic)
                if len(diff) <= 1 or np.std(diff, ddof=1) < 1e-10:
                    test_used = "Constant"
                    t_stat = np.nan
                    p_value = 1.0 if abs(np.mean(diff)) < 1e-10 else 0.0
                else:
                    try:
                        t_stat, p_value = stats.ttest_rel(v1, v2)
                        test_used = "Paired t-test"
                        if np.isnan(t_stat):
                            t_stat, p_value = 0.0, 1.0
                    except Exception:
                        t_stat, p_value = np.nan, np.nan
                        test_used = "Failed"

                # 显著性符号
                if pd.isna(p_value):
                    sig = "N/A"
                elif p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                rows.append({
                    '方法A': all_results[m1]['method_name'],
                    '方法B': all_results[m2]['method_name'],
                    '指标': metric.upper(),
                    f'A_{metric}_mean': float(np.mean(v1)),
                    f'B_{metric}_mean': float(np.mean(v2)),
                    '差异_mean(A-B)': float(np.mean(diff)),
                    '差异_std': float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                    '检验方法': test_used,
                    't_statistic': None if pd.isna(t_stat) else float(t_stat),
                    'p_value': None if pd.isna(p_value) else float(p_value),
                    '显著性': sig
                })

        self.significance_results = pd.DataFrame(rows)

        # 简洁打印
        display = self.significance_results[
            ['方法A', '方法B', '差异_mean(A-B)', 'p_value', '显著性']
        ].copy()
        display['差异_mean(A-B)'] = display['差异_mean(A-B)'].apply(lambda x: f"{x:+.4f}")
        display['p_value'] = display['p_value'].apply(
            lambda x: f"{x:.4f}" if x is not None and not pd.isna(x) else "N/A"
        )
        print(display.to_string(index=False))
        print("注: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著, N/A 不可用")

        return self.significance_results

    def save_significance_report(self):
        if self.significance_results is None:
            return None
        filename = f"significance_test_{self.file_prefix}.csv"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        self.significance_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"显著性检验已保存: {filename}")
        return filepath

    # ==========================================
    # 4. 特征重要性
    # ==========================================

    def save_feature_importance(self, all_results):
        rows = []
        for method_code in ['LightGBM', 'RidgeRegression']:
            if method_code not in all_results:
                continue
            r = all_results[method_code]
            if 'feature_importance' not in r:
                continue

            method_name = r['method_name']

            for feat, score in r['feature_importance'].items():
                rows.append({
                    'Method': method_name,
                    'Method_Code': method_code,
                    'Rounding_Mode': r['rounding_mode'],
                    'Feature_Name': feat,
                    'Importance_Score': float(score),
                    'Raw_File': os.path.basename(self.data_path),
                    'Evaluation_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        if not rows:
            return None

        df = pd.DataFrame(rows).sort_values(
            ['Method', 'Importance_Score'], ascending=[True, False]
        )
        filename = f"feature_importance_{self.file_prefix}.csv"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"特征重要性已保存: {filename}")
        return filepath

    # ==========================================
    # 5. 详细预测结果(OOF)
    # ==========================================

    def export_detailed_results(self, all_results, data):
        """导出每个样本的详细预测结果(用第一次 repeat 的 OOF 预测)"""
        print("\n=== 导出详细结果 ===")

        detailed = data.copy().reset_index(drop=True)

        for method_code, r in all_results.items():
            method_name = r['method_name']
            pred = np.asarray(r['oof_predictions_first_repeat'])
            actual = np.asarray(r['oof_actual'])

            detailed[f'{method_code}_{method_name}_pred'] = pred
            detailed[f'{method_code}_{method_name}_raw_pred'] = r['all_raw_predictions_matrix'][:, 0]
            detailed[f'{method_code}_{method_name}_abs_error'] = np.abs(actual - pred)
            detailed[f'{method_code}_{method_name}_rounding_mode'] = r['rounding_mode']

        detailed['Input_File'] = os.path.basename(self.data_path)
        detailed['Evaluation_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detailed['Total_Samples'] = len(data)

        filename = f"detailed_results_{self.file_prefix}.csv"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        detailed.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"详细结果已保存: {filename}")
        return filepath, detailed

    def export_oof_predictions(self, all_results, data):
        """保存全部 repeat 的原始/最终预测,便于核对及重算后处理。"""
        first = next(iter(all_results.values()))
        arrays = {
            'sample_index': np.arange(len(data)),
            'actual': np.asarray(first['oof_actual']),
            'n_splits': first['n_splits'],
            'n_repeats': first['n_repeats'],
            'random_state': first['random_state'],
            'input_file': os.path.basename(self.data_path)
        }
        if 'essay_id' in data.columns:
            arrays['essay_id'] = data['essay_id'].astype(str).to_numpy(dtype=str)
        for code, r in all_results.items():
            arrays[f'{code}_raw'] = r['all_raw_predictions_matrix']
            arrays[f'{code}_pred'] = r['all_predictions_matrix']
            arrays[f'{code}_rounding_mode'] = r['rounding_mode']
        filename = f"oof_predictions_{self.file_prefix}.npz"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        np.savez_compressed(filepath, **arrays)
        print(f"全部 OOF 预测已保存: {filename}")
        return filepath

    # ==========================================
    # 6. 汇总统计
    # ==========================================

    def generate_summary_statistics(self, all_results):
        print("\n=== 汇总统计 ===")
        print(f"评估模型: {self.model_name}")
        print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        first = list(all_results.values())[0]
        print(f"样本数: {first['evaluation_size']}")
        print(f"评估方式: {first['n_splits']}-fold × {first['n_repeats']}-repeat OOF")

        print("\n半分取整规则 (none = 保留原始输出):")
        for r in all_results.values():
            print(f"  {r['method_name']}: {r['rounding_mode']}")

        # 性能排名
        print("\n性能排名(按 MAE_mean):")
        ranking = sorted(
            [(r['method_name'], r['mae_mean'], r['mae_std'])
             for r in all_results.values()],
            key=lambda x: x[1]
        )
        for i, (name, m, s) in enumerate(ranking, 1):
            print(f"  {i}. {name}: {m:.3f} ± {s:.3f}")

        # Ridge alpha 信息
        if 'RidgeRegression' in all_results:
            ridge = all_results['RidgeRegression']
            if 'alpha_mean' in ridge:
                print(f"\nRidge α: {ridge['alpha_mean']:.4f} ± {ridge['alpha_std']:.4f}")


# ==========================================
# 便捷接口
# ==========================================

def quick_evaluate_all(all_results, data_path, data):
    """快速评估并保存所有结果

    Args:
        all_results: 所有模型结果
        data_path: 数据文件路径
        data: 完整数据(用于详细结果导出)

    Returns:
        (evaluator, saved_files)
    """
    evaluator = IELTSEvaluator(data_path)
    saved_files = []

    # 1. 主对比报告
    evaluator.generate_comparison_report(all_results)
    f = evaluator.save_comparison_report()
    if f:
        saved_files.append(f)

    # 2. 分数段报告
    evaluator.generate_band_report(all_results)
    f = evaluator.save_band_report()
    if f:
        saved_files.append(f)

    # 3. 配对 t 检验(MAE)
    evaluator.generate_significance_report(all_results, metric='mae')
    f = evaluator.save_significance_report()
    if f:
        saved_files.append(f)

    # 4. 特征重要性
    f = evaluator.save_feature_importance(all_results)
    if f:
        saved_files.append(f)

    # 5. 详细结果
    f, _ = evaluator.export_detailed_results(all_results, data)
    if f:
        saved_files.append(f)

    # 6. 汇总统计
    saved_files.append(evaluator.export_oof_predictions(all_results, data))
    evaluator.generate_summary_statistics(all_results)

    return evaluator, saved_files
