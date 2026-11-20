# -*- coding: utf-8 -*-
"""
IELTS评分系统 - 可视化模块
适配 RepeatedKFold OOF 框架,新增分数段图、误差箱型图等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from utils import (
    generate_file_prefix, ensure_directory_exists, get_output_filepath,
    FEATURE_NAME_MAPPING, SCORE_BAND_CONFIG, compute_band_metrics
)

# 样式设置(英文标签,无需中文字体)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class IELTSVisualizer:
    """IELTS 评分可视化器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.file_prefix = generate_file_prefix(data_path)
        self.model_name = os.path.splitext(os.path.basename(data_path))[0]

        self.method_configs = {
            'ArithmeticMean':  {'name': 'Arithmetic Mean',  'color': '#2E86AB'},
            'LightGBM':        {'name': 'LightGBM',         'color': '#A23B72'},
            'DirectLLM':       {'name': 'Direct LLM',       'color': '#F18F01'},
            'RidgeRegression': {'name': 'Ridge Regression', 'color': '#2CA02C'}
        }
        self.method_order = ['ArithmeticMean', 'LightGBM', 'DirectLLM', 'RidgeRegression']

        self.generated_files = []

    # ----- 内部工具 -----
    def _save_fig(self, name):
        filename = f"{name}_{self.file_prefix}.png"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.generated_files.append(filepath)
        print(f"图表已保存: {filename}")
        return filepath

    def _gather_methods(self, all_results):
        """按固定顺序整理可用方法的数据"""
        names, colors, results = [], [], []
        for code in self.method_order:
            if code in all_results:
                names.append(self.method_configs[code]['name'])
                colors.append(self.method_configs[code]['color'])
                results.append(all_results[code])
        return names, colors, results

    # ==========================================
    # 1. 方法对比(含 mean ± std 误差棒)
    # ==========================================
    def plot_method_comparison(self, all_results):
        """5 个核心指标的方法对比柱状图"""
        print("=== 生成方法对比图 ===")
        names, colors, results = self._gather_methods(all_results)

        metrics = [
            ('mae_mean', 'mae_std',                    'MAE',            'lower'),
            ('rmse_mean', 'rmse_std',                  'RMSE',           'lower'),
            ('correlation_mean', 'correlation_std',    'Pearson r',      'higher'),
            ('qwk_mean', 'qwk_std',                    'QWK',            'higher'),
            ('within_0_5_acc_mean', 'within_0_5_acc_std', 'Within-0.5 Acc', 'higher')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f'Method Performance Comparison - {self.model_name}',
                     fontsize=14, fontweight='bold')

        for idx, (m_key, s_key, label, direction) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            means = [r[m_key] for r in results]
            stds = [r[s_key] for r in results]

            bars = ax.bar(names, means, yerr=stds, color=colors,
                          capsize=4, edgecolor='black', linewidth=0.5)
            arrow = '↓' if direction == 'lower' else '↑'
            ax.set_title(f'{label} ({arrow} better)', fontweight='bold')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3, axis='y')

            # 数值标签
            offset = max(means) * 0.02 if max(means) > 0 else 0.01
            for bar, v, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + offset,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # 第6格:实验信息面板
        ax = axes[1, 2]
        ax.axis('off')
        first = results[0]
        info = (f"Dataset: {self.model_name}\n"
                f"Samples: {first['evaluation_size']}\n"
                f"CV: {first['n_splits']} × {first['n_repeats']} folds\n\n"
                f"Error bars = ±1 std\n"
                f"across {first['n_repeats']} repeats")
        ax.text(0.05, 0.5, info, transform=ax.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.tight_layout()
        return self._save_fig('method_comparison')

    # ==========================================
    # 2. 预测散点图(2x2)
    # ==========================================
    def plot_prediction_scatter(self, all_results):
        print("=== 生成预测散点图 ===")

        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        fig.suptitle(f'Predicted vs Actual Score - {self.model_name}',
                     fontsize=14, fontweight='bold')

        # 全局范围
        all_actual, all_pred = [], []
        for r in all_results.values():
            all_actual.extend(r['oof_actual'])
            all_pred.extend(r['oof_predictions_first_repeat'])
        if not all_actual:
            return None
        v_min = min(min(all_actual), min(all_pred)) - 0.3
        v_max = max(max(all_actual), max(all_pred)) + 0.3

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, code in enumerate(self.method_order):
            row, col = positions[idx]
            ax = axes[row, col]

            if code not in all_results:
                ax.set_visible(False)
                continue

            r = all_results[code]
            cfg = self.method_configs[code]
            actual = np.asarray(r['oof_actual'])
            pred = np.asarray(r['oof_predictions_first_repeat'])

            ax.scatter(actual, pred, alpha=0.45, color=cfg['color'], s=25)
            ax.plot([v_min, v_max], [v_min, v_max], 'r--', linewidth=1.5, alpha=0.7)

            ax.set_xlabel('Human Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f"{cfg['name']}\n"
                         f"MAE={r['mae_mean']:.3f}, "
                         f"r={r['correlation_mean']:.3f}, "
                         f"QWK={r['qwk_mean']:.3f}",
                         fontsize=10)
            ax.set_xlim(v_min, v_max)
            ax.set_ylim(v_min, v_max)
            ax.grid(True, alpha=0.3)
            ax.text(0.05, 0.95, f'n={len(actual)}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return self._save_fig('prediction_scatter')

    # ==========================================
    # 3. 误差分布(2x2)
    # ==========================================
    def plot_error_distribution(self, all_results):
        print("=== 生成误差分布图 ===")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f'Error Distribution - {self.model_name}',
                     fontsize=14, fontweight='bold')

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, code in enumerate(self.method_order):
            row, col = positions[idx]
            ax = axes[row, col]

            if code not in all_results:
                ax.set_visible(False)
                continue

            r = all_results[code]
            cfg = self.method_configs[code]
            actual = np.asarray(r['oof_actual'])
            pred = np.asarray(r['oof_predictions_first_repeat'])
            errors = pred - actual

            n, bins, _ = ax.hist(errors, bins=20, alpha=0.7, color=cfg['color'],
                                 edgecolor='black')
            mu, sigma = errors.mean(), errors.std()

            # 拟合正态曲线
            if sigma > 0:
                x = np.linspace(errors.min(), errors.max(), 100)
                y = (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                     / (sigma * np.sqrt(2 * np.pi))) * len(errors) * (bins[1] - bins[0])
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8)
            ax.axvline(0, color='black', linestyle='--', alpha=0.6)

            ax.set_xlabel('Error (Pred - Actual)')
            ax.set_ylabel('Count')
            ax.set_title(f"{cfg['name']}\nMean={mu:+.3f}, Std={sigma:.3f}",
                         fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_fig('error_distribution')

    # ==========================================
    # 4. 特征重要性(LightGBM + Ridge)
    # ==========================================
    def plot_feature_importance(self, all_results):
        has_lgb = ('LightGBM' in all_results
                   and 'feature_importance' in all_results['LightGBM'])
        has_ridge = ('RidgeRegression' in all_results
                     and 'feature_importance' in all_results['RidgeRegression'])
        if not (has_lgb or has_ridge):
            print("跳过特征重要性图(无可用数据)")
            return None

        print("=== 生成特征重要性图 ===")
        feature_keys = list(FEATURE_NAME_MAPPING.keys())
        feature_labels = list(FEATURE_NAME_MAPPING.values())
        # 缩短标签便于显示
        short_labels = ['TR', 'CC', 'LR', 'GRA']

        def normalize(d):
            vals = [d.get(k, 0) for k in feature_keys]
            mx = max(vals) if max(vals) > 0 else 1.0
            return [v / mx for v in vals]

        if has_lgb and has_ridge:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, code, color in [
                (axes[0], 'LightGBM', '#A23B72'),
                (axes[1], 'RidgeRegression', '#2CA02C')
            ]:
                vals = normalize(all_results[code]['feature_importance'])
                bars = ax.bar(short_labels, vals, color=color,
                              alpha=0.8, edgecolor='black')
                ax.set_title(f"{self.method_configs[code]['name']}", fontweight='bold')
                ax.set_ylabel('Normalized Importance')
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=10)
                ax.set_ylim(0, 1.15)
                ax.grid(True, alpha=0.3, axis='y')
            fig.suptitle(f'Feature Importance - {self.model_name}',
                         fontsize=13, fontweight='bold')
        else:
            code = 'LightGBM' if has_lgb else 'RidgeRegression'
            color = '#A23B72' if has_lgb else '#2CA02C'
            vals = normalize(all_results[code]['feature_importance'])
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(short_labels, vals, color=color,
                          alpha=0.8, edgecolor='black')
            ax.set_title(f"{self.method_configs[code]['name']} - {self.model_name}",
                         fontweight='bold')
            ax.set_ylabel('Normalized Importance')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._save_fig('feature_importance')

    # ==========================================
    # 5. NEW: 分数段分析图
    # ==========================================
    def plot_score_band_analysis(self, all_results):
        """按分数段(Low/Medium/High)的 MAE 与系统性偏差对比"""
        print("=== 生成分数段分析图 ===")
        names, colors, results = self._gather_methods(all_results)

        band_keys = ['low', 'medium', 'high']
        band_labels = [SCORE_BAND_CONFIG[b]['label'] for b in band_keys]

        mae_data = {b: [] for b in band_keys}
        bias_data = {b: [] for b in band_keys}
        sample_counts = {b: 0 for b in band_keys}

        for r in results:
            metrics = compute_band_metrics(
                r['oof_actual'], r['oof_predictions_first_repeat']
            )
            for b in band_keys:
                m = metrics[b]
                mae_data[b].append(m['mae'] if m['mae'] is not None else 0)
                bias_data[b].append(m['mean_error'] if m['mean_error'] is not None else 0)
                sample_counts[b] = m['n_samples']

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        fig.suptitle(f'Score Band Analysis - {self.model_name}',
                     fontsize=14, fontweight='bold')

        n_bands = len(band_keys)
        n_methods = len(names)
        x = np.arange(n_bands)
        width = 0.8 / n_methods

        # 左:MAE by band
        ax = axes[0]
        for i, name in enumerate(names):
            vals = [mae_data[b][i] for b in band_keys]
            offset = (i - (n_methods - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=name, color=colors[i],
                          edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        labels_with_n = [f"{lbl}\n(n={sample_counts[b]})"
                         for lbl, b in zip(band_labels, band_keys)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_with_n, fontsize=9)
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Score Band (lower is better)', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # 右:系统性偏差
        ax = axes[1]
        for i, name in enumerate(names):
            vals = [bias_data[b][i] for b in band_keys]
            offset = (i - (n_methods - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=name, color=colors[i],
                   edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_with_n, fontsize=9)
        ax.set_ylabel('Mean Error (Pred - Actual)')
        ax.set_title('Systematic Bias by Score Band\n(>0 over-est., <0 under-est.)',
                     fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._save_fig('score_band_analysis')

    # ==========================================
    # 6. NEW: 误差箱型图(按真实分数分组)
    # ==========================================
    def plot_error_boxplot_by_score(self, all_results):
        """按真实分数分组的误差箱型图,揭示按分数的偏差模式"""
        print("=== 生成误差箱型图 ===")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Error by True Score - {self.model_name}',
                     fontsize=14, fontweight='bold')

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, code in enumerate(self.method_order):
            row, col = positions[idx]
            ax = axes[row, col]

            if code not in all_results:
                ax.set_visible(False)
                continue

            r = all_results[code]
            cfg = self.method_configs[code]
            actual = np.asarray(r['oof_actual'])
            pred = np.asarray(r['oof_predictions_first_repeat'])
            errors = pred - actual

            # 按真实分数分组
            df = pd.DataFrame({'actual': actual, 'error': errors})
            grouped = df.groupby('actual')['error'].apply(list).to_dict()
            sorted_scores = sorted(grouped.keys())
            data_to_plot = [grouped[s] for s in sorted_scores]
            counts = [len(grouped[s]) for s in sorted_scores]

            bp = ax.boxplot(data_to_plot, positions=range(len(sorted_scores)),
                            patch_artist=True, widths=0.55,
                            showfliers=True, flierprops={'markersize': 3})
            for patch in bp['boxes']:
                patch.set_facecolor(cfg['color'])
                patch.set_alpha(0.6)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)

            ax.axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1)
            tick_labels = [f'{s:.1f}\n(n={c})' for s, c in zip(sorted_scores, counts)]
            ax.set_xticks(range(len(sorted_scores)))
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_xlabel('Human Score')
            ax.set_ylabel('Error (Pred - Actual)')
            ax.set_title(cfg['name'], fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._save_fig('error_boxplot_by_score')

    # ==========================================
    # 7. 综合总结图
    # ==========================================
    def plot_summary(self, all_results):
        print("=== 生成综合总结图 ===")
        names, colors, results = self._gather_methods(all_results)

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Comprehensive Summary - {self.model_name}',
                     fontsize=15, fontweight='bold')

        # 1. MAE 横向条形图(左上,跨2列)
        ax1 = plt.subplot(2, 3, (1, 2))
        means = [r['mae_mean'] for r in results]
        stds = [r['mae_std'] for r in results]
        bars = ax1.barh(names, means, xerr=stds, color=colors,
                        capsize=4, edgecolor='black')
        ax1.set_xlabel('MAE (lower is better)')
        ax1.set_title('Mean Absolute Error (with std)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        for bar, m, s in zip(bars, means, stds):
            ax1.text(bar.get_width() + s + max(means) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{m:.3f}±{s:.3f}', va='center', fontsize=9)

        # 2. QWK(右上)
        ax2 = plt.subplot(2, 3, 3)
        means = [r['qwk_mean'] for r in results]
        stds = [r['qwk_std'] for r in results]
        ax2.bar(names, means, yerr=stds, color=colors,
                capsize=3, edgecolor='black')
        ax2.set_ylabel('QWK')
        ax2.set_title('Quadratic Weighted Kappa', fontweight='bold')
        ax2.tick_params(axis='x', rotation=30)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 真实分数分布(左下)
        ax3 = plt.subplot(2, 3, 4)
        actual = np.asarray(results[0]['oof_actual'])
        ax3.hist(actual, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Human Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Score Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 统计摘要(右下,跨2列)
        ax4 = plt.subplot(2, 3, (5, 6))
        ax4.axis('off')

        ranking = sorted(
            [(r['method_name'], r['mae_mean'], r['mae_std'])
             for r in results],
            key=lambda x: x[1]
        )

        first = results[0]
        text = (
            f"Dataset: {self.model_name}\n"
            f"Total samples: {first['evaluation_size']}\n"
            f"Score range: [{actual.min():.1f}, {actual.max():.1f}]\n"
            f"Mean score: {actual.mean():.2f}\n"
            f"Evaluation: {first['n_splits']}-fold × {first['n_repeats']}-repeat OOF\n\n"
            f"Ranking by MAE (lower = better):\n"
        )
        for i, (name, m, s) in enumerate(ranking, 1):
            text += f"  {i}. {name:<22s} {m:.3f} ± {s:.3f}\n"
        text += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        ax4.text(0.02, 0.98, text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

        plt.tight_layout()
        return self._save_fig('summary')


# ==========================================
# 便捷接口
# ==========================================

def quick_visualize_all(all_results, data_path):
    """生成所有可视化图表"""
    visualizer = IELTSVisualizer(data_path)
    print("开始生成可视化图表...")

    try:
        visualizer.plot_method_comparison(all_results)
        visualizer.plot_prediction_scatter(all_results)
        visualizer.plot_error_distribution(all_results)
        visualizer.plot_feature_importance(all_results)
        visualizer.plot_score_band_analysis(all_results)
        visualizer.plot_error_boxplot_by_score(all_results)
        visualizer.plot_summary(all_results)

        print(f"\n所有图表生成完成,共 {len(visualizer.generated_files)} 个文件")
    except Exception as e:
        print(f"可视化生成出错: {str(e)}")
        import traceback
        traceback.print_exc()

    return visualizer, visualizer.generated_files