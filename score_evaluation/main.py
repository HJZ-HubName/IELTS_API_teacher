# -*- coding: utf-8 -*-
"""
IELTS评分系统 - 主程序
统一 RepeatedKFold OOF 框架,支持 direct_all + cot_all 双文件夹批处理
"""

from datetime import datetime
import warnings
import glob
import os
import pandas as pd
warnings.filterwarnings('ignore')

from data_handler import IELTSDataHandler
from models import run_all_models
from evaluation import quick_evaluate_all
from visualization import quick_visualize_all
from utils import (
    set_output_folder, create_output_folder,
    generate_file_prefix, get_output_filepath, ensure_directory_exists,
    extract_llm_name, detect_prompting_strategy,
    DEFAULT_N_SPLITS, DEFAULT_N_REPEATS, DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_MODE, DEFAULT_ROUNDING_CONFIG
)


# ==========================================
# 配置区
# ==========================================
# 处理模式
BATCH_MODE = True

# 单文件模式配置
DATA_FILE = 'score_data_all/direct_all/ielts_scoring_summary_20250709_grok-3_deduped.csv'

# 批处理模式配置
DATA_ROOT = 'score_data_all'
PROCESS_DIRECT = True              # 处理 direct_all/ 文件夹
PROCESS_COT = True                 # 处理 cot_all/ 文件夹
UNIFIED_OUTPUT_FOLDER = True       # True=统一输出文件夹, False=各自独立

# 实验参数
N_SPLITS = DEFAULT_N_SPLITS         # 5
N_REPEATS = DEFAULT_N_REPEATS       # 10
RANDOM_STATE = DEFAULT_RANDOM_STATE
ENABLE_VISUALIZATION = True

# 半分取整规则:nearest 为最近半分,中点向上,并限制在 [0, 9]
ROUNDING_M1 = DEFAULT_ROUNDING_MODE   # Arithmetic Mean
ROUNDING_M2 = DEFAULT_ROUNDING_MODE   # LightGBM
ROUNDING_M4 = DEFAULT_ROUNDING_MODE   # Ridge Regression
# Direct LLM (M3) 输出已是 0.5 步长,不参与


# ==========================================
# 单文件实验
# ==========================================

class IELTSExperiment:
    """单文件 IELTS 评分实验"""

    def __init__(self, data_file,
                 n_splits=DEFAULT_N_SPLITS,
                 n_repeats=DEFAULT_N_REPEATS,
                 random_state=DEFAULT_RANDOM_STATE,
                 enable_visualization=True,
                 rounding_config=None,
                 output_folder=None):
        self.data_file = data_file
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.enable_visualization = enable_visualization
        self.rounding_config = {**DEFAULT_ROUNDING_CONFIG, **(rounding_config or {})}

        # 识别 LLM 与 prompting 策略
        self.llm_name = extract_llm_name(data_file)
        self.prompting = detect_prompting_strategy(data_file)

        # 设置输出文件夹
        if output_folder:
            self.output_folder = set_output_folder(output_folder)
        else:
            self.output_folder = create_output_folder(data_file)

        # 状态
        self.data = None
        self.all_results = None
        self.evaluator = None
        self.generated_files = []

    def load_data(self):
        """加载数据(无需 train/test 划分,RepeatedKFold 内部处理)"""
        print("第1步: 加载数据")
        print("-" * 50)
        handler = IELTSDataHandler(self.data_file)
        handler.load_data()
        handler.validate_data()
        self.data = handler.data
        print(f"样本数: {len(self.data)}")
        print(f"LLM: {self.llm_name}")
        print(f"Prompting: {self.prompting}")
        print()

    def run_models(self):
        """运行所有4个模型(OOF CV)"""
        print("第2步: 运行模型")
        print("-" * 50)

        print("半分取整规则配置:")
        for m, mode in self.rounding_config.items():
            print(f"  {m}: {mode}")
        print()

        start = datetime.now()
        self.all_results = run_all_models(
            self.data,
            rounding_config=self.rounding_config,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        duration = (datetime.now() - start).total_seconds()
        print(f"模型运行完成,用时 {duration:.1f} 秒\n")

    def evaluate(self):
        print("第3步: 评估结果")
        print("-" * 50)
        self.evaluator, files = quick_evaluate_all(
            self.all_results, self.data_file, self.data
        )
        self.generated_files.extend(files)
        print()

    def visualize(self):
        if not self.enable_visualization:
            print("跳过可视化")
            return
        print("第4步: 生成图表")
        print("-" * 50)
        try:
            _, files = quick_visualize_all(self.all_results, self.data_file)
            self.generated_files.extend(files)
        except Exception as e:
            print(f"可视化生成失败: {e}")
        print()

    def save_summary(self):
        """保存实验总结报告"""
        if not self.evaluator or self.evaluator.comparison_results is None:
            return None

        print("第5步: 保存总结")
        print("-" * 50)

        file_prefix = generate_file_prefix(self.data_file)
        filename = f"experiment_summary_{file_prefix}.txt"
        filepath = get_output_filepath(filename)
        ensure_directory_exists(filepath)

        cmp_df = self.evaluator.comparison_results

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("IELTS Writing Auto-Scoring - Experiment Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据文件: {self.data_file}\n")
                f.write(f"LLM: {self.llm_name}\n")
                f.write(f"Prompting: {self.prompting}\n")
                f.write(f"样本数: {len(self.data)}\n")
                f.write(f"CV: {self.n_splits}-fold × {self.n_repeats}-repeat\n")
                f.write(f"随机种子: {self.random_state}\n\n")

                f.write("半分取整规则:\n")
                for m, mode in self.rounding_config.items():
                    f.write(f"  {m}: {mode}\n")
                f.write("\n")

                f.write("性能 (mean ± std):\n")
                f.write("-" * 60 + "\n")
                for _, row in cmp_df.iterrows():
                    f.write(f"\n{row['方法名称']}:\n")
                    f.write(f"  MAE:        {row['MAE']}\n")
                    f.write(f"  RMSE:       {row['RMSE']}\n")
                    f.write(f"  Pearson r:  {row['Pearson r']}\n")
                    f.write(f"  QWK:        {row['QWK']}\n")
                    f.write(f"  Within-0.5: {row['Within-0.5']}\n")

                f.write(f"\n生成文件 ({len(self.generated_files)} 个):\n")
                for fp in self.generated_files:
                    f.write(f"  {os.path.basename(fp)}\n")

            self.generated_files.append(filepath)
            print(f"总结已保存: {os.path.basename(filepath)}\n")
            return filepath
        except Exception as e:
            print(f"保存总结失败: {e}")
            return None

    def run(self):
        """主入口"""
        print("=" * 60)
        print(f"IELTS 实验: {self.llm_name} ({self.prompting})")
        print("=" * 60)
        print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"文件: {self.data_file}")
        print(f"CV: {self.n_splits}-fold × {self.n_repeats}-repeat")
        print(f"输出: {self.output_folder}")
        print()

        try:
            self.load_data()
            self.run_models()
            self.evaluate()
            self.visualize()
            self.save_summary()

            print("=" * 60)
            print(f"实验完成: {self.llm_name} ({self.prompting})")
            print("=" * 60)
            return True
        except Exception as e:
            print(f"\n实验失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# ==========================================
# 批量处理
# ==========================================

class BatchProcessor:
    """批量处理 direct_all/ 和 cot_all/ 文件夹"""

    # 跨 LLM 汇总表中,wide 格式的方法显示顺序(对齐论文 Table 2)
    METHOD_ORDER_WIDE = ['LightGBM', 'RidgeRegression', 'ArithmeticMean', 'DirectLLM']
    METHOD_SHORT = {
        'ArithmeticMean': 'AM',
        'LightGBM': 'LGB',
        'DirectLLM': 'DLM',
        'RidgeRegression': 'Ridge'
    }

    def __init__(self, data_root='score_data_all',
                 process_direct=True, process_cot=True,
                 unified_output=True,
                 n_splits=DEFAULT_N_SPLITS,
                 n_repeats=DEFAULT_N_REPEATS,
                 random_state=DEFAULT_RANDOM_STATE,
                 enable_visualization=True,
                 rounding_config=None):
        self.data_root = data_root
        self.process_direct = process_direct
        self.process_cot = process_cot
        self.unified_output = unified_output
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.enable_visualization = enable_visualization
        self.rounding_config = {**DEFAULT_ROUNDING_CONFIG, **(rounding_config or {})}

        self.unified_output_folder = None
        self.successful = []
        self.failed = []

    def setup_unified_folder(self):
        if self.unified_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            modes = set(self.rounding_config.values())
            rounding_tag = next(iter(modes)) if len(modes) == 1 else 'mixed'
            self.unified_output_folder = f"output_batch_{rounding_tag}_{timestamp}"
            os.makedirs(self.unified_output_folder, exist_ok=True)
            print(f"统一输出文件夹: {self.unified_output_folder}")

    def find_files(self):
        """扫描 direct_all/ 和 cot_all/"""
        files = []
        if self.process_direct:
            pattern = os.path.join(self.data_root, 'direct_all', '*.csv')
            for f in sorted(glob.glob(pattern)):
                files.append((f, 'Direct'))
        if self.process_cot:
            pattern = os.path.join(self.data_root, 'cot_all', '*.csv')
            for f in sorted(glob.glob(pattern)):
                files.append((f, 'CoT'))
        return files

    def get_subfolder_name(self, file_path, prompting):
        """生成单个文件结果的子文件夹名"""
        llm = extract_llm_name(file_path)
        llm_safe = llm.replace('/', '_').replace('\\', '_').replace(':', '_')
        return f"{prompting.lower()}_{llm_safe}"

    def process_one(self, file_path, prompting):
        """处理一个文件"""
        print("\n" + "=" * 60)
        print(f"处理: {os.path.basename(file_path)}")
        print(f"Prompting: {prompting}")
        print("=" * 60)

        if self.unified_output:
            subfolder = self.get_subfolder_name(file_path, prompting)
            output_folder = os.path.join(self.unified_output_folder, subfolder)
        else:
            output_folder = None

        try:
            exp = IELTSExperiment(
                data_file=file_path,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                enable_visualization=self.enable_visualization,
                rounding_config=self.rounding_config,
                output_folder=output_folder
            )
            success = exp.run()

            if success:
                self.successful.append({
                    'file': file_path,
                    'prompting': prompting,
                    'llm': extract_llm_name(file_path),
                    'experiment': exp
                })
                print(f"成功: {os.path.basename(file_path)}")
            else:
                self.failed.append({
                    'file': file_path,
                    'prompting': prompting,
                    'error': '实验执行失败'
                })
                print(f"失败: {os.path.basename(file_path)}")
        except Exception as e:
            self.failed.append({
                'file': file_path,
                'prompting': prompting,
                'error': str(e)
            })
            print(f"异常: {os.path.basename(file_path)}: {e}")

    def generate_cross_llm_summary(self):
        """生成跨 LLM 汇总(long + wide 两种格式)"""
        if not self.successful:
            return None, None

        print("\n" + "=" * 60)
        print("生成跨 LLM 汇总表")
        print("=" * 60)

        long_rows = []
        wide_rows = []

        for entry in self.successful:
            llm = entry['llm']
            prompting = entry['prompting']
            results = entry['experiment'].all_results

            wide_row = {'LLM': llm, 'Prompting': prompting}

            # Long 格式:每个 method 一行
            for code, r in results.items():
                long_rows.append({
                    'LLM': llm,
                    'Prompting': prompting,
                    'Method_Code': code,
                    'Method_Name': r['method_name'],
                    'Rounding_Mode': r['rounding_mode'],
                    'MAE_mean': r['mae_mean'],
                    'MAE_std': r['mae_std'],
                    'RMSE_mean': r['rmse_mean'],
                    'RMSE_std': r['rmse_std'],
                    'r_mean': r['correlation_mean'],
                    'r_std': r['correlation_std'],
                    'QWK_mean': r['qwk_mean'],
                    'QWK_std': r['qwk_std'],
                    'Within_0.5_mean': r['within_0_5_acc_mean'],
                    'Within_0.5_std': r['within_0_5_acc_std'],
                })

            # Wide 格式:按论文顺序填列
            for code in self.METHOD_ORDER_WIDE:
                if code not in results:
                    continue
                r = results[code]
                p = self.METHOD_SHORT[code]
                wide_row[f'{p}_Rounding_Mode'] = r['rounding_mode']
                wide_row[f'{p}_MAE'] = f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}"
                wide_row[f'{p}_RMSE'] = f"{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}"
                wide_row[f'{p}_r'] = f"{r['correlation_mean']:.3f}±{r['correlation_std']:.3f}"
                wide_row[f'{p}_QWK'] = f"{r['qwk_mean']:.3f}±{r['qwk_std']:.3f}"
                wide_row[f'{p}_Within0.5'] = f"{r['within_0_5_acc_mean']:.3f}±{r['within_0_5_acc_std']:.3f}"

            wide_rows.append(wide_row)

        long_df = pd.DataFrame(long_rows)
        wide_df = pd.DataFrame(wide_rows)

        # 排序(便于阅读)
        long_df = long_df.sort_values(['Prompting', 'LLM', 'Method_Code']).reset_index(drop=True)
        wide_df = wide_df.sort_values(['Prompting', 'LLM']).reset_index(drop=True)

        # 输出位置
        base = self.unified_output_folder if self.unified_output else '.'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        long_path = os.path.join(base, f'cross_llm_summary_long_{timestamp}.csv')
        wide_path = os.path.join(base, f'cross_llm_summary_wide_{timestamp}.csv')

        long_df.to_csv(long_path, index=False, encoding='utf-8-sig')
        wide_df.to_csv(wide_path, index=False, encoding='utf-8-sig')

        print(f"Long 格式: {long_path}")
        print(f"Wide 格式: {wide_path}")

        # 简洁预览(只显示 MAE)
        if 'LGB_MAE' in wide_df.columns:
            print("\n--- 预览: MAE (LGB / Ridge / AM / DLM) ---")
            preview_cols = ['LLM', 'Prompting', 'LGB_MAE', 'Ridge_MAE',
                            'AM_MAE', 'DLM_MAE']
            preview_cols = [c for c in preview_cols if c in wide_df.columns]
            print(wide_df[preview_cols].to_string(index=False))

        return long_path, wide_path

    def save_batch_summary(self):
        """保存批处理总结"""
        base = self.unified_output_folder if self.unified_output else '.'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(base, f'batch_summary_{timestamp}.txt')

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("IELTS 批量处理总结\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据根目录: {self.data_root}\n")
                f.write(f"CV: {self.n_splits}-fold × {self.n_repeats}-repeat\n")
                f.write(f"随机种子: {self.random_state}\n")
                f.write(f"可视化: {'启用' if self.enable_visualization else '禁用'}\n")
                f.write(f"输出模式: {'统一文件夹' if self.unified_output else '独立文件夹'}\n\n")

                f.write("半分取整规则:\n")
                for m, mode in self.rounding_config.items():
                    f.write(f"  {m}: {mode}\n")
                f.write("\n")

                total = len(self.successful) + len(self.failed)
                f.write(f"总计: {total}\n")
                f.write(f"成功: {len(self.successful)}\n")
                f.write(f"失败: {len(self.failed)}\n\n")

                if self.successful:
                    f.write("成功的实验:\n")
                    for e in self.successful:
                        f.write(f"  [{e['prompting']:6s}] {e['llm']}\n")
                    f.write("\n")

                if self.failed:
                    f.write("失败的实验:\n")
                    for e in self.failed:
                        f.write(f"  [{e['prompting']:6s}] {os.path.basename(e['file'])}: {e['error']}\n")

            print(f"批处理总结: {summary_path}")
            return summary_path
        except Exception as e:
            print(f"保存批处理总结失败: {e}")
            return None

    def run(self):
        """主入口"""
        print("=" * 60)
        print("IELTS 批量处理")
        print("=" * 60)

        files = self.find_files()
        if not files:
            print(f"未在 {self.data_root}/ 找到任何 CSV 文件!")
            return False

        print(f"找到 {len(files)} 个文件:")
        direct_count = sum(1 for _, p in files if p == 'Direct')
        cot_count = sum(1 for _, p in files if p == 'CoT')
        print(f"  Direct: {direct_count} 个")
        print(f"  CoT:    {cot_count} 个")
        print()

        print("配置:")
        print(f"  CV: {self.n_splits}-fold × {self.n_repeats}-repeat")
        print(f"  随机种子: {self.random_state}")
        print(f"  可视化: {'启用' if self.enable_visualization else '禁用'}")
        print(f"  输出模式: {'统一文件夹' if self.unified_output else '独立文件夹'}")
        print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self.setup_unified_folder()

        # 逐个处理
        for i, (path, prompting) in enumerate(files, 1):
            print(f"\n进度: {i}/{len(files)}")
            self.process_one(path, prompting)

        # 跨 LLM 汇总
        self.generate_cross_llm_summary()

        # 批处理总结
        self.save_batch_summary()

        # 收尾
        total = len(self.successful) + len(self.failed)
        print("\n" + "=" * 60)
        print(f"批量处理完成: {len(self.successful)}/{total} 成功")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.unified_output:
            print(f"输出文件夹: {self.unified_output_folder}")
        print("=" * 60)

        return True


# ==========================================
# 主入口
# ==========================================

def main():
    rounding_config = {
        'M1': ROUNDING_M1,
        'M2': ROUNDING_M2,
        'M4': ROUNDING_M4
    }

    if BATCH_MODE:
        processor = BatchProcessor(
            data_root=DATA_ROOT,
            process_direct=PROCESS_DIRECT,
            process_cot=PROCESS_COT,
            unified_output=UNIFIED_OUTPUT_FOLDER,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            enable_visualization=ENABLE_VISUALIZATION,
            rounding_config=rounding_config
        )
        return processor.run()
    else:
        exp = IELTSExperiment(
            data_file=DATA_FILE,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            enable_visualization=ENABLE_VISUALIZATION,
            rounding_config=rounding_config
        )
        return exp.run()


if __name__ == "__main__":
    main()
