# -*- coding: utf-8 -*-
"""
Aggregate score_band_analysis CSV files for IELTS LLM scoring experiments.

This script:
1. Recursively reads all score_band_analysis_*.csv files.
2. Extracts LLM and prompting information from the input file name/path.
3. Combines all files into one long-format CSV.
4. Produces aggregated band-wise summaries.
5. Produces paper-ready direct-prompting band-wise tables.

Important:
- Internal variable names avoid dots, e.g. Within05 instead of Within_0.5.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np


# ==============================
# User configuration
# ==============================


# 每次实验完成后，只修改下面的文件夹名称
RESULT_FOLDER = "output_batch_nearest_20261014_160431"

# 从当前脚本所在目录寻找结果文件夹
ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    RESULT_FOLDER
)

if not os.path.isdir(ROOT_DIR):
    raise FileNotFoundError(f"实验输出目录不存在: {ROOT_DIR}")

OUTPUT_DIR = os.path.join(ROOT_DIR, "band_analysis_aggregated")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Helper functions
# ==============================

def detect_prompting(path, input_file=None):
    """
    Detect whether a file belongs to Direct or CoT prompting.

    Detection uses both source path and original input filename.
    """
    text = str(path)
    if input_file is not None:
        text += " " + str(input_file)

    text_lower = text.replace("\\", "/").lower()

    if "cot_all" in text_lower:
        return "CoT"
    if "/cot_" in text_lower:
        return "CoT"
    if "-cot" in text_lower or "_cot" in text_lower:
        return "CoT"

    if "direct_all" in text_lower:
        return "Direct"
    if "/direct_" in text_lower:
        return "Direct"

    return "Unknown"


def extract_llm_from_input_file(input_file):
    """
    Extract LLM name from the original scoring file name.

    Example:
    ielts_scoring_summary_20250708_chatgpt-4o-latest_deduped.csv
    -> chatgpt-4o-latest

    ielts_scoring_summary_20250831_GPT-4.1-mini-CoT_deduped.csv
    -> GPT-4.1-mini
    """
    base = os.path.basename(str(input_file))
    name = base.replace(".csv", "")
    name = name.replace("ielts_scoring_summary_", "")
    name = name.replace("_deduped", "")

    # remove leading date if present
    parts = name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 8:
        name = parts[1]

    # remove CoT suffix if present
    name = name.replace("-CoT", "").replace("_CoT", "")

    return name


def standardize_columns(df):
    """
    Rename Chinese columns to safe English names.

    Note:
    - Within_0.5_Acc is renamed to Within05.
    - We avoid dots in internal column names.
    """
    col_map = {
        "方法代码": "Method_Code",
        "方法名称": "Method_Name",
        "分数段": "Score_Band",
        "分段代码": "Band_Code",
        "样本数": "N",
        "Within_0.5_Acc": "Within05",
        "平均偏差(>0高估,<0低估)": "Mean_Error",
        "输入文件": "Input_File",
    }

    df = df.rename(columns=col_map)

    return df


def method_display_name(code):
    mapping = {
        "ArithmeticMean": "Arithmetic Mean",
        "DirectLLM": "Direct LLM",
        "LightGBM": "LightGBM",
        "RidgeRegression": "Ridge Regression",
    }
    return mapping.get(code, code)


def band_display_name(code):
    mapping = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    }
    return mapping.get(code, code)


# ==============================
# 1. Read all files
# ==============================

pattern = os.path.join(ROOT_DIR, "**", "score_band_analysis_*.csv")
files = sorted(glob.glob(pattern, recursive=True))

print(f"Found {len(files)} score_band_analysis files.")

if not files:
    raise FileNotFoundError(
        f"No score_band_analysis_*.csv files found under {ROOT_DIR}"
    )

all_dfs = []

for fp in files:
    try:
        df = pd.read_csv(fp, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(fp, encoding="gbk")

    df = standardize_columns(df)
    if 'Rounding_Mode' not in df.columns or df['Rounding_Mode'].isna().any():
        raise ValueError(f"缺少取整模式记录,请使用新代码生成的结果: {fp}")

    if "Input_File" in df.columns:
        input_file = df["Input_File"].iloc[0]
        llm = extract_llm_from_input_file(input_file)
    else:
        input_file = os.path.basename(fp)
        llm = os.path.basename(os.path.dirname(fp))

    prompting = detect_prompting(fp, input_file=input_file)

    df["LLM"] = llm
    df["Prompting"] = prompting
    df["Source_File"] = fp

    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
if (combined.groupby('Method_Code')['Rounding_Mode'].nunique() > 1).any():
    raise ValueError("发现同一方法使用多种取整规则,请只指定一次实验的输出目录。")

# Ensure numeric columns
numeric_cols = ["N", "MAE", "RMSE", "Within05", "Mean_Error"]

for col in numeric_cols:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    else:
        raise KeyError(f"Required column not found after renaming: {col}")

combined["Method_Display"] = combined["Method_Code"].apply(method_display_name)
combined["Band_Display"] = combined["Band_Code"].apply(band_display_name)

# Save combined long table
combined_path = os.path.join(OUTPUT_DIR, "combined_score_band_analysis_long.csv")
combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"Saved combined long table: {combined_path}")


# ==============================
# 2. Aggregated summaries
# ==============================

summary = (
    combined
    .groupby(
        [
            "Prompting",
            "Method_Code",
            "Method_Display",
            "Rounding_Mode",
            "Band_Code",
            "Band_Display"
        ],
        as_index=False
    )
    .agg(
        N_mean=("N", "mean"),
        N_min=("N", "min"),
        N_max=("N", "max"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        Within05_mean=("Within05", "mean"),
        Within05_std=("Within05", "std"),
        Mean_Error_mean=("Mean_Error", "mean"),
        Mean_Error_std=("Mean_Error", "std"),
        Num_LLMs=("LLM", "nunique"),
        Num_Files=("Source_File", "nunique")
    )
)

summary_path = os.path.join(OUTPUT_DIR, "summary_by_prompting_method_band.csv")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"Saved summary table: {summary_path}")


# ==============================
# 3. Direct-only paper table
# ==============================

direct_summary = summary[summary["Prompting"] == "Direct"].copy()

if direct_summary.empty:
    print("Warning: No Direct prompting files detected. Check folder names or file names.")

method_order = ["DirectLLM", "ArithmeticMean", "RidgeRegression", "LightGBM"]
band_order = ["low", "medium", "high"]

direct_summary["Method_Order"] = direct_summary["Method_Code"].apply(
    lambda x: method_order.index(x) if x in method_order else 99
)

direct_summary["Band_Order"] = direct_summary["Band_Code"].apply(
    lambda x: band_order.index(x) if x in band_order else 99
)

direct_summary = direct_summary.sort_values(["Band_Order", "Method_Order"])

paper_direct = direct_summary[
    [
        "Band_Display",
        "Method_Display",
        "Rounding_Mode",
        "N_mean",
        "MAE_mean",
        "MAE_std",
        "Mean_Error_mean",
        "Mean_Error_std",
        "Within05_mean",
        "Within05_std",
        "Num_LLMs",
        "Num_Files"
    ]
].copy()

# Formatting
paper_direct["N_mean"] = paper_direct["N_mean"].round(0).astype(int)

for col in [
    "MAE_mean",
    "MAE_std",
    "Mean_Error_mean",
    "Mean_Error_std",
    "Within05_mean",
    "Within05_std"
]:
    paper_direct[col] = paper_direct[col].round(3)

paper_direct_path = os.path.join(OUTPUT_DIR, "paper_table_direct_bandwise.csv")
paper_direct.to_csv(paper_direct_path, index=False, encoding="utf-8-sig")
print(f"Saved paper-ready direct band-wise table: {paper_direct_path}")


# ==============================
# 4. CoT-only paper table
# ==============================

cot_summary = summary[summary["Prompting"] == "CoT"].copy()

if not cot_summary.empty:
    cot_summary["Method_Order"] = cot_summary["Method_Code"].apply(
        lambda x: method_order.index(x) if x in method_order else 99
    )
    cot_summary["Band_Order"] = cot_summary["Band_Code"].apply(
        lambda x: band_order.index(x) if x in band_order else 99
    )
    cot_summary = cot_summary.sort_values(["Band_Order", "Method_Order"])

    paper_cot = cot_summary[
        [
            "Band_Display",
            "Method_Display",
            "Rounding_Mode",
            "N_mean",
            "MAE_mean",
            "MAE_std",
            "Mean_Error_mean",
            "Mean_Error_std",
            "Within05_mean",
            "Within05_std",
            "Num_LLMs",
            "Num_Files"
        ]
    ].copy()

    paper_cot["N_mean"] = paper_cot["N_mean"].round(0).astype(int)

    for col in [
        "MAE_mean",
        "MAE_std",
        "Mean_Error_mean",
        "Mean_Error_std",
        "Within05_mean",
        "Within05_std"
    ]:
        paper_cot[col] = paper_cot[col].round(3)

    paper_cot_path = os.path.join(OUTPUT_DIR, "paper_table_cot_bandwise.csv")
    paper_cot.to_csv(paper_cot_path, index=False, encoding="utf-8-sig")
    print(f"Saved paper-ready CoT band-wise table: {paper_cot_path}")
else:
    print("No CoT prompting files detected.")


# ==============================
# 5. Pivot tables for quick inspection
# ==============================

def save_direct_pivot(metric_col, filename):
    """
    Save direct-prompting pivot table.

    metric_col should be one of:
    MAE_mean, Mean_Error_mean, Within05_mean, RMSE_mean
    """
    if direct_summary.empty:
        return

    pivot = direct_summary.pivot_table(
        index=["Band_Display", "Method_Display", "Rounding_Mode"],
        values=metric_col,
        aggfunc="first"
    ).reset_index()

    pivot_path = os.path.join(OUTPUT_DIR, filename)
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")
    print(f"Saved pivot: {pivot_path}")


save_direct_pivot("MAE_mean", "pivot_direct_mae_by_band.csv")
save_direct_pivot("Mean_Error_mean", "pivot_direct_mean_error_by_band.csv")
save_direct_pivot("Within05_mean", "pivot_direct_within05_by_band.csv")
save_direct_pivot("RMSE_mean", "pivot_direct_rmse_by_band.csv")


# ==============================
# 6. Low-score overestimation summary
# ==============================

if not direct_summary.empty:
    low_direct = direct_summary[direct_summary["Band_Code"] == "low"].copy()
    low_direct = low_direct.sort_values("Mean_Error_mean", ascending=False)

    risk_path = os.path.join(
        OUTPUT_DIR,
        "low_score_overestimation_summary_direct.csv"
    )

    low_direct[
        [
            "Method_Display",
            "Rounding_Mode",
            "N_mean",
            "MAE_mean",
            "MAE_std",
            "Mean_Error_mean",
            "Mean_Error_std",
            "Within05_mean",
            "Within05_std",
            "Num_LLMs",
            "Num_Files"
        ]
    ].to_csv(risk_path, index=False, encoding="utf-8-sig")

    print(f"Saved low-score overestimation summary: {risk_path}")


# ==============================
# 7. High-score underestimation summary
# ==============================

if not direct_summary.empty:
    high_direct = direct_summary[direct_summary["Band_Code"] == "high"].copy()
    high_direct = high_direct.sort_values("Mean_Error_mean", ascending=True)

    high_risk_path = os.path.join(
        OUTPUT_DIR,
        "high_score_underestimation_summary_direct.csv"
    )

    high_direct[
        [
            "Method_Display",
            "Rounding_Mode",
            "N_mean",
            "MAE_mean",
            "MAE_std",
            "Mean_Error_mean",
            "Mean_Error_std",
            "Within05_mean",
            "Within05_std",
            "Num_LLMs",
            "Num_Files"
        ]
    ].to_csv(high_risk_path, index=False, encoding="utf-8-sig")

    print(f"Saved high-score underestimation summary: {high_risk_path}")


# ==============================
# 8. Print preview
# ==============================

print("\n=== Combined data overview ===")
print(combined[["LLM", "Prompting", "Method_Code", "Band_Code", "N", "MAE", "Mean_Error", "Within05"]].head())

print("\n=== Prompting detection counts ===")
print(combined[["LLM", "Prompting", "Source_File"]].drop_duplicates()["Prompting"].value_counts())

if not direct_summary.empty:
    print("\n=== Direct prompting: paper-ready band-wise summary ===")
    print(paper_direct.to_string(index=False))
