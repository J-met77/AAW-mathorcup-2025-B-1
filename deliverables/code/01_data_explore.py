# -*- coding: utf-8 -*-
"""01_data_explore.py —— A3 数据勘察第一步：三个 Excel 的结构、字段、缺失、关键分布。
产出：output/logs/01_data_explore.log
"""
import sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\data"
LOG_PATH = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\01_data_explore.log"

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

def describe_df(name, df):
    log(f"\n{'='*70}\n### {name}  shape={df.shape}")
    log("\n-- columns & dtypes --")
    for c in df.columns:
        n_missing = df[c].isna().sum()
        log(f"  {c!r:40s} dtype={str(df[c].dtype):14s} missing={n_missing} ({n_missing/len(df)*100:.2f}%)")
    log("\n-- head(5) --")
    log(df.head(5).to_string())
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        log("\n-- numeric describe --")
        log(num.describe().T.to_string())
    cat = df.select_dtypes(exclude=[np.number])
    for c in cat.columns:
        vc = df[c].value_counts(dropna=False)
        show = vc.head(15)
        log(f"\n-- value_counts: {c}  (nunique={df[c].nunique(dropna=True)}) --")
        log(show.to_string())

log("=== 01_data_explore start ===")

# 1) Excel 底层结构（sheet 名、维度）
for fn in ["附件1.xlsx", "附件2.xlsx", "Result.xlsx"]:
    path = f"{DATA_DIR}\\{fn}"
    xl = pd.ExcelFile(path)
    log(f"\n{'#'*70}\nFILE: {fn}  sheets={xl.sheet_names}")
    for sh in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sh)
        describe_df(f"{fn} :: sheet[{sh}]", df)

log("\n=== 01_data_explore done ===")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG_PATH)
