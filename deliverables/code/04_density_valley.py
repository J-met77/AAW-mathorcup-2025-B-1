# -*- coding: utf-8 -*-
"""04_density_valley.py —— A3 数据勘察第四步：在 (赔付p, 超额u) 平面找类间"密度谷/空隙"。
   若出题方按类生成数据（合理/偏高/超额各有分布），则 u 的条件密度或其变换可能出现谷。
产出：output/logs/04_density_valley.log
"""
import sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DATA_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\data"
LOG_PATH = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\04_density_valley.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

log("=== 04_density_valley start ===")
df = pd.read_excel(rf"{DATA_DIR}\附件1.xlsx", sheet_name='sheet1')
df = df[df['线路类型'].astype(str) != 'route_type'].reset_index(drop=True)
for c in ['索赔金额','实际赔付金额']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['超额u'] = df['索赔金额'] - df['实际赔付金额']
df['赔付p'] = df['实际赔付金额']

# (1) 全局 u 细直方（20元一箱，0-1500），找跳跃/谷
log("--- (1) u 全局直方（20元箱）---")
h, edges = np.histogram(df['超额u'], bins=np.arange(0, 1500, 20))
for i in range(len(h)):
    bar = '#' * int(h[i] / 25)
    log(f"  [{edges[i]:4.0f},{edges[i+1]:4.0f}) {h[i]:5d} {bar}")

# (2) u/p 比值细直方（0-5，0.05 一箱）
log("\n--- (2) u/p 比值直方（0.05箱，上限5）---")
ratio = df['超额u'] / df['赔付p']
h2, e2 = np.histogram(ratio[ratio <= 5], bins=np.arange(0, 5.01, 0.05))
for i in range(len(h2)):
    bar = '#' * int(h2[i] / 25)
    log(f"  [{e2[i]:4.2f},{e2[i+1]:4.2f}) {h2[i]:5d} {bar}")
log(f"u/p > 5 的数量 = {(ratio>5).sum()}  u/p 中位数={ratio.median():.3f}")

# (3) r = p/(p+u) 细直方（0-1，0.02 一箱）
log("\n--- (3) 赔付率 r 直方（0.02箱）---")
r = df['赔付p'] / (df['赔付p'] + df['超额u'])
h3, e3 = np.histogram(r, bins=np.arange(0, 1.0001, 0.02))
for i in range(len(h3)):
    bar = '#' * int(h3[i] / 25)
    log(f"  [{e3[i]:4.2f},{e3[i+1]:4.2f}) {h3[i]:5d} {bar}")

# (4) 窄赔付切片内 u 的分位（100元宽切片，观察 u 条件分布平滑性）
log("\n--- (4) 窄赔付切片（100元宽）内 u 分位数 ---")
bins = np.arange(0, 1600, 100)
df['p_slice'] = pd.cut(df['赔付p'], bins)
tab = df.groupby('p_slice', observed=True)['超额u'].quantile([.5,.8,.85,.9,.95,.97,.99]).unstack()
tab.columns = ['q50','q80','q85','q90','q95','q97','q99']
tab['n'] = df.groupby('p_slice', observed=True).size()
log(tab.round(1).to_string())

# (5) 对数变换 log1p(u) 与 log1p(u/p) 的直方（找多峰）
from collections import Counter
log("\n--- (5) log1p(u) 直方（0.2箱）---")
lu = np.log1p(df['超额u'])
h5, e5 = np.histogram(lu, bins=np.arange(0, 8.5, 0.2))
for i in range(len(h5)):
    if h5[i] > 0:
        bar = '#' * int(h5[i] / 40)
        log(f"  [{e5[i]:4.1f},{e5[i+1]:4.1f}) {h5[i]:5d} {bar}")

log("\n=== 04_density_valley done ===")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG_PATH)
