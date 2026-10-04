# -*- coding: utf-8 -*-
"""03_target_structure.py —— A3 数据勘察第三步：(实际赔付金额 p, 索赔差额 d) 二维结构。
   目的：为 A4 设计标注规则提供数据证据 ——
   (a) 超额索赔 u = -d = 索赔金额-实际赔付 在赔付分箱内的分位数走势（绝对量形态？）
   (b) 比值形态：1 - 赔付/索赔 在赔付分箱内的分位数走势（比值形态？）
   (c) 混合形态：u - k*p 归一后的分位数走势（线性形态？）
   (d) 密度结构：u 在每个赔付箱内的直方，看是否"密集主体+稀疏尾部"
   (e) 目标列与候选特征的秩相关（为 Q2 特征预筛提供事实）
产出：output/logs/03_target_structure.log
"""
import sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DATA_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\data"
LOG_PATH = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\03_target_structure.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

log("=== 03_target_structure start ===")
df = pd.read_excel(rf"{DATA_DIR}\附件1.xlsx", sheet_name='sheet1')
df = df[df['线路类型'].astype(str) != 'route_type'].reset_index(drop=True)
for c in ['索赔金额','实际赔付金额','保价金额']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['索赔差额_d'] = df['实际赔付金额'] - df['索赔金额']
df['超额u'] = -df['索赔差额_d']                     # u = 索赔金额 - 实际赔付 >= 0
df['赔付率_r'] = df['实际赔付金额'] / df['索赔金额']   # r = p / (p+u) in (0,1]
log(f"n={len(df)}  u范围=[{df['超额u'].min():.2f},{df['超额u'].max():.2f}]  r范围=[{df['赔付率_r'].min():.4f},{df['赔付率_r'].max():.4f}]")

# --- (a) u 的分位数随赔付十分位 ---
log("\n--- (a) 超额u 分位数 × 赔付十分位（看绝对量阈值是否随 p 上升）---")
df['pay_bin10'] = pd.qcut(df['实际赔付金额'], 10, duplicates='drop')
qs = [.50,.75,.85,.90,.95,.97,.98,.99,.997]
tab_a = df.groupby('pay_bin10', observed=True)['超额u'].quantile(qs).unstack()
tab_a.columns = [f"u_q{int(q*1000)/1000:g}" for q in qs]
tab_a['u_max'] = df.groupby('pay_bin10', observed=True)['超额u'].max()
tab_a['pay_med'] = df.groupby('pay_bin10', observed=True)['实际赔付金额'].median()
log(tab_a.round(2).to_string())

# --- (b) r = p/(p+u) 的分位数随赔付十分位 ---
log("\n--- (b) 赔付率r 分位数 × 赔付十分位（若比值形态则各行近似水平）---")
tab_b = df.groupby('pay_bin10', observed=True)['赔付率_r'].quantile(qs).unstack()
tab_b.columns = [f"r_q{int(q*1000)/1000:g}" for q in qs]
log(tab_b.round(4).to_string())

# --- (c) u - k*p（k 取全局 u 对 p 的线性回归斜率）后的分位数 ---
k = np.polyfit(df['实际赔付金额'], df['超额u'], 1)
log(f"\n--- (c) 线性形态检验：u ≈ {k[0]:.4f}*p + {k[1]:.2f}（OLS 全局）---")
df['u_lin'] = df['超额u'] - (k[0]*df['实际赔付金额'] + k[1])
tab_c = df.groupby('pay_bin10', observed=True)['u_lin'].quantile([.85,.95,.97,.99]).unstack()
tab_c.columns = [f"ulin_q{q}" for q in [.85,.95,.97,.99]]
log(tab_c.round(2).to_string())

# --- (d) 密度结构：最低赔付箱 & 最高赔付箱的 u 直方（看密集主体+稀疏尾部）---
log("\n--- (d) 密度结构（u 等宽直方，前两个与后两个赔付箱）---")
for b in df['pay_bin10'].unique()[[0,1,-2,-1]]:
    sub = df.loc[df['pay_bin10']==b, '超额u']
    hist = pd.cut(sub, bins=np.arange(0, 1600, 100)).value_counts().sort_index()
    log(f"\n箱 {b}  (pay中位={df.loc[df['pay_bin10']==b,'实际赔付金额'].median():.0f}, n={len(sub)})")
    log("  " + " | ".join(f"[{iv.left:.0f},{iv.right:.0f}):{n}" for iv, n in hist.items() if n>0))

# --- (e) 全局 u 分位与占比对照（85% / 97% 软约束的天然切点）---
log("\n--- (e) 全局 u 分位数 ---")
log(df['超额u'].quantile([.85,.90,.95,.97,.98,.99,.995,.999]).round(2).to_string())
log(f"u<=100 占比={(df['超额u']<=100).mean()*100:.2f}%  u<=150 占比={(df['超额u']<=150).mean()*100:.2f}%  u<=200 占比={(df['超额u']<=200).mean()*100:.2f}%")

# --- (f) 目标与特征的秩相关（Q2 预筛事实）---
log("\n--- (f) 实际赔付金额 与 数值特征的 Spearman 秩相关 ---")
numfeat = ['索赔金额','保价金额','配送超时时长','妥投到进线时长','始发网点发单量','始发网点万单理赔率',
           '始发网点赔付比例','目的网点发单量','目的网点万单理赔率','目的网点赔付比例','新旧程度','始发城市','目的城市']
sp = df[numfeat].corrwith(df['实际赔付金额'], method='spearman', numeric_only=True).sort_values(key=abs, ascending=False)
log(sp.round(4).to_string())

# --- (g) 商品类型 × 赔付中位数（类别信息量）---
log("\n--- (g) 商品类型 × 实际赔付金额中位数 / 索赔金额中位数 ---")
g = df.groupby('商品类型').agg(pay_med=('实际赔付金额','median'), claim_med=('索赔金额','median'), n=('实际赔付金额','size'))
log(g.sort_values('pay_med', ascending=False).round(1).to_string())

log("\n=== 03_target_structure done ===")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG_PATH)
