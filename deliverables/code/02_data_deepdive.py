# -*- coding: utf-8 -*-
"""02_data_deepdive.py —— A3 数据勘察第二步：
   去除英文变量名行、数值转换、索赔差额=实际赔付-索赔 的符号/分布、
   二维结构初探、异常值陷阱清单（保价负数/超时聚集/进线时长混合单位/理赔率负值）、
   附件1-附件2 分布对照、Result 模板核对。
产出：output/logs/02_data_deepdive.log
"""
import sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DATA_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\data"
LOG_PATH = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\02_data_deepdive.log"

lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

pd.set_option('display.width', 200)

log("=== 02_data_deepdive start ===")
C1 = rf"{DATA_DIR}\附件1.xlsx"; C2 = rf"{DATA_DIR}\附件2.xlsx"; CR = rf"{DATA_DIR}\Result.xlsx"

df1 = pd.read_excel(C1, sheet_name='sheet1')
df2 = pd.read_excel(C2, sheet_name='Sheet1')
dfr = pd.read_excel(CR, sheet_name='Sheet1')

# --- 0) 英文变量名行识别 ---
mask1 = df1['线路类型'].astype(str) == 'route_type'
mask2 = df2['运单号'].astype(str) == 'ID'
log(f"附件1 英文行数={mask1.sum()}  附件2 英文行数={mask2.sum()}")
df1 = df1[~mask1].reset_index(drop=True)
df2 = df2[~mask2].reset_index(drop=True)
log(f"清洗后附件1 shape={df1.shape}  附件2 shape={df2.shape}")

num_cols = ['保价金额','始发城市','目的城市','寄件人id','收件人id','配送超时时长','妥投到进线时长',
            '索赔金额','新旧程度','始发网点发单量','始发网点万单理赔率','始发网点赔付比例',
            '目的网点发单量','目的网点万单理赔率','目的网点赔付比例','实际赔付金额']
for c in num_cols:
    if c in df1.columns:
        df1[c] = pd.to_numeric(df1[c], errors='coerce')
        df2[c] = pd.to_numeric(df2[c], errors='coerce') if c in df2.columns else None
log(f"\n数值转换后 NaN 统计（附件1）：\n{df1[num_cols].isna().sum()[df1[num_cols].isna().sum()>0].to_string()}")
log(f"数值转换后 NaN 统计（附件2）：\n{df2[num_cols].isna().sum()[df2[num_cols].isna().sum()>0].to_string()}")

# --- 1) 核心量：索赔差额 ---
df1['索赔差额'] = df1['实际赔付金额'] - df1['索赔金额']
log(f"\n--- 索赔差额 = 实际赔付金额 - 索赔金额 （附件1）---")
q = df1['索赔差额'].quantile([0, .001, .01, .05, .25, .5, .75, .85, .90, .95, .97, .98, .99, .999, 1])
log(f"分位数：\n{q.to_string()}")
log(f"差额>0 的样本数 = {(df1['索赔差额']>0).sum()} ({(df1['索赔差额']>0).mean()*100:.2f}%)")
log(f"差额=0 的样本数 = {(df1['索赔差额']==0).sum()}")
log(f"实际赔付>索赔 样本数 = {(df1['实际赔付金额']>df1['索赔金额']).sum()}")
log(f"实际赔付<=0 样本数 = {(df1['实际赔付金额']<=0).sum()}，索赔<=0 样本数 = {(df1['索赔金额']<=0).sum()}")
log(f"实际赔付金额分位数：\n{df1['实际赔付金额'].quantile([0,.05,.25,.5,.75,.9,.95,.99,1]).to_string()}")
log(f"索赔金额分位数：\n{df1['索赔金额'].quantile([0,.05,.25,.5,.75,.9,.95,.99,1]).to_string()}")
r = df1['实际赔付金额']/df1['索赔金额']
log(f"赔付/索赔 比值分位数：\n{r.quantile([0,.05,.25,.5,.75,.95,.99,1]).to_string()}")

# --- 2) 差额与赔付的二维结构（等频分箱看差额中位数走势）---
log("\n--- 按实际赔付金额十分位分箱，组内索赔差额分布 ---")
df1['pay_bin'] = pd.qcut(df1['实际赔付金额'], 10, duplicates='drop')
g = df1.groupby('pay_bin', observed=True)['索赔差额'].agg(
    n='count', d_med='median', d_q25=lambda s: s.quantile(.25), d_q75=lambda s: s.quantile(.75),
    d_q90=lambda s: s.quantile(.90), d_q97=lambda s: s.quantile(.97), d_min='min', pay_med='median')
log(g.to_string())

# --- 3) 陷阱清单 ---
log("\n--- 3.1 保价金额：负值结构 ---")
neg = df1['保价金额'] < 0
log(f"负值占比={neg.mean()*100:.2f}%  负值范围=[{df1.loc[neg,'保价金额'].min():.3f}, {df1.loc[neg,'保价金额'].max():.3f}]")
log(f"负值直方(20箱)：\n{pd.cut(df1.loc[neg,'保价金额'], 20).value_counts().sort_index().to_string()}")
log(f"=-1 的数量：{(df1['保价金额']==-1).sum()}")

log("\n--- 3.2 配送超时时长 ---")
s = df1['配送超时时长']
log(f"范围=[{s.min()}, {s.max()}]  在[427000,428000]内的占比={(s.between(427000,428000)).mean()*100:.2f}%")
log(f"分位数：\n{s.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).to_string()}")
log(f"<0 的数量={(s<0).sum()}")

log("\n--- 3.3 妥投到进线时长（疑似混合单位）---")
t = df1['妥投到进线时长']
log(f"范围=[{t.min()}, {t.max()}]")
log(f"分位数：\n{t.quantile([0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1]).to_string()}")
log(f"<0 数量={(t<0).sum()} ({(t<0).mean()*100:.1f}%)  >1e8 数量={(t>1e8).sum()} ({(t>1e8).mean()*100:.1f}%)")
log(f"在[-1e5,0)的数量={((t<0)&(t>-1e5)).sum()}；在[-82100,-81900]的数量={t.between(-82100,-81900).sum()}")

log("\n--- 3.4 万单理赔率负值/发单量极端值 ---")
for c in ['始发网点万单理赔率','目的网点万单理赔率']:
    log(f"{c}: 负值数={(df1[c]<0).sum()} 范围=[{df1[c].min():.2f},{df1[c].max():.2f}]")
for c in ['始发网点发单量','目的网点发单量']:
    log(f"{c}: 附件1范围=[{df1[c].min():.3g},{df1[c].max():.3g}] 附件2范围=[{df2[c].min():.3g},{df2[c].max():.3g}]")
    log(f"   附件1 分位数：{np.round(df1[c].quantile([0,.5,.9,.99,1]).values,1)}  附件2 分位数：{np.round(df2[c].quantile([0,.5,.9,.99,1]).values,1)}")

log("\n--- 3.5 完全重复行 ---")
feat_all = [c for c in df1.columns if c not in ('pay_bin','索赔差额')]
log(f"附件1 重复行数（不含自造列）={df1[feat_all].duplicated().sum()}")

# --- 4) 类别特征对照（附件1 vs 附件2）---
log("\n--- 类别/离散特征分布对照 ---")
for c in ['线路类型','是否c2c','是否生鲜妥投及时','寄件是否内部','异常原因','进线渠道','商品类型','新旧程度','寄件B/C','进线人身份']:
    v1 = df1[c].value_counts(normalize=True, dropna=False)
    v2 = df2[c].value_counts(normalize=True, dropna=False)
    keys = sorted(set(v1.index) | set(v2.index), key=str)
    log(f"\n{c}:")
    for k in keys[:14]:
        log(f"   {str(k):35s} 附件1={v1.get(k,0)*100:6.2f}%  附件2={v2.get(k,0)*100:6.2f}%")

# --- 5) ID 覆盖检查 ---
log("\n--- ID 型特征值域覆盖 ---")
for c in ['始发城市','目的城市','寄件人id','收件人id']:
    s1, s2 = set(df1[c].dropna().unique()), set(df2[c].dropna().unique())
    log(f"{c}: 附件1唯一值={len(s1)}, 附件2={len(s2)}, 附件2不在附件1的={len(s2-s1)}")

# --- 6) Result 模板核对 ---
log("\n--- Result 模板 ---")
log(f"shape={dfr.shape} 列={list(dfr.columns)}")
ids_r = dfr['运单号'].tolist()
ids_2 = pd.to_numeric(df2['运单号']).tolist()
log(f"运单号范围=[{dfr['运单号'].min()},{dfr['运单号'].max()}] 是否与附件2完全一致（顺序+取值）={ids_r==ids_2}")
log(f"重复运单号={dfr['运单号'].duplicated().sum()}")

log("\n=== 02_data_deepdive done ===")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG_PATH)
