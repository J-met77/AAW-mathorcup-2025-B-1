# -*- coding: utf-8 -*-
"""05_clean_features.py —— A3 数据工程：清洗 + 特征工程，产出可复用干净数据集。
输入：data/附件1.xlsx, data/附件2.xlsx
输出：output/tables/附件1_clean.csv, output/tables/附件2_clean.csv
日志：output/logs/05_clean_features.log
清洗决策（全部有日志事实支撑，详见 00_admin/A3_数据报告.md）：
  F1 去除嵌入英文变量名行（各1行）
  F2 数值列显式转数值
  F3 保价金额负值(7.72%, 商业上不可能) → 置0 + 保价异常标记
  F4 异常原因缺失(48.6%) → "NoReport"（业务：未提报异常）；进线渠道缺失(0.84%) → "Unknown"
  F5 配送超时时长：保留原值 + 饱和标记 [427000,428000]
  F6 妥投到进线时长：保留原值 + 负值标记 + log1p(|x|)
  F7 万单理赔率负值：保留原值 + 负值标记
  F8 网点发单量/万单理赔率：加 log1p 变换列（量纲跨度 33~1e10）
  F9 索赔金额：log1p 列 + 下限标记(=100)
  F10 类别列统一按 category 处理；高频ID列做频次编码（附件1∪附件2联合频次，无标签泄漏）
"""
import sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
DATA_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\data"
TAB_DIR = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\tables"
LOG_PATH = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\05_clean_features.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

log("=== 05_clean_features start ===")
df1 = pd.read_excel(rf"{DATA_DIR}\附件1.xlsx", sheet_name='sheet1')
df2 = pd.read_excel(rf"{DATA_DIR}\附件2.xlsx", sheet_name='Sheet1')

# F1
df1 = df1[df1['线路类型'].astype(str) != 'route_type'].reset_index(drop=True)
df2 = df2[df2['运单号'].astype(str) != 'ID'].reset_index(drop=True)
log(f"F1 去英文行后：附件1={df1.shape} 附件2={df2.shape}")

NUM = ['保价金额','配送超时时长','妥投到进线时长','索赔金额','始发网点发单量','始发网点万单理赔率',
       '始发网点赔付比例','目的网点发单量','目的网点万单理赔率','目的网点赔付比例']
CAT = ['线路类型','是否c2c','是否生鲜妥投及时','寄件是否内部','异常原因','进线渠道','商品类型',
       '新旧程度','寄件B/C','进线人身份']
IDH = ['始发城市','目的城市','寄件人id','收件人id']

def clean(df, is_train):
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # F3
    neg_mask = df['保价金额'] < 0
    df['保价异常标记'] = neg_mask.astype(int)
    df.loc[neg_mask, '保价金额'] = 0.0
    # F4
    df['异常原因'] = df['异常原因'].fillna('NoReport')
    df['进线渠道'] = df['进线渠道'].fillna('Unknown')
    # F5
    df['配送超时_饱和标记'] = df['配送超时时长'].between(427000, 428000).astype(int)
    # F6
    df['妥投进线_负值标记'] = (df['妥投到进线时长'] < 0).astype(int)
    df['妥投进线_logabs'] = np.log1p(df['妥投到进线时长'].abs())
    # F7
    for c in ['始发网点万单理赔率','目的网点万单理赔率']:
        df[c.replace('万单理赔率','理赔率负值标记')] = (df[c] < 0).astype(int)
    # F8
    for c in ['始发网点发单量','目的网点发单量','始发网点万单理赔率','目的网点万单理赔率']:
        df[c + '_log1p'] = np.log1p(df[c].clip(lower=0))
    for c in ['始发网点赔付比例','目的网点赔付比例','保价金额']:
        df[c + '_log1p'] = np.log1p(df[c].clip(lower=0))
    # F9
    df['索赔金额_log1p'] = np.log1p(df['索赔金额'])
    df['索赔金额_下限标记'] = (df['索赔金额'] <= 100.001).astype(int)
    # F10 类别
    for c in CAT:
        df[c] = df[c].astype(str)
    return df

df1 = clean(df1, True)
df2 = clean(df2, False)

# 频次编码（联合频次，不含任何标签信息）
for c in IDH + ['异常原因','商品类型']:
    freq = pd.concat([df1[c], df2[c]]).value_counts()
    df1[c + '_freq'] = df1[c].map(freq).astype(float)
    df2[c + '_freq'] = df2[c].map(freq).astype(float)
df1['城市对_freq'] = (df1['始发城市'].astype(str) + '_' + df1['目的城市'].astype(str)).map(
    pd.concat([df1['始发城市'].astype(str)+'_'+df1['目的城市'].astype(str),
               df2['始发城市'].astype(str)+'_'+df2['目的城市'].astype(str)]).value_counts()).astype(float)
df2['城市对_freq'] = (df2['始发城市'].astype(str) + '_' + df2['目的城市'].astype(str)).map(
    pd.concat([df1['始发城市'].astype(str)+'_'+df1['目的城市'].astype(str),
               df2['始发城市'].astype(str)+'_'+df2['目的城市'].astype(str)]).value_counts()).astype(float)
df1['城市对_freq'] = df1['城市对_freq'].fillna(1.0)
df2['城市对_freq'] = df2['城市对_freq'].fillna(1.0)

# 目标列
df1['实际赔付金额'] = pd.to_numeric(df1['实际赔付金额'], errors='coerce')
df1['索赔差额'] = df1['实际赔付金额'] - df1['索赔金额']
df1['超额索赔额'] = -df1['索赔差额']
assert df1['实际赔付金额'].notna().all(), "附件1 实际赔付金额存在无法转换的值"

log(f"清洗后缺失核查：附件1 缺失单元格总数 = {int(df1.isna().sum().sum())}")
log(f"清洗后缺失核查：附件2 缺失单元格总数 = {int(df2.isna().sum().sum())}")
log(f"索赔金额=100 精确计数（附件1）= {(df1['索赔金额']<=100.001).sum()} ({(df1['索赔金额']<=100.001).mean()*100:.1f}%)")
log(f"附件1 特征列数（含衍生）= {df1.shape[1]}，附件2 = {df2.shape[1]}")
log(f"新衍射列：{[c for c in df1.columns if c.endswith(('标记','log1p','freq')) or c in ('妥投进线_logabs','索赔金额_log1p')]}")

df1.to_csv(rf"{TAB_DIR}\附件1_clean.csv", index=False, encoding='utf-8-sig')
df2.to_csv(rf"{TAB_DIR}\附件2_clean.csv", index=False, encoding='utf-8-sig')
log(f"[saved] {TAB_DIR}\\附件1_clean.csv  {df1.shape}")
log(f"[saved] {TAB_DIR}\\附件2_clean.csv  {df2.shape}")
log("=== 05_clean_features done ===")
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG_PATH)
