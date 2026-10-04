# -*- coding: utf-8 -*-
"""
A3-05 附件1 vs 附件2 漂移对比（盲测 Run-2，工作区 agent_workspace_B2）
职责：类别列比例漂移、数值列分布差异（KS）、附件2 未见类别与新 ID 规模——
      只陈述事实与规模，为 A4/A6 的编码与泛化提供依据；不做建模结论。
输入：output/tables/附件1_clean.csv, output/tables/附件2_clean.csv（a3_03 基础清洗版）
输出：output/logs/a3_05_对比漂移.log
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_05_对比漂移.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.so = sys.stdout

    def write(self, s):
        self.so.write(s)
        self.f.write(s)

    def flush(self):
        self.so.flush()
        self.f.flush()


sys.stdout = Tee(LOG)
np.random.seed(20251004)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print("=" * 100)
print("A3-05 附件1 vs 附件2 漂移对比 | pandas", pd.__version__)
print("=" * 100)

d1 = pd.read_csv(WS / "output" / "tables" / "附件1_clean.csv")
d2 = pd.read_csv(WS / "output" / "tables" / "附件2_clean.csv")
print(f"[0] 附件1_clean {d1.shape}；附件2_clean {d2.shape}")

CAT_COLS = ["线路类型", "是否c2c", "是否生鲜妥投及时", "寄件是否内部", "异常原因",
            "进线渠道", "商品类型", "新旧程度", "寄件B/C", "进线人身份"]
NUM_COLS = ["保价金额", "配送超时时长", "妥投到进线时长", "索赔金额",
            "始发网点发单量", "始发网点万单理赔率", "始发网点赔付比例",
            "目的网点发单量", "目的网点万单理赔率", "目的网点赔付比例"]

# ---------------------------------------------------------- 1. 类别列漂移
print("\n[1] 类别列比例漂移（占比差 = 附件2 占比 - 附件1 占比，百分点）")
for c in CAT_COLS:
    p1 = d1[c].value_counts(normalize=True, dropna=False)
    p2 = d2[c].value_counts(normalize=True, dropna=False)
    cats = sorted(set(p1.index.fillna("__NaN__")) | set(p2.index.fillna("__NaN__")),
                  key=lambda k: -(p1.get(k, 0) + p2.get(k, 0)))
    tvd = 0.5 * sum(abs(p1.get(k, 0) - p2.get(k, 0)) for k in cats)
    print(f"  --- {c} --- 总变差距离 TVD={tvd*100:.3f}pp")
    only1 = [k for k in p1.index.fillna("__NaN__") if k not in set(p2.index.fillna("__NaN__"))]
    only2 = [k for k in p2.index.fillna("__NaN__") if k not in set(p1.index.fillna("__NaN__"))]
    if only1:
        print(f"    仅附件1 有的类别: {only1}（附件2 未见，占比合计 {p1[only1].sum()*100:.3f}%）")
    if only2:
        print(f"    仅附件2 新见类别: {only2}（占比合计 {p2[only2].sum()*100:.3f}%）")
    for k in cats[:8]:
        a, b = p1.get(k, 0) * 100, p2.get(k, 0) * 100
        print(f"    {str(k):<40} 附件1={a:8.3f}%  附件2={b:8.3f}%  差={b-a:+7.3f}pp")
    if len(cats) > 8:
        rest = cats[8:]
        s1 = sum(p1.get(k, 0) for k in rest) * 100
        s2 = sum(p2.get(k, 0) for k in rest) * 100
        print(f"    （其余 {len(rest)} 类合计          附件1={s1:8.3f}%  附件2={s2:8.3f}%  差={s2-s1:+7.3f}pp）")

# ---------------------------------------------------------- 2. 数值列分布差异
print("\n[2] 数值列两样本 KS 检验与分位差（scipy.stats.ks_2samp；仅描述差异规模）")
print(f"  {'列':<14}{'KS统计量':>10}{'p值':>12}{'附件1中位':>12}{'附件2中位':>12}{'中位差%':>10}")
for c in NUM_COLS:
    a = d1[c].dropna().astype(float)
    b = d2[c].dropna().astype(float)
    ks, p = stats.ks_2samp(a, b)
    m1, m2 = a.median(), b.median()
    rel = (m2 - m1) / m1 * 100 if m1 != 0 else float("nan")
    print(f"  {c:<14}{ks:>10.4f}{p:>12.2g}{m1:>12.4g}{m2:>12.4g}{rel:>+9.2f}%")
    q1 = a.quantile([.01, .25, .75, .99]).round(4).tolist()
    q2 = b.quantile([.01, .25, .75, .99]).round(4).tolist()
    print(f"      P1/P25/P75/P99  附件1={q1}  附件2={q2}")

# ---------------------------------------------------------- 3. ID/城市覆盖与冷启动规模
print("\n[3] 高基数 ID/城市 覆盖关系（附件2 相对附件1 的未见规模——编码与泛化依据）")
for c in ["线路类型", "始发城市", "目的城市", "寄件人id", "收件人id"]:
    s1, s2 = set(d1[c].dropna().unique()), set(d2[c].dropna().unique())
    unseen = s2 - s1
    mask = ~d2[c].isin(s1)
    n_unseen_rows = int(mask.sum())
    print(f"  {c}: 附件1 取值数={len(s1)}  附件2 取值数={len(s2)}  附件2 未见取值数={len(unseen)}"
          f"  涉及附件2 行数={n_unseen_rows} ({n_unseen_rows/len(d2)*100:.2f}%)")
    if 0 < len(unseen) <= 12:
        cnt = d2.loc[mask, c].value_counts()
        print(f"      未见取值明细: {dict(cnt)}")

# 未见寄件人/收件人的行画像（冷启动行是否同一批）
both_unseen = (~(d2["寄件人id"].isin(set(d1["寄件人id"].dropna())))) & \
              (~(d2["收件人id"].isin(set(d1["收件人id"].dropna()))))
print(f"  寄件人与收件人 id 均未见的附件2 行数={int(both_unseen.sum())} ({both_unseen.mean()*100:.2f}%)")

# 高频 ID 的跨表一致性（top 寄件人是否同批）
print("\n[4] 高频 ID 跨表一致性（top5 寄件人/收件人）")
for c in ["寄件人id", "收件人id"]:
    t1 = d1[c].value_counts().head(5)
    t2 = d2[c].value_counts().head(5)
    print(f"  {c} 附件1 top5: {dict(t1)}")
    print(f"  {c} 附件2 top5: {dict(t2)}")
    overlap = set(t1.index) & set(t2.index)
    print(f"    top5 交集: {sorted(overlap)}")

print("\nA3-05 对比漂移结束。")
