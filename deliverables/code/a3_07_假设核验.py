# -*- coding: utf-8 -*-
"""
A3-07 假设核验补充勘察（盲测 Run-2，工作区 agent_workspace_B2）
职责：为 A2 假设逐条回填补齐此前 T/U 事实未覆盖的两项：
  U3 行序漂移检验（A2-14 明确"是否存在批次/时间漂移待 A3 检验"）：
      附件1/附件2 各自行序十分位/四分位块内 目标与关键特征中位走势；
  U4 附件1 25 组完全重复行的行号间隔（重复机制旁证）。
输入：output/tables/附件1_clean.csv、附件2_clean.csv（a3_06 最终版）
输出：output/logs/a3_07_假设核验.log
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_07_假设核验.log"
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
np.random.seed(20251004)  # 无随机操作
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("=" * 100)
print("A3-07 假设核验补充勘察 | pandas", pd.__version__)
print("=" * 100)

d1 = pd.read_csv(WS / "output" / "tables" / "附件1_clean.csv")
d2 = pd.read_csv(WS / "output" / "tables" / "附件2_clean.csv")
d1["相对超额"] = (d1["索赔金额"] - d1["实际赔付金额"]) / d1["索赔金额"]
d2["相对超额_代理"] = np.nan  # 附件2 无目标，仅看特征漂移

print("\n[U3] 行序漂移检验（附表1 无绝对时间字段（列名字典 §2），以行序为唯一可用顺序代理）")
blk = pd.qcut(d1.index, 10, labels=False)
t = pd.DataFrame({
    "单数": d1.groupby(blk).size(),
    "赔付中位": d1.groupby(blk)["实际赔付金额"].median(),
    "索赔中位": d1.groupby(blk)["索赔金额"].median(),
    "相对超额中位": d1.groupby(blk)["相对超额"].median(),
    "配送超时触墙率": d1.groupby(blk)["配送超时时长_触墙"].mean(),
    "异常原因缺失率": d1.groupby(blk)["异常原因_缺失"].mean(),
})
print("  附件1 行序十分位块：")
print(t.to_string(float_format=lambda v: f"{v:.4g}"))
first, last = t["赔付中位"].iloc[0], t["赔付中位"].iloc[-1]
print(f"  首块 vs 末块：赔付中位 {first:.4g} vs {last:.4g}（差 {(last-first)/first*100:+.2f}%）；"
      f"相对超额中位 {t['相对超额中位'].iloc[0]:.4g} vs {t['相对超额中位'].iloc[-1]:.4g}")

blk2 = pd.qcut(d2.index, 4, labels=False)
t2 = pd.DataFrame({
    "单数": d2.groupby(blk2).size(),
    "索赔中位": d2.groupby(blk2)["索赔金额"].median(),
    "保价中位": d2.groupby(blk2)["保价金额"].median(),
    "配送超时触墙率": d2.groupby(blk2)["配送超时时长_触墙"].mean(),
    "异常原因缺失率": d2.groupby(blk2)["异常原因_缺失"].mean(),
})
print("  附件2 行序四分位块：")
print(t2.to_string(float_format=lambda v: f"{v:.4g}"))

# 块间差异的显著性（Kruskal-Wallis，scipy；仅描述行序是否承载结构）
from scipy import stats
g = [d1.loc[blk == k, "实际赔付金额"].values for k in range(10)]
h, p = stats.kruskal(*g)
print(f"  附件1 赔付金额按行序十块的 Kruskal-Wallis：H={h:.4g} p={p:.4g}")
g2 = [d1.loc[blk == k, "相对超额"].values for k in range(10)]
h2, p2 = stats.kruskal(*g2)
print(f"  附件1 相对超额按行序十块的 Kruskal-Wallis：H={h2:.4g} p={p2:.4g}")

print("\n[U4] 附件1 25 组完全重复行的行号跨度（重复机制旁证）")
feats = [c for c in d1.columns if c not in ("行ID", "相对超额", "配送超时时长_sl",
                                            "妥投到进线时长_sl", "始发网点发单量_log10",
                                            "目的网点发单量_log10", "网点万单理赔率_均值",
                                            "寄件人id_频率", "收件人id_频率", "始发城市_频率",
                                            "目的城市_频率", "城市对_频率", "索赔差额",
                                            "赔付索赔比", "实际赔付金额_log10", "索赔金额_log10",
                                            "保价索赔比")]
dup_all = d1.duplicated(subset=feats, keep=False)
grp = d1[dup_all].groupby(feats, dropna=False)["行ID"].apply(list)
grp = grp[grp.apply(len) == 2]
spans = sorted(b - a for a, b in grp)
print(f"  涉重行数={int(dup_all.sum())} 成对组数={len(grp)}（应与 a3_02 [8] 的 25 组一致）")
vc = pd.Series(spans).value_counts().sort_index()
print("  行号跨度（组内两行距离）分布:", {int(k): int(v) for k, v in vc.items()})
print(f"  跨度=1（物理相邻）组数={int((pd.Series(spans)==1).sum())}；跨度中位={int(np.median(spans))}；最大跨度={int(max(spans))}")

print("\nA3-07 假设核验补充勘察结束。事实编号 U3（行序漂移）、U4（重复行间隔）。")
