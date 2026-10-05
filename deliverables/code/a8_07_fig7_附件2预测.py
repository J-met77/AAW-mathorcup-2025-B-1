# -*- coding: utf-8 -*-
"""
a8_07_fig7_附件2预测.py —— fig7 附件2 预测结果分布：(a) 赔付预测 ŷ 分布；(b) 风险标注预测占比 vs 附件1 标注占比
数据来源：
  - 附件2_赔付预测.csv（2792 行，列 ŷ；df.describe 用于交叉核对）
  - 附件2_风险预测.csv（2792 行；方式2 argmax 判类）
  - 附件1_风险标注.csv（11167 行；Q1 规则标注，作为占比对照）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (CLASS_COLORS, PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig)

import matplotlib.pyplot as plt

SCRIPT = "a8_07_fig7_附件2预测"
log = get_logger(SCRIPT)
log_font_check(log)

CLASSES = ["合理诉求", "诉求偏高", "严重超额"]

pay = read_csv("附件2_赔付预测.csv")
assert len(pay) == 2792, f"附件2_赔付预测 行数异常：{len(pay)}"
yh = pay["ŷ"].to_numpy()
assert (yh > 0).all()
p50, p90 = np.percentile(yh, [50, 90])
log.info("附件2 赔付预测：n=%d mean=%.2f P50=%.2f P90=%.2f min=%.2f max=%.2f 总额=%.0f",
         len(yh), yh.mean(), p50, p90, yh.min(), yh.max(), yh.sum())

pred = read_csv("附件2_风险预测.csv")
assert len(pred) == 2792
cp = pred["风险标注"].value_counts()
lab = read_csv("附件1_风险标注.csv")
cl = lab["风险标注"].value_counts()
assert set(cp.index) <= set(CLASSES) and set(cl.index) == set(CLASSES)
share_pred = [cp[c] / len(pred) for c in CLASSES]
share_lab = [cl[c] / len(lab) for c in CLASSES]
log.info("附件2 风险预测占比：%s", {c: f"{cp[c]}({share_pred[i]:.2%})" for i, c in enumerate(CLASSES)})
log.info("附件1 标注占比（对照）：%s", {c: f"{cl[c]}({share_lab[i]:.2%})" for i, c in enumerate(CLASSES)})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.8, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

# ---- (a) ŷ 分布 ----
axa.hist(yh, bins=60, color=PAL["blue"], alpha=0.78, edgecolor="white", lw=0.4)
axa.axvline(p50, color=PAL["orange"], lw=1.6, label=f"P50 = {p50:.2f} 元")
axa.axvline(p90, color=PAL["purple"], lw=1.6, label=f"P90 = {p90:.2f} 元")
axa.set_xlabel("附件2 赔付预测金额（元）", fontfamily=YH_FONT)
axa.set_ylabel("运单数", fontfamily=YH_FONT)
axa.set_title(f"(a) 附件2 赔付预测分布（n=2,792，均值 {yh.mean():.2f} 元）\n（附件2_赔付预测.csv，列 ŷ）",
              fontsize=11.5, fontfamily=YH_FONT)
axa.legend(prop={"family": YH_FONT, "size": 9.5})
axa.text(0.98, 0.95, f"预测总额 Σŷ = {yh.sum():,.0f} 元", transform=axa.transAxes,
         ha="right", va="top", fontsize=9.5, fontfamily=YH_FONT,
         bbox=dict(boxstyle="round,pad=0.4", fc="#F5F7FA", ec="#B9C4CE", alpha=0.95))

# ---- (b) 占比对比 ----
xs = np.arange(3)
w = 0.36
b1 = axb.bar(xs - w / 2, [share_lab[i] * 100 for i in range(3)], w,
             color=[CLASS_COLORS[c] for c in CLASSES], alpha=0.55,
             edgecolor=[CLASS_COLORS[c] for c in CLASSES], lw=1.4,
             label="附件1 标注（Q1 规则，n=11,167）")
b2 = axb.bar(xs + w / 2, [share_pred[i] * 100 for i in range(3)], w,
             color=[CLASS_COLORS[c] for c in CLASSES],
             hatch="//", edgecolor=[CLASS_COLORS[c] for c in CLASSES], lw=1.0,
             label="附件2 预测（方式2，n=2,792）")
for i in range(3):
    axb.text(xs[i] - w / 2, share_lab[i] * 100 + 1.6,
             f"{share_lab[i]:.1%}\n({cl[CLASSES[i]]:,})", ha="center", fontsize=8.8, fontfamily=YH_FONT)
    axb.text(xs[i] + w / 2, share_pred[i] * 100 + 1.6,
             f"{share_pred[i]:.1%}\n({cp[CLASSES[i]]:,})", ha="center", fontsize=8.8, fontfamily=YH_FONT)
axb.set_xticks(xs, CLASSES, fontfamily=YH_FONT)
axb.set_ylim(0, 108)
axb.set_ylabel("占比（%）", fontfamily=YH_FONT)
axb.set_title("(b) 风险标注占比：附件1 规则标注 vs 附件2 预测\n（附件1_风险标注.csv / 附件2_风险预测.csv）",
              fontsize=11.5, fontfamily=YH_FONT)
axb.legend(prop={"family": YH_FONT, "size": 9}, loc="upper right")

fig.suptitle("图7　附件2 预测结果分布（数值溯源见框注与日志）", fontsize=13, y=1.02, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig7_附件2预测分布.png", log)
log.info("完成：%s", SCRIPT)
