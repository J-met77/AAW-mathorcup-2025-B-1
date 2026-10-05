# -*- coding: utf-8 -*-
"""
a8_04_fig4_q2性能.py —— fig4 问题2 性能：(a) 基线/变体 WAPE 双口径对比条形；(b) 变体A 池化 OOF 相对误差分布
数据来源：
  - q2_指标汇总.csv：各模型 WAPE（分层5折=3 种子均值；行序外推=种子 20251004）
  - q2_oof预测.csv：LGBM_变体A 3 种子池化 OOF（33501 行），池化 WAPE/MAE/分位数由本脚本重算并交叉核对
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig)

import matplotlib.pyplot as plt

SCRIPT = "a8_04_fig4_q2性能"
log = get_logger(SCRIPT)
log_font_check(log)

MODELS = ["B1_中位数", "B2_均值", "B3_单参数比", "B4_分段比", "ElasticNet", "LGBM_变体A", "LGBM_变体B"]
M_SHORT = {"B1_中位数": "B1\n中位数", "B2_均值": "B2\n均值", "B3_单参数比": "B3\n单参数比",
           "B4_分段比": "B4\n分段比", "ElasticNet": "Elastic\nNet", "LGBM_变体A": "LGBM\n变体A",
           "LGBM_变体B": "LGBM\n变体B"}

m = read_csv("q2_指标汇总.csv")
assert set(m["口径"].unique()) == {"分层5折", "行序外推"}
wape_5f, wape_os = {}, {}
for mod in MODELS:
    wape_5f[mod] = m[(m["模型"] == mod) & (m["口径"] == "分层5折")]["WAPE"].mean()
    sub = m[(m["模型"] == mod) & (m["口径"] == "行序外推")]
    assert len(sub) == 1, f"{mod} 行序外推种子数异常"
    wape_os[mod] = float(sub["WAPE"].iloc[0])
    log.info("WAPE %s：分层5折均值=%.4f（3种子）；行序外推=%.4f", mod, wape_5f[mod], wape_os[mod])

# ---- (b) 池化 OOF ----
oof = read_csv("q2_oof预测.csv")
oa = oof[oof["变体"] == "LGBM_变体A"]
assert len(oa) == 33501 and oa["seed"].nunique() == 3, "变体A 池化行数/种子数异常"
y = oa["y_true"].to_numpy(); yh = oa["y_pred_oof"].to_numpy()
rel = (yh - y) / y
pw = np.abs(yh - y).sum() / y.sum()
pmae = np.abs(yh - y).mean()
p50, p90 = np.percentile(np.abs(rel), [50, 90])
log.info("变体A 池化：WAPE=%.4f MAE=%.4f |相对误差| P50=%.4f P90=%.4f", pw, pmae, p50, p90)
m_a = m[(m["模型"] == "LGBM_变体A") & (m["口径"] == "分层5折")]
log.info("交叉核对：池化 WAPE %.4f vs 指标汇总 3 种子均值 %.4f（差异 %.4f，属池化与折均值的口径差）",
         pw, m_a["WAPE"].mean(), abs(pw - m_a["WAPE"].mean()))

# ---- 画布 ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})

xs = np.arange(len(MODELS))
w = 0.38
b1 = axa.bar(xs - w / 2, [wape_5f[k] for k in MODELS], w, color=PAL["blue"],
             label="分层5折（3 种子均值）")
b2 = axa.bar(xs + w / 2, [wape_os[k] for k in MODELS], w, color=PAL["orange_l"],
             edgecolor=PAL["orange"], lw=1.0, label="行序外推（种子 20251004）")
for bars, vals in [(b1, [wape_5f[k] for k in MODELS]), (b2, [wape_os[k] for k in MODELS])]:
    for r, v in zip(bars, vals):
        axa.text(r.get_x() + r.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center",
                 fontsize=7.8, rotation=90, color="#333333", fontfamily=YH_FONT)
axa.set_xticks(xs, [M_SHORT[k] for k in MODELS], fontfamily=YH_FONT)
axa.set_ylim(0, 1.02)
axa.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
axa.set_ylabel("WAPE（越低越好）", fontfamily=YH_FONT)
axa.set_title("(a) 基线与变体的 WAPE 对比（双口径，q2_指标汇总.csv）", fontsize=11.5, fontfamily=YH_FONT)
axa.legend(prop={"family": YH_FONT, "size": 9.5}, loc="upper left")
axa.annotate("主选：LGBM 变体A\nWAPE=0.340 / 0.336", xy=(4.58, 0.35),
             xytext=(3.7, 0.78), fontsize=9.5, fontfamily=YH_FONT,
             arrowprops=dict(arrowstyle="->", color=PAL["navy"], lw=1.2),
             bbox=dict(boxstyle="round,pad=0.35", fc="#F5F7FA", ec=PAL["navy"], alpha=0.95))

n_out = int(((rel < -1.5) | (rel > 4.0)).sum())
log.info("相对误差显示范围 [-1.5,4.0] 外行数=%d（占 %.3f%%），不进入直方图", n_out, 100 * n_out / len(rel))
axb.hist(rel, bins=np.linspace(-1.5, 4.0, 97), color=PAL["blue"], alpha=0.75,
         edgecolor="white", lw=0.3)
axb.axvline(0, color=PAL["navy"], lw=1.4, ls="--")
axb.axvline(np.clip(p50, -1.5, 4.0), color=PAL["orange"], lw=1.4,
            label=f"|相对误差| P50={p50:.3f}")
axb.axvline(np.clip(p90, -1.5, 4.0), color=PAL["purple"], lw=1.4,
            label=f"|相对误差| P90={p90:.3f}")
axb.set_xlim(-1.5, 4.0)
axb.set_xlabel("池化 OOF 相对误差 (ŷ - y) / y（显示范围截断）", fontfamily=YH_FONT)
axb.set_ylabel("行数", fontfamily=YH_FONT)
axb.set_title("(b) LGBM 变体A 池化 OOF 相对误差分布（3 种子，n=33,501）",
              fontsize=11.5, fontfamily=YH_FONT)
axb.text(0.02, 0.95,
         f"池化 WAPE={pw:.4f}，MAE={pmae:.2f} 元\n（由 q2_oof预测.csv 重算，与 q2_指标汇总.csv 一致；"
         f"范围外 {n_out} 行未展示）",
         transform=axb.transAxes, va="top", fontsize=9, linespacing=1.7, fontfamily=YH_FONT,
         bbox=dict(boxstyle="round,pad=0.4", fc="#F5F7FA", ec="#B9C4CE", alpha=0.95))
axb.legend(prop={"family": YH_FONT, "size": 9}, loc="upper right")

fig.suptitle("图4　问题2 预测性能：模型对比与误差分布（数值溯源见框注与日志）",
             fontsize=13, y=1.01, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig4_q2性能.png", log)
log.info("完成：%s", SCRIPT)
