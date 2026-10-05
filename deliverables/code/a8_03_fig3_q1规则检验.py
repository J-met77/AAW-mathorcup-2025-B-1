# -*- coding: utf-8 -*-
"""
a8_03_fig3_q1规则检验.py —— fig3 问题1 规则检验：三类超额幅度 e 的箱线/密度对比（自检表核心条目可视化）
数据来源：
  - 附件1_风险标注.csv：e=-索赔差额、风险标注（11167 行，全量）
  - q1_自检表.csv：V1 占比、V3 紧凑度 R、V4 密度差 D_geo/ρ、V5 三类 e 中位（程序化解析并交叉核对）
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (CLASS_COLORS, PAL, YH_FONT, get_logger, log_font_check,
                            read_csv, save_fig)

import matplotlib.pyplot as plt

SCRIPT = "a8_03_fig3_q1规则检验"
log = get_logger(SCRIPT)
log_font_check(log)

CLASSES = ["合理诉求", "诉求偏高", "严重超额"]

# ---- 数据：全量 e ----
lab = read_csv("附件1_风险标注.csv")
assert len(lab) == 11167
e = (-lab["索赔差额"]).to_numpy()
cls = lab["风险标注"].to_numpy()
e_by = {c: e[cls == c] for c in CLASSES}
n_by = {c: len(e_by[c]) for c in CLASSES}
med = {c: float(np.median(e_by[c])) for c in CLASSES}
log.info("全量计数=%s", n_by)
log.info("数据计算的 e 中位：合=%.4f 偏=%.4f 严=%.4f", med["合理诉求"], med["诉求偏高"], med["严重超额"])

# ---- 自检表解析（文字数字逐格溯源）----
chk = read_csv("q1_自检表.csv").set_index("编号")
v1 = str(chk.loc["V1", "数值"])
v3 = str(chk.loc["V3", "数值"])
v4 = str(chk.loc["V4", "数值"])
v5_note = str(chk.loc["V5", "说明"])
s_vals = dict(re.findall(r"s_(\w)=([\d.]+)", v1))
s_map = {"合": "合理诉求", "偏": "诉求偏高", "严": "严重超额"}
for k, c in s_map.items():
    diff = abs(float(s_vals[k]) - n_by[c] / len(lab))
    log.info("V1 交叉核对 %s：自检表 %s vs 数据 %.4f（|Δ|=%.4f）", c, s_vals[k], n_by[c] / len(lab), diff)
    assert diff < 5e-4, f"V1 占比与数据不一致：{c}"
R_vals = dict(re.findall(r"R_(\w)=([\d.]+)", v3))
D_geo = float(re.search(r"D_geo=([\d.]+)", v4).group(1))
rho = float(re.search(r"ρ=([\d.]+)", v4).group(1))
m5 = dict(re.findall(r"(合|偏|严)=([\d.]+)", v5_note))
for k, c in s_map.items():
    diff = abs(float(m5[k]) - med[c])
    log.info("V5 交叉核对 %s：自检表 e 中位 %s vs 数据 %.4f（|Δ|=%.4f）", c, m5[k], med[c], diff)
    assert diff < 0.5, f"V5 中位数与数据不一致：{c}"
log.info("自检表解析：D_geo=%.4f ρ=%.4f R=%s", D_geo, rho, R_vals)

# ---- 画布 ----
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
axa, axb = axes

# (a) 箱线（对数轴）
data = [e_by[c] for c in CLASSES]
bp = axa.boxplot(data, tick_labels=[f"{c}\nn={n_by[c]:,}（{n_by[c]/len(lab):.1%}）" for c in CLASSES],
                 patch_artist=True, showfliers=True, widths=0.52,
                 flierprops=dict(marker="o", markersize=2.5, alpha=0.35, markeredgewidth=0),
                 medianprops=dict(color="white", lw=2.0), whiskerprops=dict(color="#4D4D4D"),
                 capprops=dict(color="#4D4D4D"))
for patch, c in zip(bp["boxes"], CLASSES):
    patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.75); patch.set_edgecolor("#333333")
axa.set_yscale("log")
axa.set_ylim(0.05, 9000)
axa.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
for i, c in enumerate(CLASSES):
    axa.text(i + 1 + 0.30, med[c], f"中位\n{med[c]:.1f}", fontsize=9.5, va="center",
             color=CLASS_COLORS[c], fontfamily=YH_FONT)
axa.set_ylabel("超额幅度 e = 索赔金额 - 赔付（元，对数轴）", fontfamily=YH_FONT)
axa.set_title("(a) 三类 e 箱线对比：中位严格有序 严 > 偏 > 合（V5 业务序）",
              fontsize=11.5, fontfamily=YH_FONT)

# (b) log10(e) 密度对比
bins = np.linspace(-1.3, 3.8, 52)
for c in CLASSES:
    axb.hist(np.log10(e_by[c]), bins=bins, density=True, histtype="stepfilled",
             color=CLASS_COLORS[c], alpha=0.45, edgecolor=CLASS_COLORS[c], lw=1.2,
             label=c)
axb.set_xlabel("log10(e)", fontfamily=YH_FONT)
axb.set_ylabel("密度", fontfamily=YH_FONT)
axb.set_title("(b) 三类 log10(e) 分布：合理类集中于低超额段，严重类居长尾（C6 密度差）",
              fontsize=11.5, fontfamily=YH_FONT)
axb.legend(prop={"family": YH_FONT, "size": 9.5})
axb.text(0.985, 0.965,
         "自检表数值（q1_自检表.csv）\n"
         f"V4 密度差：D_geo={D_geo:.2f}（>1，参考标记 ≥2）\n"
         f"V4 离散度比：ρ=MAD(合理)/MAD(严重)={rho:.3f}（<1）\n"
         f"V3 紧凑度：R合={float(R_vals['合']):.3f}，R偏={float(R_vals['偏']):.3f}，R严={float(R_vals['严']):.3f}\n"
         f"V1 占比：s合={float(s_vals['合']):.3f}，s偏={float(s_vals['偏']):.3f}，s严={float(s_vals['严']):.3f}",
         transform=axb.transAxes, ha="right", va="top", fontsize=9, linespacing=1.7,
         bbox=dict(boxstyle="round,pad=0.45", fc="#F5F7FA", ec="#B9C4CE", alpha=0.95),
         fontfamily=YH_FONT)

fig.suptitle("图3　问题1 规则检验：三类分布特征与自检表核心条目（数值溯源见框注与日志）",
             fontsize=13, y=1.015, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig3_q1规则检验.png", log)
log.info("完成：%s", SCRIPT)
