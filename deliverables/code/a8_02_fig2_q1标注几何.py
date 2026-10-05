# -*- coding: utf-8 -*-
"""
a8_02_fig2_q1标注几何.py —— fig2 问题1 标注几何：x-e 平面散点（按三类着色）+ 保序阶梯边界 g1/g2 + 边界表数值
数据来源：
  - 附件1_风险标注.csv：x=实际赔付金额，e=-索赔差额（Q1 标注结果，11167 行）
  - q1_边界表.csv：箱序/x下界/x上界/g1_b/g2_b（定稿边界）
防过密：合理诉求随机抽样（seed=20251004），偏高/严重全量。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (CLASS_COLORS, PAL, YH_FONT, get_logger, log_font_check,
                            read_csv, save_fig)

import matplotlib.pyplot as plt

SCRIPT = "a8_02_fig2_q1标注几何"
log = get_logger(SCRIPT)
log_font_check(log)

# ---- 数据 ----
lab = read_csv("附件1_风险标注.csv")
assert len(lab) == 11167, f"附件1_风险标注 行数异常：{len(lab)}"
x = lab["实际赔付金额"].to_numpy()
e = (-lab["索赔差额"]).to_numpy()
cls = lab["风险标注"].to_numpy()
assert (e > 0).all(), "存在 e<=0，与 T3（差额全负）矛盾"
cnt = lab["风险标注"].value_counts()
log.info("标注计数：%s", cnt.to_dict())

rng = np.random.default_rng(20251004)
keep = np.zeros(len(lab), dtype=bool)
for c, n_cap in [("合理诉求", 2200), ("诉求偏高", 1340), ("严重超额", 225)]:
    idx = np.where(cls == c)[0]
    assert len(idx) == cnt[c], f"{c} 计数与 value_counts 不一致"
    take = idx if len(idx) <= n_cap else rng.choice(idx, size=n_cap, replace=False)
    keep[take] = True
    log.info("抽样：%s 全量%d → 展示 %d", c, len(idx), len(take))

edge = read_csv("q1_边界表.csv").sort_values("箱序")
assert len(edge) == 5, f"边界表箱数异常：{len(edge)}"
g1, g2 = edge["g1_b"].to_numpy(), edge["g2_b"].to_numpy()
assert (np.diff(g1) >= -1e-9).all() and (np.diff(g2) >= -1e-9).all() and (g2 >= g1).all(), \
    "边界非降/序关系校验失败（V2）"
xlo, xhi = edge["x下界"].to_numpy(), edge["x上界"].to_numpy()
log.info("边界表（q1_边界表.csv）：" + "；".join(
    f"箱{int(b)}: x∈[{a:.3f},{c:.3f}) g1={v1:.1f} g2={v2:.1f}"
    for b, a, c, v1, v2 in zip(edge["箱序"], xlo, xhi, g1, g2)))

# ---- 画布 ----
fig = plt.figure(figsize=(13.8, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1], wspace=0.06)
ax = fig.add_subplot(gs[0, 0])
axt = fig.add_subplot(gs[0, 1]); axt.axis("off"); axt.grid(False)

# 偏高带浅色底（g1~g2 之间）
for i in range(5):
    ax.fill([xlo[i], xhi[i], xhi[i], xlo[i]], [g1[i], g1[i], g2[i], g2[i]],
            color=PAL["orange"], alpha=0.10, zorder=0, lw=0)

order = [("合理诉求", 12, 0.40), ("诉求偏高", 14, 0.55), ("严重超额", 20, 0.85)]
for name, s, al in order:
    m = keep & (cls == name)
    ax.scatter(x[m], e[m], s=s, c=CLASS_COLORS[name], alpha=al, lw=0,
               label=f"{name}（n={cnt[name]:,}）", zorder=2)

# 阶梯边界
for i in range(5):
    ax.plot([xlo[i], xhi[i]], [g1[i], g1[i]], color=PAL["navy"], lw=2.6, zorder=4)
    ax.plot([xlo[i], xhi[i]], [g2[i], g2[i]], color=PAL["navy"], lw=2.6, ls=(0, (5, 2)), zorder=4)
    if i < 4:  # 箱间竖连接（示阶梯跳变）
        ax.plot([xhi[i], xhi[i]], [g1[i], g1[i + 1]], color=PAL["navy"], lw=1.0,
                alpha=0.6, zorder=3)
        ax.plot([xhi[i], xhi[i]], [g2[i], g2[i + 1]], color=PAL["navy"], lw=1.0,
                alpha=0.6, ls=(0, (5, 2)), zorder=3)
ax.plot([], [], color=PAL["navy"], lw=2.6, label="边界 g1(x)（诉求偏高门槛）")
ax.plot([], [], color=PAL["navy"], lw=2.6, ls=(0, (5, 2)), label="边界 g2(x)（严重超额门槛）")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.65, 3600); ax.set_ylim(0.06, 6000)
# 对数轴刻度改纯 ASCII 标签（避免 mathtext 的 U+2212 负号在 SimHei 中缺字）
from matplotlib.ticker import FuncFormatter
_fmt = FuncFormatter(lambda v, _p: f"{v:g}")
ax.xaxis.set_major_formatter(_fmt)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}" if v >= 1 else f"{v:g}"))
ax.set_xlabel("实际赔付金额 x（元，对数轴）", fontfamily=YH_FONT)
ax.set_ylabel("超额幅度 e = 索赔金额 - 赔付（元，对数轴）", fontfamily=YH_FONT)
ax.set_title("图2　问题1 标注几何：x-e 平面三类散点与保序阶梯边界（合理诉求抽样展示）",
             fontsize=12.5, fontfamily=YH_FONT)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9, prop={"family": YH_FONT, "size": 9})

# ---- 右侧：边界表数值（逐格来自 q1_边界表.csv）----
axt.text(0.5, 1.02, "边界表数值（q1_边界表.csv）", ha="center", fontsize=11.5,
         fontfamily=YH_FONT, transform=axt.transAxes)
col_x = [0.02, 0.30, 0.72, 1.02]
rows = []
for i in range(5):
    rows.append([f"{int(edge['箱序'].iloc[i])}",
                 f"[{xlo[i]:.1f}, {xhi[i]:.1f})" if i < 4 else f"[{xlo[i]:.1f}, {xhi[i]:.1f}]",
                 f"{g1[i]:,.1f}", f"{g2[i]:,.1f}"])
tbl = axt.table(cellText=rows, colLabels=["箱序", "x 区间（元）", "g1", "g2"],
                colWidths=[0.10, 0.34, 0.26, 0.26], cellLoc="center", loc="upper center",
                bbox=[0.0, 0.52, 1.0, 0.42])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_height(0.07)
    if r == 0:
        cell.set_facecolor("#E3ECF5"); cell.set_text_props(fontfamily=YH_FONT)
    else:
        cell.set_text_props(fontfamily=YH_FONT)

axt.text(0.5, 0.44, "判定路径（序贯，三步可复述）", ha="center", fontsize=10.5,
         fontfamily=YH_FONT, transform=axt.transAxes)
axt.text(0.02, 0.38, "① e ≥ g2(x) → 严重超额\n② 否则 e ≥ g1(x) → 诉求偏高\n③ 否则 → 合理诉求",
         fontsize=10, va="top", linespacing=1.8, fontfamily=YH_FONT, transform=axt.transAxes)
axt.text(0.02, 0.16,
         "边界跨箱保序（PAVA），违例数 g1=0、g2=0\n（q1_自检表.csv V2）；浅橙带为\n诉求偏高判定带 g1(x)≤e<g2(x)",
         fontsize=9, va="top", linespacing=1.8, color="#4D4D4D",
         fontfamily=YH_FONT, transform=axt.transAxes)

save_fig(fig, "fig2_q1标注几何.png", log)
log.info("完成：%s", SCRIPT)
