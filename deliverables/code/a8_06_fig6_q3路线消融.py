# -*- coding: utf-8 -*-
"""
a8_06_fig6_q3路线消融.py —— fig6 问题3 两路线对比与消融矩阵（宏F1 + 严重类召回，双口径）
数据来源：
  - q3_两路线对比.csv：方式2/方式1 的宏F1、严重类召回、标签一致率（分层5折=3 种子均值；行序外推=种子 20251004）
  - q3_消融矩阵.csv：消融 A0–A6 双口径（种子 20251004）
  - q3_指标汇总.csv：方式2 主配置（嵌套γ,s）双口径，作为第 1 组并列展示
消融配置语义取自 00_admin/A5_算法方案.md §4.5 消融矩阵清单。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig)

import matplotlib.pyplot as plt

SCRIPT = "a8_06_fig6_q3路线消融"
log = get_logger(SCRIPT)
log_font_check(log)

rl = read_csv("q3_两路线对比.csv")
s5 = rl[(rl["口径"] == "分层5折") & rl["方式2_宏F1"].notna()]
os_ = rl[(rl["口径"] == "行序外推") & rl["方式2_宏F1"].notna()]
assert len(s5) == 3 and len(os_) == 1
vals = {
    ("宏F1", "方式2"): [s5["方式2_宏F1"].mean(), float(os_["方式2_宏F1"].iloc[0])],
    ("宏F1", "方式1"): [s5["方式1_宏F1"].mean(), float(os_["方式1_宏F1"].iloc[0])],
    ("严重召回", "方式2"): [s5["方式2_严重recall"].mean(), float(os_["方式2_严重recall"].iloc[0])],
    ("严重召回", "方式1"): [s5["方式1_严重recall"].mean(), float(os_["方式1_严重recall"].iloc[0])],
}
agree = [s5["标签一致率"].mean(), float(os_["标签一致率"].iloc[0])]
log.info("两路线（分层5折均值/行序外推）：宏F1 方式2=%.4f/%.4f 方式1=%.4f/%.4f；严重召回 方式2=%.4f/%.4f 方式1=%.4f/%.4f；标签一致率=%.4f/%.4f",
         *[*vals[("宏F1", "方式2")], *vals[("宏F1", "方式1")], *vals[("严重召回", "方式2")],
           *vals[("严重召回", "方式1")], *agree])

met = read_csv("q3_指标汇总.csv")
ab = read_csv("q3_消融矩阵.csv")
AB_LABEL = ["消融A0", "消融A1", "消融A2", "消融A3", "消融A4", "消融A5", "消融A6"]
AB_NAME = {"消融A0": "A0\n无杠杆", "消融A1": "A1\n仅γ=1", "消融A2": "A2\n仅s校准",
           "消融A3": "A3\nγ=1+s", "消融A4": "A4\nγ=0.5+s", "消融A5": "A5\n随机过采样",
           "消融A6": "A6\n均衡集成"}
MAIN_LABEL = "主配置\n(嵌套γ,s)"

def get(cfg, kou, col):
    sub = ab[(ab["配置"] == cfg) & (ab["口径"] == kou) & (ab["种子"] == 20251004)]
    assert len(sub) == 1, f"{cfg}/{kou} 记录数异常"
    return float(sub[col].iloc[0])

def get_main(kou, col):
    sub = met[(met["配置"] == "方式2_主配置(嵌套γ,s)") & (met["口径"] == kou) & (met["种子"] == 20251004)]
    assert len(sub) == 1
    return float(sub[col].iloc[0])

groups = [MAIN_LABEL] + AB_LABEL
f1_5f = [get_main("分层5折", "宏F1")] + [get(c, "分层5折", "宏F1") for c in AB_LABEL]
f1_os = [get_main("行序外推", "宏F1")] + [get(c, "行序外推", "宏F1") for c in AB_LABEL]
rc_5f = [get_main("分层5折", "严重类_recall")] + [get(c, "分层5折", "严重类_recall") for c in AB_LABEL]
rc_os = [get_main("行序外推", "严重类_recall")] + [get(c, "行序外推", "严重类_recall") for c in AB_LABEL]
log.info("消融宏F1 分层5折=%s", [round(v, 4) for v in f1_5f])
log.info("消融宏F1 行序外推=%s", [round(v, 4) for v in f1_os])
log.info("消融严重召回 分层5折=%s", [round(v, 4) for v in rc_5f])
log.info("消融严重召回 行序外推=%s", [round(v, 4) for v in rc_os])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(15.0, 6.0))

# ---- (a) 两路线 ----
xs = np.arange(4)
w = 0.36
g_lab = ["宏F1\n分层5折", "宏F1\n行序外推", "严重类召回\n分层5折", "严重类召回\n行序外推"]
v_m2 = [vals[("宏F1", "方式2")][0], vals[("宏F1", "方式2")][1],
        vals[("严重召回", "方式2")][0], vals[("严重召回", "方式2")][1]]
v_m1 = [vals[("宏F1", "方式1")][0], vals[("宏F1", "方式1")][1],
        vals[("严重召回", "方式1")][0], vals[("严重召回", "方式1")][1]]
ba = axa.bar(xs - w / 2, v_m2, w, color=PAL["blue"], label="方式2 直接分类（主）")
bb = axa.bar(xs + w / 2, v_m1, w, color=PAL["orange_l"], edgecolor=PAL["orange"],
             lw=1.0, label="方式1 完整链路")
for bars in (ba, bb):
    for r in bars:
        axa.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.3f}",
                 ha="center", fontsize=8.6, fontfamily=YH_FONT)
axa.set_xticks(xs, g_lab, fontfamily=YH_FONT, fontsize=9.5)
axa.set_ylim(0, 0.80)
axa.set_ylabel("指标值", fontfamily=YH_FONT)
axa.set_title("(a) 两路线对比：方式2 优于方式1，且严重类召回更高\n（q3_两路线对比.csv）",
              fontsize=11, fontfamily=YH_FONT)
axa.legend(prop={"family": YH_FONT, "size": 9.5}, loc="upper center", ncol=2)
axa.text(0.5, 0.735,
         f"标签一致率（方式1 vs 方式2 判定）：分层5折 3 种子均值 {agree[0]:.4f}；行序外推 {agree[1]:.4f}",
         transform=axa.transAxes, ha="center", fontsize=8.8, color="#4D4D4D", fontfamily=YH_FONT)

# ---- (b) 消融 ----
xs = np.arange(len(groups))
w = 0.38
bb1 = axb.bar(xs - w / 2, f1_5f, w, color=PAL["blue"], label="宏F1·分层5折")
bb2 = axb.bar(xs + w / 2, f1_os, w, color=PAL["orange_l"], edgecolor=PAL["orange"], lw=1.0,
              label="宏F1·行序外推")
for bars in (bb1, bb2):
    for r in bars:
        axb.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.3f}",
                 ha="center", fontsize=7.6, rotation=90, color="#333333", fontfamily=YH_FONT)
axb.set_xticks(xs, [MAIN_LABEL] + [AB_NAME[c] for c in AB_LABEL], fontfamily=YH_FONT, fontsize=8.6)
axb.get_xticklabels()[0].set_color(PAL["navy"])
axb.set_ylim(0, 0.86)
axb.set_ylabel("宏F1", fontfamily=YH_FONT)
axb.set_ylim(0, 1.0)
axb2 = axb.twinx()
axb2.plot(xs - w / 2, rc_5f, "o", color=PAL["navy"], ms=7, label="严重类召回·分层5折", zorder=5)
axb2.plot(xs + w / 2, rc_os, "^", color=PAL["orange"], ms=8, mec="#8A4B0F", label="严重类召回·行序外推", zorder=5)
axb2.set_ylim(0, 1.0)
axb2.set_ylabel("严重类召回（右轴）", fontfamily=YH_FONT)
axb2.grid(False)
axb.set_title("(b) 消融矩阵与主配置：宏F1（柱）与严重类召回（点）\n"
              "（q3_消融矩阵.csv / q3_指标汇总.csv，种子 20251004）",
              fontsize=11, fontfamily=YH_FONT)
h1, l1 = axb.get_legend_handles_labels()
h2, l2 = axb2.get_legend_handles_labels()
axb.legend(h1 + h2, l1 + l2, prop={"family": YH_FONT, "size": 8.2}, loc="upper left", ncol=2,
           columnspacing=1.0, handlelength=1.4)
axb.text(0.55, 0.845, "A6 均衡集成：严重召回最高（0.96/0.98）但宏F1 大幅受损，印证主配置的取舍",
         transform=axb.transAxes, ha="center", fontsize=8.6, color="#4D4D4D", fontfamily=YH_FONT)

fig.suptitle("图6　问题3 两路线与消融对比（双指标、双口径；数值溯源见日志）",
             fontsize=13, y=1.02, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig6_q3路线消融.png", log)
log.info("完成：%s", SCRIPT)
