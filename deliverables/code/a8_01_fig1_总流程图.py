# -*- coding: utf-8 -*-
"""
a8_01_fig1_总流程图.py —— fig1 建模总流程图（三问关系：Q1 规则→Q3 标签；Q2→方式1 链路）
数据来源（图内数字全部程序化读取，不手抄）：
  - 附件1_clean.csv / 附件2_clean.csv（行数 n1/n2）
  - 附件1_风险标注.csv（三类计数与占比，Q1 输出）
  - q2_指标汇总.csv（LGBM_变体A 主口径 WAPE 均值）
  - q3_两路线对比.csv（方式2/方式1 宏F1 均值）
字体说明：D06 规定 rcParams=[SimHei, Microsoft YaHei]；本图文本含 ŷ、下标、数学负号等
SimHei 缺失字形，统一指定 Microsoft YaHei 渲染（仍为 D06 规定字体）；组合变音符号一律不用（L̃ 写作 L~）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import (PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT = "a8_01_fig1_总流程图"
log = get_logger(SCRIPT)
log_font_check(log)

# ---- 溯源数字（全部由 csv 计算）----
n1 = len(read_csv("附件1_clean.csv"))
n2 = len(read_csv("附件2_clean.csv"))
lab = read_csv("附件1_风险标注.csv")
cnt = lab["风险标注"].value_counts()
p0, p1, p2 = cnt["合理诉求"] / len(lab), cnt["诉求偏高"] / len(lab), cnt["严重超额"] / len(lab)
q2 = read_csv("q2_指标汇总.csv")
wape_a = q2[(q2["模型"] == "LGBM_变体A") & (q2["口径"] == "分层5折")]["WAPE"].mean()
rl = read_csv("q3_两路线对比.csv")
sub = rl[rl["口径"] == "分层5折"]
f1_m2 = sub["方式2_宏F1"].mean()
f1_m1 = sub["方式1_宏F1"].mean()
log.info("溯源：n1=%d n2=%d 三类占比=%.4f/%.4f/%.4f 变体A WAPE(分层5折均值)=%.4f 方式2宏F1=%.4f 方式1宏F1=%.4f",
         n1, n2, p0, p1, p2, wape_a, f1_m2, f1_m1)

# ---- 画布与框 ----
fig, ax = plt.subplots(figsize=(13.6, 7.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)

def box(x, y, w, h, text, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1A1A1A", zorder=3, linespacing=1.55, fontfamily=YH_FONT)

def arrow(x1, y1, x2, y2, color=PAL["navy"], lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))

def layer_label(x, y, t):
    ax.text(x, y, t, ha="center", va="center", fontsize=11.5,
            color=PAL["navy"], fontfamily=YH_FONT)

# --- 数据层 ---
layer_label(10.5, 93.5, "数据层")
box(2.5, 68, 16, 20,
    f"附件1（清洗后）\n{n1:,} 行\nx＝实际赔付金额\ny＝索赔金额\nd＝x-y 全负",
    "#EAF1F8", PAL["blue"], 9.5)
box(2.5, 40, 16, 18,
    f"附件2（清洗后）\n{n2:,} 行\ny＝索赔金额\n（x 不可得）",
    "#EAF1F8", PAL["blue"], 9.5)

# --- 问题求解层 ---
layer_label(37, 93.5, "问题求解层")
box(25.5, 60, 24, 28,
    "问题1：风险标注规则\n(x,e) 坐标系，e＝y-x\n等频分箱条件分位阶梯\ng₁(x)≤g₂(x)（PAVA 保序）\n"
    "序贯判定 e≥g₂→严重超额\n　　　　 e≥g₁→诉求偏高",
    "#FDF3E7", PAL["orange"], 9.5)
box(25.5, 30, 24, 20,
    f"问题2：赔付预测 ŷ\nLGBM 变体A＋smearing\n主口径 WAPE＝{wape_a:.3f}",
    "#FDF3E7", PAL["orange"], 9.5)

# --- 问题3 两路线 ---
box(55, 58, 20.5, 30,
    "方式1：完整链路\n（链路对照）\nŷ 代入问题1规则\n逐单重算标签 L~\n"
    f"宏F1＝{f1_m1:.3f}",
    "#F1EAF7", PAL["purple"], 9.5)
box(55, 26, 20.5, 26,
    "方式2：直接分类\n（主路线）\nLGBM 多分类\n＋两级杠杆(γ,s)\n训练标签＝问题1标注 L",
    "#F1EAF7", PAL["purple"], 9.5)

# --- 评估与输出层 ---
layer_label(88, 93.5, "评估与输出层")
box(78.5, 44, 19, 30,
    "双口径评估\n（D19）\n①分层 5 折×3 种子\n②行序外推 80/20\nWAPE／宏F1\n严重类召回单列",
    "#EEF3EE", PAL["gray"], 9.5)
box(78.5, 8, 19, 22,
    "Result_提交.xlsx\n附件2 赔付预测 ŷ\n附件2 风险预测\n（2792 行全覆盖）",
    "#EEF3EE", PAL["gray"], 9.5)

# --- 箭头 ---
arrow(18.5, 78, 25.5, 76)                                # 附件1→Q1
arrow(18.5, 52, 25.5, 44)                                # 附件1→Q2
arrow(49.5, 70, 55, 74)                                  # Q2 ŷ→方式1
ax.text(52.2, 75.4, "ŷ", fontsize=10, color=PAL["navy"], fontfamily=YH_FONT)
arrow(49.5, 40, 55, 38)                                  # Q1/Q2→方式2
arrow(38, 30, 61, 26, color=PAL["gray"], ls="--", rad=-0.25)
ax.text(46.5, 24.2, "OOF 特征/折对齐", fontsize=8.5, color=PAL["gray"], fontfamily=YH_FONT)
arrow(75.5, 70, 78.5, 66)                                # 方式1→评估
arrow(75.5, 39, 78.5, 52, rad=0.15)                      # 方式2→评估
arrow(88, 44, 88, 30)                                    # 评估→输出
arrow(18.5, 44, 78.5, 16, color=PAL["gray"], rad=0.12)   # 附件2→输出
ax.text(44, 12.5, "附件2 索赔金额 → 问题2 预测 / 方式2 判类", fontsize=8.5,
        color=PAL["gray"], fontfamily=YH_FONT)

# 底注（占比溯源）
ax.text(50, 3.2,
        f"问题1 标注占比（附件1_风险标注.csv）：合理诉求 {p0:.1%}｜诉求偏高 {p1:.1%}｜严重超额 {p2:.1%}"
        f"　（n1＝{n1:,}，n2＝{n2:,}，行数取自附件*_clean.csv）",
        ha="center", fontsize=9, color="#4D4D4D", fontfamily=YH_FONT)

ax.set_title("图1　三问建模总流程：Q1 规则标注 → Q2 赔付预测 → Q3 两路线分类（数字溯源见底注与日志）",
             fontsize=13, pad=14, fontfamily=YH_FONT)

save_fig(fig, "fig1_建模总流程图.png", log)
log.info("完成：%s", SCRIPT)
