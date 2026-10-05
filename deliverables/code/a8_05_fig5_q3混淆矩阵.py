# -*- coding: utf-8 -*-
"""
a8_05_fig5_q3混淆矩阵.py —— fig5 问题3 混淆矩阵热图（方式2 主配置，双口径并列，含行占比标注）
数据来源：
  - q3_混淆矩阵.csv：分层5折_主种子池化 / 行序外推 两口径的 3×3 计数
  - q3_指标汇总.csv：对应口径的宏F1（标题溯源）
行和断言：分层5折行和 = (9602,1340,225)，行序外推行和 = (1911,278,44)，与 q3_指标汇总 support 逐格核对。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig

import matplotlib.pyplot as plt

SCRIPT = "a8_05_fig5_q3混淆矩阵"
log = get_logger(SCRIPT)
log_font_check(log)

CLASSES = ["合理诉求", "诉求偏高", "严重超额"]
cm = read_csv("q3_混淆矩阵.csv")
met = read_csv("q3_指标汇总.csv")

PANELS = [("分层5折_主种子池化", "(a) 分层5折（主种子 20251004 池化）", "方式2_主配置(嵌套γ,s)", "分层5折"),
          ("行序外推", "(b) 行序外推（前 80% 训练 / 后 20% 验证）", "方式2_主配置(嵌套γ,s)", "行序外推")]

fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9))
for ax, (key, ptitle, cfg, kou) in zip(axes, PANELS):
    sub = cm[cm["口径"] == key]
    assert len(sub) == 9, f"{key} 混淆矩阵格数异常"
    M = np.zeros((3, 3), dtype=int)
    for _, r in sub.iterrows():
        M[CLASSES.index(r["真实"]), CLASSES.index(r["预测"])] = int(r["行数"])
    sup = [int(met[(met["配置"] == cfg) & (met["口径"] == kou) & (met["种子"] == 20251004)]["类%d_support" % i].iloc[0])
           for i in range(3)]
    assert (M.sum(axis=1) == np.array(sup)).all(), f"{key} 行和与 support 不一致：{M.sum(axis=1)} vs {sup}"
    f1 = float(met[(met["配置"] == cfg) & (met["口径"] == kou) & (met["种子"] == 20251004)]["宏F1"].iloc[0])
    rec2 = float(met[(met["配置"] == cfg) & (met["口径"] == kou) & (met["种子"] == 20251004)]["严重类_recall"].iloc[0])
    log.info("%s 矩阵=%s 行和=%s（support 一致）宏F1=%.4f 严重recall=%.4f", key, M.tolist(),
             M.sum(axis=1).tolist(), f1, rec2)

    im = ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1.0)
    for i in range(3):
        for j in range(3):
            share = M[i, j] / M[i].sum()
            color = "white" if share > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{M[i, j]:,}\n行占比 {share:.1%}", ha="center", va="center",
                    fontsize=10, color=color, linespacing=1.5, fontfamily=YH_FONT)
    ax.set_xticks(range(3), CLASSES, fontfamily=YH_FONT)
    ax.set_yticks(range(3), CLASSES, fontfamily=YH_FONT)
    ax.set_xlabel("预测标签", fontfamily=YH_FONT)
    ax.set_ylabel("真实标签（问题1 规则标注）", fontfamily=YH_FONT)
    ax.set_title(f"{ptitle}\n宏F1={f1:.4f}，严重类召回={rec2:.4f}", fontsize=11, fontfamily=YH_FONT)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行占比（召回视角）")

fig.suptitle("图5　问题3 混淆矩阵热图（方式2 主配置；格内为行数与行占比，数值取自 q3_混淆矩阵.csv）",
             fontsize=13, y=1.02, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig5_q3混淆矩阵.png", log)
log.info("完成：%s", SCRIPT)
