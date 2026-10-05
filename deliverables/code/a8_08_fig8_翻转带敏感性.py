# -*- coding: utf-8 -*-
"""
a8_08_fig8_翻转带敏感性.py —— fig8 问题1 标注规则翻转带敏感性（A7 实验1 产物可视化）
数据来源（只读，不修改）：
  - a7_翻转带敏感性.csv：θ 邻域 12 格（τ邻域 3×3 @B=5；B邻域 B∈{10,15,20} @τ=(0.86,0.98)），每格 bootstrap 1000 次
      flip_own=重估标签 vs 该格自身全量标签；flip_vsθ*=重估标签 vs θ* 冻结标签；确定性重标差异率=格内全量标签 vs θ* 冻结标签
  - a7_翻转带画像.csv：θ* 1000 次 bootstrap 逐行翻转频率画像（距离带 / 转移方向 / 风险集）
面板：(a) τ 邻域 flip_own 中位±IQR；(b) B 邻域三指标；(c) 距离带行均翻转频率
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a8_00_图表公共 import PAL, YH_FONT, get_logger, log_font_check, read_csv, save_fig

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT = "a8_08_fig8_翻转带敏感性"
log = get_logger(SCRIPT)
log_font_check(log)

# ================= 读数与核验 =================
sen = read_csv("a7_翻转带敏感性.csv")
assert len(sen) == 12, "敏感性网格应为 12 格"
TAU1 = [0.85, 0.86, 0.87]
TAU2 = [0.975, 0.98, 0.985]
tau_grid = sen[sen["网格"] == "τ邻域"].copy()
assert len(tau_grid) == 9 and set(tau_grid["B"]) == {5}
star = tau_grid[tau_grid["是否θ星"] == True]  # noqa: E712
assert len(star) == 1 and float(star["tau1"].iloc[0]) == 0.86 and float(star["tau2"].iloc[0]) == 0.98, "θ* 行定位异常"
V62 = float(star["flip_own_中位"].iloc[0])
assert abs(V62 - 0.01029820005372974) < 1e-12, "θ* 翻转率与 a6_01 V6②（0.010298）不一致"
log.info("θ*=(5,0.86,0.98) flip_own 中位=%.6f，与 a6_01 V6② 及 a7_01 日志一致", V62)

# (a) τ 邻域 3×3
med = np.empty((3, 3)); p25 = np.empty((3, 3)); p75 = np.empty((3, 3))
for i, t1 in enumerate(TAU1):
    for j, t2 in enumerate(TAU2):
        r = tau_grid[(tau_grid["tau1"] == t1) & (tau_grid["tau2"] == t2)]
        assert len(r) == 1, f"τ 格 ({t1},{t2}) 缺失"
        med[i, j] = r["flip_own_中位"].iloc[0] * 100
        p25[i, j] = r["flip_own_P25"].iloc[0] * 100
        p75[i, j] = r["flip_own_P75"].iloc[0] * 100
log.info("(a) τ邻域 flip_own 中位范围：%.3f%%–%.3f%%（9 格），θ*=%.4f%%",
         med.min(), med.max(), V62 * 100)
log.info("(a) τ邻域矩阵（行=τ1, 列=τ2）：\n%s", np.round(med, 3))

# (b) B 邻域（B=5 取 θ* 行，B=10/15/20 取 B邻域 行）
b_grid = sen[sen["网格"] == "B邻域"].copy()
assert list(b_grid["B"]) == [10, 15, 20]
Bs = [5, 10, 15, 20]
own_med = [V62 * 100] + (b_grid["flip_own_中位"] * 100).tolist()
own_err = np.array([[V62 * 100 - float(star["flip_own_P25"].iloc[0]) * 100,
                     float(star["flip_own_P75"].iloc[0]) * 100 - V62 * 100]] +
                   [[(m - p) * 100, (q - m) * 100] for m, p, q in
                    zip(b_grid["flip_own_中位"] * 100, b_grid["flip_own_P25"] * 100, b_grid["flip_own_P75"] * 100)]).T
vs_med = [V62 * 100] + (b_grid["flip_vsθ星_中位"] * 100).tolist()
vs_err = np.array([[V62 * 100 - float(star["flip_vsθ星_P25"].iloc[0]) * 100,
                    float(star["flip_vsθ星_P75"].iloc[0]) * 100 - V62 * 100]] +
                  [[(m - p) * 100, (q - m) * 100] for m, p, q in
                   zip(b_grid["flip_vsθ星_中位"] * 100, b_grid["flip_vsθ星_P25"] * 100, b_grid["flip_vsθ星_P75"] * 100)]).T
det = [0.0] + (b_grid["确定性重标差异率"] * 100).tolist()
log.info("(b) B 邻域：flip_own 中位=%s；flip_vsθ* 中位=%s；确定性重标差异=%s",
         np.round(own_med, 3), np.round(vs_med, 3), np.round(det, 3))

# (c) 距离带画像
pro = read_csv("a7_翻转带画像.csv")
BANDS = ["[0,0.5%)", "[0.5%,1%)", "[1%,2%)", "[2%,5%)", "[5%,10%)", "[10%,20%)", "[20%,50%)", ">=50%"]
BAND_SHORT = ["0–0.5", "0.5–1", "1–2", "2–5", "5–10", "10–20", "20–50", "≥50"]
db = pro[(pro["块"] == "距离带") & (pro["键"].isin(BANDS))].set_index("键").loc[BANDS].reset_index()
assert int(db["行数"].sum()) == 11167, "距离带 8 带行数之和应=11167"
freq = db["行均翻转频率_全量口径"] * 100
n_rows = db["行数"].astype(int)
risk = pro[pro["块"] == "风险集"].set_index("键")
n_high = int(risk.loc["高频翻转行(频率>=5%)", "行数"])
q_high = float(risk.loc["高频翻转行(频率>=5%)", "翻转质量占比_全量口径"])
n_any = int(risk.loc["任一次重估即翻转的行(频率>0)", "行数"])
log.info("(c) 距离带行数=%s（和=%d）；行均翻转频率 %%=%s", n_rows.tolist(), n_rows.sum(), np.round(freq, 2))
log.info("(c) 风险集：任一次翻转=%d 行（%.2f%%），高频翻转（≥5%%）=%d 行（%.2f%%）贡献翻转质量 %.1f%%",
         n_any, 100 * n_any / 11167, n_high, 100 * n_high / 11167, 100 * q_high)

# ================= 画布 =================
fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15.6, 5.2),
                                    gridspec_kw={"width_ratios": [1.18, 1.02, 1.0]})

# ---- (a) τ 邻域 ----
xg = np.arange(3)
w = 0.25
ramp = [PAL["blue_l"], PAL["blue"], PAL["navy"]]  # τ2 升序单色蓝阶（色盲友好）
for j, t2 in enumerate(TAU2):
    offs = (j - 1) * w
    bars = axa.bar(xg + offs, med[:, j], w - 0.02, color=ramp[j],
                   yerr=np.vstack([med[:, j] - p25[:, j], p75[:, j] - med[:, j]]),
                   error_kw=dict(ecolor="#4D4D4D", lw=1.0, capsize=2.5), label=f"τ2={t2}")
    for r, v in zip(bars, med[:, j]):
        axa.text(r.get_x() + r.get_width() / 2, 0.06, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7.2, rotation=90, color="white" if j == 2 else "#333333",
                 fontfamily=YH_FONT)
# θ* 高亮：中间组中间条描边 + 星标（橙描边，蓝底高对比、色盲友好）
y_star = p75[1, 1] + 0.07
axa.bar([1], [med[1, 1]], w - 0.02, facecolor="none", edgecolor=PAL["orange"], lw=2.0, zorder=5)
axa.scatter([1], [y_star], marker="*", s=130, color=PAL["orange"], zorder=6)
axa.annotate("θ*\n(0.86, 0.98)", xy=(1, y_star + 0.02), xytext=(1.02, y_star + 0.13),
             fontsize=8.5, fontfamily=YH_FONT, ha="center", color=PAL["orange"])
axa.axhline(1.0, color=PAL["gray"], lw=1.2, ls="--", zorder=1)
axa.text(-0.44, 1.03, "1% 约定门槛（V6②）", fontsize=8, color=PAL["gray"], fontfamily=YH_FONT,
         zorder=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
axa.set_xticks(xg, [f"τ1={t}" for t in TAU1], fontfamily=YH_FONT)
axa.set_ylim(0, 1.58)
axa.set_ylabel("bootstrap 标签翻转率 flip_own（%）", fontfamily=YH_FONT)
axa.set_title("(a) τ 邻域（B=5）：中位 ± P25–P75，9 格均近 1%\n（a7_翻转带敏感性.csv）",
              fontsize=10.5, fontfamily=YH_FONT)
axa.legend(prop={"family": YH_FONT, "size": 8.5}, loc="upper left", ncol=1)

# ---- (b) B 邻域 ----
axb.errorbar(Bs, own_med, yerr=own_err, color=PAL["blue"], marker="o", ms=5.5, lw=1.6,
             capsize=3, ecolor="#4D4D4D", elinewidth=1.0, label="flip_own（vs 该格自身全量标签）")
axb.errorbar(Bs, vs_med, yerr=vs_err, color=PAL["orange"], marker="^", ms=6, lw=1.6, ls="--",
             capsize=3, ecolor="#4D4D4D", elinewidth=1.0, label="flip_vsθ*（vs 冻结标签）")
axb.plot(Bs, det, color=PAL["purple"], marker="D", ms=4.5, lw=1.2, ls=":",
         label="确定性重标差异率（结构性）")
for b_, v_ in zip(Bs, own_med):
    axb.text(b_, v_ - 0.42, f"{v_:.2f}", ha="center", fontsize=7.8, color=PAL["blue"], fontfamily=YH_FONT)
axb.annotate("B 变化的差异以确定性重标为主\n（分箱结构改变），随机翻转仅缓升",
             xy=(15, 4.68), xytext=(8.6, 2.55), fontsize=8.5, fontfamily=YH_FONT,
             arrowprops=dict(arrowstyle="->", color=PAL["purple"], lw=1.0),
             bbox=dict(boxstyle="round,pad=0.3", fc="#F5F7FA", ec="#B9C4CE", alpha=0.95))
axb.set_xticks(Bs, [str(b) for b in Bs])
axb.set_xlim(3.2, 21.8)
axb.set_ylim(0, 6.3)
axb.set_xlabel("分箱数 B（τ 固定 = (0.86, 0.98)）", fontfamily=YH_FONT)
axb.set_ylabel("翻转率（%）", fontfamily=YH_FONT)
axb.set_title("(b) B 邻域：对自身缓升、对 θ* 的差异\n以确定性重标为主（a7_翻转带敏感性.csv）",
              fontsize=10.5, fontfamily=YH_FONT)
axb.legend(prop={"family": YH_FONT, "size": 8}, loc="upper left")

# ---- (c) 距离带画像 ----
bars = axc.bar(np.arange(8), freq, 0.72, color=PAL["blue"], alpha=0.9)
for r, v, n_ in zip(bars, freq, n_rows):
    axc.text(r.get_x() + r.get_width() / 2, v + 0.7, f"{v:.1f}", ha="center", fontsize=8,
             color="#333333", fontfamily=YH_FONT)
    axc.text(r.get_x() + r.get_width() / 2, -4.6, f"n={n_}", ha="center", fontsize=6.8,
             color=PAL["gray"], fontfamily=YH_FONT)
axc.axhline(0, color=PAL["gray_l"], lw=0.8)
axc.set_xticks(np.arange(8), BAND_SHORT, fontfamily=YH_FONT)
axc.set_xlabel("该行到最近适用边界的相对距离（%，日志轴分带）", fontfamily=YH_FONT)
axc.set_ylim(-6.5, 50)
axc.set_ylabel("行均翻转频率（%）", fontfamily=YH_FONT)
axc.set_title("(c) θ* 翻转带画像：距边界越近越易翻转\n（a7_翻转带画像.csv，1000 次 bootstrap）",
              fontsize=10.5, fontfamily=YH_FONT)
axc.text(0.98, 0.97,
         f"任一次重估即翻转：{n_any} 行（{100 * n_any / 11167:.1f}%）\n"
         f"高频翻转（频率≥5%）：{n_high} 行（{100 * n_high / 11167:.1f}%），\n贡献翻转质量 {100 * q_high:.1f}%",
         transform=axc.transAxes, va="top", ha="right", fontsize=8.2, linespacing=1.6,
         fontfamily=YH_FONT,
         bbox=dict(boxstyle="round,pad=0.35", fc="#F5F7FA", ec="#B9C4CE", alpha=0.95))

fig.suptitle("图8　问题1 标注规则的翻转带敏感性：τ 邻域稳定、B 差异属结构性、翻转集中于边界带",
             fontsize=13, y=1.03, fontfamily=YH_FONT)
fig.tight_layout()

save_fig(fig, "fig8_翻转带敏感性.png", log)
log.info("完成：%s", SCRIPT)
