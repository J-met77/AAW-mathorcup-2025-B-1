# -*- coding: utf-8 -*-
"""
a7_03_q2残差分析.py —— A7 实验3：Q2 误差分析（残差分位结构、WAPE 构成、线性族差距含义）
输入（A6 产物，只读）：
  - output/tables/q2_oof预测.csv（变体A，种子 20251004 池化 OOF，每行恰一预测）
  - output/tables/q2_指标汇总.csv（LGBM_变体A / ElasticNet / OLS / B3 / B4 双口径 WAPE——线性族对比转抄）
  - output/tables/附件1_clean.csv（行ID → 索赔金额、实际赔付金额）
内容：
  1) 复核：池化 OOF WAPE 应= 0.340649（q2_指标汇总.csv 第 2 行 / a6_03 日志行10）。
  2) 按 y_true 十分位分解 WAPE：分子（Σ|y−ŷ|）与分母（Σy）份额、区间 WAPE、APE 中位/P90、
     有向偏差中位（欠估/高估方向）；重尾证据（头部十分位对 Σ|err| 的贡献）。
  3) 异方差证据：Spearman(|残差|, y) 与 Spearman(APE, y)；逐十分位 |残差| 中位的单调性。
  4) 线性族差距含义：单特征 log-log 线性基线（log10 y ~ a + b·log10(索赔金额)，同分层 5 折、
     折内 4 折 OOF 估 smearing φ_fold，与 A5 §3.3 同式）的池化 WAPE——量化"索赔金额单一变量
     线性（对数空间）可解释多少可达标信号"，为 ElasticNet/OLS 与 LGBM 差距 ~0.002 的解读提供证据
     （T5 视角：特征信息量表述保守，D11-③）。
输出：output/tables/a7_q2残差分析.csv（镜像 output/tables/sensitivity_a7/）
日志：output/logs/a7_03_q2残差分析.log
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"
LOGD = ROOT / "output" / "logs"
MIRROR = TBL / "sensitivity_a7"
MIRROR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOGD / "a7_03_q2残差分析.log", mode="w", encoding="utf-8"),
                              logging.StreamHandler()])
LG = logging.getLogger("a7_03")
T0 = time.time()
SEED = 20251004

LG.info("=== a7_03_q2残差分析 开始 ===")
LG.info("环境：numpy=%s pandas=%s scipy=%s sklearn=%s", np.__version__, pd.__version__,
        __import__("scipy").__version__, __import__("sklearn").__version__)

oof = pd.read_csv(TBL / "q2_oof预测.csv", encoding="utf-8-sig")
oa = oof[(oof["变体"] == "LGBM_变体A") & (oof["seed"] == SEED)].copy()
assert len(oa) == 11167 and oa["行ID"].nunique() == 11167
df = pd.read_csv(TBL / "附件1_clean.csv", encoding="utf-8-sig")
m = oa.merge(df[["行ID", "索赔金额", "实际赔付金额"]], on="行ID", how="left", validate="1:1")
assert m["实际赔付金额"].notna().all()
y = m["实际赔付金额"].to_numpy(float)
yh = m["y_pred_oof"].to_numpy(float)
ycl = m["索赔金额"].to_numpy(float)
abs_err = np.abs(y - yh)
wape = abs_err.sum() / y.sum()
LG.info("复核池化 OOF WAPE=%.6f（q2_指标汇总.csv LGBM_变体A/分层5折/20251004 行=0.340649）", wape)
assert abs(wape - 0.3406492461766801) < 1e-9, "OOF 池化 WAPE 与 A6 产物不一致，停步"

rows = []
def add(块, 键, v1, v2="", v3="", v4=""):
    rows.append({"块": 块, "键": 键, "值1": v1, "值2": v2, "值3": v3, "值4": v4})

# ---------------- 1. y 十分位分解 ----------------
dec = pd.qcut(y, 10, labels=False, duplicates="drop")
K = int(dec.max()) + 1
tot_y, tot_e = y.sum(), abs_err.sum()
add("说明", "WAPE=Σ|y−ŷ|/Σy（A1 C9 主指标）；十分位按 y_true 划分", f"K={K}", f"Σy={tot_y:.0f}元", f"Σ|err|={tot_e:.0f}元")
for k in range(K):
    mk = dec == k
    n_k = int(mk.sum())
    y_lo, y_hi = float(y[mk].min()), float(y[mk].max())
    ape = abs_err[mk] / y[mk]
    bias = (yh[mk] - y[mk]) / y[mk]
    add("y十分位", f"D{k+1} y∈[{y_lo:.1f},{y_hi:.1f}]",
        f"n={n_k}", f"Σy份额={y[mk].sum()/tot_y:.4f}",
        f"Σ|err|份额={abs_err[mk].sum()/tot_e:.4f}",
        f"WAPE区间={abs_err[mk].sum()/y[mk].sum():.4f}; APE中位={np.median(ape):.4f}; "
        f"APE_P90={np.percentile(ape,90):.4f}; 有向偏差中位={np.median(bias):+.4f}")
mk_top = dec == K - 1
mk_p95 = y >= np.percentile(y, 95)
add("重尾证据", "头部十分位(D10)贡献", f"Σy份额={y[mk_top].sum()/tot_y:.4f}", f"Σ|err|份额={abs_err[mk_top].sum()/tot_e:.4f}")
add("重尾证据", "y≥P95 的 5% 行贡献", f"Σy份额={y[mk_p95].sum()/tot_y:.4f}", f"Σ|err|份额={abs_err[mk_p95].sum()/tot_e:.4f}")

# ---------------- 2. 异方差 ----------------
sp_abs = stats.spearmanr(abs_err, y)
sp_ape = stats.spearmanr(abs_err / y, y)
med_err_by_dec = [float(np.median(abs_err[dec == k])) for k in range(K)]
mono_spear = stats.spearmanr(np.arange(K), med_err_by_dec)
add("异方差", "Spearman(|残差|, y)", f"ρ={sp_abs.statistic:.4f}", f"p={sp_abs.pvalue:.2e}")
add("异方差", "Spearman(APE, y)", f"ρ={sp_ape.statistic:.4f}", f"p={sp_ape.pvalue:.2e}")
add("异方差", "|残差|中位 随十分位（D1→D10）", "→".join(f"{v:.0f}" for v in med_err_by_dec),
    f"单调 Spearman ρ={mono_spear.statistic:.4f}")
add("异方差", "解读", "绝对误差随 y 单调放大（异方差），相对误差(AGE)随 y 变化平缓" if abs(sp_ape.statistic) < abs(sp_abs.statistic) else "见数值",
    "WAPE 为加权绝对口径，误差集中在头部十分位")

# ---------------- 3. 单特征 log-log 线性基线（同分层 5 折 + 折内 smearing） ----------------
from a6_common import stratified_folds  # noqa: E402  (与 a6_03 同一折划分实现)
z = np.log10(y)
strata = np.asarray(pd.qcut(z, 10, labels=False))
zf = np.log10(ycl)
folds = stratified_folds(strata, 5, SEED)
zhat_oof = np.full(len(y), np.nan)
phi_folds = []
for tr, va in folds:
    # 折内 4 折 OOF 估 φ（A5 §3.3 同式，防训练内乐观偏差）
    inner_strata = strata[tr]
    inner_folds = stratified_folds(inner_strata, 4, SEED)
    zhat_inner = np.full(len(tr), np.nan)
    for itr, iva in inner_folds:
        lr = LinearRegression().fit(zf[tr][itr].reshape(-1, 1), z[tr][itr])
        zhat_inner[iva] = lr.predict(zf[tr][iva].reshape(-1, 1))
    resid_in = z[tr] - zhat_inner
    phi_fold = float(np.mean(10 ** resid_in))
    phi_folds.append(phi_fold)
    lr = LinearRegression().fit(zf[tr].reshape(-1, 1), z[tr])
    zhat_oof[va] = lr.predict(zf[va].reshape(-1, 1))
    # 记录系数（最后一折）
    a_b = (float(lr.intercept_), float(lr.coef_[0]))
# 修正：按验证行归属折序应用 φ_fold
phi_by_row = np.empty(len(y))
fold_id = np.empty(len(y), dtype=int)
for k, (tr, va) in enumerate(folds):
    fold_id[va] = k
phi_by_row = np.array(phi_folds)[fold_id]
yhat_log = 10 ** zhat_oof * phi_by_row
wape_log_raw = (np.abs(y - 10 ** zhat_oof).sum()) / tot_y
wape_log = np.abs(y - yhat_log).sum() / tot_y
LG.info("log-log 单特征基线：折内 φ_fold=%s；池化 WAPE（带 smearing）=%.4f（不带=%.4f）",
        [f"{v:.4f}" for v in phi_folds], wape_log, wape_log_raw)
# 全量拟合系数（报告用）
lr_full = LinearRegression().fit(zf.reshape(-1, 1), z)
r2_log = float(lr_full.score(zf.reshape(-1, 1), z))
add("loglog基线", "log10(y)=a+b·log10(索赔金额) 全量拟合",
    f"b={lr_full.coef_[0]:.4f}", f"R²={r2_log:.4f}",
    f"池化WAPE(带smearing)={wape_log:.4f}", f"池化WAPE(不带)={wape_log_raw:.4f}")
add("loglog基线", "口径", "分层5折（种子20251004，stratified_folds 与 a6_03 同实现）；φ_fold 用折内 4 折 OOF（A5 §3.3 同式）")

# ---------------- 4. 线性族对比（A6 数字转抄，q2_指标汇总.csv） ----------------
q2 = pd.read_csv(TBL / "q2_指标汇总.csv", encoding="utf-8-sig")
def wape_of(model, kou, seed=20251004):
    r = q2[(q2["模型"] == model) & (q2["口径"] == kou) & (q2["种子"] == seed)]
    return float(r["WAPE"].iloc[0])
lgbm_3 = [wape_of("LGBM_变体A", "分层5折", s) for s in (20251004, 20251005, 20251006)]
en_3 = [wape_of("ElasticNet", "分层5折", s) for s in (20251004, 20251005, 20251006)]
ols_3 = [wape_of("OLS", "分层5折", s) for s in (20251004, 20251005, 20251006)]
b3_3 = [wape_of("B3_单参数比", "分层5折", s) for s in (20251004, 20251005, 20251006)]
b4_3 = [wape_of("B4_分段比", "分层5折", s) for s in (20251004, 20251005, 20251006)]
lgbm_ro, en_ro, ols_ro = wape_of("LGBM_变体A", "行序外推"), wape_of("ElasticNet", "行序外推"), wape_of("OLS", "行序外推")
add("线性族对比", "3种子主口径WAPE均值(20251004~06)",
    f"LGBM_A={np.mean(lgbm_3):.6f}", f"ElasticNet={np.mean(en_3):.6f}", f"OLS={np.mean(ols_3):.6f}",
    f"B3={np.mean(b3_3):.6f}; B4={np.mean(b4_3):.6f}")
add("线性族对比", "差距（3种子均值）",
    f"EN−LGBM={np.mean(en_3)-np.mean(lgbm_3):+.6f}", f"OLS−LGBM={np.mean(ols_3)-np.mean(lgbm_3):+.6f}",
    f"B4−LGBM={np.mean(b4_3)-np.mean(lgbm_3):+.6f}", f"B3−LGBM={np.mean(b3_3)-np.mean(lgbm_3):+.6f}")
add("线性族对比", "3种子极差",
    f"LGBM_A={max(lgbm_3)-min(lgbm_3):.6f}", f"EN={max(en_3)-min(en_3):.6f}", f"OLS={max(ols_3)-min(ols_3):.6f}",
    f"种子区间: LGBM[{min(lgbm_3):.4f},{max(lgbm_3):.4f}] EN[{min(en_3):.4f},{max(en_3):.4f}]")
add("线性族对比", "行序口径", f"LGBM_A={lgbm_ro:.6f}", f"ElasticNet={en_ro:.6f}", f"OLS={ols_ro:.6f}",
    "行序下 OLS 0.3334 略优于 LGBM 0.3357（A6 已如实报告）")
add("线性族对比", "loglog单特征基线(本次实验)", f"池化WAPE={wape_log:.6f}",
    f"与LGBM_A差距={wape_log - np.mean(lgbm_3):+.6f}",
    f"与EN差距={wape_log - np.mean(en_3):+.6f}",
    "口径差：本次为单种子5折池化 vs A6 三种子均值，仅作量级参照")
add("结论素材", "T5 视角", "索赔金额 Spearman +0.8069 一枝独秀（A3 T5），log-log 单特征即达 ~%.3f 量级" % wape_log,
    "线性族与 LGBM 差距 ~0.002 << B4 差距 ~0.070：目标可预测部分主要由索赔金额的（近）对数线性结构承载",
    "模型族增益存在但小；特征信息量表述须保守（D11-③/A2-17）", "")

out = pd.DataFrame(rows)
out.to_csv(TBL / "a7_q2残差分析.csv", index=False, encoding="utf-8-sig")
out.to_csv(MIRROR / "a7_q2残差分析.csv", index=False, encoding="utf-8-sig")
LG.info("输出 a7_q2残差分析.csv（%d 行）+ 镜像", len(out))
LG.info("=== a7_03 完成，耗时 %.1f 秒 ===", time.time() - T0)
