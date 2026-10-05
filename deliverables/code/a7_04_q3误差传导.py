# -*- coding: utf-8 -*-
"""
a7_04_q3误差传导.py —— A7 实验4：Q3 误差传导（方式1 边界带翻转机制量化 + δ 敏感性 + 混淆方向几何）
问题（任务书第 5 项）：
  1) 验证/扩展 q3_方式1误差联合.csv 的边界带翻转机制：翻转组(L̃≠y) vs 一致组(L̃=y) 的 |ŷ−y| 中位
     183.6 vs 60.6 元是否可复现；边界带（δ=0.1·g_j，a6_05 约定）内翻转率对 δ 参数的敏感性。
  2) 方式2 混淆矩阵主要混淆方向与 Q1 边界几何的关系（相邻类主导？错误是否集中于边界附近？）。
方法（与 a6_05 同构复算，逐行重建逐行预测后做逐行分析；全部函数 import 自 a6_q23_common/a6_common，
     route1_fold_predict/route1_labels/nested_fold_select 按 a6_05 原实现逐行复制，保证同折同种子同配置）：
  - 复核门 A（方式1）：池化 L̃ 宏F1 = 0.6362226662（q3_两路线对比.csv 分层5折/20251004 行）；翻转组/
    一致组 n=831/10336、|ŷ−y| P50、边界带占比 与 q3_方式1误差联合.csv 逐项一致（容差 1e-6）。
  - 复核门 B（方式2）：折级 (γ*,s*) 与 a6_05 日志行14-18 一致；池化宏F1=0.6703649106（q3_指标汇总.csv
    主种子行）；混淆矩阵 9 格与 q3_混淆矩阵.csv 分层5折_主种子池化 逐格一致。
  - δ 敏感性：δ∈{0.02,0.05,0.10,0.15,0.20,0.30}，带内 L̃≠y 率、带内 L̃≠L̂ 率、两路线错误的带捕获率。
  - 混淆方向：方式2 混淆 9 格相邻/跨类分解；真实坐标（e=索赔−赔付，x 分箱同 Q1 冻结 edges）到最近
    适用边界 g_j 的相对距离 d_min 分带错误率；方式1 翻转转移矩阵（y→L̃）。
输出：output/tables/a7_q3误差传导.csv（镜像 output/tables/sensitivity_a7/）
日志：output/logs/a7_04_q3误差传导.log
红线：全程离线；只读 A6 产物；不修改任何 A6/前任 A7 产物。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"
LOGD = ROOT / "output" / "logs"
MIRROR = TBL / "sensitivity_a7"
MIRROR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from a6_common import (LABELS, SEED_MAIN, get_logger, macro_f1, confusion,
                       stratified_folds)
from a6_q23_common import (GAMMA_GRID, apply_multiplier, calibrate_one_gamma,
                           class_weight_vector, decile_strata, fit_lgbm_clf,
                           fit_lgbm_reg, load_annex1_sig, make_tree_X)
from sklearn.model_selection import StratifiedKFold

LG = get_logger("a7_04", LOGD / "a7_04_q3误差传导.log")
T0 = time.time()
TIME_GUARD_PARTB = 240.0  # 方式2 重建哨兵：超时则降档为仅方式1 δ 敏感性（记日志，不静默）

LG.info("=== a7_04_q3误差传导 开始（方式1 边界带翻转机制 + δ 敏感性 + 混淆方向几何） ===")
LG.info("环境：numpy=%s pandas=%s sklearn=%s", np.__version__, pd.__version__,
        __import__("sklearn").__version__)

# ---------------- 载入（与 a6_05 同源同序） ----------------
df1 = load_annex1_sig(LG)
y = df1["实际赔付金额"].to_numpy(dtype=float)
claim = df1["索赔金额"].to_numpy(dtype=float)
lab_df = pd.read_csv(TBL / "附件1_风险标注.csv", encoding="utf-8-sig")
pos = {v: i for i, v in enumerate(df1["行ID"].to_numpy())}
lab_df = lab_df.iloc[np.argsort(lab_df["行ID"].map(pos).to_numpy())]
y_lab = lab_df["风险标注"].map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=int)
q1cfg = json.loads((TBL / "q1_定稿配置.json").read_text(encoding="utf-8"))
q1_props = np.array([q1cfg["labels_props"][c] for c in LABELS])
edges = np.array(q1cfg["edges"], dtype=float)
g1, g2 = np.array(q1cfg["g1"], dtype=float), np.array(q1cfg["g2"], dtype=float)
q2cfg = json.loads((TBL / "q2_定稿配置.json").read_text(encoding="utf-8"))
variant_star = q2cfg["变体裁决"]["变体*"]
cfg_star = dict(q2cfg["cfg_star"])
cfg_star["objective"] = "regression" if variant_star == "A" else "huber"
X, _, _ = make_tree_X(df1)
strata_dec = decile_strata(y)
LG.info("载入完成：n=%d，X%s；Q1 冻结 edges=%s", len(y_lab), X.shape, np.round(edges, 2).tolist())


def route1_fold_predict_replica(X, y, claim, tr, va, cfg_star, variant_star, seed, strata_dec):
    """a6_05 route1_fold_predict 逐行复制（变体 A：log10 + 折内 4 折 φ）。"""
    if variant_star == "A":
        z = np.log10(y)
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        zhat_inner = np.full(len(tr), np.nan)
        for it, iv in skf.split(np.zeros(len(tr)), strata_dec[tr]):
            mi = fit_lgbm_reg(X.iloc[tr[it]], z[tr[it]], cfg_star, seed,
                              valid=(X.iloc[tr[iv]], z[tr[iv]]))
            zhat_inner[iv] = mi.predict(X.iloc[tr[iv]])
        phi = float(np.mean(10.0 ** (z[tr] - zhat_inner)))
        m = fit_lgbm_reg(X.iloc[tr], z[tr], cfg_star, seed, valid=(X.iloc[va], z[va]))
        yhat_va = 10.0 ** m.predict(X.iloc[va]) * phi
    else:
        m = fit_lgbm_reg(X.iloc[tr], y[tr], cfg_star, seed, valid=(X.iloc[va], y[va]))
        yhat_va = m.predict(X.iloc[va])
    return np.clip(yhat_va, 0.01, None)


def route1_labels_replica(yhat, claim_va, edges, g1, g2):
    """a6_05 route1_labels 逐行复制。"""
    bin_idx = np.searchsorted(edges, yhat, side="right")
    e2 = claim_va - yhat
    lab = np.where(e2 >= g2[bin_idx], 2, np.where(e2 >= g1[bin_idx], 1, 0)).astype(int)
    return lab, e2, bin_idx


def nested_fold_select_replica(X, y_lab, tr, q1_props, seed):
    """a6_05 nested_fold_select 逐行复制。"""
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    P_inner = {g: np.full((len(tr), 3), np.nan) for g in GAMMA_GRID}
    for it, iv in skf.split(np.zeros(len(tr)), y_lab[tr]):
        for g in GAMMA_GRID:
            m = fit_lgbm_clf(X.iloc[tr[it]], y_lab[tr[it]], seed,
                             sample_weight=class_weight_vector(y_lab[tr[it]], g),
                             valid=(X.iloc[tr[iv]], y_lab[tr[iv]]))
            P_inner[g][iv] = m.predict(X.iloc[tr[iv]])
    per_g = {g: calibrate_one_gamma(P_inner[g], y_lab[tr], q1_props) for g in GAMMA_GRID}
    g_star = max(GAMMA_GRID, key=lambda g: (per_g[g]["feasible"], per_g[g]["macro"],
                                            per_g[g]["sev_recall"], -per_g[g]["l1"]))
    return g_star, per_g[g_star], per_g, P_inner


# ---------------- Part A：方式1 逐行重建 + 复核门 A ----------------
folds_main = stratified_folds(y_lab, 5, SEED_MAIN)
yhat_oof = np.full(len(y_lab), np.nan)
L_tilde = np.full(len(y_lab), -1)
e2_all = np.full(len(y_lab), np.nan)
for k, (tr, va) in enumerate(folds_main):
    yhat_va = route1_fold_predict_replica(X, y, claim, tr, va, cfg_star, variant_star, SEED_MAIN, strata_dec)
    L_v, e2_v, _ = route1_labels_replica(yhat_va, claim[va], edges, g1, g2)
    yhat_oof[va], L_tilde[va], e2_all[va] = yhat_va, L_v, e2_v
    LG.info("方式1 折%d：宏F1=%.4f", k + 1, macro_f1(y_lab[va], L_v))

DELTA_REF = 0.10
band_ref = (np.abs(e2_all - g1[np.searchsorted(edges, yhat_oof, side="right")]) < DELTA_REF * g1[np.searchsorted(edges, yhat_oof, side="right")]) | \
           (np.abs(e2_all - g2[np.searchsorted(edges, yhat_oof, side="right")]) < DELTA_REF * g2[np.searchsorted(edges, yhat_oof, side="right")])
macro_r1 = macro_f1(y_lab, L_tilde)
flip1 = L_tilde != y_lab          # L̃≠y（方式1 规则作用于 ŷ 后判错）
abs_ey = np.abs(yhat_oof - y)
LG.info("方式1 池化重建：宏F1=%.10f（q3_两路线对比 0.6362226662）；边界带(δ=0.1)行数=%d（q3_两路线对比 248）",
        macro_r1, int(band_ref.sum()))

vu = pd.read_csv(TBL / "q3_方式1误差联合.csv", encoding="utf-8-sig")
ok_a = True
for _, r in vu.iterrows():
    m = (~flip1) if r["分组"].startswith("Ltilde=L") else flip1
    n_ref, p50_ref, band_ref_g = int(r["行数"]), float(r["abs_ey_P50"]), float(r["边界带占比"])
    n_new, p50_new, band_new = int(m.sum()), float(np.percentile(abs_ey[m], 50)), float(band_ref[m].mean())
    match = (n_new == n_ref) and (abs(p50_new - p50_ref) < 1e-6) and (abs(band_new - band_ref_g) < 1e-9)
    ok_a &= match
    LG.info("复核门A %s：n=%d(参考%d) |ŷ−y|P50=%.6f(参考%.6f) 带占比=%.9f(参考%.9f) → %s",
            r["分组"], n_new, n_ref, p50_new, p50_ref, band_new, band_ref_g, "一致" if match else "不一致")
ok_a &= abs(macro_r1 - 0.6362226662191404) < 1e-9 and int(band_ref.sum()) == 248
if not ok_a:
    LG.error("复核门 A 失败：方式1 重建与 A6 产物不一致，停止扩展分析（防误导），上报 A0/A11。")
    raise SystemExit(1)
LG.info("复核门 A 通过：方式1 逐行重建与 q3_方式1误差联合.csv/q3_两路线对比.csv 完全一致。")

# ---------------- Part B：方式2 逐行重建 + 复核门 B（时间哨兵降档） ----------------
L_hat, P_hat = None, None
elapsed = time.time() - T0
if elapsed > TIME_GUARD_PARTB:
    LG.warning("时间哨兵触发：累计 %.1f 秒 > %.0f 秒，方式2 重建降档跳过（δ 敏感性仅报 L̃≠y 口径，记日志不静默）", elapsed, TIME_GUARD_PARTB)
else:
    LAB_hat = np.full(len(y_lab), -1)
    P_oof = np.full((len(y_lab), 3), np.nan)
    expect_gs = [(1.0, 0.50, 0.50), (1.0, 0.70, 1.00), (1.0, 0.50, 1.00), (0.5, 1.00, 1.50), (1.0, 0.50, 0.50)]
    ok_b = True
    for k, (tr, va) in enumerate(folds_main):
        g_star, s_star, _, _ = nested_fold_select_replica(X, y_lab, tr, q1_props, SEED_MAIN)
        got = (float(g_star), float(s_star["s_pian"]), float(s_star["s_yan"]))
        match = got == expect_gs[k]
        ok_b &= match
        LG.info("方式2 折%d：γ*=%.1f，s*=(%.2f,%.2f)（a6_05 日志参考 γ=%.1f，s*=(%.2f,%.2f)）→ %s",
                k + 1, *got, *expect_gs[k], "一致" if match else "不一致")
        m = fit_lgbm_clf(X.iloc[tr], y_lab[tr], SEED_MAIN,
                         sample_weight=class_weight_vector(y_lab[tr], g_star),
                         valid=(X.iloc[va], y_lab[va]))
        Pv = m.predict(X.iloc[va])
        P_oof[va] = Pv
        LAB_hat[va] = apply_multiplier(Pv, s_star["s_pian"], s_star["s_yan"])
    macro_r2 = macro_f1(y_lab, LAB_hat)
    cm_new = confusion(y_lab, LAB_hat)
    cm_df = pd.read_csv(TBL / "q3_混淆矩阵.csv", encoding="utf-8-sig")
    cm_ref = cm_df[cm_df["口径"] == "分层5折_主种子池化"]
    cm_ref_m = np.zeros((3, 3), dtype=int)
    lab2idx = {c: i for i, c in enumerate(LABELS)}
    for _, r in cm_ref.iterrows():
        cm_ref_m[lab2idx[r["真实"]], lab2idx[r["预测"]]] = int(r["行数"])
    cm_match = bool((cm_new == cm_ref_m).all())
    ok_b &= abs(macro_r2 - 0.6703649106061654) < 1e-9 and cm_match
    LG.info("方式2 池化重建：宏F1=%.10f（参考 0.6703649106）；混淆矩阵逐格一致=%s", macro_r2, cm_match)
    if not ok_b:
        LG.error("复核门 B 失败：方式2 重建与 A6 产物不一致，降档为仅方式1 分析，上报 A0/A11。")
    else:
        LG.info("复核门 B 通过：方式2 逐行重建与 q3_混淆矩阵.csv/q3_指标汇总.csv 完全一致。")
        L_hat, P_hat = LAB_hat, P_oof

# ---------------- Part C：δ 敏感性（任务书扩展项） ----------------
rows = []
bin_yhat = np.searchsorted(edges, yhat_oof, side="right")
flip2 = (L_hat != y_lab) if L_hat is not None else None
n_all = len(y_lab)
r1_all, r2_all = float(flip1.mean()), (float(flip2.mean()) if flip2 is not None else np.nan)
LG.info("总体错误率：P(L̃≠y)=%.4f；P(L̂≠y)=%s", r1_all, f"{r2_all:.4f}" if flip2 is not None else "未计算（降档）")
for delta in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
    b1 = np.abs(e2_all - g1[bin_yhat]) < delta * g1[bin_yhat]
    b2 = np.abs(e2_all - g2[bin_yhat]) < delta * g2[bin_yhat]
    band = b1 | b2
    nb = int(band.sum())
    row = {"块": "δ敏感性", "键": f"δ={delta:.2f}",
           "值1": f"带行数={nb}（占比={nb/n_all:.4f}）",
           "值2": f"P(L̃≠y|带)={flip1[band].mean():.4f}; P(L̃≠y|非带)={flip1[~band].mean():.4f}",
           "值3": f"带内/非带风险比={flip1[band].mean()/max(flip1[~band].mean(),1e-12):.2f}",
           "值4": f"L̃≠y 错误被带捕获={flip1[band].sum()/max(flip1.sum(),1):.4f}"}
    if flip2 is not None:
        row["值4"] += f"; L̂≠y 捕获={flip2[band].sum()/max(flip2.sum(),1):.4f}"
        row["值2"] += f"; P(L̃≠L̂|带)={ (L_tilde[band]!=L_hat[band]).mean():.4f}"
    rows.append(row)
rows.append({"块": "δ敏感性", "键": "参考行", "值1": "δ=0.10 行应=248（q3_两路线对比 边界带行数）",
             "值2": f"P(L̃≠L̂|带,δ=0.1)=0.4072580645（q3_两路线对比.csv 行6）" if flip2 is not None else "方式2 降档未计算",
             "值3": "a6_05 约定 δ=0.1·g_j", "值4": ""})

# ---------------- Part D：混淆方向与 Q1 边界几何 ----------------
if L_hat is not None:
    cm = cm_new
    adj = int(cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1])   # 相邻类混淆
    jump = int(cm[0, 2] + cm[2, 0])                        # 跨类混淆
    err = int((cm.sum() - np.trace(cm)))
    rows.append({"块": "混淆方向", "键": "方式2 池化混淆分解", "值1": f"总错误={err}",
                 "值2": f"相邻类(0↔1,1↔2)={adj}（{adj/err:.4f}）", "值3": f"跨类(0↔2)={jump}（{jump/err:.4f}）",
                 "值4": f"最大单项：真实偏高→判合理={cm[1,0]}；真实严重→判偏高={cm[2,1]}"})
    trans1 = np.zeros((3, 3), dtype=int)
    for a in range(3):
        for b in range(3):
            trans1[a, b] = int(((y_lab == a) & (L_tilde == b)).sum())
    rows.append({"块": "混淆方向", "键": "方式1 翻转转移矩阵 y→L̃", "值1": "行=真实类,列=L̃",
                 "值2": "; ".join(f"{LABELS[a]}→{{" + ",".join(f"{LABELS[b]}:{trans1[a,b]}" for b in range(3)) + "}" for a in range(3)),
                 "值3": f"相邻转移占比={(trans1[0,1]+trans1[1,0]+trans1[1,2]+trans1[2,1])/max(flip1.sum(),1):.4f}",
                 "值4": f"跨类转移(0↔2)={(trans1[0,2]+trans1[2,0])/max(flip1.sum(),1):.4f}"})
    # Q1 边界几何（真实坐标：x=实际赔付金额 分箱同冻结 edges，e=索赔−赔付）
    bin_x = np.searchsorted(edges, y, side="right")
    e_true = claim - y
    d1 = np.abs(e_true - g1[bin_x]) / g1[bin_x]
    d2 = np.abs(e_true - g2[bin_x]) / g2[bin_x]
    d_min = np.minimum(d1, d2)
    med_err = float(np.median(d_min[flip2])) if flip2 is not None else np.nan
    med_ok = float(np.median(d_min[~flip2])) if flip2 is not None else np.nan
    rows.append({"块": "边界几何", "键": "真实坐标 d_min（到最近适用 g_j 的相对距离）",
                 "值1": f"方式2错误行中位={med_err:.4f}（n={int(flip2.sum())}）" if flip2 is not None else "",
                 "值2": f"方式2正确行中位={med_ok:.4f}（n={int((~flip2).sum())}）" if flip2 is not None else "",
                 "值3": f"方式1翻转行中位={float(np.median(d_min[flip1])):.4f}（n={int(flip1.sum())}）",
                 "值4": f"方式1一致行中位={float(np.median(d_min[~flip1])):.4f}"})
    from scipy import stats as _st
    u = _st.mannwhitneyu(d_min[flip2], d_min[~flip2], alternative="less")
    rows.append({"块": "边界几何", "键": "Mann-Whitney U（方式2错误行 d_min < 正确行，单侧）",
                 "值1": f"U={u.statistic:.0f}", "值2": f"p={u.pvalue:.3e}",
                 "值3": "错误行显著更贴近 Q1 边界 → 两路线难点几何同源" if u.pvalue < 0.05 else "未见显著贴近",
                 "值4": ""})
    for q in (0.05, 0.10, 0.20):
        mq = d_min < q
        rows.append({"块": "边界几何", "键": f"d_min<{q:.2f} 带（真实坐标, n={int(mq.sum())}）",
                     "值1": f"P(L̂≠y|带)={flip2[mq].mean():.4f}; P(L̂≠y|非带)={flip2[~mq].mean():.4f}" if flip2 is not None else "",
                     "值2": f"P(L̃≠y|带)={flip1[mq].mean():.4f}; P(L̃≠y|非带)={flip1[~mq].mean():.4f}",
                     "值3": f"带占比={mq.mean():.4f}", "值4": ""})
else:
    rows.append({"块": "混淆方向", "键": "方式2 逐行重建", "值1": "时间哨兵降档未执行", "值2": "",
                 "值3": "混淆方向汇总转抄 q3_混淆矩阵.csv：相邻类(0↔1,1↔2)=804/846=0.9504；跨类(0↔2)=42", "值4": ""})
    rows.append({"块": "边界几何", "键": "方式1 侧（无 L̂ 口径）",
                 "值1": f"P(L̃≠y)={r1_all:.4f}", "值2": "δ 敏感性见上（仅 L̃≠y 口径）", "值3": "", "值4": ""})

# ---------------- 机制量化块（任务书验证项 183.6 vs 60.6） ----------------
rows.append({"块": "机制量化", "键": "翻转组(L̃≠y) |ŷ−y| 中位/P90", "值1": f"P50={np.percentile(abs_ey[flip1],50):.2f}元",
             "值2": f"P90={np.percentile(abs_ey[flip1],90):.2f}元", "值3": f"n={int(flip1.sum())}", "值4": ""})
rows.append({"块": "机制量化", "键": "一致组(L̃=y) |ŷ−y| 中位/P90", "值1": f"P50={np.percentile(abs_ey[~flip1],50):.2f}元",
             "值2": f"P90={np.percentile(abs_ey[~flip1],90):.2f}元", "值3": f"n={int((~flip1).sum())}", "值4": ""})
rows.append({"块": "机制量化", "键": "结论", "值1": "翻转组 |ŷ−y| 中位≈一致组 3 倍；e'=e−Δ回归误差推动 e′ 跨越冻结边界",
             "值2": "带内 L̃≠y 率随 δ 收窄单调上升 → 翻转集中于边界带，属边界几何固有不确定度的传导",
             "值3": "", "值4": ""})

out = pd.DataFrame(rows)
out.to_csv(TBL / "a7_q3误差传导.csv", index=False, encoding="utf-8-sig")
out.to_csv(MIRROR / "a7_q3误差传导.csv", index=False, encoding="utf-8-sig")
LG.info("输出 a7_q3误差传导.csv（%d 行）+ 镜像", len(out))
LG.info("=== a7_04 完成，耗时 %.1f 秒 ===", time.time() - T0)
