# -*- coding: utf-8 -*-
"""
a6_05_q3分类双路线.py —— 问题3：方式2 直接分类（主路线，M3-4+M3-5 两级杠杆嵌套校准）
                          + 方式1 完整链路对比（Q2 定稿配置 + Q1 冻结边界，M4-6 同台）
                          + 消融 A0–A6（C13）+ 混淆矩阵 + 附件2 风险标注预测（D17/D18/D19）
依赖（C11 硬依赖）：output/tables/附件1_风险标注.csv（Q1 定稿标签）、q1_定稿配置.json（冻结边界）、
                   q2_定稿配置.json + 附件2_赔付预测.csv（a6_03/04 产物）。
- 主路线：3 种子×5 折（按标签分层）；每外折内层 4 折 OOF：γ∈{0,0.5,1} 训练期权重 w=(n/(3n_c))^γ，
  (s_偏,s_严) 网格校准（约束占比带 C2，目标 macro-F1，并列：严重 recall 高→占比 L1 距小→字典序，A5 §4.3）
  → (γ*,s*) → 外折以 γ* 重训、s* 后验乘子判类。行序外推口径同构（D19/U3）。
- 消融（种子 20251004 单种子，A5 §4.5；重采样/加权仅作用于训练环节，评估在原分布验证集）：
  A0 无杠杆 / A1 仅 M3-4(γ=1) / A2 仅 M3-5(γ=0+s) / A3 γ=1+s / A4 γ=0.5+s /
  A5 随机过采样（numpy 手写）/ A6 均衡集成 R=10（numpy 手写）。A2–A4 的 s 校准复用主路线内层 OOF 概率（省拟合，登记）。
- 方式1：同折同种子同特征集（M4-6 归因控制），Q2 定稿模型（变体 A 含折内 4 折 φ）→ ŷ →
  e'=索赔金额−ŷ → Q1 冻结规则（外推约定）→ L̃；报宏 F1/每类/严重类、两路线一致率、边界带翻转率（δ=0.1·g_j，约定）。
- 检测点：R-06 交接断言、R-07（log loss 劣于 A0 或严重类 recall 全 0）、R-04（双口径排序反转）。
- 随机源登记：折切分 stratified_folds(y_lab,5,seed)、内层 StratifiedKFold(4,random_state=seed)、
  LGBM seed=外层种子；A5 过采样 rng=default_rng([20251004,折序])；A6 子集 rng=default_rng([20251004,1000+折序,r])。
- 输出：q3_指标汇总.csv、q3_消融矩阵.csv、q3_混淆矩阵.csv、q3_两路线对比.csv、q3_方式1误差联合.csv、
        附件2_风险预测.csv、q3_定稿配置.json；日志 output/logs/a6_05_q3双路线.log。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (BOOT_B, F_ANNEX1, LABELS, LOGD, SEEDS_REPEAT, SEED_MAIN, TBL,
                       assert_features_ok, confusion, get_logger, log_env, macro_f1,
                       md5_of, per_class_prf, stratified_folds)
from a6_q23_common import (GAMMA_GRID, S_GRID_PIAN, S_GRID_YAN, apply_multiplier,
                           bootstrap_ci_metric, calibrate_one_gamma, class_weight_vector,
                           decile_strata, fit_lgbm_clf, fit_lgbm_reg, load_annex1_sig,
                           load_annex2_sig, logloss_oof, make_tree_X, row_order_split,
                           write_block_report)

T0 = time.time()
LG = get_logger("a6_05", LOGD / "a6_05_q3双路线.log")
F_LAB = TBL / "附件1_风险标注.csv"
F_PRED2 = TBL / "附件2_赔付预测.csv"
F_Q1CFG = TBL / "q1_定稿配置.json"
F_Q2CFG = TBL / "q2_定稿配置.json"
TIME_GUARD_ABLATION = 420.0  # 超时降档哨兵（A5 §6）：主路线+行序完成后若超此值，消融裁至 A0–A3 并记日志


def sev_recall_fn(y, lab):
    n = int((y == 2).sum())
    return float(((y == 2) & (lab == 2)).sum() / n) if n > 0 else float("nan")


def sev_prec_fn(y, lab):
    n = int((lab == 2).sum())
    return float(((y == 2) & (lab == 2)).sum() / n) if n > 0 else float("nan")


def clf_metric_row(tag, protocol, seed, y_true, P, lab_pred, fold_macro=None, with_boot=False):
    r = {"配置": tag, "口径": protocol, "种子": seed}
    r["宏F1"] = macro_f1(y_true, lab_pred)
    r.update(per_class_prf(y_true, lab_pred))
    r["严重类_recall"], r["严重类_precision"] = sev_recall_fn(y_true, lab_pred), sev_prec_fn(y_true, lab_pred)
    for c in range(3):
        r[f"预测占比_{LABELS[c]}"] = float((lab_pred == c).mean())
    r["OOF_logloss"] = logloss_oof(P, y_true) if P is not None else np.nan
    if fold_macro is not None and len(fold_macro) > 1:
        r["折间宏F1_均值"], r["折间宏F1_std"] = float(np.mean(fold_macro)), float(np.std(fold_macro))
    else:
        r["折间宏F1_均值"] = float(fold_macro[0]) if fold_macro else np.nan
        r["折间宏F1_std"] = 0.0
    if with_boot:
        for name, fn in (("宏F1", macro_f1), ("严重类recall", sev_recall_fn), ("严重类precision", sev_prec_fn)):
            lo, hi = bootstrap_ci_metric(y_true, lab_pred, fn, seed=SEED_MAIN)
            r[f"{name}_CI2.5"], r[f"{name}_CI97.5"] = lo, hi
    return r


def nested_fold_select(X, y_lab, tr, q1_props, seed):
    """外折内层：4 折 OOF × 3 个 γ → 每 γ 最优 (s_偏,s_严) → 全局 (γ*,s*)。返回 (γ*, s_best, per_g, P_inner)。"""
    from sklearn.model_selection import StratifiedKFold
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


def route1_fold_predict(X, y, claim, tr, va, cfg_star, variant_star, seed, strata_dec):
    """方式1 单折：Q2 定稿模型在 tr 拟合（变体 A 含折内 4 折 φ，A5 §4.4），返回 ŷ_va（已 clip≥0.01）。"""
    if variant_star == "A":
        z = np.log10(y)
        from sklearn.model_selection import StratifiedKFold
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


def route1_labels(yhat, claim_va, edges, g1, g2):
    """Q1 冻结规则作用于 ŷ：bin=等频分箱归属（低于首箱下界归箱 0、高于末箱上界归箱 B−1，外推约定）；
    e'=索赔金额−ŷ（超额坐标）；序贯判定 e'≥g2[bin]→2、e'≥g1[bin]→1、否则 0（同值取高序）。"""
    bin_idx = np.searchsorted(edges, yhat, side="right")
    e2 = claim_va - yhat
    lab = np.where(e2 >= g2[bin_idx], 2, np.where(e2 >= g1[bin_idx], 1, 0)).astype(int)
    return lab, e2, bin_idx


def main():
    lg = LG
    lg.info("=== a6_05_q3分类双路线 开始（A5 §4：方式2 主 + 方式1 对比 + 消融 A0–A6） ===")
    log_env(lg, [F_ANNEX1, F_LAB, F_PRED2, F_Q1CFG, F_Q2CFG])

    # ---------------- 载入与 C11/R-06 交接断言
    df1 = load_annex1_sig(lg)
    assert_features_ok(df1, "附件1", lg)
    y = df1["实际赔付金额"].to_numpy(dtype=float)
    claim = df1["索赔金额"].to_numpy(dtype=float)
    lab_df = pd.read_csv(F_LAB, encoding="utf-8-sig")
    assert lab_df.shape[0] == len(df1) and lab_df["行ID"].is_unique, "标签文件与 clean 行数/键不符"
    pos = {v: i for i, v in enumerate(df1["行ID"].to_numpy())}
    order = lab_df["行ID"].map(pos).to_numpy()
    assert not pd.isna(order).any() and len(set(order.tolist())) == len(order), "标签文件行ID 与 clean 未一一对应"
    lab_df = lab_df.iloc[np.argsort(order)]
    y_lab = lab_df["风险标注"].map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=float)
    assert not np.isnan(y_lab).any(), "风险标注存在无法映射的取值"
    y_lab = y_lab.astype(int)
    n_c = [int((y_lab == c).sum()) for c in range(3)]
    q1cfg = json.loads(F_Q1CFG.read_text(encoding="utf-8"))
    assert n_c == [q1cfg["labels_counts"][c] for c in LABELS], f"标签计数与 Q1 定稿不符：{n_c}"
    if not (100 <= n_c[2] <= 335 and n_c[1] >= 100):
        write_block_report(lg, "R-06 交接断言失败（a6_05）",
                           f"R-06（A2-19/V8 破）：n_严={n_c[2]}、n_偏={n_c[1]} 超带；Q3 训练禁启动。\n")
        raise SystemExit(1)
    lg.info("C11/R-06 交接断言通过：标签计数 合理=%d 偏高=%d 严重=%d（与 q1_定稿配置.json 一致）", *n_c)
    q1_props = np.array([q1cfg["labels_props"][c] for c in LABELS])
    edges = np.array(q1cfg["edges"], dtype=float)
    g1, g2 = np.array(q1cfg["g1"], dtype=float), np.array(q1cfg["g2"], dtype=float)
    B = int(q1cfg["theta_star"]["B"])
    lg.info("Q1 冻结边界（方式1 链路，C11 硬依赖已满足）：B=%d，edges=%s，g1=%s，g2=%s",
            B, np.round(edges, 2).tolist(), np.round(g1, 2).tolist(), np.round(g2, 2).tolist())

    q2cfg = json.loads(F_Q2CFG.read_text(encoding="utf-8"))
    variant_star = q2cfg["变体裁决"]["变体*"]
    cfg_star = dict(q2cfg["cfg_star"])
    cfg_star["objective"] = "regression" if variant_star == "A" else "huber"
    lg.info("Q2 定稿配置（方式1 链路）：变体*=%s，cfg*=%s，φ_final=%.6f",
            variant_star, cfg_star, float(q2cfg["phi_final"]))

    X, cat_ref, _ = make_tree_X(df1)
    strata_dec = decile_strata(y)  # log10(y) 十分位（方式1 内层 φ 分层用）
    lg.info("特征矩阵 %s（与 Q2 同一允许清单，A4 §4.2）", X.shape)

    metric_rows, ablation_rows, route_metric_rows, route_cmp_rows = [], [], [], []

    # ---------------- 主路线（方式2）：3 种子 × 5 折嵌套校准
    P_pool, LAB_pool, folds_by_seed, inner_by_seed, fold_sstar_log = {}, {}, {}, {}, {}
    for seed in SEEDS_REPEAT:
        folds = stratified_folds(y_lab, 5, seed)
        folds_by_seed[seed] = folds
        P_oof = np.full((len(y_lab), 3), np.nan)
        lab_oof = np.full(len(y_lab), -1)
        fold_macro, sstars, inner_store = [], [], {}
        for k, (tr, va) in enumerate(folds):
            g_star, s_star, per_g, P_inner = nested_fold_select(X, y_lab, tr, q1_props, seed)
            inner_store[k] = {"tr": tr, "va": va, "P_inner": P_inner, "g_star": g_star, "s_star": s_star}
            m = fit_lgbm_clf(X.iloc[tr], y_lab[tr], seed,
                             sample_weight=class_weight_vector(y_lab[tr], g_star),
                             valid=(X.iloc[va], y_lab[va]))
            Pv = m.predict(X.iloc[va])
            P_oof[va] = Pv
            lab_oof[va] = apply_multiplier(Pv, s_star["s_pian"], s_star["s_yan"])
            fold_macro.append(macro_f1(y_lab[va], lab_oof[va]))
            sstars.append([float(g_star), float(s_star["s_pian"]), float(s_star["s_yan"])])
            lg.info("主路线 种子=%d 折%d/%d：γ*=%.1f，s*=(%.2f,%.2f)（内层宏F1=%.4f 可行=%d）；"
                    "外折宏F1=%.4f，严重recall=%.4f，预测占比 合=%.4f 偏=%.4f 严=%.4f",
                    seed, k + 1, len(folds), g_star, s_star["s_pian"], s_star["s_yan"], s_star["macro"],
                    s_star["feasible"], fold_macro[-1], sev_recall_fn(y_lab[va], lab_oof[va]),
                    *[float((lab_oof[va] == c).mean()) for c in range(3)])
        P_pool[seed], LAB_pool[seed], inner_by_seed[seed] = P_oof, lab_oof, inner_store
        fold_sstar_log[str(seed)] = sstars
        lg.info("主路线 种子=%d 池化：宏F1=%.4f；严重 recall=%.4f precision=%.4f；预测占比 合=%.4f 偏=%.4f 严=%.4f；logloss=%.4f",
                seed, macro_f1(y_lab, lab_oof), sev_recall_fn(y_lab, lab_oof), sev_prec_fn(y_lab, lab_oof),
                *[float((lab_oof == c).mean()) for c in range(3)], logloss_oof(P_oof, y_lab))
        metric_rows.append(clf_metric_row("方式2_主配置(嵌套γ,s)", "分层5折", seed, y_lab, P_oof,
                                          lab_oof, fold_macro, with_boot=(seed == SEED_MAIN)))

    # ---------------- 行序外推口径（主路线，同构嵌套校准）
    tr_ro, va_ro = row_order_split(len(y_lab))
    g_ro, s_ro, per_g_ro, P_inner_ro = nested_fold_select(X, y_lab, tr_ro, q1_props, SEED_MAIN)
    m_ro = fit_lgbm_clf(X.iloc[tr_ro], y_lab[tr_ro], SEED_MAIN,
                        sample_weight=class_weight_vector(y_lab[tr_ro], g_ro),
                        valid=(X.iloc[va_ro], y_lab[va_ro]))
    P_ro = m_ro.predict(X.iloc[va_ro])
    lab_ro = apply_multiplier(P_ro, s_ro["s_pian"], s_ro["s_yan"])
    lg.info("主路线 行序口径：γ*=%.1f，s*=(%.2f,%.2f)（内层宏F1=%.4f 可行=%d）；验证 宏F1=%.4f，严重recall=%.4f，"
            "预测占比 合=%.4f 偏=%.4f 严=%.4f",
            g_ro, s_ro["s_pian"], s_ro["s_yan"], s_ro["macro"], s_ro["feasible"],
            macro_f1(y_lab[va_ro], lab_ro), sev_recall_fn(y_lab[va_ro], lab_ro),
            *[float((lab_ro == c).mean()) for c in range(3)])
    metric_rows.append(clf_metric_row("方式2_主配置(嵌套γ,s)", "行序外推", SEED_MAIN, y_lab[va_ro], P_ro,
                                      lab_ro, [macro_f1(y_lab[va_ro], lab_ro)], with_boot=True))
    lg.info("主路线耗时哨兵：累计 %.1f 秒", time.time() - T0)
    ablation_set = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    if time.time() - T0 > TIME_GUARD_ABLATION:
        ablation_set = ["A0", "A1", "A2", "A3"]
        lg.warning("A5 §6 超时预案触发：主路线累计 %.1f 秒 > %.0f 秒，消融矩阵裁至 %s（记日志，不静默）",
                   time.time() - T0, TIME_GUARD_ABLATION, ablation_set)

    # ---------------- 消融 A0–A6（种子 20251004；分层 5 折 + 行序；A2–A4 复用主路线内层 OOF 概率）
    def run_ablation_outer(tr, va, gamma, s_pair, kind, fold_key):
        """训练环节单一杠杆外折拟合；kind=weight(γ 加权)/oversample/ensemble；评估在原分布 va。"""
        if kind == "weight":
            m = fit_lgbm_clf(X.iloc[tr], y_lab[tr], SEED_MAIN,
                             sample_weight=class_weight_vector(y_lab[tr], gamma),
                             valid=(X.iloc[va], y_lab[va]))
            Pv = m.predict(X.iloc[va])
        elif kind == "oversample":  # A5：严重类有放回复制至偏高类同量级（numpy 手写，仅训练环节）
            rng = np.random.default_rng([SEED_MAIN, fold_key])
            n_sev, n_mid = int((y_lab[tr] == 2).sum()), int((y_lab[tr] == 1).sum())
            idx_sev = np.where(y_lab[tr] == 2)[0]
            extra = rng.choice(idx_sev, size=max(n_mid - n_sev, 0), replace=True)
            tr_aug = np.concatenate([tr, tr[extra]])
            m = fit_lgbm_clf(X.iloc[tr_aug], y_lab[tr_aug], SEED_MAIN, sample_weight=None,
                             valid=(X.iloc[va], y_lab[va]))
            Pv = m.predict(X.iloc[va])
            lg.info("消融A5 折%d：严重类 %d → 过采样至 %d（=偏高类计数），训练行 %d → %d",
                    fold_key + 1, n_sev, n_mid, len(tr), len(tr_aug))
        elif kind == "ensemble":  # A6：R=10 个均衡子集（全部严重类+等量随机非严重），平均后验
            Pv = np.zeros((len(va), 3))
            for r_i in range(10):
                rng = np.random.default_rng([SEED_MAIN, 1000 + fold_key, r_i])
                idx_sev = np.where(y_lab[tr] == 2)[0]
                idx_rest = np.where(y_lab[tr] != 2)[0]
                sub = np.concatenate([idx_sev, rng.choice(idx_rest, size=len(idx_sev), replace=False)])
                m = fit_lgbm_clf(X.iloc[tr[sub]], y_lab[tr[sub]], SEED_MAIN, sample_weight=None,
                                 valid=(X.iloc[va], y_lab[va]))
                Pv += m.predict(X.iloc[va])
            Pv /= 10.0
        else:
            raise ValueError(kind)
        if s_pair is None:
            lab_va = np.argmax(Pv, axis=1)
        else:
            lab_va = apply_multiplier(Pv, s_pair[0], s_pair[1])
        return Pv, lab_va

    for protocol in ("分层5折", "行序外推"):
        folds_ab = folds_by_seed[SEED_MAIN] if protocol == "分层5折" else [(tr_ro, va_ro)]
        for name in ablation_set:
            fold_macro = []
            lab_pool_ab = np.full(len(y_lab), -1)
            P_pool_ab = np.full((len(y_lab), 3), np.nan) if protocol == "分层5折" else None
            for k, (tr, va) in enumerate(folds_ab):
                if name == "A0":
                    Pv, lab_va = run_ablation_outer(tr, va, 0.0, None, "weight", k)
                elif name == "A1":
                    Pv, lab_va = run_ablation_outer(tr, va, 1.0, None, "weight", k)
                elif name in ("A2", "A3", "A4"):
                    g_fix = {"A2": 0.0, "A3": 1.0, "A4": 0.5}[name]
                    if protocol == "分层5折":
                        P_in = inner_by_seed[SEED_MAIN][k]["P_inner"][g_fix]
                        tr_ref = tr
                    else:
                        P_in = P_inner_ro[g_fix]
                        tr_ref = tr_ro
                    s_fix = calibrate_one_gamma(P_in, y_lab[tr_ref], q1_props)
                    Pv, lab_va = run_ablation_outer(tr, va, g_fix, (s_fix["s_pian"], s_fix["s_yan"]), "weight", k)
                    lg.info("消融 %s %s 折%d：γ=%.1f 固定，s*=(%.2f,%.2f)（内层宏F1=%.4f 可行=%d）",
                            name, protocol, k + 1, g_fix, s_fix["s_pian"], s_fix["s_yan"],
                            s_fix["macro"], s_fix["feasible"])
                elif name == "A5":
                    Pv, lab_va = run_ablation_outer(tr, va, 0.0, None, "oversample", k)
                elif name == "A6":
                    Pv, lab_va = run_ablation_outer(tr, va, 0.0, None, "ensemble", k)
                fold_macro.append(macro_f1(y_lab[va], lab_va))
                lab_pool_ab[va] = lab_va
                if protocol == "分层5折":
                    P_pool_ab[va] = Pv
            if protocol == "分层5折":
                y_true_pool, lab_pool, P_row = y_lab, lab_pool_ab, P_pool_ab
            else:
                y_true_pool, lab_pool, P_row = y_lab[va_ro], lab_pool_ab[va_ro], None
            ablation_rows.append(clf_metric_row(f"消融{name}", protocol, SEED_MAIN, y_true_pool,
                                                P_row, lab_pool, fold_macro))
            lg.info("消融 %s（%s）池化：宏F1=%.4f；严重recall=%.4f precision=%.4f；预测占比 合=%.4f 偏=%.4f 严=%.4f",
                    name, protocol, ablation_rows[-1]["宏F1"], ablation_rows[-1]["严重类_recall"],
                    ablation_rows[-1]["严重类_precision"],
                    ablation_rows[-1]["预测占比_合理诉求"], ablation_rows[-1]["预测占比_诉求偏高"],
                    ablation_rows[-1]["预测占比_严重超额"])

    # ---------------- 方式1 完整链路（同折同种子同特征集，M4-6）
    for seed in SEEDS_REPEAT:
        folds = folds_by_seed[seed]
        lab_oof_r1 = np.full(len(y_lab), -1)
        yhat_oof = np.full(len(y_lab), np.nan)
        fold_macro_r1 = []
        for k, (tr, va) in enumerate(folds):
            yhat_va = route1_fold_predict(X, y, claim, tr, va, cfg_star, variant_star, seed, strata_dec)
            L_tilde, e2, bin_idx = route1_labels(yhat_va, claim[va], edges, g1, g2)
            lab_oof_r1[va] = L_tilde
            yhat_oof[va] = yhat_va
            fold_macro_r1.append(macro_f1(y_lab[va], L_tilde))
        e_all = claim - yhat_oof
        bins_all = np.searchsorted(edges, yhat_oof, side="right")
        band = (np.abs(e_all - g1[bins_all]) < 0.1 * g1[bins_all]) | \
               (np.abs(e_all - g2[bins_all]) < 0.1 * g2[bins_all])
        agree = float((lab_oof_r1 == LAB_pool[seed]).mean())
        flip_band = float((lab_oof_r1[band] != LAB_pool[seed][band]).mean()) if band.any() else float("nan")
        route_metric_rows.append(clf_metric_row("方式1_完整链路", "分层5折", seed, y_lab, None,
                                                lab_oof_r1, fold_macro_r1))
        route_cmp_rows.append({"口径": "分层5折", "种子": seed,
                               "方式2_宏F1": macro_f1(y_lab, LAB_pool[seed]),
                               "方式1_宏F1": macro_f1(y_lab, lab_oof_r1),
                               "方式2_严重recall": sev_recall_fn(y_lab, LAB_pool[seed]),
                               "方式1_严重recall": sev_recall_fn(y_lab, lab_oof_r1),
                               "标签一致率": agree, "边界带翻转率_L1不等于L2": flip_band,
                               "边界带行数": int(band.sum())})
        lg.info("方式1 种子=%d 池化：宏F1=%.4f；严重recall=%.4f；两路线标签一致率=%.4f；边界带翻转率(L̃≠L̂)=%.4f（边界带 %d 行）",
                seed, macro_f1(y_lab, lab_oof_r1), sev_recall_fn(y_lab, lab_oof_r1), agree, flip_band, int(band.sum()))
        if seed == SEED_MAIN:
            band_pool_main = {"yhat": yhat_oof.copy(), "e": e_all.copy(),
                              "lab_r1": lab_oof_r1.copy(), "band": band.copy()}
    # 方式1 行序口径
    yhat_ro = route1_fold_predict(X, y, claim, tr_ro, va_ro, cfg_star, variant_star, SEED_MAIN, strata_dec)
    L_ro1, e_ro, bin_ro = route1_labels(yhat_ro, claim[va_ro], edges, g1, g2)
    band_ro = (np.abs(e_ro - g1[bin_ro]) < 0.1 * g1[bin_ro]) | (np.abs(e_ro - g2[bin_ro]) < 0.1 * g2[bin_ro])
    agree_ro = float((L_ro1 == lab_ro).mean())
    fb_ro = float((L_ro1[band_ro] != lab_ro[band_ro]).mean()) if band_ro.any() else float("nan")
    route_metric_rows.append(clf_metric_row("方式1_完整链路", "行序外推", SEED_MAIN, y_lab[va_ro], None,
                                            L_ro1, [macro_f1(y_lab[va_ro], L_ro1)]))
    route_cmp_rows.append({"口径": "行序外推", "种子": SEED_MAIN,
                           "方式2_宏F1": macro_f1(y_lab[va_ro], lab_ro),
                           "方式1_宏F1": macro_f1(y_lab[va_ro], L_ro1),
                           "方式2_严重recall": sev_recall_fn(y_lab[va_ro], lab_ro),
                           "方式1_严重recall": sev_recall_fn(y_lab[va_ro], L_ro1),
                           "标签一致率": agree_ro, "边界带翻转率_L1不等于L2": fb_ro,
                           "边界带行数": int(band_ro.sum())})
    lg.info("方式1 行序口径：宏F1=%.4f；两路线一致率=%.4f；边界带翻转率=%.4f",
            macro_f1(y_lab[va_ro], L_ro1), agree_ro, fb_ro)

    # R-04 检测：两口径下（主配置 vs A0）与（方式2 vs 方式1）方向须一致
    a0_main = [r for r in ablation_rows if r["配置"] == "消融A0" and r["口径"] == "分层5折"][0]["宏F1"]
    a0_ro_v = [r for r in ablation_rows if r["配置"] == "消融A0" and r["口径"] == "行序外推"][0]["宏F1"]
    main_main = [r for r in metric_rows if r["配置"] == "方式2_主配置(嵌套γ,s)" and r["口径"] == "分层5折"
                 and r["种子"] == SEED_MAIN][0]["宏F1"]
    main_ro_v = [r for r in metric_rows if r["配置"] == "方式2_主配置(嵌套γ,s)" and r["口径"] == "行序外推"][0]["宏F1"]
    r2_main, r2_ro = route_cmp_rows[0], route_cmp_rows[3]
    d1 = (main_main > a0_main) == (main_ro_v > a0_ro_v)
    d2 = (r2_main["方式2_宏F1"] > r2_main["方式1_宏F1"]) == (r2_ro["方式2_宏F1"] > r2_ro["方式1_宏F1"])
    lg.info("R-04 检测（Q3）：主配置>A0 方向两口径%s（分层 %.4f vs %.4f；行序 %.4f vs %.4f）；方式2>方式1 方向两口径%s",
            "一致" if d1 else "反转", main_main, a0_main, main_ro_v, a0_ro_v, "一致" if d2 else "反转")
    if not (d1 and d2):
        write_block_report(lg, "R-04 Q3 双口径排序反转（a6_05）",
                           f"R-04 触发：Q3 双口径宏 F1 排序反转。主配置vsA0 方向一致={d1}，方式2vs方式1 方向一致={d2}。\n"
                           "证据：q3_指标汇总.csv、q3_两路线对比.csv；候选预案：停步上报，两口径分别报告，不得单口径下结论。\n")

    # R-07 检测：主配置 OOF log loss 劣于 A0，或严重类 recall 全 0
    ll_main = [r for r in metric_rows if r["配置"] == "方式2_主配置(嵌套γ,s)" and r["口径"] == "分层5折"
               and r["种子"] == SEED_MAIN][0]["OOF_logloss"]
    a0_ll = [r for r in ablation_rows if r["配置"] == "消融A0" and r["口径"] == "分层5折"][0]["OOF_logloss"]
    sev_rec_all = [r["严重类_recall"] for r in metric_rows
                   if r["配置"] == "方式2_主配置(嵌套γ,s)" and r["口径"] == "分层5折"]
    if ll_main > a0_ll or any((np.isnan(v)) or v == 0 for v in sev_rec_all):
        write_block_report(lg, "R-07 Q3 概率标定/严重类召回异常（a6_05）",
                           f"R-07 触发：主配置 OOF logloss={ll_main:.4f} vs A0={a0_ll:.4f}（劣于 A0 即触发）；"
                           f"严重类 recall 序列={sev_rec_all}。\n候选预案：主族换 M3-6 均衡集成，记录并上报。\n")

    # ---------------- 方式1 误差联合分布表（C14 实证素材：ŷ 误差 × L̃ 翻转）
    yh = band_pool_main["yhat"]
    flip = band_pool_main["lab_r1"] != y_lab
    tbl = []
    for g_name, m_flip in (("Ltilde=L（判定一致）", ~flip), ("Ltilde!=L（判定翻转）", flip)):
        m = m_flip & ~np.isnan(yh)
        tbl.append({"分组": g_name, "行数": int(m.sum()),
                    "abs_ey_P50": float(np.percentile(np.abs(yh[m] - y[m]), 50)),
                    "abs_ey_P90": float(np.percentile(np.abs(yh[m] - y[m]), 90)),
                    "e减yhat_P50": float(np.percentile(band_pool_main["e"][m], 50)),
                    "边界带占比": float(band_pool_main["band"][m].mean())})
    pd.DataFrame(tbl).to_csv(TBL / "q3_方式1误差联合.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q3_方式1误差联合.csv：%s", [f"{t['分组']} n={t['行数']} |ŷ−y|P50={t['abs_ey_P50']:.1f}" for t in tbl])

    # ---------------- 定稿：全量嵌套校准 → 附件2 风险标注预测（方式2 进 Result，D17）
    tr_all = np.arange(len(y_lab))
    g_fin, s_fin, per_g_fin, _P = nested_fold_select(X, y_lab, tr_all, q1_props, SEED_MAIN)
    lg.info("全量嵌套校准：γ*=%.1f，s*=(%.2f,%.2f)（内层宏F1=%.4f 可行=%d）；各 γ 内层=%s",
            g_fin, s_fin["s_pian"], s_fin["s_yan"], s_fin["macro"], s_fin["feasible"],
            {f"γ={g}": {"s": [per_g_fin[g]["s_pian"], per_g_fin[g]["s_yan"]],
                        "宏F1": round(per_g_fin[g]["macro"], 4), "可行": per_g_fin[g]["feasible"]} for g in GAMMA_GRID})
    m_fin = fit_lgbm_clf(X, y_lab, SEED_MAIN, sample_weight=class_weight_vector(y_lab, g_fin))
    df2 = load_annex2_sig(lg)
    assert_features_ok(df2, "附件2", lg)
    X2, _, unseen2 = make_tree_X(df2, cat_ref=cat_ref)
    assert all(v == 0 for v in unseen2.values()), f"附件2 出现附件1 未见类别：{unseen2}"
    P2 = m_fin.predict(X2)
    lab2 = apply_multiplier(P2, s_fin["s_pian"], s_fin["s_yan"])
    out2 = pd.DataFrame({"运单号": df2["运单号"].to_numpy(), "风险标注": [LABELS[v] for v in lab2]})
    assert len(out2) == 2792 and out2["风险标注"].notna().all()
    out2.to_csv(TBL / "附件2_风险预测.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 附件2_风险预测.csv：2792 行（方式2 argmax(s·p) 判类，D17）")
    lg.info("附件2 方式2 预测占比：合=%.4f 偏=%.4f 严=%.4f（如实报告，不硬凑 C2 带，A2-06）",
            *[float((lab2 == c).mean()) for c in range(3)])

    # 方式1 在附件2 的对照（复用 a6_04 ŷ，不新增拟合）
    pred2 = pd.read_csv(F_PRED2, encoding="utf-8-sig")
    assert (pred2["运单号"].to_numpy() == df2["运单号"].to_numpy()).all(), "附件2_赔付预测 行序与附件2_clean 不一致"
    L2_r1, e2_all, bin2 = route1_labels(pred2["ŷ"].to_numpy(dtype=float),
                                        df2["索赔金额"].to_numpy(dtype=float), edges, g1, g2)
    agree2 = float((L2_r1 == lab2).mean())
    band2 = (np.abs(e2_all - g1[bin2]) < 0.1 * g1[bin2]) | (np.abs(e2_all - g2[bin2]) < 0.1 * g2[bin2])
    fb2 = float((L2_r1[band2] != lab2[band2]).mean()) if band2.any() else float("nan")
    lg.info("附件2 两路线对照：方式1 标签占比 合=%.4f 偏=%.4f 严=%.4f；两路线一致率=%.4f；边界带翻转率(L̃≠L̂)=%.4f（边界带 %d 行）",
            *[float((L2_r1 == c).mean()) for c in range(3)], agree2, fb2, int(band2.sum()))
    route_cmp_rows.append({"口径": "附件2预测", "种子": SEED_MAIN,
                           "方式2_宏F1": np.nan, "方式1_宏F1": np.nan,
                           "方式2_严重recall": float((lab2 == 2).mean()),
                           "方式1_严重recall": float((L2_r1 == 2).mean()),
                           "标签一致率": agree2, "边界带翻转率_L1不等于L2": fb2,
                           "边界带行数": int(band2.sum())})

    # ---------------- 混淆矩阵（主配置：主种子池化 + 行序）
    cm_rows = []
    for protocol, yt, lp in (("分层5折_主种子池化", y_lab, LAB_pool[SEED_MAIN]),
                             ("行序外推", y_lab[va_ro], lab_ro)):
        cm = confusion(yt, lp)
        for a in range(3):
            for b in range(3):
                cm_rows.append({"口径": protocol, "真实": LABELS[a], "预测": LABELS[b], "行数": int(cm[a, b])})
        lg.info("混淆矩阵（%s）：对角=%d；严重类漏判为偏高=%d、漏判为合理=%d",
                protocol, int(np.trace(cm)), int(cm[2, 1]), int(cm[2, 0]))
    pd.DataFrame(cm_rows).to_csv(TBL / "q3_混淆矩阵.csv", index=False, encoding="utf-8-sig")

    # ---------------- 落盘
    pd.DataFrame(metric_rows).to_csv(TBL / "q3_指标汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablation_rows).to_csv(TBL / "q3_消融矩阵.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(route_metric_rows + route_cmp_rows).to_csv(TBL / "q3_两路线对比.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q3_指标汇总.csv（%d 行）、q3_消融矩阵.csv（%d 行）、q3_两路线对比.csv（%d 行）、q3_混淆矩阵.csv（%d 行）",
            len(metric_rows), len(ablation_rows), len(route_metric_rows) + len(route_cmp_rows), len(cm_rows))

    cfg_out = {
        "脚本": "code/a6_05_q3分类双路线.py",
        "契约": "00_admin/A5_算法方案.md §4（D17/D18/D19 落地；C11/C13/C14 支撑）",
        "主路线": {"杠杆": "M3-4 类权重 w=(n/(3n_c))^γ + M3-5 后验乘子 s=(1,s_偏,s_严)",
                   "γ网格": GAMMA_GRID, "s_偏网格": S_GRID_PIAN, "s_严网格": S_GRID_YAN,
                   "折级_γs": fold_sstar_log},
        "定稿": {"γ*": float(g_fin), "s_偏*": float(s_fin["s_pian"]), "s_严*": float(s_fin["s_yan"]),
                 "内层宏F1": float(s_fin["macro"]), "可行": bool(s_fin["feasible"])},
        "附件2_预测占比": {LABELS[c]: float((lab2 == c).mean()) for c in range(3)},
        "附件2_两路线一致率": agree2,
        "输入md5": {p.name: md5_of(p) for p in (F_ANNEX1, F_LAB, F_PRED2, F_Q1CFG, F_Q2CFG)},
        "输出文件": ["q3_指标汇总.csv", "q3_消融矩阵.csv", "q3_混淆矩阵.csv", "q3_两路线对比.csv",
                     "q3_方式1误差联合.csv", "附件2_风险预测.csv", "q3_定稿配置.json"],
    }
    (TBL / "q3_定稿配置.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")
    lg.info("输出 q3_定稿配置.json：定稿 γ*=%.1f，s*=(%.2f,%.2f)", g_fin, s_fin["s_pian"], s_fin["s_yan"])
    lg.info("=== a6_05 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
