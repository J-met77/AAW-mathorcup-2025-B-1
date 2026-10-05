# -*- coding: utf-8 -*-
"""
a6_03_q2双口径评估.py —— 问题2：终评 5 折×3 种子 + 行序外推 8934/2233，基线 B1–B4，变体裁决（A5 §3.6/§3.7/§3.5）
- 口径①主：分层 5 折（按 log10 目标十分位）×种子 {20251004,20251005,20251006}；口径②稳健：行序外推
  前 8934 训练/后 2233 验证（U3/D19/D11-②）。两口径结论须定性一致（R-04 检测点）。
- 参评模型：LGBM 变体 A（log10+smearing，φ 折内 4 折 OOF）/ 变体 B（Huber α）、XGBoost 同族对照
  （选中超参等价映射，主种子双口径）、ElasticNetCV、OLS、强制基线 B1–B4（D16）。
- 指标：WAPE（主）/MAE/RMSE/SMAPE_保护；折间均值±std；池化 OOF bootstrap 95% CI（B=1000，约定）。
- 判据（D16/D11-③）：变体* 在双口径 WAPE 均低于 B3 且低于 B4、3 种子方向一致，否则触发 R-05 停步上报。
- 输出：q2_指标汇总.csv、q2_oof预测.csv（变体 A/B 全部种子折级）、q2_定稿配置.json（变体裁决+定稿轮数+φ_final）。
- 日志：output/logs/a6_03_q2双口径评估.log。
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from a6_common import (BOOT_B, F_ANNEX1, FEATURE_COLS, LABELS, LOGD, SEEDS_REPEAT, SEED_MAIN,
                       TBL, all_reg_metrics, assert_features_ok, bootstrap_ci, get_logger,
                       log_env, md5_of, stratified_folds, wape)
from a6_q23_common import (N_ROWORDER_TRAIN, baseline_oof, cv_q2_variant, cv_xgb_variant,
                           decile_strata, load_annex1_sig, make_linear_model, make_tree_X,
                           row_order_split, write_block_report)

T0 = time.time()
LG = get_logger("a6_03", LOGD / "a6_03_q2双口径评估.log")

METRICS = ["WAPE", "MAE", "RMSE"]


def metric_row(model, protocol, seed, y_true, y_pred, fold_wapes=None, n_valid=None):
    r = {"模型": model, "口径": protocol, "种子": seed}
    m = all_reg_metrics(np.asarray(y_true, float), np.asarray(y_pred, float))
    r.update(m)
    for name, fn in (("WAPE", lambda a, b: all_reg_metrics(a, b)["WAPE"]),
                     ("MAE", lambda a, b: all_reg_metrics(a, b)["MAE"]),
                     ("RMSE", lambda a, b: all_reg_metrics(a, b)["RMSE"])):
        lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=BOOT_B, seed=SEED_MAIN)
        r[f"{name}_CI2.5"] = lo
        r[f"{name}_CI97.5"] = hi
    if fold_wapes is not None and len(fold_wapes) > 1:
        r["折间WAPE_均值"] = float(np.mean(fold_wapes))
        r["折间WAPE_std"] = float(np.std(fold_wapes))
    else:
        r["折间WAPE_均值"] = fold_wapes[0] if fold_wapes else np.nan
        r["折间WAPE_std"] = 0.0
    r["验证行数"] = int(len(y_true)) if n_valid is None else int(n_valid)
    return r


def main():
    lg = LG
    lg.info("=== a6_03_q2双口径评估 开始（A5 §3.6/§3.7：终评双口径 + B1–B4 + 变体裁决） ===")
    log_env(lg, [F_ANNEX1])
    df1 = load_annex1_sig(lg)
    assert_features_ok(df1, "附件1", lg)
    sel = json.loads((TBL / "q2_搜索选优.json").read_text(encoding="utf-8"))
    cfgA, cfgB = sel["cfg_star_A"]["cfg"], sel["cfg_star_B"]["cfg"]
    lg.info("读入 a6_02 选优：cfg*_A=%s（3折WAPE=%.6f）；cfg*_B=%s（3折WAPE=%.6f）",
            cfgA, sel["cfg_star_A"]["oof_wape_mean"], cfgB, sel["cfg_star_B"]["oof_wape_mean"])

    y = df1["实际赔付金额"].to_numpy(dtype=float)
    claim = df1["索赔金额"].to_numpy(dtype=float)
    ratio = df1["赔付索赔比"].to_numpy(dtype=float)
    rid = df1["行ID"].to_numpy()
    X, cat_ref, _ = make_tree_X(df1)
    strata = decile_strata(y)
    lg.info("特征矩阵 %s；目标 y 中位=%.1f；赔付索赔比全表中位=%.4f（T4，仅报告）", X.shape, np.median(y), np.median(ratio))

    # ---------------- 口径①：分层 5 折 × 3 种子（LGBM 双变体）
    oof_rows = []          # q2_oof预测.csv 累积（变体 A/B，全部种子）
    cv_store = {}          # (variant) -> {seed: res}
    for variant, cfg in (("A", cfgA), ("B", cfgB)):
        cv_store[variant] = {}
        for seed in SEEDS_REPEAT:
            folds = stratified_folds(strata, 5, seed)
            res = cv_q2_variant(X, y, variant, cfg, folds, seed, strata, do_inner_phi=True)
            cv_store[variant][seed] = {"res": res, "folds": folds}
            pooled = res["oof"]
            lg.info("口径① 变体%s 种子=%d：池化 OOF %s；折间 WAPE 均值=%.6f±%.6f；φ_fold=%s；best_iter=%s",
                    variant, seed, {k: round(v, 6) for k, v in all_reg_metrics(y, pooled).items()},
                    np.mean(res["fold_wape"]), np.std(res["fold_wape"]),
                    np.round(res["phi"], 6).tolist(), res["best_iters"])
            for k, (tr, va) in enumerate(folds, 1):
                for i in va:
                    oof_rows.append({"行ID": int(rid[i]), "y_true": y[i],
                                     "y_pred_oof": pooled[i], "fold": k, "seed": seed,
                                     "变体": f"LGBM_变体{variant}",
                                     "z_pred_oof_log10": res["oof_z"][i] if variant == "A" else np.nan,
                                     "phi_fold": res["phi"][k - 1] if variant == "A" else np.nan})

    # ---------------- 口径②：行序外推（单次切分，主种子）
    ro = {}
    tr_ro, va_ro = row_order_split(len(y))
    lg.info("口径② 行序外推切分：训练=%d 行（行ID %d~%d），验证=%d 行（行ID %d~%d）（U3/D19）",
            len(tr_ro), rid[tr_ro][0], rid[tr_ro][-1], len(va_ro), rid[va_ro][0], rid[va_ro][-1])
    for variant, cfg in (("A", cfgA), ("B", cfgB)):
        res = cv_q2_variant(X, y, variant, cfg, [(tr_ro, va_ro)], SEED_MAIN, strata, do_inner_phi=True)
        ro[variant] = res
        lg.info("口径② 变体%s：验证集 %s；φ=%s；best_iter=%s",
                variant, {k: round(v, 6) for k, v in all_reg_metrics(y[va_ro], res["oof"][va_ro]).items()},
                np.round(res["phi"], 6).tolist(), res["best_iters"])

    # ---------------- 基线 B1–B4 与线性族（同折同种子，D16）
    Xlin = df1[FEATURE_COLS]  # 线性族矩阵：列名由 ColumnTransformer 按名选取；数值 NaN 折内中位插补
    base_store, lin_store = {}, {}
    for seed in SEEDS_REPEAT:
        folds = stratified_folds(strata, 5, seed)
        for kind in ("B1", "B2", "B3", "B4"):
            base_store[(kind, seed)] = baseline_oof(kind, y, claim, ratio, folds)
        for kind in ("elasticnet", "ols"):
            oof_lin = np.full(len(y), np.nan)
            fw = []
            for tr, va in folds:
                mdl = make_linear_model(kind, seed)
                mdl.fit(Xlin.iloc[tr], y[tr])
                oof_lin[va] = np.clip(mdl.predict(Xlin.iloc[va]), 0.01, None)
                fw.append(wape(y[va], oof_lin[va]))
            lin_store[(kind, seed)] = {"oof": oof_lin, "fold_wape": fw}
        lg.info("种子=%d 基线/线性族 OOF 完成（同折同种子）", seed)
    # 行序口径（主种子）
    base_ro, lin_ro = {}, {}
    for kind in ("B1", "B2", "B3", "B4"):
        base_ro[kind] = baseline_oof(kind, y, claim, ratio, [(tr_ro, va_ro)])
    for kind in ("elasticnet", "ols"):
        mdl = make_linear_model(kind, SEED_MAIN)
        mdl.fit(Xlin.iloc[tr_ro], y[tr_ro])
        oof_lin = np.full(len(y), np.nan)
        oof_lin[va_ro] = np.clip(mdl.predict(Xlin.iloc[va_ro]), 0.01, None)
        lin_ro[kind] = oof_lin

    # ---------------- XGBoost 同族对照（选中超参等价映射，主种子双口径；A5 §3.4）
    xgb_store, xgb_ro = {}, {}
    for variant, cfg in (("A", cfgA), ("B", cfgB)):
        folds = stratified_folds(strata, 5, SEED_MAIN)
        res = cv_xgb_variant(X, y, variant, cfg, folds, SEED_MAIN, strata)
        xgb_store[variant] = {"res": res, "folds": folds}
        lg.info("XGB 对照 变体%s（口径①主种子）：池化 OOF WAPE=%.6f；折间=%.6f±%.6f",
                variant, wape(y, res["oof"]), np.mean(res["fold_wape"]), np.std(res["fold_wape"]))
        res_ro = cv_xgb_variant(X, y, variant, cfg, [(tr_ro, va_ro)], SEED_MAIN, strata)
        xgb_ro[variant] = res_ro
        lg.info("XGB 对照 变体%s（口径②行序）：验证 WAPE=%.6f", variant, wape(y[va_ro], res_ro["oof"][va_ro]))

    # ---------------- 指标汇总表
    rows = []
    for variant in ("A", "B"):
        for seed in SEEDS_REPEAT:
            res = cv_store[variant][seed]["res"]
            rows.append(metric_row(f"LGBM_变体{variant}", "分层5折", seed, y, res["oof"],
                                   res["fold_wape"]))
    for seed in SEEDS_REPEAT:
        for kind, name in (("B1", "B1_中位数"), ("B2", "B2_均值"), ("B3", "B3_单参数比"), ("B4", "B4_分段比")):
            oofb = base_store[(kind, seed)]
            fwb = [wape(y[va], oofb[va]) for _, va in stratified_folds(strata, 5, seed)]
            rows.append(metric_row(name, "分层5折", seed, y, oofb, fwb))
        for kind, name in (("elasticnet", "ElasticNet"), ("ols", "OLS")):
            st = lin_store[(kind, seed)]
            rows.append(metric_row(name, "分层5折", seed, y, st["oof"], st["fold_wape"]))
    for variant in ("A", "B"):
        res = xgb_store[variant]["res"]
        rows.append(metric_row(f"XGB_变体{variant}对照", "分层5折", SEED_MAIN, y, res["oof"],
                               res["fold_wape"]))
    # 口径②行
    for variant in ("A", "B"):
        rows.append(metric_row(f"LGBM_变体{variant}", "行序外推", SEED_MAIN, y[va_ro], ro[variant]["oof"][va_ro],
                               [ro[variant]["fold_wape"][0]], n_valid=len(va_ro)))
    for kind, name in (("B1", "B1_中位数"), ("B2", "B2_均值"), ("B3", "B3_单参数比"), ("B4", "B4_分段比")):
        rows.append(metric_row(name, "行序外推", SEED_MAIN, y[va_ro], base_ro[kind][va_ro],
                               n_valid=len(va_ro)))
    for kind, name in (("elasticnet", "ElasticNet"), ("ols", "OLS")):
        rows.append(metric_row(name, "行序外推", SEED_MAIN, y[va_ro], lin_ro[kind][va_ro], n_valid=len(va_ro)))
    for variant in ("A", "B"):
        res = xgb_ro[variant]
        rows.append(metric_row(f"XGB_变体{variant}对照", "行序外推", SEED_MAIN, y[va_ro], res["oof"][va_ro],
                               n_valid=len(va_ro)))
    summ = pd.DataFrame(rows)
    summ.to_csv(TBL / "q2_指标汇总.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q2_指标汇总.csv：%d 行（模型×口径×种子，含 bootstrap %.0f%% CI，B=%d）",
            len(summ), 100 * 0.95, BOOT_B)

    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(TBL / "q2_oof预测.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q2_oof预测.csv：%d 行（变体 A/B × 3 种子 × 5 折，池化每行恰一预测/种子）", len(oof_df))

    # ---------------- 变体裁决（A5 §3.3：WAPE 主、MAE 辅）与 D16 判据
    main_w = {v: float(np.mean([all_reg_metrics(y, cv_store[v][s]["res"]["oof"])["WAPE"] for s in SEEDS_REPEAT]))
              for v in ("A", "B")}
    main_m = {v: float(np.mean([all_reg_metrics(y, cv_store[v][s]["res"]["oof"])["MAE"] for s in SEEDS_REPEAT]))
              for v in ("A", "B")}
    ro_w = {v: float(all_reg_metrics(y[va_ro], ro[v]["oof"][va_ro])["WAPE"]) for v in ("A", "B")}
    variant_star = "A" if (main_w["A"], main_m["A"]) <= (main_w["B"], main_m["B"]) else "B"
    lg.info("变体裁决（3 种子主口径 WAPE 均值，MAE 辅）：A=%.6f/MAE=%.4f，B=%.6f/MAE=%.4f → 变体*=%s",
            main_w["A"], main_m["A"], main_w["B"], main_m["B"], variant_star)
    lg.info("行序口径对照：A=%.6f，B=%.6f → 排序%s（R-04 检测：两口径变体排序须一致）",
            ro_w["A"], ro_w["B"], "一致" if (ro_w["A"] < ro_w["B"]) == (main_w["A"] < main_w["B"]) else "反转")

    variant_star_w = {s: all_reg_metrics(y, cv_store[variant_star][s]["res"]["oof"])["WAPE"] for s in SEEDS_REPEAT}
    b3_w = {s: all_reg_metrics(y, base_store[("B3", s)] )["WAPE"] for s in SEEDS_REPEAT}
    b4_w = {s: all_reg_metrics(y, base_store[("B4", s)])["WAPE"] for s in SEEDS_REPEAT}
    b3_ro = all_reg_metrics(y[va_ro], base_ro["B3"][va_ro])["WAPE"]
    b4_ro = all_reg_metrics(y[va_ro], base_ro["B4"][va_ro])["WAPE"]
    vs_ro = all_reg_metrics(y[va_ro], ro[variant_star]["oof"][va_ro])["WAPE"]
    cond_main = all(variant_star_w[s] < b3_w[s] and variant_star_w[s] < b4_w[s] for s in SEEDS_REPEAT)
    cond_ro = vs_ro < b3_ro and vs_ro < b4_ro
    crit = {"主口径_逐种子": bool(cond_main),
            "主口径_均值": bool(np.mean(list(variant_star_w.values())) < np.mean(list(b3_w.values()))
                          and np.mean(list(variant_star_w.values())) < np.mean(list(b4_w.values()))),
            "行序口径": bool(cond_ro)}
    crit["综合"] = bool(crit["主口径_逐种子"] and crit["主口径_均值"] and crit["行序口径"])
    lg.info("D16 判据“稳定优于 B3/B4”：主口径 3 种子 WAPE 变体*=%s vs B3=%s vs B4=%s",
            {s: round(v, 6) for s, v in variant_star_w.items()},
            {s: round(v, 6) for s, v in b3_w.items()}, {s: round(v, 6) for s, v in b4_w.items()})
    lg.info("D16 判据（行序口径）：变体*=%.6f vs B3=%.6f vs B4=%.6f；逐项=%s → 综合=%s",
            vs_ro, b3_ro, b4_ro, crit, "成立" if crit["综合"] else "不成立")
    if not crit["综合"]:
        msg = ("R-05（A2-17）触发：最优 Q2 模型不稳定劣于 B3/B4（A5 §3.5 判据不成立）。\n"
               f"证据：主口径 3 种子 WAPE 变体*={variant_star_w} vs B3={b3_w} vs B4={b4_w}；"
               f"行序口径 变体*={vs_ro:.6f} vs B3={b3_ro:.6f} vs B4={b4_ro:.6f}。\n"
               "候选预案：触发强化聚合特征回退预案，上报主会话协同，不静默换法。\n")
        write_block_report(lg, "R-05 模型未稳定优于 B3/B4（a6_03）", msg)
    if (ro_w["A"] < ro_w["B"]) != (main_w["A"] < main_w["B"]):
        msg = ("R-04（A2-14/U3+D11-②）触发：Q2 双口径变体排序反转。\n"
               f"证据：主口径 WAPE 均值 A={main_w['A']:.6f} B={main_w['B']:.6f}；行序口径 A={ro_w['A']:.6f} B={ro_w['B']:.6f}。\n"
               "候选预案：停步上报，两口径分别报告，不得单口径下结论。\n")
        write_block_report(lg, "R-04 Q2 双口径变体排序反转（a6_03）", msg)

    # ---------------- 定稿配置（a6_04 全量重训输入）
    cfg_star = cfgA if variant_star == "A" else cfgB
    folds_main = stratified_folds(strata, 5, SEED_MAIN)
    bi_main = cv_store[variant_star][SEED_MAIN]["res"]["best_iters"]
    n_final = min(2000, int(round(np.mean(bi_main) * 1.1)))
    lg.info("定稿轮数：主种子 5 折 best_iter=%s → 均值=%.1f ×1.1 → %d（上限 2000，A5 §3.4）",
            bi_main, np.mean(bi_main), n_final)
    if variant_star == "A":
        oof_df_vs = oof_df[(oof_df["变体"] == "LGBM_变体A") & (oof_df["seed"] == SEED_MAIN)]
        oof_df_vs = oof_df_vs.sort_values("行ID", kind="mergesort")
        assert (oof_df_vs["行ID"].to_numpy() == rid).all(), "OOF 表行ID 与附件1 行序未对齐"
        zhat = oof_df_vs["z_pred_oof_log10"].to_numpy(float)
        phi_final = float(np.mean(10.0 ** (np.log10(y) - zhat)))
        lg.info("φ_final（变体 A，主种子 5 折池化 OOF 残差按行ID 对齐，A5 §3.3）=%.6f（折级 φ 参考=%s）",
                phi_final, np.round(cv_store["A"][SEED_MAIN]["res"]["phi"], 6).tolist())
    else:
        phi_final = 1.0
        lg.info("变体 B 无 smearing：φ_final=1.0")
    cfg_json = {
        "脚本": "code/a6_03_q2双口径评估.py",
        "契约": "00_admin/A5_算法方案.md §3.3/§3.5/§3.6/§3.7（D15/D16/D19/D21 落地）",
        "变体裁决": {"变体*": variant_star,
                     "主口径_WAPE均值": main_w, "主口径_MAE均值": main_m, "行序_WAPE": ro_w,
                     "XGB对照_主口径WAPE": {v: wape(y, xgb_store[v]["res"]["oof"]) for v in ("A", "B")}},
        "cfg_star": cfg_star,
        "best_iter": {"主种子5折": bi_main, "均值": float(np.mean(bi_main)), "定稿轮数": n_final},
        "phi_final": phi_final,
        "D16_判据": crit,
        "输入md5": {"附件1_clean.csv": md5_of(F_ANNEX1),
                    "q2_搜索选优.json": md5_of(TBL / "q2_搜索选优.json")},
        "输出文件": ["q2_指标汇总.csv", "q2_oof预测.csv", "q2_定稿配置.json"],
    }
    (TBL / "q2_定稿配置.json").write_text(json.dumps(cfg_json, ensure_ascii=False, indent=2), encoding="utf-8")
    lg.info("输出 q2_定稿配置.json：变体*=%s，cfg*=%s，定稿轮数=%d，φ_final=%.6f",
            variant_star, cfg_star, n_final, phi_final)
    lg.info("=== a6_03 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
