# -*- coding: utf-8 -*-
"""
a6_02_q2搜索.py —— 问题2：LGBM 双变体随机搜索（A5 §3.3/§3.4/§3.7 选优段落地）
- 双变体：A = log10+smearing（目标 z=log10(实际赔付金额)，φ 折内 4 折 OOF 估计）；B = 原尺度 Huber（α∈{0.5,0.9} 入搜索）。
- 搜索空间（A5 §3.4，协议约定）：num_leaves∈{15,31,63}，min_child_samples∈{20,50}，feature_fraction∈{0.7,0.9}，
  bagging_fraction∈{0.7,0.9}+bagging_freq=1，reg_lambda∈{0.1,1.0}，reg_alpha∈{0,0.1}，lr=0.05 固定，
  n_estimators 上限 2000，早停 100（度量 l1，约定）。
- 搜索方式：随机搜索 24 组/变体（种子 20251004，产品空间内无放回抽样）× 3 折（种子 20251004，
  按 log10 目标十分位分层），以 OOF WAPE（折间均值）选优；选优折数 3 与终评 5 分离（防选择偏置混入终评）。
- 输出：output/tables/q2_搜索日志.csv（48 行）、output/tables/q2_搜索选优.json（cfg*_A/cfg*_B）。
- 日志：output/logs/a6_02_q2搜索.log。回退触发器：R-09（LGBM 报错/不可复现）。
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (F_ANNEX1, F_ANNEX2, LOGD, SEED_MAIN, TBL, get_logger,
                       log_env, md5_of, stratified_folds)
from a6_q23_common import (capture_cat_ref, cv_q2_variant, decile_strata,
                           load_annex1_sig, load_annex2_sig, make_tree_X)

T0 = time.time()
LG = get_logger("a6_02", LOGD / "a6_02_q2搜索.log")
N_CFG = 24          # A5 §3.4 约定；超时降档 24→12（A5 §6，触发时记日志）
N_FOLD_SEL = 3      # 选优折数（与终评 5 折分离）

SPACE_BASE = dict(
    num_leaves=[15, 31, 63],
    min_child_samples=[20, 50],
    feature_fraction=[0.7, 0.9],
    bagging_fraction=[0.7, 0.9],
    reg_lambda=[0.1, 1.0],
    reg_alpha=[0.0, 0.1],
)
SPACE_B_EXTRA = dict(alpha=[0.5, 0.9])  # 变体 B 的 Huber α 入搜索（A5 §3.3）


def sample_configs(rng, space, n):
    keys = list(space.keys())
    grid = list(product(*[space[k] for k in keys]))
    pick = rng.choice(len(grid), size=n, replace=False)
    return [dict(zip(keys, grid[i])) for i in sorted(pick)]


def main():
    lg = LG
    lg.info("=== a6_02_q2搜索 开始（A5 §3.4：双变体随机搜索 24 组×3 折） ===")
    log_env(lg, [F_ANNEX1, F_ANNEX2])
    df1 = load_annex1_sig(lg)
    df2 = load_annex2_sig(lg)  # 载入即断言（D21/D22），a6_02 不改动附件2
    from a6_common import assert_features_ok
    assert_features_ok(df1, "附件1", lg)

    y = df1["实际赔付金额"].to_numpy(dtype=float)
    X, cat_ref, _unseen = make_tree_X(df1)
    strata = decile_strata(y)
    lg.info("特征矩阵：%d 行 × %d 列（允许清单 %d 列，其中类别 %d 列）；log10 目标十分位分层完成",
            X.shape[0], X.shape[1], len(X.columns), sum(c in cat_ref for c in X.columns))
    lg.info("目标：实际赔付金额 y（中位=%.1f，P5=%.2f，max=%.2f，T1）；变体 A 目标 z=log10(y)∈[%.3f, %.3f]",
            np.median(y), np.percentile(y, 5), y.max(), np.log10(y).min(), np.log10(y).max())

    folds = stratified_folds(strata, N_FOLD_SEL, SEED_MAIN)
    lg.info("选优折：%d 折分层（按 log10(y) 十分位），种子=%d；各折验证规模=%s",
            N_FOLD_SEL, SEED_MAIN, [len(va) for _, va in folds])

    rng = np.random.default_rng(SEED_MAIN)
    space_A = dict(SPACE_BASE)
    space_B = dict(SPACE_BASE, **SPACE_B_EXTRA)
    cfgs_A = sample_configs(rng, space_A, N_CFG)
    cfgs_B = sample_configs(rng, space_B, N_CFG)
    lg.info("随机搜索配置抽样（无放回）：变体 A 空间 %d 组合抽 %d 组；变体 B 空间 %d 组合抽 %d 组（种子=%d）",
            len(list(product(*space_A.values()))), len(cfgs_A),
            len(list(product(*space_B.values()))), len(cfgs_B), SEED_MAIN)

    rows = []
    best = {}
    for variant, cfg_list in (("A", cfgs_A), ("B", cfgs_B)):
        lg.info("---- 变体 %s 搜索开始（%d 组 × %d 折；A 含折内 4 折 φ） ----", variant, len(cfg_list), N_FOLD_SEL)
        for ci, cfg in enumerate(cfg_list, 1):
            try:
                res = cv_q2_variant(X, y, variant, cfg, folds, SEED_MAIN, strata, do_inner_phi=True)
            except Exception as ex:  # R-09 前置检测：LGBM 报错即停步上报
                msg = (f"R-09（环境）触发：a6_02 变体 {variant} 第 {ci} 组配置 LGBM 拟合报错。\n"
                       f"配置={cfg}\n异常={ex!r}\n候选预案：降级 sklearn HistGradientBoosting 等价族并上报，不静默换法。\n")
                from a6_q23_common import write_block_report
                write_block_report(lg, "R-09 LGBM 拟合报错（a6_02）", msg)
                raise
            fw = res["fold_wape"]
            fm = res["fold_mae"]
            row = {"变体": variant, "组号": ci, **cfg,
                   "折1_WAPE": fw[0], "折2_WAPE": fw[1], "折3_WAPE": fw[2],
                   "OOF_WAPE_均值": float(np.mean(fw)), "OOF_WAPE_std": float(np.std(fw)),
                   "OOF_MAE_均值": float(np.mean(fm)),
                   "best_iter_均值": float(np.mean(res["best_iters"])),
                   "φ_折值": "|".join(f"{v:.6f}" for v in res["phi"])}
            rows.append(row)
            lg.info("变体%s 组%02d/%02d %s → OOF WAPE 均值=%.6f（折值 %s），MAE=%.4f，best_iter均=%.0f",
                    variant, ci, len(cfg_list),
                    {k: v for k, v in cfg.items()}, row["OOF_WAPE_均值"],
                    np.round(fw, 6).tolist(), row["OOF_MAE_均值"], row["best_iter_均值"])
        sub = pd.DataFrame([r for r in rows if r["变体"] == variant])
        bidx = sub.sort_values(["OOF_WAPE_均值", "OOF_MAE_均值"], kind="mergesort").index[0]
        brow = sub.loc[bidx]
        cfg_star = {k: brow[k] for k in (SPACE_BASE if variant == "A" else {**SPACE_BASE, "alpha": SPACE_B_EXTRA["alpha"]})}
        cfg_star = {k: (int(v) if k in ("num_leaves", "min_child_samples") else float(v)) for k, v in cfg_star.items()}
        best[variant] = {"cfg": cfg_star,
                         "oof_wape_mean": float(brow["OOF_WAPE_均值"]),
                         "oof_mae_mean": float(brow["OOF_MAE_均值"]),
                         "best_iter_mean": float(brow["best_iter_均值"])}
        lg.info("变体 %s 选优完成：cfg*=%s，OOF WAPE 均值=%.6f（3 折，种子=%d）",
                variant, cfg_star, best[variant]["oof_wape_mean"], SEED_MAIN)

    grid_out = pd.DataFrame(rows)
    grid_out.to_csv(TBL / "q2_搜索日志.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q2_搜索日志.csv：%d 行（变体×24 组×3 折指标）", len(grid_out))

    sel = {
        "脚本": "code/a6_02_q2搜索.py",
        "契约": "00_admin/A5_算法方案.md §3.3/§3.4/§3.7（D15/D16 落地）",
        "种子": {"主种子": SEED_MAIN, "搜索抽样": SEED_MAIN, "选优折": SEED_MAIN},
        "搜索设定": {"每组折数": N_FOLD_SEL, "组数": N_CFG,
                     "空间": {"A": space_A, "B": space_B},
                     "早停": "100 轮无改善，度量 l1（约定）"},
        "cfg_star_A": best["A"], "cfg_star_B": best["B"],
        "说明": "选优指标=3 折 OOF WAPE 折间均值（MAE 并列辅）；变体裁决在 a6_03 终评（5 折×3 种子）进行，本脚本不裁决变体",
        "输入md5": {"附件1_clean.csv": md5_of(F_ANNEX1), "附件2_clean.csv": md5_of(F_ANNEX2)},
        "输出文件": ["q2_搜索日志.csv", "q2_搜索选优.json"],
    }
    (TBL / "q2_搜索选优.json").write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    lg.info("输出 q2_搜索选优.json：cfg*_A=%s（WAPE=%.6f）；cfg*_B=%s（WAPE=%.6f）",
            best["A"]["cfg"], best["A"]["oof_wape_mean"], best["B"]["cfg"], best["B"]["oof_wape_mean"])
    lg.info("=== a6_02 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
