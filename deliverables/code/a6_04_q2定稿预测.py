# -*- coding: utf-8 -*-
"""
a6_04_q2定稿预测.py —— 问题2：定稿配置全量重训 → 附件2 2792 行逐单预测实际赔付金额（A5 §3.6 最终交付）
- 读 q2_定稿配置.json（a6_03 变体裁决：变体*=A，cfg*，定稿轮数=主种子 5 折 best_iter 均值×1.1，φ_final）。
- 全量 n1=11167 重训 LGBM（不设内部留出，A5 §3.4）；变体 A：附件2 预测 ŷ=10^ẑ×φ_final。
- 约束落地：预测 clip≥0.01 并四舍五入两位小数（金额语义）；禁止任何 min(·, 保价) 截断（T9）。
- 输出：output/tables/附件2_赔付预测.csv（运单号, 索赔金额, ŷ；2792 行）；更新 q2_定稿配置.json 执行字段。
- 回退触发器：R-08（附件2 索赔金额缺失——载入器已断言）、R-09（LGBM 报错停步上报）。
- 日志：output/logs/a6_04_q2定稿预测.log。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (F_ANNEX1, F_ANNEX2, LOGD, SEED_MAIN, TBL, assert_features_ok,
                       get_logger, log_env, md5_of)
from a6_q23_common import (EPS_CLIP, capture_cat_ref, fit_lgbm_reg, load_annex1_sig,
                           load_annex2_sig, make_tree_X, write_block_report)

T0 = time.time()
LG = get_logger("a6_04", LOGD / "a6_04_q2定稿预测.log")


def main():
    lg = LG
    lg.info("=== a6_04_q2定稿预测 开始（A5 §3.6 最终交付：全量重训 + 附件2 预测） ===")
    log_env(lg, [F_ANNEX1, F_ANNEX2])
    cfgj = json.loads((TBL / "q2_定稿配置.json").read_text(encoding="utf-8"))
    variant_star = cfgj["变体裁决"]["变体*"]
    cfg_star = dict(cfgj["cfg_star"])
    n_round = int(cfgj["best_iter"]["定稿轮数"])
    phi_final = float(cfgj["phi_final"])
    cfg_star["objective"] = "regression" if variant_star == "A" else "huber"
    lg.info("读入定稿配置：变体*=%s，cfg*=%s，定稿轮数=%d，φ_final=%.6f",
            variant_star, cfg_star, n_round, phi_final)

    df1 = load_annex1_sig(lg)
    assert_features_ok(df1, "附件1", lg)
    df2 = load_annex2_sig(lg)
    assert_features_ok(df2, "附件2", lg)

    y = df1["实际赔付金额"].to_numpy(dtype=float)
    X1, cat_ref, _ = make_tree_X(df1)
    X2, _, unseen = make_tree_X(df2, cat_ref=cat_ref)
    lg.info("特征矩阵：附件1 %s，附件2 %s（类别域对齐附件1，未见类别计数=%s，预期全 0）",
            X1.shape, X2.shape, {k: v for k, v in unseen.items() if v != 0} or "无")
    assert all(v == 0 for v in unseen.values()), "附件2 出现附件1 未见类别值，停止并上报"

    lg.info("全量重训（n1=%d，无内部留出，num_boost_round=%d，种子=%d，deterministic=True）",
            len(y), n_round, SEED_MAIN)
    try:
        if variant_star == "A":
            z = np.log10(y)
            mdl = fit_lgbm_reg(X1, z, cfg_star, SEED_MAIN, valid=None, num_boost_round=n_round)
        else:
            mdl = fit_lgbm_reg(X1, y, cfg_star, SEED_MAIN, valid=None, num_boost_round=n_round)
    except Exception as ex:  # R-09
        write_block_report(lg, "R-09 LGBM 全量重训报错（a6_04）",
                           f"R-09（环境）触发：全量重训报错。\n异常={ex!r}\n候选预案：降级 sklearn HistGradientBoosting 等价族并上报，不静默换法。\n")
        raise
    lg.info("全量重训完成（训练目标=%s）", "z=log10(y)" if variant_star == "A" else "y（Huber）")

    if variant_star == "A":
        zhat2 = mdl.predict(X2)
        yhat2 = 10.0 ** zhat2 * phi_final
    else:
        yhat2 = mdl.predict(X2)
    yhat2 = np.clip(yhat2, EPS_CLIP, None)
    yhat2_round = np.round(yhat2, 2)

    out = pd.DataFrame({"运单号": df2["运单号"].to_numpy(),
                        "索赔金额": df2["索赔金额"].to_numpy(dtype=float),
                        "ŷ": yhat2_round})
    assert len(out) == 2792 and out["运单号"].is_unique and out["ŷ"].notna().all()
    assert (out["ŷ"] >= EPS_CLIP).all(), "预测金额存在低于 0.01 的值（clip 失效？）"
    out.to_csv(TBL / "附件2_赔付预测.csv", index=False, encoding="utf-8-sig")

    lg.info("附件2 预测统计：n=%d，min=%.2f，P25=%.2f，中位=%.2f，均值=%.2f，P75=%.2f，P95=%.2f，max=%.2f",
            len(yhat2_round), yhat2_round.min(), np.percentile(yhat2_round, 25),
            np.median(yhat2_round), yhat2_round.mean(), np.percentile(yhat2_round, 75),
            np.percentile(yhat2_round, 95), yhat2_round.max())
    lg.info("对照（附件1 实际赔付 y，T1）：中位=%.2f，均值=%.2f，P95=%.2f，max=%.2f —— 预测分布与训练目标同量级（ sanity 报告，不设定阈值）",
            np.median(y), y.mean(), np.percentile(y, 95), y.max())
    lg.info("禁止项核验：未做 min(·, 保价) 截断（T9）；保价金额仅作特征（非后处理约束）")
    lg.info("输出 附件2_赔付预测.csv：%d 行（运单号, 索赔金额, ŷ 两位小数）；md5 待 a6_06 复核",
            len(out))

    cfgj["执行"] = {
        "脚本": "code/a6_04_q2定稿预测.py",
        "全量重训": {"n": int(len(y)), "num_boost_round": n_round, "seed": SEED_MAIN,
                     "objective": cfg_star["objective"]},
        "phi_final_用于反变换": phi_final if variant_star == "A" else 1.0,
        "预测统计": {"min": float(yhat2_round.min()), "P25": float(np.percentile(yhat2_round, 25)),
                     "中位": float(np.median(yhat2_round)), "均值": float(yhat2_round.mean()),
                     "P75": float(np.percentile(yhat2_round, 75)),
                     "P95": float(np.percentile(yhat2_round, 95)), "max": float(yhat2_round.max())},
        "输入md5": {"附件1_clean.csv": md5_of(F_ANNEX1), "附件2_clean.csv": md5_of(F_ANNEX2)},
        "输出文件": ["附件2_赔付预测.csv", "q2_定稿配置.json（本文件，追加执行字段）"],
    }
    (TBL / "q2_定稿配置.json").write_text(json.dumps(cfgj, ensure_ascii=False, indent=2), encoding="utf-8")
    lg.info("q2_定稿配置.json 已追加执行字段（重训轮数/φ/预测统计）")
    lg.info("=== a6_04 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
