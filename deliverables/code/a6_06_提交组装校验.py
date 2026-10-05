# -*- coding: utf-8 -*-
"""
a6_06_提交组装校验.py —— 组装 output/Result_提交.xlsx 并做 F3/F4/F6 程序化自检（A5 §6/§8.2）
- 唯一允许写 output/Result_提交.xlsx 的脚本；模板 data/Result.xlsx 只读，不改动运单号（F3/题面）。
- 数据源：附件2_赔付预测.csv（a6_04，实际赔付金额）+ 附件2_风险预测.csv（a6_05 方式2，风险标注，D17）。
- 自检清单（task_board A6 门禁 + F2/F3/F4/F6）：
  ①行数=2792 且与 附件2_clean.csv 逐行同序；②运单号与模板/附件2_clean 逐行一致零改动；
  ③两列（实际赔付金额/风险标注）无空值；④金额非负（且符合 clip≥0.01、两位小数口径）；
  ⑤标签取值合法（三分类域）；⑥列名与列序与模板完全一致。
- 幂等复跑比对：写盘后回读逐格比对（2792×3），日志留数字一致证据。
- 日志：output/logs/a6_06_提交组装校验.log。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (LABELS, LOGD, ROOT, TBL, F_RESULT_TPL, get_logger, log_env, md5_of)
from a6_q23_common import load_annex2_sig

T0 = time.time()
LG = get_logger("a6_06", LOGD / "a6_06_提交组装校验.log")
F_PAY = TBL / "附件2_赔付预测.csv"
F_RISK = TBL / "附件2_风险预测.csv"
F_OUT = ROOT / "output" / "Result_提交.xlsx"


def selfcheck(res: pd.DataFrame, tpl: pd.DataFrame, df2: pd.DataFrame, lg) -> list[str]:
    errs = []
    # ① 行数
    if len(res) != 2792 or len(tpl) != 2792 or len(df2) != 2792:
        errs.append(f"①行数不符：Result={len(res)} 模板={len(tpl)} 附件2_clean={len(df2)}（应均为 2792）")
    # ② 运单号逐行一致零改动（模板序 == 附件2_clean 序 == Result 序）
    t_ids = tpl["运单号"].to_numpy()
    a_ids = df2["运单号"].to_numpy()
    r_ids = res["运单号"].to_numpy()
    if not np.array_equal(t_ids, a_ids):
        n_diff = int((t_ids != a_ids).sum())
        errs.append(f"②a 模板运单号与附件2_clean 逐行不一致：{n_diff} 行")
    if not np.array_equal(t_ids, r_ids):
        n_diff = int((t_ids != r_ids).sum())
        errs.append(f"②b Result 运单号与模板逐行不一致（改动 {n_diff} 行）——违反 F3/题面")
    # ③ 两列无空值
    for col in ("实际赔付金额", "风险标注"):
        n_na = int(res[col].isna().sum())
        if n_na:
            errs.append(f"③ {col} 存在 {n_na} 个空值")
    # ④ 金额非负 + 口径
    amt = res["实际赔付金额"].to_numpy(dtype=float)
    if np.isnan(amt).any():
        errs.append("④a 金额列含 NaN")
    else:
        if (amt < 0).any():
            errs.append(f"④b 金额存在负值：min={amt.min():.2f}")
        if (amt < 0.01).any():
            errs.append(f"④c 金额低于 clip 下限 0.01：{int((amt < 0.01).sum())} 行")
        bad_round = int((np.abs(amt - np.round(amt, 2)) > 1e-9).sum())
        if bad_round:
            errs.append(f"④d 金额非两位小数：{bad_round} 行")
    # ⑤ 标签取值合法
    lab = res["风险标注"]
    bad_lab = set(lab.dropna().unique()) - set(LABELS)
    if bad_lab:
        errs.append(f"⑤ 风险标注存在非法取值：{bad_lab}")
    # ⑥ 列名与列序
    if list(res.columns) != list(tpl.columns):
        errs.append(f"⑥ 列名/列序与模板不符：{list(res.columns)} vs {list(tpl.columns)}")
    return errs


def main():
    lg = LG
    lg.info("=== a6_06_提交组装校验 开始（组装 output/Result_提交.xlsx + F2/F3/F4/F6 自检 + 幂等复跑比对） ===")
    log_env(lg, [TBL / "附件1_clean.csv", TBL / "附件2_clean.csv", F_PAY, F_RISK, F_RESULT_TPL])
    tpl = pd.read_excel(F_RESULT_TPL)
    lg.info("模板（只读）：%s，列=%s；运单号范围 %s~%s",
            tpl.shape, list(tpl.columns), tpl["运单号"].iloc[0], tpl["运单号"].iloc[-1])
    df2 = load_annex2_sig(lg)
    pay = pd.read_csv(F_PAY, encoding="utf-8-sig")
    risk = pd.read_csv(F_RISK, encoding="utf-8-sig")
    for tag, d in (("附件2_赔付预测", pay), ("附件2_风险预测", risk)):
        assert len(d) == 2792 and d["运单号"].is_unique, f"{tag} 行数/唯一性断言失败"
        assert (d["运单号"].to_numpy() == df2["运单号"].to_numpy()).all(), f"{tag} 运单号与附件2_clean 不同序"
    lg.info("数据源断言通过：两份预测表 2792 行且与附件2_clean 同序（运单号 %s~%s）",
            df2["运单号"].iloc[0], df2["运单号"].iloc[-1])

    pay_map = dict(zip(pay["运单号"].to_numpy(), pay["ŷ"].to_numpy(dtype=float)))
    risk_map = dict(zip(risk["运单号"].to_numpy(), risk["风险标注"].to_numpy()))
    res = pd.DataFrame({
        "运单号": tpl["运单号"].to_numpy(),
        "实际赔付金额": [pay_map[v] for v in tpl["运单号"].to_numpy()],
        "风险标注": [risk_map[v] for v in tpl["运单号"].to_numpy()],
    })
    errs = selfcheck(res, tpl, df2, lg)
    if errs:
        for e in errs:
            lg.error("自检失败：%s", e)
        raise SystemExit("Result 自检失败，禁止写出（详见日志）")
    lg.info("程序化自检全部通过：①2792 行×3 列；②运单号与模板/附件2_clean 逐行一致零改动；"
            "③两列无空值；④金额非负 min=%.2f 且两位小数；⑤标签取值域=%s；⑥列名列序与模板一致",
            float(res["实际赔付金额"].min()), sorted(set(res["风险标注"])))
    lab_counts = res["风险标注"].value_counts()
    lg.info("Result 最终占比（如实报告，A2-06）：%s；金额：min=%.2f 中位=%.2f 均值=%.2f max=%.2f",
            {k: int(v) for k, v in lab_counts.items()},
            res["实际赔付金额"].min(), res["实际赔付金额"].median(),
            res["实际赔付金额"].mean(), res["实际赔付金额"].max())
    lg.info("金额占比细节：合理诉求 %.4f / 诉求偏高 %.4f / 严重超额 %.4f",
            lab_counts.get("合理诉求", 0) / 2792, lab_counts.get("诉求偏高", 0) / 2792,
            lab_counts.get("严重超额", 0) / 2792)

    res.to_excel(F_OUT, index=False, engine="openpyxl")
    lg.info("写出 output/Result_提交.xlsx：%s（md5=%s；注：xlsx 内嵌时间戳，跨次运行 md5 可能不同，以逐格比对为准）",
            F_OUT.name, md5_of(F_OUT))

    # ---------------- 幂等复跑比对：回读逐格比对（2792×3）
    back = pd.read_excel(F_OUT)
    ok_ids = np.array_equal(back["运单号"].to_numpy(), res["运单号"].to_numpy())
    ok_amt = np.array_equal(back["实际赔付金额"].to_numpy(dtype=float),
                            res["实际赔付金额"].to_numpy(dtype=float))
    ok_lab = (back["风险标注"].astype(str).to_numpy() == res["风险标注"].astype(str).to_numpy()).all()
    n_cells = back.shape[0] * back.shape[1]
    if ok_ids and ok_amt and ok_lab and back.shape == res.shape:
        lg.info("幂等复跑比对：一致（回读 %d 行×%d 列=%d 格全部逐格相同；运单号零改动、金额/标签逐格相同）",
                back.shape[0], back.shape[1], n_cells)
    else:
        lg.error("幂等复跑比对失败：运单号=%s 金额=%s 标签=%s 形状=%s vs %s",
                 ok_ids, ok_amt, ok_lab, back.shape, res.shape)
        raise SystemExit("回读比对失败")
    lg.info("=== a6_06 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
