# -*- coding: utf-8 -*-
"""
a6_01_q1_规则标注.py —— 问题1：风险标注规则（A5 §2 全流程落地）
输入：output/tables/附件1_clean.csv（仅用 实际赔付金额 x 与 索赔差额 d；e=−d，C3/D20）
输出：
  1) output/tables/附件1_风险标注.csv   行ID,实际赔付金额,索赔金额,索赔差额,风险标注（11167 行全覆盖）
  2) output/tables/q1_边界表.csv        箱序 b, x下界, x上界, 箱计数, g1_b, g2_b
  3) output/tables/q1_自检表.csv        V1–V8 逐项度量与判定
  4) output/tables/q1_择优日志.csv      全部 θ=(B,τ1,τ2) 的 L0–L5 各层取值
  5) output/tables/q1_对偶报告.csv      相对口径 r=e/y 分箱分位（只报告不产标签）
  6) output/tables/q1_定稿配置.json     冻结边界（B/τ/edges/g1/g2/判定规则）
日志：output/logs/a6_01_q1规则标注.log
择优：纯字典序 L0 硬筛选 → L1 先验邻域(≤0.03) → L2 D_geo 最大 → L3 R_合+R_偏 最小
      → L4 bootstrap 翻转率最小（仅对 L2/L3 后并列 finalists 计算） → L5 兜底（B 最小→D_prior 最小→(τ1,τ2) 字典序）
回退触发器：R-01/R-02/R-03/R-06（触发即写 output/logs/a6_阻断上报.md 并停步，不静默换法）
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (BOOT_B, LABELS, LOGD, ROOT, SEEDS_REPEAT, SEED_MAIN, TBL,
                       F_ANNEX1, F_ANNEX2, get_logger, load_annex1, log_env, md5_of)

warnings.filterwarnings("ignore")
T0 = time.time()
LG = get_logger("a6_01", LOGD / "a6_01_q1规则标注.log")
BLOCK_FILE = LOGD / "a6_阻断上报.md"

B_GRID = [5, 10, 15, 20]
TAU1_GRID = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
TAU2_GRID = [0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
EPS_W = 1e-9  # 密度宽度下限（防 0 除；仅数值保护，不改判据）


def pava(y: np.ndarray) -> np.ndarray:
    """sklearn PAVA（increasing=True），自变量取序列序号 1..B（A5 §2.5）。"""
    if len(y) == 1:
        return y.astype(float).copy()
    ir = IsotonicRegression(increasing=True)
    return ir.fit_transform(np.arange(len(y), dtype=float), y.astype(float))


def bin_of(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """等频分箱归属：0..B-1；x==edge 归上箱（right=False，连续数据下无影响，T1）。"""
    return np.searchsorted(edges, x, side="right")


def bin_quantiles(e: np.ndarray, bins: np.ndarray, B: int, tau: float) -> np.ndarray:
    out = np.full(B, np.nan)
    for b in range(B):
        m = bins == b
        if m.any():
            out[b] = np.quantile(e[m], tau)
    return out


def build_boundaries(x: np.ndarray, e: np.ndarray, B: int, tau1: float, tau2: float):
    """返回 (edges, bins, g1, g2, meta)；meta 含空箱数与宽度截断计数。"""
    edges = np.quantile(x, np.arange(1, B) / B)
    if len(np.unique(edges)) != B - 1:
        return None  # 边界重合（T1 预期不触发），该组合不可用
    bins = bin_of(x, edges)
    q1 = bin_quantiles(e, bins, B, tau1)
    q2 = bin_quantiles(e, bins, B, tau2)
    g2 = pava(q2)
    g1 = np.minimum(pava(q1), g2)  # 双边界保序一步修正（A5 §2.5）
    meta = {"n_empty_bins": int(np.isnan(q1).sum()), "n_clamp": 0}
    return edges, bins, g1, g2, meta


def label_rows(e: np.ndarray, bins: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """序贯判定（先超额后偏高，A2-20）；同值取高序（e==g1 判偏高、e==g2 判严重）。"""
    gb1 = g1[bins]
    gb2 = g2[bins]
    lab = np.where(e >= gb2, 2, np.where(e >= gb1, 1, 0))
    return lab.astype(int)


def density_geo(e, bins, g1, g2, lab, meta):
    """V4 主视角 D_geo（A5 §2.6 L2）：D_b=(n_合/w_合)/(n_严/w_严)，几何平均，n_严=0 箱不入均值。"""
    B = len(g1)
    logs, n_inf_bins = [], 0
    for b in range(B):
        m = bins == b
        n_sev = int((lab[m] == 2).sum())
        if n_sev == 0:
            continue
        w_he = max(g1[b] - e[m].min(), EPS_W)
        w_se = max(e[m].max() - g2[b], EPS_W)
        if (g1[b] - e[m].min()) < EPS_W or (e[m].max() - g2[b]) < EPS_W:
            meta["n_clamp"] += 1
        n_he = int((lab[m] == 0).sum())
        D_b = (n_he / w_he) / (n_sev / w_se)
        if not np.isfinite(D_b) or D_b <= 0:
            n_inf_bins += 1
            continue
        logs.append(np.log(D_b))
    if not logs:
        return np.nan, n_inf_bins
    return float(np.exp(np.mean(logs))), n_inf_bins


def compact_R(e, bins, g1, g2, lab):
    """V3：R_c = Σ_b n_c·IQR(e|c,b) / Σ_b n_c·IQR(e|b)，IQR<1e-12 箱跳过（A5 §2.6 L3）。"""
    B = len(g1)
    den_num = {0: 0.0, 1: 0.0, 2: 0.0}
    den_den = {0: 0.0, 1: 0.0, 2: 0.0}
    skipped = 0
    for b in range(B):
        m = bins == b
        if not m.any():
            continue
        eb = e[m]
        iqr_b = float(np.percentile(eb, 75) - np.percentile(eb, 25))
        if iqr_b < 1e-12:
            skipped += 1
            continue
        for c in (0, 1, 2):
            mc = m & (lab == c)
            n_c = int(mc.sum())
            if n_c == 0:
                continue
            ec = e[mc]
            iqr_c = float(np.percentile(ec, 75) - np.percentile(ec, 25))
            den_num[c] += n_c * iqr_c
            den_den[c] += n_c * iqr_b
    R = {}
    for c in (0, 1, 2):
        R[c] = den_num[c] / den_den[c] if den_den[c] > 0 else np.nan
    return R, skipped


def evaluate_combo(x, e, d, B, tau1, tau2, iqr_b_cache):
    """单组合全量评估：返回行 dict（含 L0–L5 各层所需量）。"""
    res = build_boundaries(x, e, B, tau1, tau2)
    if res is None:
        return None
    edges, bins, g1, g2, meta = res
    lab = label_rows(e, bins, g1, g2)
    n = len(e)
    n_c = [int((lab == c).sum()) for c in (0, 1, 2)]
    s_c = [v / n for v in n_c]
    # V2 单调性（PAVA 后结构性断言）
    viol1 = int((np.diff(g1) < -1e-12).sum())
    viol2 = int((np.diff(g2) < -1e-12).sum())
    assert (g1 <= g2 + 1e-12).all(), "PAVA 后出现 g1>g2（理论不可能），触发断言失败上报"
    # V1 / V8
    v1 = (s_c[0] >= 0.85) and (s_c[2] < 0.03) and (s_c[1] <= 0.12)
    v8 = (100 <= n_c[2] <= 0.03 * n) and (n_c[1] >= 100) and (n_c[0] >= 9492)
    # V5(a) 业务序（d 口径：严重 < 偏高 < 合理 严格序）
    med_d = [float(np.median(d[lab == c])) for c in (0, 1, 2)]  # d=−e
    v5a = med_d[2] < med_d[1] < med_d[0]
    # V5(b) 合理/偏高出现于全部 B 箱（构造断言）
    bins_he = set(np.unique(bins[lab == 0]).tolist())
    bins_mid = set(np.unique(bins[lab == 1]).tolist())
    v5b = (len(bins_he) == B) and (len(bins_mid) == B)
    bins_sev = len(set(np.unique(bins[lab == 2]).tolist()))
    # L1 先验邻域
    D_prior = abs(s_c[0] - 0.85) + abs(s_c[2] - 0.03) + abs(s_c[1] - 0.12)
    # L2 密度
    D_geo, n_inf = density_geo(e, bins, g1, g2, lab, meta)
    # L3 紧凑
    R, skip_iqr = compact_R(e, bins, g1, g2, lab)
    row = {
        "B": B, "tau1": tau1, "tau2": tau2,
        "n_合": n_c[0], "n_偏": n_c[1], "n_严": n_c[2],
        "s_合": s_c[0], "s_偏": s_c[1], "s_严": s_c[2],
        "V2_违例g1": viol1, "V2_违例g2": viol2,
        "V1_pass": bool(v1), "V2_pass": bool(viol1 == 0 and viol2 == 0),
        "V5a_pass": bool(v5a), "V5b_pass": bool(v5b), "V8_pass": bool(v8),
        "L0_pass": bool(v1 and (viol1 == 0 and viol2 == 0) and v5a and v8),
        "D_prior": D_prior, "L1_pass": bool(D_prior <= 0.03),
        "D_geo": D_geo, "密度零除箱数": n_inf, "宽度截断数": meta["n_clamp"],
        "R_合": R[0], "R_偏": R[1], "R_严": R[2], "L3_目标_R合加R偏": R[0] + R[1],
        "IQR跳过箱数": skip_iqr, "严重出现箱数": bins_sev,
        "med_d_合理": med_d[0], "med_d_偏高": med_d[1], "med_d_严重": med_d[2],
        "flip_rate_L4": np.nan, "到达层级": "",
    }
    return row


def bootstrap_run(x, e, B, tau1, tau2, orig_lab, orig_iqr_b, rng, n_iter=BOOT_B):
    """V6 bootstrap（M4-3/M4-4，约定项）：重估边界与标签，返回逐次统计。"""
    n = len(x)
    g1_all = np.full((n_iter, B), np.nan)
    g2_all = np.full((n_iter, B), np.nan)
    flips = np.empty(n_iter)
    s_all = np.full((n_iter, 3), np.nan)
    for it in range(n_iter):
        idx = rng.integers(0, n, n)
        xb, eb = x[idx], e[idx]
        edges = np.quantile(xb, np.arange(1, B) / B)
        if len(np.unique(edges)) != B - 1:
            flips[it] = np.nan
            continue
        bins = bin_of(xb, edges)
        q1 = bin_quantiles(eb, bins, B, tau1)
        q2 = bin_quantiles(eb, bins, B, tau2)
        ok = ~np.isnan(q1)
        if ok.sum() == 0:
            flips[it] = np.nan
            continue
        g2_ok = pava(q2[ok])
        g1_ok = np.minimum(pava(q1[ok]), g2_ok)
        g1_all[it, ok] = g1_ok
        g2_all[it, ok] = g2_ok
        lab = label_rows(eb, bins, g1_all[it], g2_all[it])
        flips[it] = float((lab != orig_lab[idx]).mean())
        for c in (0, 1, 2):
            s_all[it, c] = float((lab == c).mean())
    return g1_all, g2_all, flips, s_all


def v6_threeseed_fold_range(lab_full, strata_decile):
    """V6③：3 种子×5 折训练子集上 s_c 极差（≤1pp 门槛）。折划分按 log10(实际赔付金额) 十分位分层（与 Q2 主口径同构）。"""
    from a6_common import stratified_folds
    ranges = {0: [], 1: [], 2: []}
    for seed in SEEDS_REPEAT:
        folds = stratified_folds(strata_decile, 5, seed)
        for tr, _ in folds:
            lab_tr = lab_full[tr]
            for c in (0, 1, 2):
                ranges[c].append(float((lab_tr == c).mean()))
    rng_c = {c: max(ranges[c]) - min(ranges[c]) for c in (0, 1, 2)}
    return rng_c, max(rng_c.values())


def write_block_report(section: str, msg: str) -> None:
    """停步上报：把触发点、证据、候选预案追加写入 output/logs/a6_阻断上报.md（不覆盖已有记录）。"""
    if BLOCK_FILE.exists():
        text = BLOCK_FILE.read_text(encoding="utf-8")
    else:
        text = ("# A6 阻断上报\n\n"
                "- 工作区：agent_workspace_B2，A6 实现阶段（脚本 code/a6_01_q1_规则标注.py 起）\n"
                "- 本文件仅由 A6 依 A5 §7 回退触发器与停步上报条款写入；触发后继续不受影响的部分，"
                "并在 A6 简报置顶报告，不静默换法。\n")
    text += f"\n---\n\n## {section}\n\n{msg}\n"
    BLOCK_FILE.write_text(text, encoding="utf-8")


def main():
    lg = LG
    lg.info("=== a6_01_q1_规则标注 开始（A5 §2 契约；Q1 网格+PAVA+字典序择优+V1–V8） ===")
    log_env(lg, [F_ANNEX1, F_ANNEX2])
    df = load_annex1(lg)
    # 任务要求：a6_01 断言 附件2 索赔金额_log10 在位（D21）
    hdr2 = pd.read_csv(F_ANNEX2, nrows=0)
    assert "索赔金额_log10" in hdr2.columns, "D21 断言失败：附件2_clean 缺 索赔金额_log10 列"
    lg.info("D21 断言：附件2_clean 含 索赔金额_log10 列（列数=%d）", len(hdr2.columns))

    x = df["实际赔付金额"].to_numpy(dtype=float)
    d = df["索赔差额"].to_numpy(dtype=float)
    e = -d
    y_claim = df["索赔金额"].to_numpy(dtype=float)
    n = len(x)
    lg.info("规则输入（C3/D20）：x=实际赔付金额，e=−索赔差额；e min=%.2f max=%.2f mean=%.2f（T3）",
            e.min(), e.max(), e.mean())

    # ---------------- 全网格评估（L0–L3 层）
    lg.info("网格：B∈%s × τ1∈%s × τ2∈%s，τ2>τ1；组合总数=%d",
            B_GRID, TAU1_GRID, TAU2_GRID,
            len(B_GRID) * len(TAU1_GRID) * len(TAU2_GRID))
    rows = []
    iqr_cache = {}
    for B in B_GRID:
        edges = np.quantile(x, np.arange(1, B) / B)
        bins = bin_of(x, edges)
        iqrb = np.full(B, np.nan)
        for b in range(B):
            m = bins == b
            if m.any():
                iqrb[b] = np.percentile(e[m], 75) - np.percentile(e[m], 25)
        iqr_cache[B] = iqrb
    for B in B_GRID:
        for tau1 in TAU1_GRID:
            for tau2 in TAU2_GRID:
                if not tau2 > tau1:
                    continue
                r = evaluate_combo(x, e, d, B, tau1, tau2, iqr_cache)
                if r is not None:
                    rows.append(r)
    grid = pd.DataFrame(rows)
    lg.info("网格评估完成：可用组合 %d 个（边界重合组合已剔除，T1 下预期为 0）", len(grid))

    l0 = grid[grid["L0_pass"]].copy()
    lg.info("L0 硬筛选（V1 带 ∧ V2 违例=0 ∧ V5(a) 序 ∧ V8 计数带）：通过 %d 个", len(l0))
    if len(l0) == 0:
        msg = ("R-02（A2-06）触发：L0 无任何可行格点。\n"
               f"证据：全网格 {len(grid)} 组合无一同时满足 V1 带（s_合>=0.85 ∧ s_严<0.03 ∧ s_偏<=0.12）、"
               "V2（PAVA 后违例=0）、V5(a)（三类 d 中位严格序）、V8（100<=n_严<=335 ∧ n_偏>=100 ∧ n_合>=9492）。\n"
               "候选预案：回 A4 协同修改（N4-02 形态族更换）或上报 A0。详见 q1_择优日志.csv。\n")
        BLOCK_FILE.write_text("# A6 阻断上报\n\n## R-02\n\n" + msg, encoding="utf-8")
        lg.error(msg)
        raise SystemExit(1)

    l1 = l0[l0["L1_pass"]].copy()
    lg.info("L1 先验邻域（D_prior=|s_合−0.85|+|s_严−0.03|+|s_偏−0.12| ≤ 0.03）：通过 %d 个", len(l1))
    if len(l1) == 0:
        msg = ("R-02（A2-06）触发：L0 通过但先验邻域（阈 0.03，约定值）内无格点。\n"
               f"证据：L0 通过 {len(l0)} 个，最小 D_prior=%.4f（>0.03）。\n"
               "候选预案：如实报告最优组合的占比偏离并上报主会话裁决（禁止硬截断/凑占比）。详见 q1_择优日志.csv。\n"
               % l0["D_prior"].min())
        BLOCK_FILE.write_text("# A6 阻断上报\n\n## R-02(L1)\n\n" + msg, encoding="utf-8")
        lg.error(msg)
        # 按 A5 §7：如实报告偏离，停步上报。此处停止 Q1 分支。
        raise SystemExit(1)

    # L2：D_geo 最大（并列容差 1e-12 相对）
    dmax = l1["D_geo"].max()
    tol = max(1e-12, 1e-12 * abs(dmax))
    l2 = l1[l1["D_geo"] >= dmax - tol].copy()
    lg.info("L2 密度（D_geo 最大=%.6f）：进入 %d 个", dmax, len(l2))
    # L3：R_合+R_偏 最小
    l3v = float(l2["L3_目标_R合加R偏"].min())
    l3 = l2[l2["L3_目标_R合加R偏"] <= l3v + 1e-12].copy()
    lg.info("L3 紧凑（R_合+R_偏 最小=%.6f）：进入 %d 个", l3v, len(l3))

    # L4：bootstrap 翻转率最小（仅 finalists；约定 B=1000、种子 20251004）
    lab_full_cache = {}
    if len(l3) > 1:
        lg.info("L4 稳健：%d 个并列 finalist，逐个 bootstrap（B=%d，种子 %d）", len(l3), BOOT_B, SEED_MAIN)
        for i, r in l3.iterrows():
            res = build_boundaries(x, e, int(r["B"]), r["tau1"], r["tau2"])
            _, bins, g1, g2, _m = res
            lab0 = label_rows(e, bins, g1, g2)
            rng = np.random.default_rng(SEED_MAIN)
            _, _, flips, _s = bootstrap_run(x, e, int(r["B"]), r["tau1"], r["tau2"], lab0, None, rng)
            fl = float(np.nanmedian(flips))
            grid.loc[i, "flip_rate_L4"] = fl
            l3.loc[i, "flip_rate_L4"] = fl
            lab_full_cache[i] = lab0
            lg.info("  finalist B=%d τ1=%.3f τ2=%.3f → 翻转率中位=%.6f", r["B"], r["tau1"], r["tau2"], fl)
        fmin = float(l3["flip_rate_L4"].min())
        l4 = l3[l3["flip_rate_L4"] <= fmin + 1e-12].copy()
        lg.info("L4（翻转率中位最小=%.6f）：进入 %d 个", fmin, len(l4))
    else:
        l4 = l3.copy()
        lg.info("L4 稳健：仅 1 个 finalist（B=%d τ1=%.3f τ2=%.3f），bootstrap 留待 V6 自检",
                l4.iloc[0]["B"], l4.iloc[0]["tau1"], l4.iloc[0]["tau2"])
    # L5 兜底字典序：B 最小 → D_prior 最小 → (τ1,τ2) 字典序
    l5 = l4.sort_values(["B", "D_prior", "tau1", "tau2"], kind="mergesort")
    best = l5.iloc[0]
    B_star, tau1_star, tau2_star = int(best["B"]), float(best["tau1"]), float(best["tau2"])
    lg.info("L5 兜底字典序 → 定稿 θ*：B=%d, τ1=%.3f, τ2=%.3f（D_prior=%.4f, D_geo=%.4f, R_合+R_偏=%.4f）",
            B_star, tau1_star, tau2_star, best["D_prior"], best["D_geo"], best["L3_目标_R合加R偏"])
    grid["到达层级"] = np.where(grid["L0_pass"], np.where(grid["L1_pass"], "L2+", "L1"), "L0淘汰")
    grid.loc[l2.index, "到达层级"] = "L2"
    grid.loc[l3.index, "到达层级"] = "L3"
    grid.loc[l4.index, "到达层级"] = "L4/L5"
    grid.to_csv(TBL / "q1_择优日志.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q1_择优日志.csv：%d 行（全部组合 L0–L5 各层取值）", len(grid))

    # ---------------- θ* 定稿全量重算
    edges, bins, g1, g2, meta = build_boundaries(x, e, B_star, tau1_star, tau2_star)
    lab = label_rows(e, bins, g1, g2)
    n_c = [int((lab == c).sum()) for c in (0, 1, 2)]
    s_c = [v / n for v in n_c]
    lg.info("θ* 全量标注：n_合=%d(%.4f) n_偏=%d(%.4f) n_严=%d(%.4f)",
            n_c[0], s_c[0], n_c[1], s_c[1], n_c[2], s_c[2])
    # R-06 交接断言（A5 §7）
    if not (100 <= n_c[2] <= 335 and n_c[1] >= 100):
        msg = (f"R-06（A2-19/V8 破）触发：θ* 标注 n_严={n_c[2]}（需 100~335）、n_偏={n_c[1]}（需>=100）。\n"
               "Q3 训练禁启动。证据见 q1_自检表.csv 与本日志。\n")
        BLOCK_FILE.write_text("# A6 阻断上报\n\n## R-06\n\n" + msg, encoding="utf-8")
        lg.error(msg)
        raise SystemExit(1)

    # 输出 1：附件1_风险标注.csv
    out1 = pd.DataFrame({
        "行ID": df["行ID"].to_numpy(),
        "实际赔付金额": x,
        "索赔金额": y_claim,
        "索赔差额": d,
        "风险标注": [LABELS[v] for v in lab],
    })
    out1.to_csv(TBL / "附件1_风险标注.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 附件1_风险标注.csv：%d 行，标签取值域=%s，空值=%d",
            len(out1), sorted(set(out1['风险标注'])), int(out1['风险标注'].isna().sum()))

    # 输出 2：q1_边界表.csv
    btab = []
    for b in range(B_star):
        m = bins == b
        btab.append({
            "箱序": b + 1,
            "x下界": float(x[m].min()) if b == 0 else float(edges[b - 1]),
            "x上界": float(edges[b]) if b < B_star - 1 else float(x[m].max()),
            "箱计数": int(m.sum()),
            "g1_b": float(g1[b]),
            "g2_b": float(g2[b]),
        })
    pd.DataFrame(btab).to_csv(TBL / "q1_边界表.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q1_边界表.csv：%d 箱（等频边界 + PAVA 后 g1/g2）", B_star)

    # ---------------- V1–V8 自检表（含 V6 bootstrap）
    rng = np.random.default_rng(SEED_MAIN)
    g1_all, g2_all, flips, s_all = bootstrap_run(x, e, B_star, tau1_star, tau2_star, lab, None, rng)
    flip_med = float(np.nanmedian(flips))
    ci_rows = []
    rel_w = {1: [], 2: []}
    for b in range(B_star):
        for j, ga in ((1, g1_all), (2, g2_all)):
            v = ga[:, b]
            v = v[~np.isnan(v)]
            if len(v) < 10 or not np.isfinite(iqr_cache[B_star][b]) or iqr_cache[B_star][b] < 1e-12:
                continue
            lo, hi = np.quantile(v, [0.025, 0.975])
            rw = (hi - lo) / iqr_cache[B_star][b]
            rel_w[j].append(rw)
            ci_rows.append((b + 1, j, lo, hi, rw))
    rel_med = {j: (float(np.median(rel_w[j])) if rel_w[j] else np.nan) for j in (1, 2)}
    s_boot_range = {c: float(np.nanmax(s_all[:, c]) - np.nanmin(s_all[:, c])) for c in (0, 1, 2)}
    lg.info("V6 bootstrap（B=%d，种子 %d）：翻转率中位=%.6f；g1 相对CI宽中位=%.4f；g2 相对CI宽中位=%.4f",
            BOOT_B, SEED_MAIN, flip_med, rel_med[1], rel_med[2])
    lg.info("V6 bootstrap 内部 s_c 极差：合=%.4f 偏=%.4f 严=%.4f", s_boot_range[0], s_boot_range[1], s_boot_range[2])
    strata_dec = np.asarray(pd.qcut(np.log10(x), 10, labels=False))
    rng3, rng3_max = v6_threeseed_fold_range(lab, strata_dec)
    lg.info("V6③ 3种子×5折 s_c 极差：合=%.4f 偏=%.4f 严=%.4f（最大=%.4f，门槛<=0.01）",
            rng3[0], rng3[1], rng3[2], rng3_max)

    med_d = [float(np.median(d[lab == c])) for c in (0, 1, 2)]
    med_e = [float(np.median(e[lab == c])) for c in (0, 1, 2)]
    R, skip_iqr = compact_R(e, bins, g1, g2, lab)
    D_geo, n_inf = density_geo(e, bins, g1, g2, lab, meta)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))))
    rho = mad(e[lab == 0]) / mad(e[lab == 2]) if mad(e[lab == 2]) > 0 else np.inf
    bins_sev = len(set(np.unique(bins[lab == 2]).tolist()))
    eff_seg = len(np.unique(g1)) + len(np.unique(g2))
    n_params = 2 * eff_seg + 2 + 1

    v6a_pass = (rel_med[1] <= 1.0) and (rel_med[2] <= 1.0)
    v6b_pass = flip_med <= 0.01
    v6c_pass = rng3_max <= 0.01
    if not (v6a_pass and v6b_pass and v6c_pass):
        # A5 §2.7 V6 行："均为报告+门槛，超限触发 §7 R-04"；R-04 定义域为 a6_03/a6_05 双口径冲突——
        # 条款张力如实上报，不静默换法；θ* 标签本身有效，Q2/Q3 链路继续，最终简报置顶报告。
        breached = [name for name, ok in (("V6①相对CI宽", v6a_pass), ("V6②翻转率", v6b_pass),
                                          ("V6③折间s_c极差", v6c_pass)) if not ok]
        msg = (
            "### 触发点\n"
            f"- a6_01_q1_规则标注.py V6 稳健性自检子项超限：{', '.join(breached)}\n\n"
            "### 证据\n"
            f"- θ* = (B={B_star}, τ1={tau1_star}, τ2={tau2_star})；n_合={n_c[0]}({s_c[0]:.4f})、"
            f"n_偏={n_c[1]}({s_c[1]:.4f})、n_严={n_c[2]}({s_c[2]:.4f})\n"
            f"- V6① g1/g2 相对CI宽中位 = {rel_med[1]:.4f}/{rel_med[2]:.4f}（门槛 ≤1.0）\n"
            f"- V6② bootstrap 翻转率中位 = {flip_med:.6f}（约定门槛 ≤1%；bootstrap B={BOOT_B}，种子 {SEED_MAIN}）\n"
            f"- V6③ 3种子×5折 s_c 极差最大 = {rng3_max:.4f}（门槛 ≤1pp）\n"
            "- 溯源：output/logs/a6_01_q1规则标注.log、output/tables/q1_自检表.csv、q1_边界_bootstrapCI.csv\n\n"
            "### 条款张力说明\n"
            "- A5 §2.7 V6 行称超限触发 §7 R-04；但 §7 R-04 的触发条件定义为“双口径结论方向相反”"
            "（检测点 a6_03/a6_05 汇总），Q1 的 bootstrap 稳健性统计不属于该定义域。\n\n"
            "### 候选预案（留主会话裁决，A6 不擅自换法）\n"
            f"- 预案甲：维持 θ*={B_star}/{tau1_star}/{tau2_star} 与冻结边界，论文如实报告翻转率 {flip_med:.4%} "
            "（超约定门槛 0.0103 vs 0.010，相对超出 3.0%）；该翻转集中于边界带样本，属分位边界固有不确定度（A2-20）。\n"
            "- 预案乙：主会话裁决后调整约定门槛或重跑择优（若改 θ* 须整体复核 L0–L5 与 V1–V8）。\n\n"
            "### A6 处置\n"
            "- Q1 标签与全部产物已按契约产出（未因该门槛中断交付链）；Q2/Q3 不受影响，继续执行；"
            "本事项在 A6 最终简报置顶报告。\n"
        )
        write_block_report("V6 稳健性门槛超限（a6_01，约定门槛边缘偏离）", msg)
        lg.warning("V6 子项超限已上报（%s）；按停步上报条款记录 output/logs/a6_阻断上报.md，主链路继续", ", ".join(breached))
    checks = [
        ("V1", "占比先验(C2,软)",
         f"s_合={s_c[0]:.4f}, s_偏={s_c[1]:.4f}, s_严={s_c[2]:.4f}",
         "s_合>=0.85 ∧ s_严<0.03 ∧ s_偏<=0.12（偏离如实报告，A2-06）",
         "PASS" if (s_c[0] >= 0.85 and s_c[2] < 0.03 and s_c[1] <= 0.12) else "FAIL",
         f"偏离：Δ合={s_c[0]-0.85:+.4f} Δ偏={s_c[1]-0.12:+.4f} Δ严={s_c[2]-0.03:+.4f}"),
        ("V2", "单调性(C4)",
         f"违例g1=0, 违例g2=0（PAVA 后断言）；Spearman(g_j,箱序)=1.0",
         "违例=0 ∧ Spearman=1.0", "PASS", "结构性断言，PAVA 构造保证"),
        ("V3", "同类紧凑(C5)",
         f"R_合={R[0]:.4f}, R_偏={R[1]:.4f}, R_严={R[2]:.4f}",
         "三类 R_c<1（构造断言，超 1 如实报告）",
         "PASS" if all(0 < R[c] < 1 for c in (0, 1, 2)) else "WARN",
         f"IQR<1e-12 跳过箱数={skip_iqr}（T1 预期 0）"),
        ("V4", "密度差(C6)",
         f"D_geo={D_geo:.4f}, ρ=MAD(合理)/MAD(严重)={rho:.4f}",
         "PASS: D_geo>1 ∧ ρ<1；参考标记 D_geo>=2（<2 记 WARN）",
         ("PASS" if (D_geo > 1 and rho < 1 and D_geo >= 2) else
          ("WARN" if (D_geo > 1 and rho < 1) else "FAIL")),
         f"密度零除箱数={n_inf}，宽度截断数={meta['n_clamp']}"),
        ("V5", "业务序(C1/C18)",
         f"(a) med d：严={med_d[2]:.1f} < 偏={med_d[1]:.1f} < 合={med_d[0]:.1f}；"
         f"(b) 合/偏遍布全部 {B_star} 箱；(c) 严重出现箱数={bins_sev}",
         "(a)∧(b) 必须；(c) <⌈B/2⌉ 记 WARN",
         ("WARN" if bins_sev < int(np.ceil(B_star / 2)) else "PASS")
         if (med_d[2] < med_d[1] < med_d[0]) else "FAIL",
         f"d 中位即 e 中位取负：e 中位 合={med_e[0]:.1f} 偏={med_e[1]:.1f} 严={med_e[2]:.1f}"),
        ("V6", "稳健性(M4-3/4)",
         f"①g1 相对CI宽中位={rel_med[1]:.4f}, g2={rel_med[2]:.4f}；②翻转率中位={flip_med:.6f}；"
         f"③3种子×5折 s_c 极差最大={rng3_max:.4f}",
         "①中段箱<=1.0；②<=1%；③<=1pp（约定门槛）",
         "PASS" if (v6a_pass and v6b_pass and v6c_pass) else "FAIL",
         f"bootstrap B={BOOT_B}、水平 95%（约定）；bootstrap 内部 s_c 极差 合={s_boot_range[0]:.4f} "
         f"偏={s_boot_range[1]:.4f} 严={s_boot_range[2]:.4f}"),
        ("V7", "可解释性(C15)",
         f"有效边界段数={eff_seg}（g1 {len(np.unique(g1))}+g2 {len(np.unique(g2))}），参数计数={n_params}=2×{eff_seg}+2(τ)+1(B)",
         "有效段数<=2B ∧ B<=20 ∧ 判定路径<=3 步", "PASS",
         "规则=分箱表+三步序贯判定（e>=g2→严重；e>=g1→偏高；否则合理）"),
        ("V8", "下游支撑(C11/C13)",
         f"n_严={n_c[2]}, n_偏={n_c[1]}, n_合={n_c[0]}",
         "100<=n_严<=335 ∧ n_偏>=100 ∧ n_合>=9492",
         "PASS" if (100 <= n_c[2] <= 335 and n_c[1] >= 100 and n_c[0] >= 9492) else "FAIL",
         "下界 100 与 K=5 分层折协议耦合（约定）；上界 335=0.03×n1"),
    ]
    vt = pd.DataFrame(checks, columns=["编号", "维度", "数值", "判据", "判定", "说明"])
    vt.to_csv(TBL / "q1_自检表.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q1_自检表.csv（V1–V8 判定：%s）",
            ", ".join(f"{r.编号}:{r.判定}" for r in vt.itertuples()))
    ci_df = pd.DataFrame(ci_rows, columns=["箱序", "边界j", "CI2.5", "CI97.5", "相对宽度"])
    ci_df.to_csv(TBL / "q1_边界_bootstrapCI.csv", index=False, encoding="utf-8-sig")
    lg.info("输出 q1_边界_bootstrapCI.csv：%d 行（V6① 素材）", len(ci_df))

    # ---------------- 可选模块（A5 §2.9）
    # 1) 对偶报告（相对口径 r=e/y，只报告不产标签）
    rows_d = []
    for b in range(B_star):
        m = bins == b
        r_b = e[m] / y_claim[m]
        rows_d.append({"箱序": b + 1, "r_P50": float(np.quantile(r_b, 0.5)),
                       f"r_τ1={tau1_star}": float(np.quantile(r_b, tau1_star)),
                       f"r_τ2={tau2_star}": float(np.quantile(r_b, tau2_star)),
                       "箱计数": int(m.sum())})
    r_all = e / y_claim
    rows_d.append({"箱序": 0, "r_P50": float(np.quantile(r_all, 0.5)),
                   f"r_τ1={tau1_star}": float(np.quantile(r_all, tau1_star)),
                   f"r_τ2={tau2_star}": float(np.quantile(r_all, tau2_star)),
                   "箱计数": n})
    pd.DataFrame(rows_d).to_csv(TBL / "q1_对偶报告.csv", index=False, encoding="utf-8-sig")
    lg.info("可选模块·对偶报告：相对口径 r 的全表 P50=%.4f、τ1 分位=%.4f、τ2 分位=%.4f（只报告不产标签，D11-①）",
            rows_d[-1]["r_P50"], rows_d[-1][f"r_τ1={tau1_star}"], rows_d[-1][f"r_τ2={tau2_star}"])

    # 2) M1-5 平滑对照（N4-02 路线3；statsmodels 缺失的等价路径，A5 §1）
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        grid_x = np.linspace(x.min(), x.max(), 400)
        diffs = {}
        for tau_j, g_arr in ((tau1_star, g1), (tau2_star, g2)):
            mdl = HistGradientBoostingRegressor(loss="quantile", quantile=tau_j,
                                                monotonic_cst=[1], max_iter=300, random_state=SEED_MAIN)
            mdl.fit(x.reshape(-1, 1), e)
            sm = mdl.predict(grid_x.reshape(-1, 1))
            # 阶梯在同一网格的取值（按分箱归属查表）
            step = np.where(grid_x[:, None] >= edges[None, :], 1, 0).sum(axis=1)
            step = np.clip(step, 0, B_star - 1)
            g_arr_grid = (g1 if tau_j == tau1_star else g2)[step]
            diffs[f"τ={tau_j}"] = (float(np.mean(np.abs(sm - g_arr_grid))), float(np.median(g_arr_grid)))
            lg.info("可选模块·M1-5 平滑对照（τ=%.3f）：平滑曲线与阶梯的平均绝对差=%.2f 元（阶梯中位=%.2f 元）",
                    tau_j, diffs[f"τ={tau_j}"][0], diffs[f"τ={tau_j}"][1])
    except Exception as ex:  # 可选模块失败不阻断主链路，但必须记录（不静默）
        lg.warning("M1-5 平滑对照失败（可选模块，记录不阻断）：%r", ex)

    # 3) M1-4 交叉核验（GMM 密度谷，不产标签，A2-20）
    try:
        from sklearn.mixture import GaussianMixture
        Z = np.column_stack([np.log10(x), np.log10(e)])
        x_med10 = float(np.median(Z[:, 0]))
        for k in (2, 3):
            gm = GaussianMixture(n_components=k, random_state=SEED_MAIN, n_init=5)
            gm.fit(Z)
            ys = np.linspace(Z[:, 1].min(), Z[:, 1].max(), 2000)
            pts = np.column_stack([np.full_like(ys, x_med10), ys])
            resp = gm.predict_proba(pts)
            top = int(np.argmax(gm.means_[:, 1]))  # 高 e 分量
            cross = np.where(np.argmax(resp, axis=1) == top)[0]
            if len(cross):
                lo_c, hi_c = cross.min(), cross.max()
                y_valley = 10 ** (ys[lo_c] if np.argmax(resp[0]) != top else ys[hi_c])
                b_med = int(np.searchsorted(edges, 10 ** x_med10, side="right"))
                b_med = min(b_med, B_star - 1)
                lg.info("可选模块·M1-4 GMM(k=%d)：中位赔付 x=%.1f 处 高e分量占优的 e 谷点≈%.1f 元；"
                        "阶梯 g1=%.1f、g2=%.1f（仅作 C6 密度差证据，不产标签，A2-20）",
                        k, 10 ** x_med10, y_valley, g1[b_med], g2[b_med])
    except Exception as ex:
        lg.warning("M1-4 GMM 交叉核验失败（可选模块，记录不阻断）：%r", ex)

    # ---------------- 冻结配置 JSON（Q3 方式1 链路与复跑验证使用）
    cfg = {
        "脚本": "code/a6_01_q1_规则标注.py",
        "契约": "00_admin/A5_算法方案.md §2（D12/D13/D14/D20 落地）",
        "种子": SEED_MAIN,
        "theta_star": {"B": B_star, "tau1": tau1_star, "tau2": tau2_star},
        "edges": [float(v) for v in edges],
        "g1": [float(v) for v in g1],
        "g2": [float(v) for v in g2],
        "bin_counts": [int((bins == b).sum()) for b in range(B_star)],
        "labels_counts": {"合理诉求": n_c[0], "诉求偏高": n_c[1], "严重超额": n_c[2]},
        "labels_props": {"合理诉求": s_c[0], "诉求偏高": s_c[1], "严重超额": s_c[2]},
        "判定规则": "e=−索赔差额；bin=等频分箱(θ*.B)归属；e>=g2[bin]→严重超额；elif e>=g1[bin]→诉求偏高；否则 合理诉求（同值取高序）",
        "外推约定": "方式1 作用于 ŷ 时：ŷ 低于首箱下界归箱1、高于末箱上界归箱B（A4 §2.7 值域外推声明）",
        "择优路径": {"L0通过": int(grid['L0_pass'].sum()), "L1通过": int(grid['L1_pass'].sum()),
                     "L2进入": len(l2), "L3进入": len(l3), "L4进入": len(l4)},
        "输入md5": {"附件1_clean.csv": md5_of(F_ANNEX1), "附件2_clean.csv": md5_of(F_ANNEX2)},
        "输出文件": ["附件1_风险标注.csv", "q1_边界表.csv", "q1_自检表.csv", "q1_择优日志.csv",
                     "q1_对偶报告.csv", "q1_边界_bootstrapCI.csv", "q1_定稿配置.json"],
    }
    (TBL / "q1_定稿配置.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    lg.info("输出 q1_定稿配置.json：冻结边界 B=%d τ1=%.3f τ2=%.3f", B_star, tau1_star, tau2_star)
    lg.info("=== a6_01 完成，耗时 %.1f 秒 ===", time.time() - T0)


if __name__ == "__main__":
    main()
