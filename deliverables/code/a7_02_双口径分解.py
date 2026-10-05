# -*- coding: utf-8 -*-
"""
a7_02_双口径分解.py —— A7 实验2：D24 双口径差异深挖（Q3 杠杆收益反转机制）
现象（A6 实测，q3_指标汇总.csv / q3_消融矩阵.csv）：
  方式2 主配置(嵌套γ,s) vs 消融A0(无杠杆) 宏F1：分层5折 0.6704>0.6529（杠杆 +0.0175）；
  行序外推 0.6710<0.6922（杠杆 −0.0212）→ 方向反转（R-04，D24 限定结论于分层口径）。
本实验回答两点：
  (1) 机制分解：两口径下验证集的标签分布/特征（赔付 x）分布差异有多大？类级 F1 分解显示
      杠杆收益/损失来自哪里？
  (2) 统计显著性：行序口径仅单次 8934/2233 切分、严重类验证样本仅 44 行——"反转"本身
      是否在小样本噪声范围内？（按真实类逐行多项式 bootstrap 的配对比较）
方法：
  - 主配置混淆矩阵取自 A6 q3_混淆矩阵.csv（精确）；
  - A0 混淆矩阵由 q3_消融矩阵.csv 的每类 P/R/支撑数做"行列合计精确重构"（格内分配存在
    一个自由度，取可行域两端各跑一遍，报告区间——不作静默假设）；
  - 配对 bootstrap：n'_c ~ Multinomial(n, q_true)，每类预测按该配置该类的预测分布重抽，
    B=2000，种子 20251004；比较 Δ宏F1 = F1_主 − F1_A0 的 CI 与 P(Δ>0)。
输出：output/tables/a7_双口径分解.csv（镜像 output/tables/sensitivity_a7/）
日志：output/logs/a7_02_双口径分解.log
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"
LOGD = ROOT / "output" / "logs"
MIRROR = TBL / "sensitivity_a7"
MIRROR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOGD / "a7_02_双口径分解.log", mode="w", encoding="utf-8"),
                              logging.StreamHandler()])
LG = logging.getLogger("a7_02")
T0 = time.time()
SEED, NBOOT = 20251004, 2000

LG.info("=== a7_02_双口径分解 开始（D24 指定：Q3 杠杆收益反转机制） ===")
LG.info("环境：numpy=%s pandas=%s scipy=%s；bootstrap B=%d，种子 %d", np.__version__, pd.__version__,
        __import__("scipy").__version__, NBOOT, SEED)

# ---------------- 1. 两口径验证集的分布对比 ----------------
df = pd.read_csv(TBL / "附件1_clean.csv", encoding="utf-8-sig")
ann = pd.read_csv(TBL / "附件1_风险标注.csv", encoding="utf-8-sig")
df = df.merge(ann[["行ID", "风险标注"]], on="行ID", how="inner")
assert len(df) == 11167
n_tr, n_va = 8934, 2233  # A5 §3.6：前 80% 训练 / 后 20% 验证（a6_03 日志行22 同口径）
is_valid = df["行ID"] > n_tr
x = df["实际赔付金额"].to_numpy(float)
ycl = df["索赔金额"].to_numpy(float)
e = -df["索赔差额"].to_numpy(float)
lab = df["风险标注"]
CLS = ["合理诉求", "诉求偏高", "严重超额"]

dist_rows = []
def add_dist(块, 键, v_tr, v_va, fmt="{:.4f}"):
    dist_rows.append({"块": 块, "指标": 键, "行序_训练(1~8934)": fmt.format(v_tr),
                      "行序_验证(8935~11167)": fmt.format(v_va)})

q_tr = np.array([(lab[~is_valid] == c).mean() for c in CLS])
q_va = np.array([(lab[is_valid] == c).mean() for c in CLS])
q_pool = np.array([(lab == c).mean() for c in CLS])
for i, c in enumerate(CLS):
    add_dist("标签占比", c, q_tr[i], q_va[i])
add_dist("x 中位(元)", "实际赔付金额", float(np.median(x[~is_valid])), float(np.median(x[is_valid])), "{:.1f}")
add_dist("y 中位(元)", "索赔金额", float(np.median(ycl[~is_valid])), float(np.median(ycl[is_valid])), "{:.1f}")
add_dist("e 中位(元)", "超额幅度=−差额", float(np.median(e[~is_valid])), float(np.median(e[is_valid])), "{:.1f}")
ks_x = stats.ks_2samp(np.log10(x[~is_valid]), np.log10(x[is_valid]))
ks_y = stats.ks_2samp(np.log10(ycl[~is_valid]), np.log10(ycl[is_valid]))
ks_e = stats.ks_2samp(np.log10(e[~is_valid]), np.log10(e[is_valid]))
dist_rows.append({"块": "KS检验(训练vs验证)", "指标": "log10 实际赔付金额", "行序_训练(1~8934)": f"stat={ks_x.statistic:.4f}",
                  "行序_验证(8935~11167)": f"p={ks_x.pvalue:.4f}"})
dist_rows.append({"块": "KS检验(训练vs验证)", "指标": "log10 索赔金额", "行序_训练(1~8934)": f"stat={ks_y.statistic:.4f}",
                  "行序_验证(8935~11167)": f"p={ks_y.pvalue:.4f}"})
dist_rows.append({"块": "KS检验(训练vs验证)", "指标": "log10 超额幅度e", "行序_训练(1~8934)": f"stat={ks_e.statistic:.4f}",
                  "行序_验证(8935~11167)": f"p={ks_e.pvalue:.4f}"})
kw_x = stats.kruskal(*[np.log10(x[(df["行ID"] - 1) // 1117 == k]) for k in range(10)])
dist_rows.append({"块": "U3复核", "指标": "行序十分位块 log10(x) KW", "行序_训练(1~8934)": f"H={kw_x.statistic:.2f}",
                  "行序_验证(8935~11167)": f"p={kw_x.pvalue:.4f}"})
dist_rows.append({"块": "对照", "指标": "分层5折口径", "行序_训练(1~8934)": "训练/验证同分布（按标签分层随机）",
                  "行序_验证(8935~11167)": f"池化验证=全表 11167 行，类占比 {q_pool[0]:.4f}/{q_pool[1]:.4f}/{q_pool[2]:.4f}"})
dist_df = pd.DataFrame(dist_rows)
LG.info("分布对比：行序验证 x 中位=%.1f（训练 %.1f，差 %+.1f%%）；KS log10(x) p=%.4f；KW 十分位 p=%.4f（U3 p=0.0091 同向）",
        np.median(x[is_valid]), np.median(x[~is_valid]),
        100 * (np.median(x[is_valid]) / np.median(x[~is_valid]) - 1), ks_x.pvalue, kw_x.pvalue)
LG.info("行序验证类占比 合/偏/严 = %.4f/%.4f/%.4f（全表 %.4f/%.4f/%.4f）——标签分布几乎相同，不是反转主因",
        *q_va, *q_pool)

# ---------------- 2. 类级 F1 分解（A6 数字转抄 + Δ） ----------------
summ = pd.read_csv(TBL / "q3_指标汇总.csv", encoding="utf-8-sig")
abl = pd.read_csv(TBL / "q3_消融矩阵.csv", encoding="utf-8-sig")
main_l = summ[(summ["配置"] == "方式2_主配置(嵌套γ,s)") & (summ["口径"] == "分层5折") & (summ["种子"] == 20251004)].iloc[0]
main_r = summ[(summ["配置"] == "方式2_主配置(嵌套γ,s)") & (summ["口径"] == "行序外推")].iloc[0]
a0_l = abl[(abl["配置"] == "消融A0") & (abl["口径"] == "分层5折")].iloc[0]
a0_r = abl[(abl["配置"] == "消融A0") & (abl["口径"] == "行序外推")].iloc[0]
dec_rows = []
for tag, m, a in (("分层5折", main_l, a0_l), ("行序外推", main_r, a0_r)):
    for i, c in enumerate(CLS, start=0):
        f1m, f1a = m[f"类{i}_f1"], a[f"类{i}_f1"]
        dec_rows.append({"口径": tag, "类": c, "F1_主配置": round(float(f1m), 6), "F1_A0": round(float(f1a), 6),
                         "ΔF1(主−A0)": round(float(f1m - f1a), 6),
                         "recall_主": round(float(m[f"类{i}_recall"]), 6), "recall_A0": round(float(a[f"类{i}_recall"]), 6),
                         "precision_主": round(float(m[f"类{i}_precision"]), 6),
                         "precision_A0": round(float(a[f"类{i}_precision"]), 6),
                         "支撑数": int(m[f"类{i}_support"])})
    dec_rows.append({"口径": tag, "类": "宏F1", "F1_主配置": round(float(m["宏F1"]), 6),
                     "F1_A0": round(float(a["宏F1"]), 6), "ΔF1(主−A0)": round(float(m["宏F1"] - a["宏F1"]), 6),
                     "recall_主": np.nan, "recall_A0": np.nan, "precision_主": np.nan, "precision_A0": np.nan,
                     "支撑数": int(m["类0_support"] + m["类1_support"] + m["类2_support"])})
dec_df = pd.DataFrame(dec_rows)
LG.info("类级分解：分层 Δ宏F1=%+.4f（主要来自严重类 ΔF1=%+.4f：recall 0.2133→0.3200）；行序 Δ宏F1=%+.4f（严重类 %+.4f、偏高类 %+.4f 均为负）",
        main_l["宏F1"] - a0_l["宏F1"], main_l["类2_f1"] - a0_l["类2_f1"],
        main_r["宏F1"] - a0_r["宏F1"], main_r["类2_f1"] - a0_r["类2_f1"], main_r["类1_f1"] - a0_r["类1_f1"])

# ---------------- 3. 混淆矩阵：主配置（精确） + A0（行列合计精确重构） ----------------
cm = pd.read_csv(TBL / "q3_混淆矩阵.csv", encoding="utf-8-sig")
def cm_of(口径):
    sub = cm[cm["口径"] == 口径]
    M = np.zeros((3, 3), dtype=int)
    for _, r in sub.iterrows():
        M[CLS.index(r["真实"]), CLS.index(r["预测"])] = int(r["行数"])
    return M
M_main_l, M_main_r = cm_of("分层5折_主种子池化"), cm_of("行序外推")

def reconstruct_A0(row):
    """由每类 P/R/support 重构混淆矩阵：TP=round(R*n_c)、FN=n_c−TP、列合计 col=round(TP/P)、FP=col−TP。
    FP 按列向量分配到行存在一个自由度，返回 (自由度区间, 分配函数)。"""
    sup = np.array([int(row[f"类{i}_support"]) for i in range(3)])
    rec = np.array([row[f"类{i}_recall"] for i in range(3)])
    pre = np.array([row[f"类{i}_precision"] for i in range(3)])
    TP = np.array([round(float(r * s)) for r, s in zip(rec, sup)])
    FN = sup - TP
    COL = np.array([round(float(t / p)) for t, p in zip(TP, pre)])
    FP = COL - TP
    assert FP.sum() == FN.sum(), "P/R 与支撑数不自洽，停止重构"
    return TP, FN, FP, COL

def build_A0(sup, rec, pre, t):
    TP = np.array([round(float(r * s)) for r, s in zip(rec, sup)])
    FN = sup - TP
    COL = np.array([round(float(tp / p)) for tp, p in zip(TP, pre)])
    FP = COL - TP
    FN0, FN1, FN2 = FN
    FP0, FP1, FP2 = FP
    r01 = t
    r02 = FN0 - t
    r12 = FP2 - r02
    r10 = FN1 - r12
    r20 = FP0 - r10
    r21 = FN2 - r20
    checks = [r01 >= 0, r02 >= 0, r12 >= 0, r10 >= 0, r20 >= 0, r21 >= 0,
              r01 <= FP1, r02 <= FP2, r10 <= FP0, r20 <= FP0, r12 <= FP2, r21 <= FP1]
    if not all(checks):
        return None
    M = np.array([[TP[0], r01, r02], [r10, TP[1], r12], [r20, r21, TP[2]]], dtype=int)
    assert (M.sum(axis=1) == sup).all() and (M.sum(axis=0) == COL).all()
    return M

def macro_f1_from_cm(M):
    f1s = []
    for c in range(3):
        tp = M[c, c]
        col, row = M[:, c].sum(), M[c, :].sum()
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))

# 校验：主配置混淆矩阵重算宏F1 应与 q3_指标汇总 一致
f1_check_l, f1_check_r = macro_f1_from_cm(M_main_l), macro_f1_from_cm(M_main_r)
LG.info("主配置混淆矩阵复核：分层宏F1=%.6f（汇总 0.670365）；行序宏F1=%.6f（汇总 0.670963）",
        f1_check_l, f1_check_r)

# A0 重构可行域
sup_l = np.array([int(a0_l[f"类{i}_support"]) for i in range(3)])
rec_l = np.array([a0_l[f"类{i}_recall"] for i in range(3)])
pre_l = np.array([a0_l[f"类{i}_precision"] for i in range(3)])
sup_r = np.array([int(a0_r[f"类{i}_support"]) for i in range(3)])
rec_r = np.array([a0_r[f"类{i}_recall"] for i in range(3)])
pre_r = np.array([a0_r[f"类{i}_precision"] for i in range(3)])
feas_l = [t for t in range(0, 600) if build_A0(sup_l, rec_l, pre_l, t) is not None]
feas_r = [t for t in range(0, 600) if build_A0(sup_r, rec_r, pre_r, t) is not None]
LG.info("A0 重构自由参数可行域：分层 t∈[%d,%d]（%d 个解）；行序 t∈[%d,%d]（%d 个解）——取两端为端界情形",
        feas_l[0], feas_l[-1], len(feas_l), feas_r[0], feas_r[-1], len(feas_r))
f1_a0_l_ends = [macro_f1_from_cm(build_A0(sup_l, rec_l, pre_l, t)) for t in (feas_l[0], feas_l[-1])]
f1_a0_r_ends = [macro_f1_from_cm(build_A0(sup_r, rec_r, pre_r, t)) for t in (feas_r[0], feas_r[-1])]
LG.info("A0 重构宏F1 端界：分层 %.6f~%.6f（消融实测 0.652902）；行序 %.6f~%.6f（消融实测 0.692190）",
        *f1_a0_l_ends, *f1_a0_r_ends)

# ---------------- 4. 配对多项式 bootstrap（B=2000，种子 20251004） ----------------
rng = np.random.default_rng(SEED)
def boot_delta(M_main, M_a0, tag):
    n = M_main.sum()
    q_true = M_main.sum(axis=1) / n
    rowd_main = M_main / M_main.sum(axis=1, keepdims=True)
    rowd_a0 = M_a0 / M_a0.sum(axis=1, keepdims=True)
    deltas, f1m_all, f1a_all = [], [], []
    for _ in range(NBOOT):
        n_c = rng.multinomial(n, q_true)
        Mm = np.zeros((3, 3), int)
        Ma = np.zeros((3, 3), int)
        for c in range(3):
            Mm[c, :] = rng.multinomial(int(n_c[c]), rowd_main[c])
            Ma[c, :] = rng.multinomial(int(n_c[c]), rowd_a0[c])
        f1m_all.append(macro_f1_from_cm(Mm))
        f1a_all.append(macro_f1_from_cm(Ma))
        deltas.append(f1m_all[-1] - f1a_all[-1])
    deltas = np.array(deltas)
    out = {"口径": tag, "B": NBOOT,
           "Δ宏F1_点估计": round(float(macro_f1_from_cm(M_main) - macro_f1_from_cm(M_a0)), 6),
           "Δ宏F1_中位": round(float(np.median(deltas)), 6),
           "Δ宏F1_CI2.5": round(float(np.percentile(deltas, 2.5)), 6),
           "Δ宏F1_CI97.5": round(float(np.percentile(deltas, 97.5)), 6),
           "P_Δ大于0": round(float((deltas > 0).mean()), 4),
           "F1_主_中位": round(float(np.median(f1m_all)), 6),
           "F1_主_CI2.5": round(float(np.percentile(f1m_all, 2.5)), 6),
           "F1_主_CI97.5": round(float(np.percentile(f1m_all, 97.5)), 6),
           "F1_A0_中位": round(float(np.median(f1a_all)), 6),
           "F1_A0_CI2.5": round(float(np.percentile(f1a_all, 2.5)), 6),
           "F1_A0_CI97.5": round(float(np.percentile(f1a_all, 97.5)), 6)}
    return out

boot_rows = []
for tag, M_main, (sup, rec, pre, feas, f1_ends) in (
        ("分层5折", M_main_l, (sup_l, rec_l, pre_l, feas_l, f1_a0_l_ends)),
        ("行序外推", M_main_r, (sup_r, rec_r, pre_r, feas_r, f1_a0_r_ends))):
    for tmark, t in (("重构端界A(自由参数下界)", feas[0]), ("重构端界B(自由参数上界)", feas[-1])):
        M_a0 = build_A0(sup, rec, pre, t)
        row = boot_delta(M_main, M_a0, tag)
        row["A0_重构方案"] = tmark
        row["A0_重构宏F1"] = round(float(macro_f1_from_cm(M_a0)), 6)
        boot_rows.append(row)
        LG.info("bootstrap %s %s：Δ宏F1 点估计=%+.4f，95%%CI=[%+.4f, %+.4f]，P(Δ>0)=%.3f",
                tag, tmark, row["Δ宏F1_点估计"], row["Δ宏F1_CI2.5"], row["Δ宏F1_CI97.5"], row["P_Δ大于0"])

# ---------------- 落盘 ----------------
out1 = pd.DataFrame(dist_rows)
out2 = dec_df
out3 = pd.DataFrame(boot_rows)
out1.to_csv(TBL / "a7_双口径分解_分布对比.csv", index=False, encoding="utf-8-sig")
out2.to_csv(TBL / "a7_双口径分解.csv", index=False, encoding="utf-8-sig")
out3.to_csv(TBL / "a7_双口径分解_bootstrap.csv", index=False, encoding="utf-8-sig")
out1.to_csv(MIRROR / "a7_双口径分解_分布对比.csv", index=False, encoding="utf-8-sig")
out2.to_csv(MIRROR / "a7_双口径分解.csv", index=False, encoding="utf-8-sig")
out3.to_csv(MIRROR / "a7_双口径分解_bootstrap.csv", index=False, encoding="utf-8-sig")
LG.info("输出 a7_双口径分解.csv（类级分解 %d 行）、a7_双口径分解_分布对比.csv（%d 行）、a7_双口径分解_bootstrap.csv（%d 行）+ 镜像",
        len(out2), len(out1), len(out3))
LG.info("=== a7_02 完成，耗时 %.1f 秒 ===", time.time() - T0)
