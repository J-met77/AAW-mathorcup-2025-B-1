# -*- coding: utf-8 -*-
"""
a7_01_q1翻转带敏感性.py —— A7 实验1：Q1 V6② 翻转带敏感性（D23 指定）
问题：θ*=(B=5, τ1=0.86, τ2=0.98) 的 bootstrap 标签翻转率中位 1.0298%（a6_01，超 1% 约定门槛）
      是否为"分位边界固有不确定度"而非"择优偶然"？
方法（全部与 a6_01_q1_规则标注.py 同构，直接 import 其函数保证逐行一致）：
  1) θ 邻域网格 bootstrap：τ1∈{0.85,0.86,0.87}×τ2∈{0.975,0.98,0.985}（B=5）与 B∈{5,10,15,20}（τ=0.86/0.98），
     每格 1000 次重抽（种子 20251004，与 a6_01 同源）：flip_own=重估标签 vs 该格自身全量标签；
     flip_vsθ*=重估标签 vs θ* 冻结标签；确定性重标差异率=格内全量标签 vs θ* 冻结标签。
  2) θ* 翻转样本画像：1000 次 bootstrap 逐次记录边界，逐行统计全量口径翻转频率
     （该次边界作用于该行 (x,e) 的标签 ≠ 冻结标签；in-bag 口径另行池化复核 1.0298%）；
     按"到最近适用边界的相对距离"分带给出行均翻转频率与翻转质量占比；
     冻结标签→重估标签转移矩阵；高频翻转行集中度。
输出：output/tables/a7_翻转带敏感性.csv、a7_翻转带画像.csv、a7_翻转带画像_边界CI.csv
     （镜像于 output/tables/sensitivity_a7/，红线与任务路径双兼容）
日志：output/logs/a7_01_翻转带敏感性.log
红线：全程离线；只读 A6 产物与附件1_clean；不修改任何 A6 产物。
"""
import importlib.util
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

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOGD / "a7_01_翻转带敏感性.log", mode="w", encoding="utf-8"),
                              logging.StreamHandler()])
LG = logging.getLogger("a7_01")

T0 = time.time()
SEED = 20251004
NDRAW = 1000  # 与 a6_01 V6 bootstrap 同为 B=1000（约定项）

# ---- 复用 a6_01 的边界构造函数（保证与 A6 逐行同构）----
spec = importlib.util.spec_from_file_location("a6_01_mod", ROOT / "code" / "a6_01_q1_规则标注.py")
a6_01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a6_01)

LG.info("=== a7_01_q1翻转带敏感性 开始（D23 指定：θ*=(5,0.86,0.98) 翻转带敏感性） ===")
LG.info("环境：numpy=%s pandas=%s；bootstrap 每格 %d 次，种子 %d", np.__version__, pd.__version__, NDRAW, SEED)

df = pd.read_csv(TBL / "附件1_clean.csv", encoding="utf-8-sig")
x = df["实际赔付金额"].to_numpy(dtype=float)
d = df["索赔差额"].to_numpy(dtype=float)
e = -d
n = len(x)
LG.info("读入 附件1_clean（utf-8-sig）：%d 行；e=−索赔差额 min=%.2f max=%.2f", n, e.min(), e.max())

B_STAR, T1_STAR, T2_STAR = 5, 0.86, 0.98
edges_s, bins_s, g1_s, g2_s, _meta = a6_01.build_boundaries(x, e, B_STAR, T1_STAR, T2_STAR)
lab_star = a6_01.label_rows(e, bins_s, g1_s, g2_s)
n_star = [int((lab_star == c).sum()) for c in (0, 1, 2)]
LG.info("θ* 冻结复算（溯源校验）：n_合=%d n_偏=%d n_严=%d（应=9602/1340/225，q1_定稿配置.json）", *n_star)
assert n_star == [9602, 1340, 225], "θ* 冻结标签复算与 A6 产物不一致，停步"

# ---- θ* 的 bootstrap（与 a6_01 同式同种子：复核翻转率中位）----
rng = np.random.default_rng(SEED)
g1_all, g2_all, flips_star, s_all = a6_01.bootstrap_run(x, e, B_STAR, T1_STAR, T2_STAR, lab_star, None, rng, NDRAW)
flip_med_star = float(np.nanmedian(flips_star))
LG.info("复核 a6_01 V6②：θ* bootstrap 翻转率中位=%.6f（a6_01 日志行21=0.010298）", flip_med_star)

# ---- 逐行画像：重放同一 bootstrap 序列，逐次保留边界，作用于全部行 ----
rng2 = np.random.default_rng(SEED)
flip_cnt = np.zeros(n)          # 全量口径：该行在多少次边界重估下标签改变
trans = np.zeros((3, 3))        # 冻结标签 → 重估标签（池化，每次对全部行累计）
inbag_flip = 0                  # in-bag 口径翻转总次数（与 a6_01 flips[it] 同定义，复核用）
inbag_total = 0
valid_draws = 0
g1_all2 = np.full((NDRAW, B_STAR), np.nan)
g2_all2 = np.full((NDRAW, B_STAR), np.nan)
for it in range(NDRAW):
    idx = rng2.integers(0, n, n)
    xb, eb = x[idx], e[idx]
    ed = np.quantile(xb, np.arange(1, B_STAR) / B_STAR)
    if len(np.unique(ed)) != B_STAR - 1:
        continue
    bins_d = a6_01.bin_of(xb, ed)
    q1 = a6_01.bin_quantiles(eb, bins_d, B_STAR, T1_STAR)
    q2 = a6_01.bin_quantiles(eb, bins_d, B_STAR, T2_STAR)
    ok = ~np.isnan(q1)
    if ok.sum() == 0:
        continue
    g2d = a6_01.pava(q2[ok])
    g1d = np.minimum(a6_01.pava(q1[ok]), g2d)
    g1_all2[it, ok] = g1d
    g2_all2[it, ok] = g2d
    valid_draws += 1
    # 全量口径：该次边界作用于全部行
    bins_full = a6_01.bin_of(x, ed)
    lab_d = a6_01.label_rows(e, bins_full, g1_all2[it], g2_all2[it])
    diff = lab_d != lab_star
    flip_cnt += diff
    for c0 in range(3):
        for c1 in range(3):
            trans[c0, c1] += int(((lab_star == c0) & (lab_d == c1)).sum())
    # in-bag 口径（复核）
    lab_b = a6_01.label_rows(eb, bins_d, g1_all2[it], g2_all2[it])
    inbag_flip += int((lab_b != lab_star[idx]).sum())
    inbag_total += len(idx)
flip_freq_full = flip_cnt / valid_draws
LG.info("重放完成：有效 %d/%d 次；全量口径平均翻转频率=%.6f；in-bag 口径池化翻转率=%.6f（a6_01 中位=%.6f）",
        valid_draws, NDRAW, flip_freq_full.mean(), inbag_flip / inbag_total, flip_med_star)

# ---- 距离带画像：到最近适用边界的相对距离 ----
d1 = np.abs(e - g1_s[bins_s]) / g1_s[bins_s]
d2 = np.abs(e - g2_s[bins_s]) / g2_s[bins_s]
near_g1 = d1 <= d2
dist_near = np.where(near_g1, d1, d2)
band_edges = [0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, np.inf]
band_names = ["[0,0.5%)", "[0.5%,1%)", "[1%,2%)", "[2%,5%)", "[5%,10%)", "[10%,20%)", "[20%,50%)", ">=50%"]
prof_rows = []
for lo, hi, nm in zip(band_edges[:-1], band_edges[1:], band_names):
    m = (dist_near >= lo) & (dist_near < hi)
    if m.sum() == 0:
        continue
    prof_rows.append({"块": "距离带", "键": nm, "行数": int(m.sum()),
                      "行占比": float(m.mean()),
                      "行均翻转频率_全量口径": float(flip_freq_full[m].mean()),
                      "翻转质量占比_全量口径": float(flip_cnt[m].sum() / flip_cnt.sum())})
m2 = dist_near < 0.02
prof_rows.append({"块": "距离带", "键": "距离<2%中贴g1(偏高界)", "行数": int((m2 & near_g1).sum()),
                  "行占比": float((m2 & near_g1).mean()),
                  "行均翻转频率_全量口径": float(flip_freq_full[m2 & near_g1].mean()) if (m2 & near_g1).any() else np.nan,
                  "翻转质量占比_全量口径": float(flip_cnt[m2 & near_g1].sum() / flip_cnt.sum())})
prof_rows.append({"块": "距离带", "键": "距离<2%中贴g2(严重界)", "行数": int((m2 & ~near_g1).sum()),
                  "行占比": float((m2 & ~near_g1).mean()),
                  "行均翻转频率_全量口径": float(flip_freq_full[m2 & ~near_g1].mean()) if (m2 & ~near_g1).any() else np.nan,
                  "翻转质量占比_全量口径": float(flip_cnt[m2 & ~near_g1].sum() / flip_cnt.sum())})

# ---- 转移方向（池化，全量口径）----
LABN = {0: "合理诉求", 1: "诉求偏高", 2: "严重超额"}
tot_flip_trans = trans.sum() - np.trace(trans)
for c0 in range(3):
    for c1 in range(3):
        if c0 == c1:
            continue
        prof_rows.append({"块": "转移方向", "键": f"{LABN[c0]}→{LABN[c1]}", "行数": int(trans[c0, c1]),
                          "行占比": np.nan,
                          "行均翻转频率_全量口径": np.nan,
                          "翻转质量占比_全量口径": float(trans[c0, c1] / tot_flip_trans)})

# ---- 边界 bootstrap 95% CI（V6① 同式，供风险集与论文引用）----
ci_rows = []
for b in range(B_STAR):
    for j, ga in ((1, g1_all2), (2, g2_all2)):
        v = ga[:, b]
        v = v[~np.isnan(v)]
        eb_b = e[bins_s == b]
        iqr_b = float(np.percentile(eb_b, 75) - np.percentile(eb_b, 25))
        lo, hi = np.quantile(v, [0.025, 0.975])
        ci_rows.append({"块": "边界CI", "键": f"箱{b+1}_g{j}", "行数": int((bins_s == b).sum()),
                        "行占比": np.nan,
                        "行均翻转频率_全量口径": np.nan,
                        "翻转质量占比_全量口径": np.nan,
                        "CI2.5": lo, "CI97.5": hi, "相对宽度": (hi - lo) / iqr_b})

# ---- 风险集与集中度 ----
risk = flip_freq_full > 0
risk5 = flip_freq_full >= 0.05
top10 = np.sort(flip_freq_full)[::-1][n // 10 - 1]  # 按行秩取前10%阈值（频率零膨胀，不能用分位数）
m_top = flip_freq_full >= top10
prof_rows.append({"块": "风险集", "键": "任一次重估即翻转的行(频率>0)", "行数": int(risk.sum()),
                  "行占比": float(risk.mean()),
                  "行均翻转频率_全量口径": float(flip_freq_full[risk].mean()) if risk.any() else np.nan,
                  "翻转质量占比_全量口径": np.nan})
prof_rows.append({"块": "风险集", "键": "高频翻转行(频率>=5%)", "行数": int(risk5.sum()),
                  "行占比": float(risk5.mean()),
                  "行均翻转频率_全量口径": float(flip_freq_full[risk5].mean()) if risk5.any() else np.nan,
                  "翻转质量占比_全量口径": float(flip_cnt[risk5].sum() / flip_cnt.sum())})
prof_rows.append({"块": "风险集", "键": "翻转频率按行秩前10%行", "行数": int(m_top.sum()),
                  "行占比": float(m_top.mean()), "行均翻转频率_全量口径": float(flip_freq_full[m_top].mean()),
                  "翻转质量占比_全量口径": float(flip_cnt[m_top].sum() / flip_cnt.sum())})
prof_rows.append({"块": "复核", "键": "a6_01 V6② 翻转率中位（in-bag）", "行数": n, "行占比": flip_med_star,
                  "行均翻转频率_全量口径": float(flip_freq_full.mean()),
                  "翻转质量占比_全量口径": np.nan})

# ---- 邻域网格 bootstrap ----
grid_rows = []
tau_cells = [(t1, t2) for t1 in (0.85, 0.86, 0.87) for t2 in (0.975, 0.98, 0.985)]
b_cells = [(B, 0.86, 0.98) for B in (10, 15, 20)]
cells = [("τ邻域", B_STAR, t1, t2) for (t1, t2) in tau_cells] + \
        [("B邻域", B, t1, t2) for (B, t1, t2) in b_cells]
for tag, B, t1, t2 in cells:
    res = a6_01.build_boundaries(x, e, B, t1, t2)
    if res is None:
        continue
    _, bins_c, g1_c, g2_c, _m = res
    lab_c = a6_01.label_rows(e, bins_c, g1_c, g2_c)
    rng_c = np.random.default_rng(SEED)
    _, _, fl_own, _ = a6_01.bootstrap_run(x, e, B, t1, t2, lab_c, None, rng_c, NDRAW)
    rng_v = np.random.default_rng(SEED)
    _, _, fl_v, _ = a6_01.bootstrap_run(x, e, B, t1, t2, lab_star, None, rng_v, NDRAW)
    grid_rows.append({
        "网格": tag, "B": B, "tau1": t1, "tau2": t2, "draws": NDRAW,
        "flip_own_中位": float(np.nanmedian(fl_own)),
        "flip_own_P25": float(np.nanpercentile(fl_own, 25)),
        "flip_own_P75": float(np.nanpercentile(fl_own, 75)),
        "flip_vsθ星_中位": float(np.nanmedian(fl_v)),
        "flip_vsθ星_P25": float(np.nanpercentile(fl_v, 25)),
        "flip_vsθ星_P75": float(np.nanpercentile(fl_v, 75)),
        "确定性重标差异率": float((lab_c != lab_star).mean()),
        "是否θ星": (B == B_STAR and t1 == T1_STAR and t2 == T2_STAR),
    })
    LG.info("格 %s B=%d τ=(%.3f,%.3f)：flip_own中位=%.6f  flip_vsθ*中位=%.6f  确定性重标差异=%.6f",
            tag, B, t1, t2, grid_rows[-1]["flip_own_中位"], grid_rows[-1]["flip_vsθ星_中位"],
            grid_rows[-1]["确定性重标差异率"])

grid_df = pd.DataFrame(grid_rows)
prof_df = pd.DataFrame(prof_rows)
ci_df = pd.DataFrame(ci_rows)
grid_df.to_csv(TBL / "a7_翻转带敏感性.csv", index=False, encoding="utf-8-sig")
prof_df.to_csv(TBL / "a7_翻转带画像.csv", index=False, encoding="utf-8-sig")
ci_df.to_csv(TBL / "a7_翻转带画像_边界CI.csv", index=False, encoding="utf-8-sig")
grid_df.to_csv(MIRROR / "a7_翻转带敏感性.csv", index=False, encoding="utf-8-sig")
prof_df.to_csv(MIRROR / "a7_翻转带画像.csv", index=False, encoding="utf-8-sig")
ci_df.to_csv(MIRROR / "a7_翻转带画像_边界CI.csv", index=False, encoding="utf-8-sig")
LG.info("输出 a7_翻转带敏感性.csv（%d 格）、a7_翻转带画像.csv（%d 行）、a7_翻转带画像_边界CI.csv（%d 行）+ sensitivity_a7 镜像",
        len(grid_df), len(prof_df), len(ci_df))
LG.info("=== a7_01 完成，耗时 %.1f 秒 ===", time.time() - T0)
