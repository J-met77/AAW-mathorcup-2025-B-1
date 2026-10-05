# -*- coding: utf-8 -*-
"""
a7_05_稳健性汇总.py —— A7 实验5：三问稳健性汇总（任务书第 6 项，≤5 分钟）
内容：
  1) 汇总 A6 已有 bootstrap CI 与 3 种子极差（Q1 V6 三项、Q2 WAPE、Q3 宏F1/严重类 recall），
     逐项注明溯源（文件+行）。
  2) 补充：严重类 recall 的逐种子 bootstrap CI（二项自助：n=225 严重行内 k 次正确重抽 B=2000，
     百分位法；种子 20251004；主种子另有 A6 行级池化 bootstrap CI 为权威口径，本表为轻量逐种子补充）。
     同法补 方式1 严重类 recall 与 行序口径（n=44）的参考区间。
输出：output/tables/a7_稳健性汇总.csv（镜像 output/tables/sensitivity_a7/）
日志：output/logs/a7_05_稳健性汇总.log
红线：全程离线；只读 A6/前任 A7 产物；不修改任何既有产物。
"""
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

from a6_common import get_logger

LG = get_logger("a7_05", LOGD / "a7_05_稳健性汇总.log")
T0 = time.time()
B_BOOT, RNG_SEED = 2000, 20251004

LG.info("=== a7_05_稳健性汇总 开始 ===")
LG.info("环境：numpy=%s pandas=%s；二项自助 B=%d，种子 %d", np.__version__, pd.__version__, B_BOOT, RNG_SEED)


def binom_boot_ci(k: int, n: int, b: int = B_BOOT, seed: int = RNG_SEED):
    """二项自助：n 个严重行中 k 行判对，重抽 b 次的 recall 百分位 CI（轻量口径，忽略折间相关）。"""
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, k / n, size=b) / n
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


rows = []


def add(问题, 项目, 口径, 数值, 区间或极差, 溯源, 判定=""):
    rows.append({"问题": 问题, "项目": 项目, "口径": 口径, "数值": 数值,
                 "区间或极差": 区间或极差, "溯源": 溯源, "判定": 判定})


# ---------------- Q1（V1–V8 汇总，重点 V6） ----------------
v = pd.read_csv(TBL / "q1_自检表.csv", encoding="utf-8-sig")
v6 = v[v["编号"] == "V6"].iloc[0]
add("Q1", "V6① 边界CI相对宽度中位", "g1 / g2（bootstrap B=1000，95%，约定）",
    f"{v6['数值']}".split("；")[0],
    "门槛中段箱 ≤1.0", "q1_自检表.csv 行7（V6）", "PASS")
ci = pd.read_csv(TBL / "q1_边界_bootstrapCI.csv", encoding="utf-8-sig")
add("Q1", "V6① 逐箱相对宽度范围", "g1: 箱1~5；g2: 箱1~5",
    f"g1∈[{ci[ci['边界j']==1]['相对宽度'].min():.4f},{ci[ci['边界j']==1]['相对宽度'].max():.4f}]",
    f"g2∈[{ci[ci['边界j']==2]['相对宽度'].min():.4f},{ci[ci['边界j']==2]['相对宽度'].max():.4f}]",
    "q1_边界_bootstrapCI.csv 全表", "g2 箱1 最宽 0.8102（严重界小计数箱）")
add("Q1", "V6② bootstrap 标签翻转率中位", "θ*=(5,0.86,0.98)，in-bag 口径",
    "0.010298", "约定门槛 ≤1%（相对超出 3.0%，D23 如实报告）",
    "q1_自检表.csv 行7（V6）；a6_01 日志行21", "FAIL（超限，D23 维持冻结）")
sen = pd.read_csv(TBL / "a7_翻转带敏感性.csv", encoding="utf-8-sig")
tau_own = sen[sen["网格"] == "τ邻域"]["flip_own_中位"]
b_own = sen[sen["网格"] == "B邻域"]["flip_own_中位"]
add("Q1", "V6② τ 邻域翻转率范围（own 口径）", "B=5，τ1∈{0.85,0.86,0.87}×τ2∈{0.975,0.98,0.985}，每格 1000 次",
    f"[{tau_own.min():.6f},{tau_own.max():.6f}]",
    "9 格全部 ≤0.0108（与 θ* 的 0.0103 同量级）→ 非择优偶然，属边界固有不确定度",
    "a7_翻转带敏感性.csv 行2-10（a7_01）", "佐证 D23 预案甲")
add("Q1", "V6② B 邻域翻转率范围（own 口径）", "τ=(0.86,0.98)，B∈{10,15,20}",
    f"[{b_own.min():.6f},{b_own.max():.6f}]",
    "B 增大翻转率上升（箱变小、边界变密）→ θ*.B=5 为稳健性最优端",
    "a7_翻转带敏感性.csv 行11-13（a7_01）", "")
add("Q1", "V6③ 3种子×5折 s_c 极差最大", "类占比 s_c", "0.0051", "门槛 ≤1pp",
    "q1_自检表.csv 行7（V6）", "PASS")
add("Q1", "占比（V1，软先验）", "s_合/s_偏/s_严", "0.8599/0.1200/0.0201",
    "偏离 C2 带：Δ合=+0.0099、Δ严=-0.0099（如实报告）", "q1_定稿配置.json labels_props", "PASS")

# ---------------- Q2（3 种子 + bootstrap CI） ----------------
q2 = pd.read_csv(TBL / "q2_指标汇总.csv", encoding="utf-8-sig")


def w3(model, kou="分层5折"):
    r = q2[(q2["模型"] == model) & (q2["口径"] == kou)].sort_values("种子")["WAPE"].to_numpy(float)
    return r


for model, note in (("LGBM_变体A", "定稿变体"), ("ElasticNet", "线性族对照"), ("OLS", "线性族对照"),
                    ("B3_单参数比", "基线"), ("B4_分段比", "基线"), ("LGBM_变体B", "Huber 淘汰变体")):
    r = w3(model)
    ci_row = q2[(q2["模型"] == model) & (q2["口径"] == "分层5折") & (q2["种子"] == 20251004)]
    ci_s = f"[{float(ci_row['WAPE_CI2.5'].iloc[0]):.4f},{float(ci_row['WAPE_CI97.5'].iloc[0]):.4f}]" \
        if not np.isnan(float(ci_row["WAPE_CI2.5"].iloc[0])) else "—"
    add("Q2", f"{model} 3种子分层WAPE", "20251004/05/06 池化 OOF",
        f"均值={r.mean():.4f}", f"范围[{r.min():.4f},{r.max():.4f}]，极差={r.max()-r.min():.4f}；主种子95%CI {ci_s}",
        "q2_指标汇总.csv（模型行）", note)
r_ro = q2[(q2["模型"] == "LGBM_变体A") & (q2["口径"] == "行序外推")].iloc[0]
add("Q2", "LGBM_变体A 行序外推 WAPE", "前 80% 训练 / 后 20% 验证（U3/D19）",
    f"{float(r_ro['WAPE']):.4f}", f"95%CI [{float(r_ro['WAPE_CI2.5']):.4f},{float(r_ro['WAPE_CI97.5']):.4f}]",
    "q2_指标汇总.csv LGBM_变体A/行序外推 行", "低于分层口径，行序更苛未见恶化")
add("Q2", "行序口径线性族对照", "OLS/EN/LGBM", "OLS=0.3334 / EN=0.3341 / LGBM=0.3357",
    "行序下 OLS 略优 0.0023（如实报告）", "q2_指标汇总.csv 行序外推行", "如实报告项")

# ---------------- Q3（3 种子 + bootstrap CI + 严重类 recall 补充 CI） ----------------
q3 = pd.read_csv(TBL / "q3_指标汇总.csv", encoding="utf-8-sig")
main = q3[(q3["配置"] == "方式2_主配置(嵌套γ,s)")]
f1_3 = main[main["口径"] == "分层5折"].sort_values("种子")["宏F1"].to_numpy(float)
f1_row = main[main["口径"] == "行序外推"].iloc[0]
add("Q3", "方式2 主配置 3种子分层宏F1", "20251004/05/06 池化 OOF",
    f"均值={f1_3.mean():.4f}", f"范围[{f1_3.min():.4f},{f1_3.max():.4f}]，极差={f1_3.max()-f1_3.min():.4f}",
    "q3_指标汇总.csv 方式2主配置/分层5折 行", "极差 0.0183 主要来自严重类小样本波动")
add("Q3", "方式2 主种子宏F1 bootstrap CI", "行级重抽 B=1000（A6 已有）",
    f"{float(f1_3[0]):.4f}", f"[{float(main.iloc[0]['宏F1_CI2.5']):.4f},{float(main.iloc[0]['宏F1_CI97.5']):.4f}]",
    "q3_指标汇总.csv 主种子行 宏F1_CI 列", "3 种子极差 0.0183 与 CI 宽 0.046 同量级 → 种子不改变结论")
add("Q3", "方式2 行序外推宏F1", "前 80%/后 20%", f"{float(f1_row['宏F1']):.4f}",
    f"[{float(f1_row['宏F1_CI2.5']):.4f},{float(f1_row['宏F1_CI97.5']):.4f}]",
    "q3_指标汇总.csv 行序行", "与分层 0.6704 一致（D19 双口径一致）")
# 严重类 recall：A6 已有行级 CI（主种子/行序）+ 本次逐种子二项自助补充
m04 = main[main["种子"] == 20251004].iloc[0]
add("Q3", "严重类 recall 行级 bootstrap CI（权威）", "分层5折 主种子（A6 已有）",
    f"{float(m04['严重类_recall']):.4f}",
    f"[{float(m04['严重类recall_CI2.5']):.4f},{float(m04['严重类recall_CI97.5']):.4f}]",
    "q3_指标汇总.csv 主种子行 严重类recall_CI 列", "n_严=225，CI 宽 0.123")
add("Q3", "严重类 recall 行级 bootstrap CI（权威）", "行序外推（A6 已有）",
    f"{float(f1_row['严重类_recall']):.4f}",
    f"[{float(f1_row['严重类recall_CI2.5']):.4f},{float(f1_row['严重类recall_CI97.5']):.4f}]",
    "q3_指标汇总.csv 行序行 严重类recall_CI 列", "n_严=44，CI 宽 0.293（极宽）")
rec_seeds = main[main["口径"] == "分层5折"].sort_values("种子")
for _, r in rec_seeds.iterrows():
    n_sev = int(r["类2_support"])
    k = int(round(float(r["严重类_recall"]) * n_sev))
    lo, hi = binom_boot_ci(k, n_sev)
    add("Q3", "严重类 recall 二项自助 CI（补充）", f"分层5折 种子={int(r['种子'])}；n={n_sev}, k={k}",
        f"{float(r['严重类_recall']):.4f}", f"[{lo:.4f},{hi:.4f}]",
        "a7_05 二项自助 B=2000（严重行内重抽，忽略折间相关，轻量口径）；recall 溯源 q3_指标汇总.csv",
        "主种子以 A6 行级 CI 为权威")
rt = pd.read_csv(TBL / "q3_两路线对比.csv", encoding="utf-8-sig")
r1 = rt[(rt["配置"] == "方式1_完整链路") & (rt["口径"] == "分层5折") & (rt["种子"] == 20251004)].iloc[0]
k1, n1 = int(round(float(r1["严重类_recall"]) * int(r1["类2_support"]))), int(r1["类2_support"])
lo1, hi1 = binom_boot_ci(k1, n1)
add("Q3", "方式1 严重类 recall 二项自助 CI（补充）", f"分层5折 主种子；n={n1}, k={k1}",
    f"{float(r1['严重类_recall']):.4f}", f"[{lo1:.4f},{hi1:.4f}]",
    "a7_05 二项自助 B=2000；recall 溯源 q3_指标汇总.csv 方式1行", "方式1 严重类 recall 显著低于方式2")
ab = pd.read_csv(TBL / "q3_消融矩阵.csv", encoding="utf-8-sig")
a0_s = float(ab[(ab["配置"] == "消融A0") & (ab["口径"] == "分层5折")]["宏F1"].iloc[0])
a0_r = float(ab[(ab["配置"] == "消融A0") & (ab["口径"] == "行序外推")]["宏F1"].iloc[0])
add("Q3", "杠杆收益双口径（D24 限定）", "主配置−消融A0", f"分层 +{0.6703649106061654-a0_s:.4f} / 行序 {0.6709632075571341-a0_r:+.4f}",
    f"A0：分层 {a0_s:.4f} / 行序 {a0_r:.4f}（方向反转）", "q3_消融矩阵.csv 消融A0 行；a7_双口径分解.csv",
    "杠杆提升宏F1 的结论仅限分层口径（D24）")
add("Q3", "概率标定（D25 局限）", "主配置 vs 消融A0 的 OOF log loss", "0.2386 vs 0.2208",
    "M3-4 均衡权重的结构性失准（R-07）", "q3_指标汇总.csv/q3_消融矩阵.csv OOF_logloss 列", "记录为局限")
add("Q3", "两路线一致率 / 边界带翻转率", "分层5折 主种子池化", "一致率 0.9756；边界带(δ=0.1)248 行翻转率 0.4073",
    "δ 敏感性与机制量化见 a7_04", "q3_两路线对比.csv 行6；a7_q3误差传导.csv", "")

# ---------------- 附件2（预测占比，无抽样 CI） ----------------
add("附件2", "方式2 预测占比", "合/偏/严 = 2518/254/20", "0.9019/0.0910/0.0072",
    "严重 20 行支撑极小；为确定性预测无抽样 CI，不确定性由 Q3 模型 CI 传导",
    "附件2_风险预测.csv value_counts；q3_两路线对比.csv 行10", "严重 0.72% 低于 C2 带上限（局限清单项）")
add("附件2", "赔付预测范围", "min/max", "8.12 / 1455.73 元", "无保价截断（T9）",
    "附件2_赔付预测.csv；q2_定稿配置.json 执行.预测统计", "")

out = pd.DataFrame(rows)
out.to_csv(TBL / "a7_稳健性汇总.csv", index=False, encoding="utf-8-sig")
out.to_csv(MIRROR / "a7_稳健性汇总.csv", index=False, encoding="utf-8-sig")
LG.info("输出 a7_稳健性汇总.csv（%d 行）+ 镜像", len(out))
LG.info("=== a7_05 完成，耗时 %.1f 秒 ===", time.time() - T0)
