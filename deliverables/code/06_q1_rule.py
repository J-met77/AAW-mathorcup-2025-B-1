# -*- coding: utf-8 -*-
"""06_q1_rule.py —— A6 实现 E1：问题1 风险标注规则估计与择优（A5 §1 规格）。
算法：
  1) p 等频 K=20 分箱 → 箱内 u 分位数（τ1, τ2）
  2) M1: PAVA 保序 + 线性插值/端点常数外推
     M2: T(p)=α+β·p^γ（α≥0 约束），γ∈{1,1.2,1.5,2} 加权LS
  3) (τ1,τ2)×形态 网格 → S1-S4 校验 → 词典序择优 → 冻结 rule_final.json
输出：
  output/tables/q1_rule_grid.csv        网格校验表
  output/tables/rule_final.json         冻结规则
  output/tables/q1_labels_附件1.csv     全量标注
  output/tables/q1_class_profile.csv    三类画像（占比对照/差额分布/密度）
  output/logs/06_q1_rule.log
"""
import sys, io, json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TAB = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\tables"
LOG = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\06_q1_rule.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

log("=== 06_q1_rule start ===")
df = pd.read_csv(rf"{TAB}\附件1_clean.csv")
df['超额u'] = df['索赔金额'] - df['实际赔付金额']
p = df['实际赔付金额'].values.astype(float)
u = df['超额u'].values.astype(float)
log(f"n={len(df)}  p∈[{p.min():.2f},{p.max():.2f}]  u∈[{u.min():.2f},{u.max():.2f}]")

K = 20
pbin = pd.qcut(p, K, labels=False, duplicates='drop')
K = int(pbin.max()) + 1
nodes_p = np.array([np.median(p[pbin == k]) for k in range(K)])
counts = np.array([(pbin == k).sum() for k in range(K)])
log(f"K={K} 等频箱，箱中位赔付=[{nodes_p[0]:.2f} ... {nodes_p[-1]:.2f}]，箱规模 min={counts.min()}")

def pava(x, w):
    """加权 PAVA，返回非降拟合。"""
    x = np.asarray(x, float); w = np.asarray(w, float)
    val = x.copy(); wt = w.copy(); idx = list(range(len(x)))
    blocks = [[v, t, [i]] for v, t, i in zip(val, wt, idx)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i+1][0]:
            v = (blocks[i][0]*blocks[i][1] + blocks[i+1][0]*blocks[i+1][1]) / (blocks[i][1]+blocks[i+1][1])
            t = blocks[i][1] + blocks[i+1][1]
            mem = blocks[i][2] + blocks[i+1][2]
            blocks[i:i+2] = [[v, t, mem]]
            i = max(i-1, 0)
        else:
            i += 1
    out = np.empty(len(x))
    for v, t, mem in blocks:
        out[mem] = v
    return out

def m1_nodes(tau):
    q = np.array([np.quantile(u[pbin == k], tau) for k in range(K)])
    return pava(q, counts.astype(float))

def m2_fit(tau, gamma):
    q = np.array([np.quantile(u[pbin == k], tau) for k in range(K)])
    X = np.column_stack([np.ones(K), nodes_p ** gamma])
    beta, *_ = np.linalg.lstsq(X, q, rcond=None)
    if beta[0] < 0:  # α≥0 约束：过原点比例形态
        X0 = (nodes_p ** gamma).reshape(-1, 1)
        b0, *_ = np.linalg.lstsq(X0, q, rcond=None)
        beta = np.array([0.0, b0[0]])
    return beta  # (α, β)

def eval_rule(T1, T2, name):
    def label_fn(pp, uu):
        return np.where(uu <= T1(pp), 0, np.where(uu <= T2(pp), 1, 2)).astype(int)
    y = label_fn(p, u)
    prop = np.array([(y == k).mean() for k in range(3)]) * 100
    # S2 单调性网格扫描
    grid = np.linspace(p.min(), p.max(), 200)
    T1g = T1(grid); T2g = T2(grid)
    mono_viol = int(((np.diff(T1g) < -1e-12).sum() + (np.diff(T2g) < -1e-12).sum())
                    + (T2g <= T1g + 1e-12).sum())
    # S3-v3 类内紧凑（区间占比）：赔付十分位内，类样本 u 极差 / 箱全体 u 极差
    dec = pd.qcut(p, 10, labels=False, duplicates='drop')
    compact = {0: [], 1: [], 2: []}
    for d in range(10):
        sel_d = dec == d
        pool_rng = u[sel_d].max() - u[sel_d].min()
        if pool_rng <= 0:
            continue
        for k in range(3):
            sel = sel_d & (y == k)
            if sel.sum() < 10:
                continue
            uk = u[sel]
            compact[k].append((uk.max() - uk.min()) / pool_rng)
    comp_mean = {k: float(np.mean(compact[k])) if compact[k] else np.nan for k in range(3)}
    s3_ok = all(v < 1 for v in comp_mean.values() if not np.isnan(v))
    # S4-v2 密度对比：每赔付十分位内，全体u做KDE，取类样本处密度均值，再对箱平均
    def kde_at(samples, ref):
        n = len(ref); sd = ref.std(ddof=1)
        if sd <= 0 or n < 2:
            return np.nan
        h = 0.9 * sd * n ** (-0.2)
        sub = ref if n <= 4000 else np.random.default_rng(2025).choice(ref, 4000, replace=False)
        d_i = np.exp(-0.5 * ((samples[:, None] - sub[None, :]) / h) ** 2).sum(1) / (len(sub) * h * np.sqrt(2 * np.pi))
        return float(d_i.mean())
    dens = {0: [], 1: [], 2: []}
    for d in range(10):
        sel_d = dec == d
        ud = u[sel_d]
        for k in range(3):
            sel = sel_d & (y == k)
            if sel.sum() >= 10:
                dens[k].append(kde_at(u[sel], ud))
    dens_mean = {k: float(np.mean([v for v in dens[k] if not np.isnan(v)])) for k in range(3)}
    return dict(name=name, pct_合理=prop[0], pct_偏高=prop[1], pct_严重=prop[2],
                mono_viol=mono_viol,
                compact_r=comp_mean[0], compact_p=comp_mean[1], compact_s=comp_mean[2],
                s3_ok=bool(s3_ok),
                dens_r=dens_mean[0], dens_p=dens_mean[1], dens_s=dens_mean[2],
                dens_ratio_rs=dens_mean[0] / max(dens_mean[2], 1e-12),
                dens_ratio_rp=dens_mean[0] / max(dens_mean[1], 1e-12)), y

rows = []
label_store = {}
TAUS1 = [0.85, 0.86, 0.88, 0.90]
TAUS2 = [0.97, 0.98, 0.99]
for tau1 in TAUS1:
    for tau2 in TAUS2:
        # M1
        n1 = m1_nodes(tau1); n2 = m1_nodes(tau2)
        T1 = lambda pp, n=n1: np.interp(pp, nodes_p, n)
        T2 = lambda pp, n=n2: np.interp(pp, nodes_p, n)
        r, y = eval_rule(T1, T2, f"M1|τ1={tau1},τ2={tau2}")
        r.update(form="M1", tau1=tau1, tau2=tau2)
        rows.append(r); label_store[r['name']] = y
        # M2 各 γ
        for gamma in [1.0, 1.2, 1.5, 2.0]:
            b1 = m2_fit(tau1, gamma); b2 = m2_fit(tau2, gamma)
            T1 = lambda pp, b=b1: b[0] + b[1]*np.power(np.maximum(pp, 1e-9), gamma)
            T2 = lambda pp, b=b2: b[0] + b[1]*np.power(np.maximum(pp, 1e-9), gamma)
            r, y = eval_rule(T1, T2, f"M2(γ={gamma})|τ1={tau1},τ2={tau2}")
            r.update(form=f"M2(g={gamma})", tau1=tau1, tau2=tau2)
            rows.append(r); label_store[r['name']] = y

grid = pd.DataFrame(rows)
log("\n--- S1-S4(v3) 网格校验表（节选：S1 合规[合理>=85.5%,严重∈[1.5,3)] 且 S2 零违例 且 S3-v3 成立 且 密度比>1.1）---")
ok = grid[(grid.pct_合理 >= 85.5) & (grid.pct_合理 <= 90) & (grid.pct_严重 < 3) & (grid.pct_严重 >= 1.5)
          & (grid.mono_viol == 0) & (grid.s3_ok) & (grid.dens_ratio_rs > 1.1)]
log(ok.to_string(index=False))
grid.to_csv(rf"{TAB}\q1_rule_grid.csv", index=False, encoding='utf-8-sig')

# 词典序择优：S1合规+S2零违例+S3有序 中 dens_ratio_rs 最大；并列(<5%)取 M2 简洁形态
cand = ok.copy()
assert len(cand) > 0, "无满足 S1-S3 的规则配置，需回退（记录 decisions）"
best_d = cand.dens_ratio_rs.max()
final_set = cand[cand.dens_ratio_rs >= best_d * 0.95]
prefer_m2 = final_set[final_set.form.str.startswith("M2")]
pick = (prefer_m2 if len(prefer_m2) else final_set).sort_values('dens_ratio_rs', ascending=False).iloc[0]
log(f"\n>>> 冻结规则：{pick['form']}  τ1={pick.tau1}, τ2={pick.tau2}  "
    f"占比(合理/偏高/严重)={pick.pct_合理:.2f}/{pick.pct_偏高:.2f}/{pick.pct_严重:.2f}%  "
    f"密度比(合理/严重)={pick.dens_ratio_rs:.1f}")

tau1, tau2 = float(pick.tau1), float(pick.tau2)
if pick.form.startswith("M1"):
    n1 = m1_nodes(tau1); n2 = m1_nodes(tau2)
    rule = dict(form="M1", tau1=tau1, tau2=tau2, K=K, nodes_p=nodes_p.tolist(),
                T1_nodes=n1.tolist(), T2_nodes=n2.tolist())
else:
    gamma = float(pick.form.split("(g=")[1].split(")")[0])
    b1 = m2_fit(tau1, gamma); b2 = m2_fit(tau2, gamma)
    rule = dict(form="M2", gamma=gamma, tau1=tau1, tau2=tau2, alpha1=float(b1[0]), beta1=float(b1[1]),
                alpha2=float(b2[0]), beta2=float(b2[1]))

def apply_rule(pp, uu, rule):
    if rule["form"] == "M1":
        T1 = np.interp(pp, rule["nodes_p"], rule["T1_nodes"])
        T2 = np.interp(pp, rule["nodes_p"], rule["T2_nodes"])
    else:
        g = rule["gamma"]
        T1 = rule["alpha1"] + rule["beta1"]*np.power(np.maximum(pp,1e-9), g)
        T2 = rule["alpha2"] + rule["beta2"]*np.power(np.maximum(pp,1e-9), g)
    return np.where(uu <= T1, 0, np.where(uu <= T2, 1, 2)).astype(int)

y = apply_rule(p, u, rule)
df['风险标注'] = pd.Categorical.from_codes(y, categories=['合理诉求','诉求偏高','严重超额'])
prop = df['风险标注'].value_counts(normalize=True) * 100
log("\n--- 附件1 标注占比（题面软约束对照）---")
log(f"合理诉求: {prop['合理诉求']:.2f}%  (题面 ≥85%)")
log(f"诉求偏高: {prop['诉求偏高']:.2f}%")
log(f"严重超额: {prop['严重超额']:.2f}%  (题面 <3%)")

df[['实际赔付金额','索赔金额','超额u','风险标注']].to_csv(
    rf"{TAB}\q1_labels_附件1.csv", index=False, encoding='utf-8-sig')

# 三类画像
prof = df.groupby('风险标注', observed=True).agg(
    n=('超额u','size'), u_med=('超额u','median'), u_q25=('超额u', lambda s: s.quantile(.25)),
    u_q75=('超额u', lambda s: s.quantile(.75)), u_max=('超额u','max'),
    p_med=('实际赔付金额','median'), p_mean=('实际赔付金额','mean'))
prof['占比%'] = df['风险标注'].value_counts(normalize=True) * 100
log("\n--- 三类画像 ---")
log(prof.round(2).to_string())
prof.reset_index().to_csv(rf"{TAB}\q1_class_profile.csv", index=False, encoding='utf-8-sig')

with open(rf"{TAB}\rule_final.json", "w", encoding="utf-8") as f:
    json.dump(rule, f, ensure_ascii=False, indent=2)
log(f"[saved] {TAB}\\rule_final.json")
log("=== 06_q1_rule done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
