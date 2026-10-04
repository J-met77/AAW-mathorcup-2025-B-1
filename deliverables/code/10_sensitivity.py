# -*- coding: utf-8 -*-
"""10_sensitivity.py —— A7 输入 E5：敏感性实验。
 (a) 规则参数 τ1/τ2 扰动 → 三类占比、密度比（frozen 形态 M2 γ=1）
 (b) 标签扰动（3 组代表性 τ 配置）→ Q3 宏F1 稳定性（同折 StratifiedKFold，方案=过采样 0.5/0.5）
 (c) Q2 特征组消融（5折 MAE）：全量 / 无索赔金额特征 / 无网点行为 / 无频次编码 / 仅索赔金额
 (d) 参考：p̂ = 索赔金额 − 全局中位超额 的 MAE（claim−median(u) 基线）
输出：output/tables/sensitivity_tau.csv, sensitivity_q3_labels.csv, q2_ablation.csv, output/logs/10_sensitivity.log
"""
import sys, io, json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TAB = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\tables"
LOG = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\10_sensitivity.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))
SEED = 2025
CLS = ['合理诉求', '诉求偏高', '严重超额']

log("=== 10_sensitivity start ===")
df = pd.read_csv(rf"{TAB}\附件1_clean.csv")
p = df['实际赔付金额'].values.astype(float)
c = df['索赔金额'].values.astype(float)
u = c - p
cfg2 = json.load(open(rf"{TAB}\q2_model_config.json", encoding="utf-8"))

def rule_M2_linear(tau1, tau2):
    """frozen 形态（M2, γ=1）：对 u 重估 α+βp 线性边界。"""
    K = 20
    pbin = pd.qcut(p, K, labels=False, duplicates='drop')
    nodes_p = np.array([np.median(p[pbin == k]) for k in range(K)])
    counts = np.bincount(pbin, minlength=K).astype(float)
    out = {}
    for tau in (tau1, tau2):
        q = np.array([np.quantile(u[pbin == k], tau) for k in range(K)])
        X = np.column_stack([np.ones(K), nodes_p])
        beta, *_ = np.linalg.lstsq(X, q, rcond=None)
        if beta[0] < 0:
            b0, *_ = np.linalg.lstsq(nodes_p.reshape(-1, 1), q, rcond=None)
            beta = np.array([0.0, b0[0]])
        out[tau] = beta
    return out[tau1], out[tau2]

def kde_at(samples, ref):
    n = len(ref); sd = ref.std(ddof=1)
    if sd <= 0 or n < 2: return np.nan
    h = 0.9 * sd * n ** (-0.2)
    sub = ref if n <= 4000 else np.random.default_rng(SEED).choice(ref, 4000, replace=False)
    d = np.exp(-0.5*((samples[:, None]-sub[None, :])/h)**2).sum(1)/(len(sub)*h*np.sqrt(2*np.pi))
    return float(d.mean())

# (a) τ 扰动
rows = []
for tau1 in [0.85, 0.86, 0.88, 0.90]:
    for tau2 in [0.97, 0.98, 0.99]:
        (a1, b1), (a2, b2) = rule_M2_linear(tau1, tau2)
        T1 = a1 + b1*p; T2 = a2 + b2*p
        y = np.where(u <= T1, 0, np.where(u <= T2, 1, 2))
        dec = pd.qcut(p, 10, labels=False, duplicates='drop')
        dr = []
        for d in range(10):
            sel = dec == d
            dr.append(kde_at(u[sel][y[sel] == 0], u[sel]) / max(kde_at(u[sel][y[sel] == 2], u[sel]), 1e-12))
        rows.append(dict(tau1=tau1, tau2=tau2,
                         pct_合理=(y == 0).mean()*100, pct_偏高=(y == 1).mean()*100, pct_严重=(y == 2).mean()*100,
                         密度比_合理严重=float(np.mean(dr)), n_严重=int((y == 2).sum())))
sens = pd.DataFrame(rows)
log("--- (a) τ 扰动敏感性 ---")
log(sens.round(3).to_string(index=False))
sens.to_csv(rf"{TAB}\sensitivity_tau.csv", index=False, encoding='utf-8-sig')

# (b) 标签扰动 → Q3 稳定性（同折、过采样 0.5/0.5）
FEATS, CAT = cfg2['features'], cfg2['cat_features']
X = df[FEATS].copy()
for cc in CAT:
    X[cc] = X[cc].astype('category')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
q3_rows = []
for tau1, tau2 in [(0.86, 0.98), (0.85, 0.97), (0.88, 0.98)]:
    (a1, b1), (a2, b2) = rule_M2_linear(tau1, tau2)
    T1 = a1 + b1*p; T2 = a2 + b2*p
    y = np.where(u <= T1, 0, np.where(u <= T2, 1, 2))
    f1s = []
    for tri, vai in skf.split(X, y):
        rng = np.random.default_rng(SEED)
        cnt = np.bincount(y[tri], minlength=3); n0 = cnt[0]
        idx_all = [np.where(y[tri] == k)[0] for k in range(3)]
        take = [idx_all[0]]
        for k, ratio in [(1, .5), (2, .5)]:
            need = int(round(n0*ratio)) - cnt[k]
            take.append(np.concatenate([idx_all[k], rng.choice(idx_all[k], need, replace=True)]) if need > 0 else idx_all[k])
        idx = np.concatenate(take); rng.shuffle(idx)
        clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, n_estimators=300, random_state=SEED,
                                 deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)
        clf.fit(X.iloc[tri].iloc[idx].reset_index(drop=True), y[tri][idx])
        f1s.append(f1_score(y[vai], clf.predict(X.iloc[vai]), average='macro'))
    q3_rows.append(dict(tau1=tau1, tau2=tau2, pct_严重=(y == 2).mean()*100,
                        macroF1_mean=float(np.mean(f1s)), macroF1_std=float(np.std(f1s))))
    log(f"(b) τ=({tau1},{tau2}) 严重占比={(y==2).mean()*100:.2f}%  Q3宏F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
pd.DataFrame(q3_rows).to_csv(rf"{TAB}\sensitivity_q3_labels.csv", index=False, encoding='utf-8-sig')

# (c) Q2 特征组消融（5折 KFold）
groups = {
 '全量': FEATS,
 '无索赔金额特征': [f for f in FEATS if not f.startswith('索赔金额')],
 '无网点行为特征': [f for f in FEATS if '网点' not in f],
 '无频次编码特征': [f for f in FEATS if not f.endswith('_freq')],
 '仅索赔金额特征': ['索赔金额', '索赔金额_log1p'],
}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
ab_rows = []
for gname, feats in groups.items():
    Xg = X[feats]
    maes = []
    for tri, vai in kf.split(Xg):
        m = lgb.LGBMRegressor(objective='regression_l1', metric='l1', learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, n_estimators=198, random_state=SEED,
                              deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)
        m.fit(Xg.iloc[tri], p[tri])
        maes.append(np.abs(p[vai] - np.maximum(m.predict(Xg.iloc[vai]), 0)).mean())
    ab_rows.append(dict(feature_set=gname, n_feat=len(feats), MAE_mean=float(np.mean(maes)), MAE_std=float(np.std(maes))))
    log(f"(c) 消融 {gname:14s} (n_feat={len(feats)}): MAE={np.mean(maes):.2f}±{np.std(maes):.2f}")
pd.DataFrame(ab_rows).to_csv(rf"{TAB}\q2_ablation.csv", index=False, encoding='utf-8-sig')

# (d) claim − median(u) 参考
base_mae = np.abs(u - np.median(u)).mean()
log(f"(d) p̂=索赔金额−中位超额(282.94) 的 MAE = {base_mae:.2f}")
pd.DataFrame([dict(baseline='claim−median(u)', MAE=float(base_mae))]).to_csv(
    rf"{TAB}\baseline_claim_minus_median_u.csv", index=False, encoding='utf-8-sig')

log("=== 10_sensitivity done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
