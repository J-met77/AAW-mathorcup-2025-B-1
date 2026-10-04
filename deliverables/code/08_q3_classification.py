# -*- coding: utf-8 -*-
"""08_q3_classification.py —— A6 实现 E3/E4：问题3 直接分类（三不均衡方案）+ 路线B 对比（A5 §3 规格）。
折协议：StratifiedKFold(5, seed=2025)，路线A/B 共用（折索引落盘）。
方案：A-clsW 类权重 / B-resample 训练折内随机过采样 / C-thr 先验校正阈值移动；路线B=Q2回归+冻结规则。
输出：q3_cv_metrics.csv, q3_confusion_routeA.csv, q3_confusion_routeB.csv, q3_model_config.json,
      fold_indices.csv, output/logs/08_q3_classification.log
"""
import sys, io, json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TAB = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\tables"
LOG = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\08_q3_classification.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))
SEED = 2025
CLS = ['合理诉求', '诉求偏高', '严重超额']

log("=== 08_q3_classification start ===")
df = pd.read_csv(rf"{TAB}\附件1_clean.csv")
lab = pd.read_csv(rf"{TAB}\q1_labels_附件1.csv")
y = lab['风险标注'].map({c: i for i, c in enumerate(CLS)}).values
assert len(y) == len(df)
cfg2 = json.load(open(rf"{TAB}\q2_model_config.json", encoding="utf-8"))
rule = json.load(open(rf"{TAB}\rule_final.json", encoding="utf-8"))
log(f"标签分布：{pd.Series(y).value_counts().sort_index().to_dict()}  "
    f"(合理 {np.mean(y==0)*100:.2f}% / 偏高 {np.mean(y==1)*100:.2f}% / 严重 {np.mean(y==2)*100:.2f}%)")

FEATS, CAT = cfg2['features'], cfg2['cat_features']
X = df[FEATS].copy()
for c in CAT:
    X[c] = X[c].astype('category')
claim = df['索赔金额'].values.astype(float)

def make_clf(class_weight=None, seed=SEED):
    return lgb.LGBMClassifier(objective='multiclass', num_class=3,
                              learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              n_estimators=300, class_weight=class_weight,
                              random_state=seed, deterministic=True, force_row_wise=True,
                              verbose=-1, n_jobs=-1)

def apply_rule(pp, uu):
    if rule["form"] == "M1":
        T1 = np.interp(pp, rule["nodes_p"], rule["T1_nodes"]); T2 = np.interp(pp, rule["nodes_p"], rule["T2_nodes"])
    else:
        g = rule["gamma"]
        T1 = rule["alpha1"] + rule["beta1"]*np.power(np.maximum(pp,1e-9), g)
        T2 = rule["alpha2"] + rule["beta2"]*np.power(np.maximum(pp,1e-9), g)
    return np.where(uu <= T1, 0, np.where(uu <= T2, 1, 2)).astype(int)

def random_oversample(Xtr, ytr, r1, r2, seed=SEED):
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr, minlength=3)
    n0 = counts[0]
    idx_all = [np.where(ytr == k)[0] for k in range(3)]
    take = [idx_all[0]]
    for k, ratio in [(1, r1), (2, r2)]:
        need = int(round(n0 * ratio)) - counts[k]
        if need > 0:
            extra = rng.choice(idx_all[k], need, replace=True)
            take.append(np.concatenate([idx_all[k], extra]))
        else:
            take.append(idx_all[k])
    idx = np.concatenate(take)
    rng.shuffle(idx)
    return Xtr.iloc[idx].reset_index(drop=True), ytr[idx]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))
pd.DataFrame({'fold': [f for f, (tr, va) in enumerate(folds) for _ in va],
              'index': [i for tr, va in folds for i in va]}).to_csv(
    rf"{TAB}\fold_indices.csv", index=False, encoding='utf-8-sig')

schemes = []
# A-clsW
schemes.append(dict(name='A_clsW_balanced', kind='clsw', class_weight='balanced'))
# B-resample 4 配比
for r1 in [0.5, 1.0]:
    for r2 in [0.3, 0.5]:
        schemes.append(dict(name=f'B_resample_r1={r1},r2={r2}', kind='resample', r1=r1, r2=r2))
# C-thr 先验校正
for w in [0.25, 0.5, 1.0]:
    schemes.append(dict(name=f'C_prior_corr_w={w}', kind='prior', w=w))

rows = []
conf_best, best_key = None, None
for sc in schemes:
    oof = np.zeros(len(y), dtype=int)
    f1s, accs, rec2s = [], [], []
    for f, (tri, vai) in enumerate(folds):
        clf = make_clf('balanced' if sc['kind'] == 'clsw' else None)
        if sc['kind'] == 'resample':
            Xtr, ytr = random_oversample(X.iloc[tri], y[tri], sc['r1'], sc['r2'])
            clf.fit(Xtr, ytr)
        else:
            clf.fit(X.iloc[tri], y[tri])
        proba = clf.predict_proba(X.iloc[vai])
        if sc['kind'] == 'prior':
            pi = np.bincount(y[tri], minlength=3) / len(tri)
            q = proba / np.power(pi, sc['w'])
            pred = np.argmax(q / q.sum(1, keepdims=True), axis=1)
        else:
            pred = np.argmax(proba, axis=1)
        oof[vai] = pred
        f1s.append(f1_score(y[vai], pred, average='macro'))
        accs.append(accuracy_score(y[vai], pred))
        rec2s.append((pred[y[vai] == 2] == 2).mean())
    f1m, f1s_ = float(np.mean(f1s)), float(np.std(f1s))
    cm = confusion_matrix(y, oof, labels=[0, 1, 2])
    prec, rec, f1c, sup = precision_recall_fscore_support(y, oof, labels=[0, 1, 2], zero_division=0)
    rows.append(dict(scheme=sc['name'], macroF1_mean=f1m, macroF1_std=f1s_,
                     acc_mean=float(np.mean(accs)), recall_severe_mean=float(np.mean(rec2s)),
                     prec_severe=float(prec[2]), f1_severe=float(f1c[2])))
    log(f"{sc['name']:28s} 宏F1={f1m:.4f}±{f1s_:.4f} acc={np.mean(accs):.4f} "
        f"严重召回={np.mean(rec2s):.3f} 严重精确={prec[2]:.3f} 严重F1={f1c[2]:.3f}")
    if best_key is None or f1m > best_key:
        best_key, best_scheme, conf_best = f1m, sc['name'], cm

cv_df = pd.DataFrame(rows)
log("\n--- 方案汇总 ---")
log(cv_df.to_string(index=False))
cv_df.to_csv(rf"{TAB}\q3_cv_metrics.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(conf_best, index=[f'真_{c}' for c in CLS], columns=[f'预测_{c}' for c in CLS]).to_csv(
    rf"{TAB}\q3_confusion_routeA.csv", encoding='utf-8-sig')
log(f"\n>>> 路线A 最优方案：{best_scheme}（宏F1={best_key:.4f}）")

# --- 路线B：Q2 回归 + 冻结规则（同折） ---
oof_b = np.zeros(len(y), dtype=int)
for f, (tri, vai) in enumerate(folds):
    reg = lgb.LGBMRegressor(
        objective='regression_l1' if cfg2['objective'] == 'l1' else 'regression',
        metric='l1' if cfg2['objective'] == 'l1' else 'l2',
        learning_rate=cfg2['learning_rate'], num_leaves=cfg2['num_leaves'],
        min_child_samples=cfg2['min_child_samples'], n_estimators=cfg2['n_estimators'],
        random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)
    reg.fit(X.iloc[tri], y[tri])
    phat = np.maximum(reg.predict(X.iloc[vai]), 0)
    ub = np.maximum(claim[vai] - phat, 0)
    oof_b[vai] = apply_rule(phat, ub)
f1b = f1_score(y, oof_b, average='macro')
cm_b = confusion_matrix(y, oof_b, labels=[0, 1, 2])
prec, rec, f1c, sup = precision_recall_fscore_support(y, oof_b, labels=[0, 1, 2], zero_division=0)
log(f"\n路线B(回归+规则)          宏F1={f1b:.4f} acc={accuracy_score(y, oof_b):.4f} "
    f"严重召回={rec[2]:.3f} 严重精确={prec[2]:.3f} 严重F1={f1c[2]:.3f}")
log("路线B 混淆矩阵：")
log(pd.DataFrame(cm_b, index=[f'真_{c}' for c in CLS], columns=[f'预测_{c}' for c in CLS]).to_string())
pd.DataFrame(cm_b, index=[f'真_{c}' for c in CLS], columns=[f'预测_{c}' for c in CLS]).to_csv(
    rf"{TAB}\q3_confusion_routeB.csv", encoding='utf-8-sig')

# 路线A OOF 混淆矩阵（最优方案重跑一遍 OOF 已有 conf_best；补打印）
log("\n路线A 最优方案 OOF 混淆矩阵：")
log(pd.DataFrame(conf_best, index=[f'真_{c}' for c in CLS], columns=[f'预测_{c}' for c in CLS]).to_string())

config3 = dict(best_scheme=best_scheme, macroF1=float(best_key), routeB_macroF1=float(f1b),
               clf_params=dict(learning_rate=0.05, num_leaves=31, min_child_samples=20,
                               n_estimators=300, class_weight=('balanced' if best_scheme.startswith('A') else None)),
               resample=(dict(r1=schemes[[s['name'] for s in schemes].index(best_scheme)].get('r1'),
                              r2=schemes[[s['name'] for s in schemes].index(best_scheme)].get('r2'))
                         if best_scheme.startswith('B') else None),
               prior_w=(float(best_scheme.split('w=')[1]) if best_scheme.startswith('C') else None),
               seed=SEED)
with open(rf"{TAB}\q3_model_config.json", "w", encoding="utf-8") as f:
    json.dump(config3, f, ensure_ascii=False, indent=2)
log(f"[saved] q3_model_config.json")
log("=== 08_q3_classification done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
