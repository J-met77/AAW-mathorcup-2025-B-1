# -*- coding: utf-8 -*-
"""07_q2_regression.py —— A6 实现 E2：问题2 实际赔付金额回归（A5 §2 规格）。
模型：B0 中位数基线 / B1 log-log 线性 / M-L2 / M-L1 / M-LOG（LightGBM 小网格）
协议：KFold(5, shuffle, seed=2025)；超参按 fold-1 预筛 top3 → 5 折全评 → 80/20 留出复验
输出：output/tables/q2_cv_metrics.csv, q2_model_config.json, output/logs/07_q2_regression.log
"""
import sys, io, json, time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
TAB = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\tables"
LOG = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\output\logs\07_q2_regression.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))
SEED = 2025

log("=== 07_q2_regression start ===")
df = pd.read_csv(rf"{TAB}\附件1_clean.csv")
y = df['实际赔付金额'].values.astype(float)

CAT = ['线路类型','是否c2c','是否生鲜妥投及时','寄件是否内部','异常原因','进线渠道','商品类型','新旧程度','寄件B/C','进线人身份']
NUM = ['保价金额','保价金额_log1p','保价异常标记','配送超时时长','配送超时_饱和标记','妥投到进线时长',
       '妥投进线_负值标记','妥投进线_logabs','索赔金额','索赔金额_log1p','索赔金额_下限标记',
       '始发网点发单量','始发网点发单量_log1p','始发网点万单理赔率','始发网点万单理赔率_log1p','始发网点理赔率负值标记',
       '始发网点赔付比例','始发网点赔付比例_log1p','目的网点发单量','目的网点发单量_log1p','目的网点万单理赔率',
       '目的网点万单理赔率_log1p','目的网点理赔率负值标记','目的网点赔付比例','目的网点赔付比例_log1p',
       '始发城市_freq','目的城市_freq','寄件人id_freq','收件人id_freq','异常原因_freq','商品类型_freq','城市对_freq']
FEATS = NUM + CAT
X = df[FEATS].copy()
for c in CAT:
    X[c] = X[c].astype('category')
cat_idx = [FEATS.index(c) for c in CAT]
log(f"特征数={len(FEATS)}（数值{len(NUM)}+类别{len(CAT)}），n={len(X)}")

def metrics(y_true, y_pred, tag):
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_all = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    m = y_true >= 1
    mape1 = float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)
    wape = float(np.abs(y_true - y_pred).sum() / y_true.sum() * 100)
    over = float((y_pred > 0).mean())  # 非负占比（恒1）
    return dict(model=tag, MAE=mae, RMSE=rmse, MAPE_all=mape_all, MAPE_pge1=mape1, WAPE=wape)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

# --- B0 / B1 基线 ---
res = []
pred_b0 = np.full(len(y), np.median(y))
res.append(metrics(y, pred_b0, "B0_中位数基线"))
lr = np.log(y)
Xb1 = np.log(df['索赔金额'].values.astype(float))
a, b = np.polyfit(Xb1, lr, 1)
pred_b1 = np.exp(a + b * Xb1)
res.append(metrics(y, pred_b1, "B1_loglog线性"))
log("\n--- 基线 ---")
log(pd.DataFrame(res).to_string(index=False))

# --- LGBM 候选网格 ---
base = dict(learning_rate=0.1, num_leaves=31, min_child_samples=20, n_estimators=2000)
def make_model(obj, lr, leaves, mcs, log_target=False):
    return lgb.LGBMRegressor(
        objective='regression_l1' if obj == 'l1' else 'regression',
        metric='l1' if obj == 'l1' else 'l2',
        learning_rate=lr, num_leaves=leaves, min_child_samples=mcs,
        n_estimators=base['n_estimators'], random_state=SEED, deterministic=True,
        force_row_wise=True, verbose=-1, n_jobs=-1)

grid = []
for obj in ['l2', 'l1']:
    for lr in [0.05, 0.1]:
        for leaves in [31, 63]:
            for mcs in [20, 50]:
                grid.append(dict(obj=obj, lr=lr, leaves=leaves, mcs=mcs, log_target=False))
for lr in [0.05, 0.1]:  # M-LOG：对 log1p(y) 做 L2
    for leaves in [31, 63]:
        grid.append(dict(obj='l2', lr=lr, leaves=leaves, mcs=20, log_target=True))
log(f"\nLGBM 候选配置数={len(grid)}（fold-1 预筛）")

tr_idx, va_idx = folds[0]
ytr, yva = y[tr_idx], y[va_idx]
screen = []
t0 = time.time()
for i, g in enumerate(grid):
    yt = np.log1p(ytr) if g['log_target'] else ytr
    m = make_model(g['obj'], g['lr'], g['leaves'], g['mcs'])
    m.fit(X.iloc[tr_idx], yt, eval_set=[(X.iloc[va_idx], np.log1p(yva) if g['log_target'] else yva)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pv = m.predict(X.iloc[va_idx])
    pv = np.expm1(pv) if g['log_target'] else pv
    mm = metrics(yva, pv, f"cfg{i}")
    screen.append(dict(cfg=i, **g, fold1_MAE=mm['MAE'], best_iter=m.best_iteration_ or base['n_estimators']))
    log(f"  cfg{i} obj={g['obj']} lr={g['lr']} leaves={g['leaves']} mcs={g['mcs']} log={g['log_target']} -> MAE={mm['MAE']:.3f} ({time.time()-t0:.0f}s)")
screen_df = pd.DataFrame(screen).sort_values('fold1_MAE')
top3 = screen_df.head(3)
log("\n--- 预筛 top3 ---")
log(top3.to_string(index=False))

# --- top3 完整 5 折 ---
cv_rows = []
for _, r in top3.iterrows():
    i = int(r['cfg']); g = grid[i]
    maes, rmses, mapes, iter_eff = [], [], [], []
    oof = np.zeros(len(y))
    for f, (tri, vai) in enumerate(folds):
        yt = np.log1p(y[tri]) if g['log_target'] else y[tri]
        m = make_model(g['obj'], g['lr'], g['leaves'], g['mcs'])
        m.fit(X.iloc[tri], yt, eval_set=[(X.iloc[vai], np.log1p(y[vai]) if g['log_target'] else y[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = m.predict(X.iloc[vai]); pv = np.expm1(pv) if g['log_target'] else pv
        oof[vai] = pv
        mm = metrics(y[vai], pv, f"cfg{i}_fold{f}")
        maes.append(mm['MAE']); rmses.append(mm['RMSE']); mapes.append(mm['MAPE_pge1'])
        iter_eff.append(m.best_iteration_ or base['n_estimators'])
    full = metrics(y, oof, f"cfg{i}_OOF")
    cv_rows.append(dict(cfg=i, obj=g['obj'], lr=g['lr'], leaves=g['leaves'], mcs=g['mcs'],
                        log_target=g['log_target'], MAE_mean=np.mean(maes), MAE_std=np.std(maes),
                        RMSE_mean=np.mean(rmses), MAPE_pge1_mean=np.mean(mapes),
                        OOF_MAE=full['MAE'], OOF_RMSE=full['RMSE'], OOF_MAPE_pge1=full['MAPE_pge1'],
                        OOF_WAPE=full['WAPE'], best_iter_median=int(np.median(iter_eff))))
cv_df = pd.DataFrame(cv_rows).sort_values('OOF_MAE')
log("\n--- top3 五折 CV + OOF ---")
log(cv_df.to_string(index=False))

best = cv_df.iloc[0]
g = grid[int(best['cfg'])]
log(f"\n>>> 选中配置：cfg{int(best['cfg'])} obj={g['obj']} lr={g['lr']} leaves={g['leaves']} "
    f"mcs={g['mcs']} log_target={g['log_target']}  OOF MAE={best['OOF_MAE']:.3f}")

# --- 80/20 留出复验 ---
tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
yt = np.log1p(y[tri]) if g['log_target'] else y[tri]
m = make_model(g['obj'], g['lr'], g['leaves'], g['mcs'])
m.fit(X.iloc[tri], yt, eval_set=[(X.iloc[vai], np.log1p(y[vai]) if g['log_target'] else y[vai])],
      callbacks=[lgb.early_stopping(100, verbose=False)])
pv = m.predict(X.iloc[vai]); pv = np.expm1(pv) if g['log_target'] else pv
hold = metrics(y[vai], pv, "holdout8020")
log(f"留出复验(80/20, seed=2025)：MAE={hold['MAE']:.3f} RMSE={hold['RMSE']:.3f} "
    f"MAPE(p>=1)={hold['MAPE_pge1']:.2f}% WAPE={hold['WAPE']:.2f}%")

# --- 冻结配置 ---
config = dict(features=FEATS, cat_features=CAT, objective=('l1' if g['obj']=='l1' else 'l2'),
              log_target=bool(g['log_target']), learning_rate=float(g['lr']),
              num_leaves=int(g['leaves']), min_child_samples=int(g['mcs']),
              n_estimators=int(np.clip(best['best_iter_median']*1.2, 100, 2000)),
              seed=SEED, cv=dict(OOF_MAE=float(best['OOF_MAE']), OOF_RMSE=float(best['OOF_RMSE']),
                                 OOF_MAPE_pge1=float(best['OOF_MAPE_pge1']), OOF_WAPE=float(best['OOF_WAPE']),
                                 MAE_fold_std=float(best['MAE_std'])),
              holdout=dict(MAE=hold['MAE'], RMSE=hold['RMSE'], MAPE_pge1=hold['MAPE_pge1'], WAPE=hold['WAPE']),
              baselines=dict(B0_MAE=float(metrics(y, pred_b0, 'b')['MAE']),
                             B1_MAE=float(metrics(y, pred_b1, 'b')['MAE'])))
with open(rf"{TAB}\q2_model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
res_df = pd.concat([pd.DataFrame(res), cv_df], ignore_index=True)
res_df.to_csv(rf"{TAB}\q2_cv_metrics.csv", index=False, encoding='utf-8-sig')
log(f"[saved] q2_model_config.json, q2_cv_metrics.csv")
log("=== 07_q2_regression done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
