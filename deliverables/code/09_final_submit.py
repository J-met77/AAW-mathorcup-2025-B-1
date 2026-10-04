# -*- coding: utf-8 -*-
"""09_final_submit.py —— A6 实现 E6：全量重训最优配置 → 附件2 推断 → Result_提交.xlsx（A5 §4 规格）。
严格以 data/Result.xlsx 为模板：只填 实际赔付金额、风险标注 两列，运单号列不触碰。
自检：行数/运单号逐行一致/无空值/标签合法/赔付非负。
输出：output/Result_提交.xlsx, output/tables/附件2_predictions.csv, output/logs/09_final_submit.log
"""
import sys, io, json
import pandas as pd
import numpy as np
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
WS = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1"
TAB = rf"{WS}\output\tables"
LOG = rf"{WS}\output\logs\09_final_submit.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))
SEED = 2025
CLS = ['合理诉求', '诉求偏高', '严重超额']

log("=== 09_final_submit start ===")
df1 = pd.read_csv(rf"{TAB}\附件1_clean.csv")
df2 = pd.read_csv(rf"{TAB}\附件2_clean.csv")
cfg2 = json.load(open(rf"{TAB}\q2_model_config.json", encoding="utf-8"))
cfg3 = json.load(open(rf"{TAB}\q3_model_config.json", encoding="utf-8"))
rule = json.load(open(rf"{TAB}\rule_final.json", encoding="utf-8"))

FEATS, CAT = cfg2['features'], cfg2['cat_features']

def buildXY(dfs):
    """类别列统一编码（train∪test 全集水平，保证 train/test 码一致）。"""
    cats_map = {c: sorted(set(df1[c].astype(str)) | set(df2[c].astype(str))) for c in CAT}
    Xs = []
    for d in dfs:
        X = d[FEATS].copy()
        for c in CAT:
            X[c] = pd.Categorical(d[c].astype(str), categories=cats_map[c])
        Xs.append(X)
    return Xs

X1, X2 = buildXY([df1, df2])
log(f"训练矩阵 {X1.shape}，推断矩阵 {X2.shape}")

# ---- 问题2：全量重训（Q2 最优配置 cfg8: L1, lr=0.05, leaves=31, mcs=20, 198树）----
y_pay = df1['实际赔付金额'].values.astype(float)
reg = lgb.LGBMRegressor(objective='regression_l1', metric='l1',
                        learning_rate=cfg2['learning_rate'], num_leaves=cfg2['num_leaves'],
                        min_child_samples=cfg2['min_child_samples'], n_estimators=cfg2['n_estimators'],
                        random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=-1)
reg.fit(X1, y_pay)
pred_pay = np.maximum(reg.predict(X2), 0)
log(f"问题2 推断：n={len(pred_pay)}  min={pred_pay.min():.2f} max={pred_pay.max():.2f} "
    f"median={np.median(pred_pay):.2f} mean={pred_pay.mean():.2f}")

# ---- 问题3：全量重训（Q3 最优方案 B_resample r1=0.5, r2=0.5）----
y_lab = pd.read_csv(rf"{TAB}\q1_labels_附件1.csv")['风险标注'].map({c: i for i, c in enumerate(CLS)}).values
if cfg3['best_scheme'].startswith('B'):
    rng = np.random.default_rng(SEED)
    counts = np.bincount(y_lab, minlength=3); n0 = counts[0]
    idx_all = [np.where(y_lab == k)[0] for k in range(3)]
    take = [idx_all[0]]
    for k, ratio in [(1, cfg3['resample']['r1']), (2, cfg3['resample']['r2'])]:
        need = int(round(n0 * ratio)) - counts[k]
        take.append(np.concatenate([idx_all[k], rng.choice(idx_all[k], need, replace=True)]) if need > 0 else idx_all[k])
    idx = np.concatenate(take); rng.shuffle(idx)
    Xtr, ytr = X1.iloc[idx].reset_index(drop=True), y_lab[idx]
    log(f"问题3 过采样后训练集：{np.bincount(ytr, minlength=3).tolist()}（原 {counts.tolist()}）")
else:
    Xtr, ytr = X1, y_lab
clf = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                         learning_rate=cfg3['clf_params']['learning_rate'],
                         num_leaves=cfg3['clf_params']['num_leaves'],
                         min_child_samples=cfg3['clf_params']['min_child_samples'],
                         n_estimators=cfg3['clf_params']['n_estimators'],
                         class_weight=cfg3['clf_params']['class_weight'],
                         random_state=SEED, deterministic=True, force_row_wise=True,
                         verbose=-1, n_jobs=-1)
clf.fit(Xtr, ytr)
pred_lab = clf.predict(X2)
dist = {CLS[k]: float((pred_lab == k).mean() * 100) for k in range(3)}
log(f"问题3 推断分布：合理={dist['合理诉求']:.2f}% 偏高={dist['诉求偏高']:.2f}% 严重={dist['严重超额']:.2f}%")

# 路线B 参考输出（不提交，供论文对比）：Q2预测值按规则判类
def apply_rule(pp, uu):
    g = rule["gamma"]
    T1 = rule["alpha1"] + rule["beta1"]*np.power(np.maximum(pp,1e-9), g)
    T2 = rule["alpha2"] + rule["beta2"]*np.power(np.maximum(pp,1e-9), g)
    return np.where(uu <= T1, 0, np.where(uu <= T2, 1, 2)).astype(int)
claim2 = df2['索赔金额'].values.astype(float)
lab_routeB = apply_rule(pred_pay, np.maximum(claim2 - pred_pay, 0))
log(f"路线B 参考（不提交）：合理={np.mean(lab_routeB==0)*100:.2f}% 偏高={np.mean(lab_routeB==1)*100:.2f}% 严重={np.mean(lab_routeB==2)*100:.2f}%")

# ---- 填充 Result 模板（运单号列零改动）----
tmpl = pd.read_excel(rf"{WS}\data\Result.xlsx", sheet_name='Sheet1')
ids_tmpl = tmpl['运单号'].tolist()
ids_att2 = pd.to_numeric(df2['运单号']).tolist()
assert ids_tmpl == ids_att2, "模板运单号与附件2不一致，禁止填充"
out = tmpl.copy()
out['实际赔付金额'] = np.round(pred_pay, 2)
out['风险标注'] = [CLS[k] for k in pred_lab]
outpath = rf"{WS}\output\Result_提交.xlsx"
out.to_excel(outpath, index=False)

# ---- 自检 ----
chk = pd.read_excel(outpath, sheet_name='Sheet1')
assert chk.shape == (2792, 3), f"形状异常 {chk.shape}"
assert chk['运单号'].tolist() == ids_tmpl, "运单号被改动！"
assert chk[['实际赔付金额','风险标注']].notna().all().all(), "存在空值"
assert set(chk['风险标注'].unique()) <= set(CLS), "标签越界"
assert (chk['实际赔付金额'] >= 0).all(), "赔付为负"
log("\n--- 提交自检 ---")
log(f"形状={chk.shape} 运单号一致=True 无空值=True 标签合法=True 赔付非负=True")
log(f"提交分布：合理={np.mean(chk['风险标注']=='合理诉求')*100:.2f}% "
    f"偏高={np.mean(chk['风险标注']=='诉求偏高')*100:.2f}% 严重={np.mean(chk['风险标注']=='严重超额')*100:.2f}%")
log(f"[saved] {outpath}")

pd.DataFrame({'运单号': ids_tmpl, '预测实际赔付金额': np.round(pred_pay, 2),
              '路线A_风险标注': [CLS[k] for k in pred_lab],
              '路线B_风险标注_参考': [CLS[k] for k in lab_routeB],
              '索赔金额': claim2}).to_csv(rf"{TAB}\附件2_predictions.csv", index=False, encoding='utf-8-sig')
log(f"[saved] {TAB}\\附件2_predictions.csv")
log("=== 09_final_submit done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
