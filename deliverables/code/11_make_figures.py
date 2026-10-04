# -*- coding: utf-8 -*-
"""11_make_figures.py —— A8：论文级图表（≥6 张，中文、300dpi、统一配色）。
统一配色：合理诉求=#2E7D32 偏高=#F9A825 严重超额=#C62828，主色=#1F4E79。
产出：output/figures/fig1_标注规则二维散点.png ... fig8_规则敏感性热图.png
"""
import sys, io, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
WS = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1"
TAB = rf"{WS}\output\tables"
FIG = rf"{WS}\output\figures"
LOG = rf"{WS}\output\logs\11_make_figures.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

CC = {'合理诉求': '#2E7D32', '诉求偏高': '#F9A825', '严重超额': '#C62828'}
MAIN = '#1F4E79'
CLS = ['合理诉求', '诉求偏高', '严重超额']

log("=== 11_make_figures start ===")
df = pd.read_csv(rf"{TAB}\附件1_clean.csv")
lab = pd.read_csv(rf"{TAB}\q1_labels_附件1.csv")
rule = json.load(open(rf"{TAB}\rule_final.json", encoding="utf-8"))
pred2 = pd.read_csv(rf"{TAB}\附件2_predictions.csv")
cmA = pd.read_csv(rf"{TAB}\q3_confusion_routeA.csv", index_col=0)
cmB = pd.read_csv(rf"{TAB}\q3_confusion_routeB.csv", index_col=0)
cv3 = pd.read_csv(rf"{TAB}\q3_cv_metrics.csv")
sens = pd.read_csv(rf"{TAB}\sensitivity_tau.csv")

p = df['实际赔付金额'].values.astype(float)
u = df['超额索赔额'].values.astype(float)
y = lab['风险标注'].map({c: i for i, c in enumerate(CLS)}).values

# ---------- 图1 标注规则二维散点 ----------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
rng = np.random.default_rng(2025)
for k, cname in enumerate(CLS):
    m = y == k
    idx = np.where(m)[0]
    if len(idx) > 3000:
        idx = rng.choice(idx, 3000, replace=False)
    ax.scatter(p[idx], u[idx], s=6, c=CC[cname], alpha=0.35, label=f'{cname}（n={m.sum()}）', edgecolors='none')
pp = np.linspace(0.74, p.max(), 300)
ax.plot(pp, rule['alpha1'] + rule['beta1']*pp, 'k--', lw=1.8, label='T1(p)=264.87+2.482p（合理上界）')
ax.plot(pp, rule['alpha2'] + rule['beta2']*pp, 'k-', lw=1.8, label='T2(p)=850.93+3.269p（严重下界）')
ax.set_xlabel('实际赔付金额（元）'); ax.set_ylabel('超额索赔额 u = 索赔金额 − 实际赔付（元）')
ax.set_title('图1  附件1 风险标注结果与划分边界（随机抽稀展示）')
ax.set_xlim(-20, 3250); ax.set_ylim(-50, 4600)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
fig.tight_layout(); fig.savefig(rf"{FIG}\fig1_标注规则二维散点.png"); plt.close(fig)
log("[fig1] 标注规则二维散点 saved")

# ---------- 图2 三类 u 分布密度（C7 密度对比） ----------
fig, ax = plt.subplots(figsize=(7.2, 4.8))
xs = np.linspace(0, 3000, 600)
def kde_curve(sample, xs):
    sample = np.asarray(sample); sd = sample.std(ddof=1)
    h = 0.9 * sd * len(sample) ** (-0.2)
    sub = sample if len(sample) <= 4000 else np.random.default_rng(2025).choice(sample, 4000, replace=False)
    return np.exp(-0.5*((xs[:, None]-sub[None, :])/h)**2).sum(1)/(len(sub)*h*np.sqrt(2*np.pi))
for k, cname in enumerate(CLS):
    ax.plot(xs, kde_curve(u[y == k], xs), color=CC[cname], lw=2, label=cname)
    ax.fill_between(xs, kde_curve(u[y == k], xs), color=CC[cname], alpha=0.12)
ax.set_xlabel('超额索赔额 u（元）'); ax.set_ylabel('核密度')
ax.set_title('图2  三类运单超额索赔额分布密度（合理密集、严重稀疏）')
ax.set_xlim(0, 3000); ax.legend()
fig.tight_layout(); fig.savefig(rf"{FIG}\fig2_三类差额密度对比.png"); plt.close(fig)
log("[fig2] 三类差额密度对比 saved")

# ---------- 图3 赔付金额分布：训练 vs 附件2预测 ----------
fig, ax = plt.subplots(figsize=(7.2, 4.8))
bins = np.linspace(0, 1600, 65)
ax.hist(np.clip(p, 0, 1600), bins=bins, density=True, alpha=0.55, color=MAIN, label='附件1 实际赔付（训练）')
ax.hist(np.clip(pred2['预测实际赔付金额'], 0, 1600), bins=bins, density=True, alpha=0.55, color='#E07B39', label='附件2 预测赔付')
ax.axvline(np.median(p), color=MAIN, ls='--', lw=1.2, label=f'训练中位数 {np.median(p):.0f} 元')
ax.axvline(np.median(pred2['预测实际赔付金额']), color='#E07B39', ls='--', lw=1.2, label=f'预测中位数 {np.median(pred2["预测实际赔付金额"]):.0f} 元')
ax.set_xlabel('实际赔付金额（元）'); ax.set_ylabel('密度')
ax.set_title('图3  实际赔付金额分布：训练集与附件2预测对比')
ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(rf"{FIG}\fig3_赔付金额分布对比.png"); plt.close(fig)
log("[fig3] 赔付金额分布对比 saved")

# ---------- 图4 Q2 预测-真值 与 分箱误差（80/20 留出） ----------
cfg2 = json.load(open(rf"{TAB}\q2_model_config.json", encoding="utf-8"))
FEATS, CAT = cfg2['features'], cfg2['cat_features']
X = df[FEATS].copy()
for c in CAT:
    X[c] = X[c].astype('category')
ytr_all = p
tri, vai = train_test_split(np.arange(len(p)), test_size=0.2, random_state=2025)
reg = lgb.LGBMRegressor(objective='regression_l1', metric='l1', learning_rate=0.05, num_leaves=31,
                        min_child_samples=20, n_estimators=198, random_state=2025, deterministic=True,
                        force_row_wise=True, verbose=-1, n_jobs=-1)
reg.fit(X.iloc[tri], p[tri])
pv = reg.predict(X.iloc[vai]); pt = p[vai]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
ax = axes[0]
ax.scatter(pt, pv, s=6, c=MAIN, alpha=0.3, edgecolors='none')
lim = [0, 1700]
ax.plot(lim, lim, 'k--', lw=1.2, label='y = x')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('实际赔付（元）'); ax.set_ylabel('预测赔付（元）')
ax.set_title(f'(a) 留出集预测-真值（MAE={np.abs(pt-pv).mean():.1f} 元）')
ax.legend()
ax = axes[1]
qcut = pd.qcut(pt, 8, labels=False, duplicates='drop')
mids = [np.median(pt[qcut == k]) for k in range(8)]
bias = [np.median(pv[qcut == k] - pt[qcut == k]) for k in range(8)]
mae8 = [np.abs(pv[qcut == k] - pt[qcut == k]).mean() for k in range(8)]
ax.bar(range(8), mae8, color=MAIN, alpha=0.75, label='MAE')
ax.set_xticks(range(8)); ax.set_xticklabels([f'{m:.0f}' for m in mids])
ax.set_xlabel('实际赔付分箱中位数（元）'); ax.set_ylabel('MAE（元）')
ax2 = ax.twinx()
ax2.plot(range(8), bias, 'o-', color='#C62828', lw=1.6, label='偏差中位数')
ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.set_ylabel('偏差中位数（元，红）')
ax.set_title('(b) 分箱误差与偏差（收缩效应）')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8.5, loc='upper left')
fig.suptitle('图4  问题2 回归误差分析（80/20 留出）', y=1.0)
fig.tight_layout(); fig.savefig(rf"{FIG}\fig4_回归误差分析.png"); plt.close(fig)
log("[fig4] 回归误差分析 saved")

# ---------- 图5 两路线混淆矩阵 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, cm, title in zip(axes, [cmA, cmB], ['路线A：直接分类（宏F1=0.635）', '路线B：回归+规则（宏F1=0.559）']):
    arr = cm.values.astype(float)
    arrn = arr / arr.sum(axis=1, keepdims=True)
    im = ax.imshow(arrn, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{arr[i, j]:.0f}\n({arrn[i, j]*100:.1f}%)', ha='center', va='center',
                    fontsize=8.5, color='white' if arrn[i, j] > 0.5 else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(CLS, fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLS, fontsize=8.5)
    ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
    ax.set_title(title, fontsize=10.5)
    ax.grid(False)
fig.suptitle('图5  两路线 OOF 混淆矩阵（行归一化）', y=1.0)
fig.tight_layout(); fig.savefig(rf"{FIG}\fig5_两路线混淆矩阵.png"); plt.close(fig)
log("[fig5] 两路线混淆矩阵 saved")

# ---------- 图6 不均衡处理方案对比 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.8))
names = [s.replace('_resample', '\n过采样').replace('_clsW', '\n类权重').replace('_prior_corr', '\n先验校正')
         for s in cv3['scheme']]
x = np.arange(len(cv3))
ax.bar(x - 0.2, cv3['macroF1_mean'], width=0.4, color=MAIN, label='宏F1')
ax.bar(x + 0.2, cv3['recall_severe_mean'], width=0.4, color='#C62828', label='严重类召回')
ax.axhline(0.309, color='#455A64', ls=':', lw=1.4)
ax.text(3.5, 0.695, '多数类基线宏F1=0.309（灰色点线）', fontsize=8.5, color='#263238',
        fontweight='bold', ha='center',
        bbox=dict(facecolor='white', edgecolor='#455A64', boxstyle='round,pad=0.25', alpha=0.95))
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylim(0, 0.75); ax.set_ylabel('得分')
ax.set_title('图6  “严重超额”不均衡处理方案对比（StratifiedKFold-5）')
ax.legend()
fig.tight_layout(); fig.savefig(rf"{FIG}\fig6_不均衡方案对比.png"); plt.close(fig)
log("[fig6] 不均衡方案对比 saved")

# ---------- 图7 特征消融 ----------
ab = pd.read_csv(rf"{TAB}\q2_ablation.csv")
base = pd.read_csv(rf"{TAB}\q2_cv_metrics.csv")
fig, ax = plt.subplots(figsize=(7.6, 4.4))
colors = [MAIN, '#C62828', '#F9A825', '#5B8DB8', '#7B7B7B']
ax.barh(ab['feature_set'][::-1], ab['MAE_mean'][::-1],
        xerr=ab['MAE_std'][::-1], color=colors[:len(ab)][::-1], alpha=0.85, capsize=3)
ax.axvline(199.02, color='k', ls=':', lw=1.2)
ax.text(200, len(ab)-1.1, '中位数基线 MAE=199.0', fontsize=8, color='gray')
ax.set_xlabel('5折 CV MAE（元）')
ax.set_title('图7  问题2 特征组消融（索赔金额特征为主力信号）')
fig.tight_layout(); fig.savefig(rf"{FIG}\fig7_特征消融.png"); plt.close(fig)
log("[fig7] 特征消融 saved")

# ---------- 图8 规则敏感性热图 ----------
from matplotlib.colors import Normalize
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, col, title in zip(axes, ['pct_合理', 'pct_严重'], ['合理诉求占比（%）', '严重超额占比（%）']):
    cmap_name = 'YlGnBu' if col == 'pct_合理' else 'OrRd'
    cmap = plt.get_cmap(cmap_name)
    mat = sens.pivot(index='tau1', columns='tau2', values=col)
    norm = Normalize(vmin=mat.values.min(), vmax=mat.values.max())
    im = ax.imshow(mat.values, cmap=cmap_name, norm=norm)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            r, g, b, _ = cmap(norm(mat.values[i, j]))   # 按底色亮度自动黑/白，保证数字可读
            lum = 0.299*r + 0.587*g + 0.114*b
            ax.text(j, i, f'{mat.values[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white' if lum < 0.55 else 'black')
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels([f'τ2={c}' for c in mat.columns], fontsize=8.5)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([f'τ1={r}' for r in mat.index], fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.grid(False)
fig.suptitle('图8  标注规则参数 (τ1, τ2) 敏感性（frozen 形态 M2）', y=1.0)
fig.tight_layout(); fig.savefig(rf"{FIG}\fig8_规则敏感性热图.png"); plt.close(fig)
log("[fig8] 规则敏感性热图 saved")

log("=== 11_make_figures done ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
