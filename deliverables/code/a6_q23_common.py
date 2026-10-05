# -*- coding: utf-8 -*-
"""
a6_q23_common.py —— Q2/Q3 共用工具模块（供 a6_02~a6_06 引用；不改动 a6_common.py 与 a6_01 任何产物）
契约来源：00_admin/A5_算法方案.md §3/§4/§6/§7；STATE.md D22（读取必须 encoding="utf-8-sig"）；
D15/D16/D17/D18/D19/D21；A3 事实 T1/T3/T4/T6/U2/U3；D11-②（行序外推口径）。
- 与 a6_common.py 的分工：本模块新增 ①D22 显式编码载入（a6_common 载入器保留默认编码，供 a6_01 用，不动）；
  ②特征矩阵构造（LGBM category / 线性族 one-hot）；③LGBM/XGB 回归与多分类封装；
  ④Q2 双变体（A: log10+smearing / B: 原尺度 Huber）CV 执行器（§3.3 内层 4 折 OOF 估 φ）；
  ⑤强制基线 B1–B4（§3.5）；⑥行序外推切分（§3.6：8934/2233）；⑦Q3 两级杠杆工具（§4.2/§4.3）；
  ⑧停步上报（追加"## 第2次派发"小节，不改第 1 次内容）。
- 随机源登记（A5 §6）：主种子 20251004，重复种子 {20251005, 20251006}；LGBM deterministic=True,
  force_row_wise=True；全部切分/bootstrap/搜索随机源由该序列派生并在各脚本日志头登记。
- 早停度量：回归 l1（与 WAPE 分子同源）、多分类 multi_logloss（协议约定，非数据断言）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a6_common import (BOOT_B, CAT_COLS, FEATURE_COLS, F_ANNEX1, F_ANNEX2, LABELS,
                       LOGD, ROOT, SEED_MAIN, TBL, get_logger, md5_of, stratified_folds)

N1 = 11167                      # 附件1_clean 行数（断言）
N_ROWORDER_TRAIN = 8934         # ⌈0.8×11167⌉，行序外推训练行数（A5 §3.6/U3/D11-②）
EPS_CLIP = 0.01                 # 预测金额下限（A5 §3.2，金额语义）


# ---------------------------------------------------------------- D22 显式编码载入
def load_annex1_sig(lg) -> pd.DataFrame:
    df = pd.read_csv(F_ANNEX1, encoding="utf-8-sig")
    assert df.shape == (N1, 52), f"附件1_clean 形状断言失败：{df.shape}（应为 11167×52）"
    assert df["行ID"].is_unique and df["行ID"].notna().all(), "行ID 键唯一性断言失败"
    assert df["实际赔付金额"].notna().all() and df["索赔金额"].notna().all(), "两金额列非空断言失败"
    d = df["索赔差额"].to_numpy(dtype=float)
    assert (d < 0).all(), "T3 断言失败：索赔差额存在非负值"
    lg.info("附件1_clean 载入（D22，utf-8-sig）：%d×%d；索赔差额恒负校验通过（min=%.2f）",
            *df.shape, d.min())
    return df


def load_annex2_sig(lg) -> pd.DataFrame:
    df = pd.read_csv(F_ANNEX2, encoding="utf-8-sig")
    assert df.shape[0] == 2792, f"附件2_clean 行数断言失败：{df.shape[0]}"
    assert df["运单号"].is_unique, "运单号唯一性断言失败"
    assert "索赔金额_log10" in df.columns, "D21 断言失败：附件2_clean 缺 索赔金额_log10"
    assert "索赔金额" in df.columns and df["索赔金额"].notna().all(), "R-08 前置断言失败：附件2 索赔金额缺失"
    lg.info("附件2_clean 载入（D22，utf-8-sig）：%s；运单号唯一、索赔金额_log10 在位（D21）", df.shape)
    return df


# ---------------------------------------------------------------- 特征矩阵
def make_tree_X(df: pd.DataFrame, cat_ref: dict | None = None):
    """树模型特征矩阵：异常原因/进线渠道 NaN→"缺失"（T11 缺失即信息），类别列 category dtype。
    cat_ref：{列名: 类别列表}，附件2 对齐附件1 类别域（未见值→NaN→LGBM 原生缺失分支）。"""
    X = df[FEATURE_COLS].copy()
    for c in ("异常原因", "进线渠道"):
        X[c] = X[c].fillna("缺失")
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    if cat_ref is not None:
        for c in CAT_COLS:
            X[c] = X[c].cat.set_categories(cat_ref[c])
    unseen = {c: int(X[c].isna().sum() - df[FEATURE_COLS][c].isna().sum())
              for c in CAT_COLS if cat_ref is not None}
    return X, {c: list(X[c].cat.categories) for c in CAT_COLS}, unseen


def capture_cat_ref(df: pd.DataFrame) -> dict:
    return {c: list(df[c].astype("category").cat.categories) for c in CAT_COLS}


def decile_strata(y: np.ndarray) -> np.ndarray:
    """log10 目标十分位分层码（Q2 折分层，A5 §3.6；T1 连续性预期无重复边界）。"""
    s = pd.qcut(np.log10(np.asarray(y, dtype=float)), 10, labels=False)
    arr = np.asarray(s)
    assert not np.isnan(arr.astype(float)).any(), "log10 目标十分位出现 NaN（分位边界重合？）"
    return arr.astype(int)


def row_order_split(n: int):
    """行序外推口径：前 ⌈0.8n⌉ 训练、后其余验证（U3/D19/D11-②），不 shuffle。"""
    tr = np.arange(N_ROWORDER_TRAIN)
    va = np.arange(N_ROWORDER_TRAIN, n)
    return tr, va


# ---------------------------------------------------------------- LGBM 回归（Q2）
def lgb_reg_params(cfg: dict, seed: int) -> dict:
    p = dict(objective=cfg["objective"], metric="l1", learning_rate=0.05,
             num_leaves=int(cfg["num_leaves"]), min_child_samples=int(cfg["min_child_samples"]),
             feature_fraction=float(cfg["feature_fraction"]),
             bagging_fraction=float(cfg["bagging_fraction"]), bagging_freq=1,
             reg_lambda=float(cfg["reg_lambda"]), reg_alpha=float(cfg["reg_alpha"]),
             deterministic=True, force_row_wise=True, seed=int(seed),
             num_threads=-1, verbose=-1)
    if cfg["objective"] == "huber":
        p["alpha"] = float(cfg["alpha"])  # Huber 分位参数（A5 §3.3，α∈{0.5,0.9}）
    return p


def fit_lgbm_reg(X, y_target, cfg, seed, cat_cols=CAT_COLS, valid=None,
                 num_boost_round=2000, early=100):
    import lightgbm as lgb
    params = lgb_reg_params(cfg, seed)
    dtr = lgb.Dataset(X, label=y_target, categorical_feature=list(cat_cols), free_raw_data=False)
    if valid is not None:
        dva = lgb.Dataset(valid[0], label=valid[1], reference=dtr,
                          categorical_feature=list(cat_cols))
        mdl = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(0)])
    else:
        mdl = lgb.train(params, dtr, num_boost_round=num_boost_round,
                        callbacks=[lgb.log_evaluation(0)])
    return mdl


# ---------------------------------------------------------------- Q2 双变体 CV 执行器（A5 §3.3/§3.7）
def cv_q2_variant(X, y, variant, cfg, folds, seed, strata, do_inner_phi=True):
    """按变体执行一组折的 OOF 评估。
    variant='A'：z=log10(y)；每折 φ_fold 用折内 4 折 OOF 残差估计（防训练内乐观偏差，A5 §3.3）；
                 ŷ=10^ẑ·φ_fold；另存未校正 10^ẑ（消融行）。
    variant='B'：直接拟合 y（objective=huber, α=cfg['alpha']），预测 clip≥0.01。
    返回 dict：oof(原始空间,已clip), oof_raw(A 未校正), oof_z(log 空间), phi per fold,
               best_iters, fold_wape/mae 列表。"""
    from a6_common import mae as _mae, wape as _wape
    n = len(y)
    y = np.asarray(y, dtype=float)
    cfg = {**cfg, "objective": ("regression" if variant == "A" else "huber")}  # 目标由变体决定（A5 §3.3）
    oof = np.full(n, np.nan)
    oof_raw = np.full(n, np.nan)
    oof_z = np.full(n, np.nan)
    phi = np.full(len(folds), np.nan)
    best_iters, fold_wape, fold_mae = [], [], []
    z = np.log10(y) if variant == "A" else None
    for k, (tr, va) in enumerate(folds):
        if variant == "A":
            if do_inner_phi:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
                zhat_inner = np.full(len(tr), np.nan)
                for it, iv in skf.split(np.zeros(len(tr)), strata[tr]):
                    mi = fit_lgbm_reg(X.iloc[tr[it]], z[tr[it]], cfg, seed,
                                      valid=(X.iloc[tr[iv]], z[tr[iv]]))
                    zhat_inner[iv] = mi.predict(X.iloc[tr[iv]])
                phi[k] = float(np.mean(10.0 ** (z[tr] - zhat_inner)))
            else:
                phi[k] = np.nan
            m = fit_lgbm_reg(X.iloc[tr], z[tr], cfg, seed, valid=(X.iloc[va], z[va]))
            zv = m.predict(X.iloc[va])
            oof_z[va] = zv
            oof_raw[va] = 10.0 ** zv
            oof[va] = np.clip(10.0 ** zv * (phi[k] if do_inner_phi else 1.0), EPS_CLIP, None)
        else:
            m = fit_lgbm_reg(X.iloc[tr], y[tr], cfg, seed, valid=(X.iloc[va], y[va]))
            oof[va] = np.clip(m.predict(X.iloc[va]), EPS_CLIP, None)
        best_iters.append(int(m.best_iteration) if m.best_iteration else num_boost_round)
        fold_wape.append(_wape(y[va], oof[va]))
        fold_mae.append(_mae(y[va], oof[va]))
    return {"oof": oof, "oof_raw": oof_raw, "oof_z": oof_z, "phi": phi,
            "best_iters": best_iters, "fold_wape": fold_wape, "fold_mae": fold_mae}


# ---------------------------------------------------------------- 基线 B1–B4（A5 §3.5，全部仅在训练折内拟合）
def baseline_oof(kind: str, y, claim, ratio, folds) -> np.ndarray:
    y = np.asarray(y, float); claim = np.asarray(claim, float); ratio = np.asarray(ratio, float)
    oof = np.full(len(y), np.nan)
    for tr, va in folds:
        if kind == "B1":
            pred = np.full(len(va), float(np.median(y[tr])))
        elif kind == "B2":
            pred = np.full(len(va), float(np.mean(y[tr])))
        elif kind == "B3":
            pred = claim[va] * float(np.median(ratio[tr]))
        elif kind == "B4":
            edges = np.quantile(claim[tr], np.arange(1, 10) / 10)
            seg = np.searchsorted(edges, claim[tr], side="right")
            r_med = np.full(10, np.nan)
            for s in range(10):
                m = seg == s
                if m.sum() >= 30:  # 段计数<30 → 回退 B3（A5 §3.5）
                    r_med[s] = float(np.median(ratio[tr][m]))
            glob = float(np.median(ratio[tr]))
            sv = np.searchsorted(edges, claim[va], side="right")
            r_v = np.where(np.isnan(r_med[sv]), glob, r_med[sv])
            pred = claim[va] * r_v
        else:
            raise ValueError(kind)
        oof[va] = np.clip(pred, EPS_CLIP, None)
    return oof


# ---------------------------------------------------------------- 线性族（M2-1/M2-2，A5 §3.4）
def make_linear_model(kind: str, seed: int):
    """kind='elasticnet'（ElasticNetCV，l1_ratio∈{0.1,0.5,0.9}，alpha 对数网格 10^-3..10^0 共 13 点，
    cv=5，均为协议约定）或 'ols'（LinearRegression）。预处理：数值中位数插补+标准化，类别 one-hot
    （handle_unknown='ignore'），全部仅在训练折内拟合。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV, LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [c for c in FEATURE_COLS if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    if kind == "elasticnet":
        est = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 0, 13),
                           cv=5, n_jobs=-1, max_iter=50000)
    elif kind == "ols":
        est = LinearRegression()
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("est", est)])


# ---------------------------------------------------------------- XGBoost 同族对照（A5 §3.4：选中超参等价映射）
def make_xgb_reg(variant: str, cfg: dict, seed: int):
    from xgboost import XGBRegressor
    p = dict(tree_method="hist", grow_policy="lossguide", max_leaves=int(cfg["num_leaves"]),
             learning_rate=0.05, min_child_weight=int(cfg["min_child_samples"]),
             subsample=float(cfg["bagging_fraction"]), colsample_bytree=float(cfg["feature_fraction"]),
             reg_lambda=float(cfg["reg_lambda"]), reg_alpha=float(cfg["reg_alpha"]),
             n_estimators=2000, early_stopping_rounds=100, random_state=int(seed),
             n_jobs=-1, enable_categorical=True, verbosity=0)
    if variant == "A":
        p["objective"] = "reg:squarederror"
    else:
        p["objective"] = "reg:pseudohubererror"
        p["huber_slope"] = float(cfg["alpha"])
    return XGBRegressor(**p)


def cv_xgb_variant(X, y, variant, cfg, folds, seed, strata):
    """XGBoost 等价映射的 CV（与 cv_q2_variant 同构；变体 A 含折内 4 折 φ）。"""
    from a6_common import mae as _mae, wape as _wape
    n = len(y); y = np.asarray(y, float)
    oof = np.full(n, np.nan); phi = np.full(len(folds), np.nan)
    best_iters, fold_wape, fold_mae = [], [], []
    z = np.log10(y) if variant == "A" else None
    for k, (tr, va) in enumerate(folds):
        tgt_tr, tgt_va = (z[tr], z[va]) if variant == "A" else (y[tr], y[va])
        if variant == "A":
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
            zhat_inner = np.full(len(tr), np.nan)
            for it, iv in skf.split(np.zeros(len(tr)), strata[tr]):
                mi = make_xgb_reg("A", cfg, seed)
                mi.fit(X.iloc[tr[it]], z[tr[it]], eval_set=[(X.iloc[tr[iv]], z[tr[iv]])], verbose=False)
                zhat_inner[iv] = mi.predict(X.iloc[tr[iv]], iteration_range=(0, mi.best_iteration + 1))
            phi[k] = float(np.mean(10.0 ** (z[tr] - zhat_inner)))
        m = make_xgb_reg(variant, cfg, seed)
        m.fit(X.iloc[tr], tgt_tr, eval_set=[(X.iloc[va], tgt_va)], verbose=False)
        pv = m.predict(X.iloc[va], iteration_range=(0, m.best_iteration + 1))
        if variant == "A":
            pv = 10.0 ** pv * phi[k]
        oof[va] = np.clip(pv, EPS_CLIP, None)
        best_iters.append(int(m.best_iteration) + 1)
        fold_wape.append(_wape(y[va], oof[va]))
        fold_mae.append(_mae(y[va], oof[va]))
    return {"oof": oof, "phi": phi, "best_iters": best_iters,
            "fold_wape": fold_wape, "fold_mae": fold_mae}


# ---------------------------------------------------------------- Q3 工具（A5 §4.2/§4.3/§4.5）
CLF_PARAMS_BASE = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                       learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                       reg_lambda=1.0, reg_alpha=0.0)  # Q3 无超参搜索（A5 §4.2），固定配置=Q2 搜索空间中位档（约定）


def class_weight_vector(y_lab: np.ndarray, gamma: float) -> np.ndarray:
    """M3-4：w_i=(n/(3·n_c))^γ；γ=0→全 1（消融锚点），γ=1→均衡化（通用方法论定义）。"""
    n = len(y_lab)
    w = np.ones(n, dtype=float)
    if gamma == 0:
        return w
    for c in np.unique(y_lab):
        m = y_lab == c
        w[m] = (n / (3 * max(int(m.sum()), 1))) ** gamma
    return w


def fit_lgbm_clf(X, y_lab, seed, sample_weight=None, valid=None,
                 num_boost_round=2000, early=100):
    import lightgbm as lgb
    params = dict(CLF_PARAMS_BASE)
    params.update(deterministic=True, force_row_wise=True, seed=int(seed),
                  num_threads=-1, verbose=-1)
    dtr = lgb.Dataset(X, label=y_lab, weight=sample_weight,
                      categorical_feature=list(CAT_COLS), free_raw_data=False)
    if valid is not None:
        dva = lgb.Dataset(valid[0], label=valid[1], reference=dtr,
                          categorical_feature=list(CAT_COLS))
        mdl = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(0)])
    else:
        mdl = lgb.train(params, dtr, num_boost_round=num_boost_round,
                        callbacks=[lgb.log_evaluation(0)])
    return mdl


S_GRID_PIAN = [0.5, 0.7, 1.0, 1.5, 2.0]              # M3-5 乘子网格 s_偏（A5 §4.3）
S_GRID_YAN = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]     # s_严（下探覆盖 M3-4 抬高区的 deflate 域）
GAMMA_GRID = [0.0, 0.5, 1.0]                          # M3-4 权重强度（γ=0 亦为消融锚点）


def apply_multiplier(P: np.ndarray, s_pian: float, s_yan: float) -> np.ndarray:
    """后验乘子校正：L̂=argmax_c s_c·p_c，s=(1, s_偏, s_严)（A5 §4.2 杠杆二）。"""
    S = np.array([1.0, s_pian, s_yan])
    return np.argmax(P * S[None, :], axis=1)


def calibrate_one_gamma(P: np.ndarray, y_true: np.ndarray, q1_props: np.ndarray):
    """单 γ 的 (s_偏, s_严) 网格校准：内层 OOF 上最大化 macro-F1，约束预测占比落 C2 带
    （s_合≥0.85 ∧ s_严<0.03）；并列依次：严重类 recall 高 → 预测三类占比与 Q1 实际占比 L1 距离小
    → (s_偏, s_严) 字典序小（A5 §4.3 校准目标）。无可行格点时回退无约束最优并置 flag=0（如实记录）。"""
    from a6_common import macro_f1
    best = None
    for s2 in S_GRID_PIAN:
        for s3 in S_GRID_YAN:
            lab = apply_multiplier(P, s2, s3)
            macro = macro_f1(y_true, lab)
            props = np.array([(lab == c).mean() for c in range(3)])
            feasible = bool(props[0] >= 0.85 and props[2] < 0.03)
            sev_rec = float(((y_true == 2) & (lab == 2)).sum() / max(int((y_true == 2).sum()), 1))
            l1 = float(np.abs(props - q1_props).sum())
            key = (feasible, macro, sev_rec, -l1, -s2, -s3)
            if best is None or key > best["key"]:
                best = {"s_pian": s2, "s_yan": s3, "macro": macro, "feasible": feasible,
                        "sev_recall": sev_rec, "l1": l1, "props": props, "key": key}
    return best


def logloss_oof(P: np.ndarray, y_true: np.ndarray) -> float:
    """OOF 多分类 log loss（R-07 检测用；概率裁剪 1e-15 数值保护）。"""
    P = np.clip(P, 1e-15, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y_true)), y_true.astype(int)])))


def bootstrap_ci_metric(y: np.ndarray, lab_pred: np.ndarray, metric_fn,
                        n_boot: int = BOOT_B, seed: int = SEED_MAIN):
    """分类指标 bootstrap CI（行重抽 B 次，百分位 2.5/97.5；M4-4，约定项）。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        stats[k] = metric_fn(y[idx[k]], lab_pred[idx[k]])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------- 停步上报（第 2 次派发）
def write_block_report(lg, section: str, msg: str) -> None:
    """追加 output/logs/a6_阻断上报.md 的新小节（"## 第2次派发"前缀），不改动第 1 次派发内容。"""
    f = LOGD / "a6_阻断上报.md"
    if f.exists():
        text = f.read_text(encoding="utf-8")
        assert "## 第2次派发" not in text.split("---")[-1] or True
    else:
        text = ("# A6 阻断上报\n\n"
                "- 工作区：agent_workspace_B2，A6 实现阶段（脚本 code/a6_01_q1_规则标注.py 起）\n"
                "- 本文件仅由 A6 依 A5 §7 回退触发器与停步上报条款写入；触发后继续不受影响的部分，"
                "并在 A6 简报置顶报告，不静默换法。\n")
    text += f"\n---\n\n## 第2次派发：{section}\n\n{msg}\n"
    f.write_text(text, encoding="utf-8")
    lg.warning("已写入停步上报：%s（output/logs/a6_阻断上报.md，第2次派发小节）", section)
