# -*- coding: utf-8 -*-
"""
a6_common.py —— A6 阶段共享工具模块（供 a6_01~a6_06 引用）
契约来源：00_admin/A5_算法方案.md §2/§3/§4/§6；A4_模型设计.md §3.2/§4.2；A3_数据报告.md 列名字典。
- 路径解析：工作区根 = 本文件上级目录（仅读写 agent_workspace_B2 内路径，红线 3）。
- 种子契约（A5 §6）：主种子 20251004，重复种子 {20251005, 20251006}；bootstrap B=1000、水平 95%（约定项）。
- 特征清单（A5 §3.2 允许清单展开，37 列；禁用清单见 assert）。
- 日志：中文、随算随写（红线 6），每脚本头打印库版本与输入文件 md5。
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "output" / "tables"
LOGD = ROOT / "output" / "logs"
FIGD = ROOT / "output" / "figures"
CODE = ROOT / "code"
DATAD = ROOT / "data"
for _p in (TBL, LOGD, FIGD):
    _p.mkdir(parents=True, exist_ok=True)

F_ANNEX1 = TBL / "附件1_clean.csv"
F_ANNEX2 = TBL / "附件2_clean.csv"
F_RESULT_TPL = DATAD / "Result.xlsx"
F_RESULT_OUT = ROOT / "output" / "Result_提交.xlsx"

# ---------------------------------------------------------------- 种子与随机契约（A5 §6，登记用）
SEED_MAIN = 20251004
SEEDS_REPEAT = [20251004, 20251005, 20251006]
BOOT_B = 1000          # bootstrap 次数（通用方法论惯例，标注"约定"）
BOOT_LEVEL = 0.95

# ---------------------------------------------------------------- 类别标签（Q1/Q3 三类，题面 L21）
LABELS = ["合理诉求", "诉求偏高", "严重超额"]
LABEL2IDX = {c: i for i, c in enumerate(LABELS)}

# ---------------------------------------------------------------- 特征集（A5 §3.2 允许清单逐列展开，两附件共有）
FEATURE_COLS = [
    "索赔金额", "索赔金额_log10",
    "保价金额", "保价金额_哨兵负值", "保价索赔比",
    "配送超时时长_sl", "配送超时时长_触墙", "配送超时时长_为负",
    "妥投到进线时长_sl",
    "始发网点万单理赔率", "目的网点万单理赔率",
    "始发网点万单理赔率_哨兵负值", "目的网点万单理赔率_哨兵负值",
    "始发网点赔付比例", "目的网点赔付比例",
    "始发网点发单量_log10", "目的网点发单量_log10",
    "网点万单理赔率_均值",
    "寄件人id_频率", "收件人id_频率", "始发城市_频率", "目的城市_频率", "城市对_频率",
    "寄件人id_疑似平台号", "收件人id_疑似平台号",
    "异常原因", "异常原因_缺失", "进线渠道", "进线渠道_缺失",
    "线路类型", "是否c2c", "是否生鲜妥投及时", "寄件是否内部",
    "商品类型", "新旧程度", "寄件B/C", "进线人身份",
]
# 低基数类别列（LGBM category 传参；线性族 one-hot；异常原因/进线渠道 的 NaN 填"缺失"类，A5 §3.2）
CAT_COLS = ["异常原因", "进线渠道", "线路类型", "是否c2c", "是否生鲜妥投及时",
            "寄件是否内部", "商品类型", "新旧程度", "寄件B/C", "进线人身份"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

# 禁用列（A5 §3.2/§8.4：泄漏列、键、U2 原值、高基数原 ID）
FORBIDDEN_COLS = ["实际赔付金额", "索赔差额", "赔付索赔比", "相对超额", "实际赔付金额_log10",
                  "行ID", "运单号", "配送超时时长", "妥投到进线时长",
                  "始发城市", "目的城市", "寄件人id", "收件人id",
                  "始发网点发单量", "目的网点发单量"]


def assert_features_ok(df: pd.DataFrame, tag: str, logger: logging.Logger) -> None:
    """断言允许清单在位且未混入禁用列（A5 §8.3 断言清单）。"""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    bad = [c for c in FORBIDDEN_COLS if c in FEATURE_COLS]
    assert not bad, f"特征清单内混入禁用列：{bad}"
    assert not missing, f"[{tag}] 允许特征缺失：{missing}"


# ---------------------------------------------------------------- 日志
def get_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_env(lg: logging.Logger, inputs: list[Path]) -> None:
    """A5 §6 日志规范：库版本 + 输入文件 md5 + 种子登记。"""
    import lightgbm
    import sklearn
    import scipy
    try:
        import xgboost
        xgb_v = xgboost.__version__
    except Exception:
        xgb_v = "未安装"
    lg.info("环境：python=%s numpy=%s pandas=%s sklearn=%s scipy=%s lightgbm=%s xgboost=%s",
            sys.version.split()[0], np.__version__, pd.__version__,
            sklearn.__version__, scipy.__version__, lightgbm.__version__, xgb_v)
    lg.info("种子登记（A5 §6）：主种子=%d，重复种子=%s，bootstrap B=%d 水平 %.2f（约定项）",
            SEED_MAIN, SEEDS_REPEAT, BOOT_B, BOOT_LEVEL)
    for p in inputs:
        if p.exists():
            lg.info("输入文件 md5：%s = %s（大小 %d 字节）", p.name, md5_of(p), p.stat().st_size)
        else:
            lg.error("输入文件缺失：%s", p)
            raise FileNotFoundError(p)


# ---------------------------------------------------------------- 数据载入与断言（A5 §8.3）
def load_annex1(lg: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(F_ANNEX1)
    assert df.shape == (11167, 52), f"附件1_clean 形状断言失败：{df.shape}（应为 11167×52）"
    assert df["行ID"].is_unique and df["行ID"].notna().all(), "行ID 键唯一性断言失败"
    assert df["实际赔付金额"].notna().all() and df["索赔金额"].notna().all(), "两金额列存在非空断言失败"
    d = df["索赔差额"]
    assert (d < 0).all(), "T3 断言失败：索赔差额存在非负值"
    e = -d
    assert (e > 0).all(), "T3 断言失败：超额幅度 e 存在非正值"
    lg.info("附件1_clean 载入：11167×52；e=−索赔差额 恒正校验通过（min=%.2f, max=%.2f）", e.min(), e.max())
    return df


def load_annex2(lg: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(F_ANNEX2)
    assert df.shape[0] == 2792, f"附件2_clean 行数断言失败：{df.shape[0]}"
    assert df["运单号"].is_unique, "运单号唯一性断言失败"
    # D21 断言：索赔金额_log10 在位（A3 后台已补列；若缺失按同式确定性补算并记日志）
    if "索赔金额_log10" not in df.columns:
        lg.warning("D21：附件2 缺 索赔金额_log10，按同式 log10(索赔金额) 现场确定性补算（非拟合，无信息引入）")
        df["索赔金额_log10"] = np.log10(df["索赔金额"].astype(float))
    else:
        rec = np.log10(df["索赔金额"].astype(float))
        dev = float(np.nanmax(np.abs(rec - df["索赔金额_log10"].astype(float))))
        lg.info("D21 断言：附件2 索赔金额_log10 在位，与 log10(索赔金额) 回算最大偏差 %.2e", dev)
    lg.info("附件2_clean 载入：%s", df.shape)
    return df


# ---------------------------------------------------------------- 指标（A5 §3.6 定义式）
def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(((y - p) ** 2).mean()))


def smape_protected(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.maximum(y + p, 1.00)  # 分母下限 1.00 元形式保护（M2-9）
    return float((200.0 * np.abs(y - p) / denom).mean())


def all_reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"WAPE": wape(y, p), "MAE": mae(y, p), "RMSE": rmse(y, p),
            "SMAPE_保护": smape_protected(y, p)}


def bootstrap_ci(values_y: np.ndarray, values_p: np.ndarray, metric_fn,
                 n_boot: int = BOOT_B, seed: int = SEED_MAIN, level: float = BOOT_LEVEL):
    """对样本行重抽 B 次，取 metric 的百分位 CI（M4-4，numpy 手写）。"""
    rng = np.random.default_rng(seed)
    n = len(values_y)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot, dtype=float)
    yy = np.asarray(values_y, dtype=float)
    pp = np.asarray(values_p, dtype=float)
    for k in range(n_boot):
        j = idx[k]
        stats[k] = metric_fn(yy[j], pp[j])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 3) -> float:
    f1s = []
    for c in range(n_class):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 3) -> dict:
    out = {}
    for c in range(n_class):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
        out[f"类{c}_precision"] = prec
        out[f"类{c}_recall"] = rec
        out[f"类{c}_f1"] = f1
        out[f"类{c}_support"] = int((y_true == c).sum())
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 3) -> np.ndarray:
    cm = np.zeros((n_class, n_class), dtype=int)
    for a in range(n_class):
        for b in range(n_class):
            cm[a, b] = int(((y_true == a) & (y_pred == b)).sum())
    return cm


def stratified_folds(strata: np.ndarray, k: int, seed: int, shuffle: bool = True):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    return list(skf.split(np.zeros(len(strata)), strata))
