# -*- coding: utf-8 -*-
"""
A3-03 清洗（盲测 Run-2，工作区 agent_workspace_B2）
职责：以 a3_01/a3_02 勘察结论为依据执行清洗，产出基础清洗表
      output/tables/附件1_clean.csv、output/tables/附件2_clean.csv（本脚本版本=基础清洗；
      a3_06 特征工程将在其上追加衍生列后覆写为最终版）。
清洗总原则（每条规则在日志中记录【规则+理由+影响行数】）：
  R1 双表头剥离（中文表头取为列名；第一数据行为嵌入英文表头行，剥离）
  R2 类型规范化（数值列 float64 / 整数编码列 int64 / 类别列字符串；强转失败=0，见 a3_02 [0b]）
  R3 哨兵值处理：不可能为负的数值列出现 [-1,0) 噪声负值或 ==-1（a3_02 [3][4] 证实为缺失掩蔽模式）
     → 原列值改 NaN，同时增设 <列名>_哨兵负值 0/1 标记列（信息不丢失，不臆造替代值）
  R4 真缺失保留：异常原因/进线渠道 NaN 原样保留（缺失即信息，指示列由 a3_06 衍生）
  R5 重复行保留：附件1 25 组完全重复行不删（无主键无法判错，a3_02 [8]），仅记录
  R6 主键补设：附件1 无运单号（a3_01 [6]），增设稳定行键 行ID=1..N（原行序）
硬约束：附件2_clean 行数 == 2792（a3_02 [10]）；清洗不丢任何运单。
输入：data/附件1.xlsx, data/附件2.xlsx
输出：output/tables/附件1_clean.csv, output/tables/附件2_clean.csv, output/logs/a3_03_清洗.log
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_03_清洗.log"
OUT = WS / "output" / "tables"
LOG.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.so = sys.stdout

    def write(self, s):
        self.so.write(s)
        self.f.write(s)

    def flush(self):
        self.so.flush()
        self.f.flush()


sys.stdout = Tee(LOG)
np.random.seed(20251004)  # 全链路统一种子；本清洗无随机操作
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("=" * 100)
print("A3-03 清洗 | pandas", pd.__version__, "| 种子 20251004（本脚本无随机操作，登记用）")
print("表头行口径（写死）：第1行=中文表头(header=0)，第2行=嵌入英文表头行（数据首行，剥离）")
print("=" * 100)

F1_P = WS / "data" / "附件1.xlsx"
F2_P = WS / "data" / "附件2.xlsx"

SHARED_FEATS = ["线路类型", "是否c2c", "是否生鲜妥投及时", "保价金额", "始发城市", "目的城市",
                "寄件人id", "收件人id", "寄件是否内部", "配送超时时长", "异常原因", "进线渠道",
                "妥投到进线时长", "索赔金额", "商品类型", "新旧程度", "寄件B/C", "进线人身份",
                "始发网点发单量", "始发网点万单理赔率", "始发网点赔付比例",
                "目的网点发单量", "目的网点万单理赔率", "目的网点赔付比例"]

NUM_FLOAT = ["保价金额", "配送超时时长", "妥投到进线时长", "索赔金额",
             "始发网点发单量", "始发网点万单理赔率", "始发网点赔付比例",
             "目的网点发单量", "目的网点万单理赔率", "目的网点赔付比例", "实际赔付金额"]
NUM_INT = ["线路类型", "是否c2c", "是否生鲜妥投及时", "始发城市", "目的城市",
           "寄件人id", "收件人id", "寄件是否内部", "新旧程度"]
CAT_COLS = ["异常原因", "进线渠道", "商品类型", "寄件B/C", "进线人身份"]

# 哨兵处理清单：{列: (谓词描述, 谓词函数)}，依据 a3_02 [3][4] 证据
SENTINEL_RULES = {
    "保价金额":            ("值<0（100%落[-1,0)，保价金额按附注L83为申报价值不可能为负）", lambda s: s < 0),
    "始发网点万单理赔率":   ("值<0（100%落(-1,0)，理赔率按附注L77定义>=0）",              lambda s: s < 0),
    "目的网点万单理赔率":   ("值<0（100%落(-1,0)，理赔率按附注L77定义>=0）",              lambda s: s < 0),
    "始发网点发单量":       ("值==-1（发单量为计数不可能为-1；a3_02[4]证实仅取-1）",       lambda s: s == -1),
}


def load_and_strip(path: Path, tag: str) -> pd.DataFrame:
    """R1：中文表头 + 剥离嵌入英文表头数据行。"""
    df = pd.read_excel(path, sheet_name=0, header=0, engine="openpyxl")
    n0 = len(df)
    en = pd.read_excel(path, header=None, nrows=2).iloc[1].tolist()
    assert df.iloc[0].tolist() == en, f"{tag} 第一数据行不是英文表头行，口径漂移！"
    df = df.iloc[1:].reset_index(drop=True)
    print(f"[R1] {tag}: 剥离嵌入英文表头行 1 行，{n0} -> {len(df)}。理由：a3_01 [3] 证实双行表头（中/英）。")
    return df


def log_rule(tag: str, col: str, rule: str, reason: str, n: int, total: int):
    print(f"[R3] 【{tag}.{col}】规则: {rule} | 理由: {reason} | 影响行数: {n} ({n/total*100:.3f}%)")


def clean(df: pd.DataFrame, tag: str, has_id: bool, has_target: bool) -> pd.DataFrame:
    total = len(df)
    # ---- R2 类型规范化
    coerced_fail = 0
    for c in NUM_FLOAT:
        if c not in df.columns:
            continue
        before_nn = int(df[c].notna().sum())
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_nan = int(df[c].isna().sum())
        coerced_fail += after_nan  # 原始无缺失（a3_01[8]），NaN 必为强转失败
        df[c] = df[c].astype("float64")
    for c in NUM_INT:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        assert s.notna().all(), f"{tag}.{c} 出现不可解析值"
        assert bool(((s % 1) == 0).all()), f"{tag}.{c} 含非整数值，不能安全转 int64"
        df[c] = s.astype("int64")
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    if has_id:
        s = pd.to_numeric(df["运单号"], errors="coerce")
        assert s.notna().all() and bool(((s % 1) == 0).all())
        df["运单号"] = s.astype("int64")
    print(f"[R2] {tag}: 类型规范化完成（float 列 {len([c for c in NUM_FLOAT if c in df.columns])} 个、"
          f"int 列 {len([c for c in NUM_INT if c in df.columns])} 个、类别列 {len([c for c in CAT_COLS if c in df.columns])} 个）；"
          f"数值强转失败计数={coerced_fail}（a3_02 [0b] 已证为 0）。")

    # ---- R3 哨兵值处理：原列改 NaN + 标记列（信息不丢失）
    for col, (rule, pred) in SENTINEL_RULES.items():
        if col not in df.columns:
            continue
        s = df[col].astype("float64")
        m = pred(s)
        n = int(m.sum())
        df[col + "_哨兵负值"] = m.astype("int64")
        df.loc[m, col] = np.nan
        log_rule(tag, col, f"命中谓词的值置 NaN，增设 {col}_哨兵负值 标记列", rule, n, total)

    # ---- R4/R5 说明（无操作，记录口径）
    print(f"[R4] {tag}: 异常原因/进线渠道 NaN 原样保留，不插补。理由：异常原因为员工提报字段"
          f"（附表1 L55），未提报本身是信息；指示列由 a3_06 衍生。影响行数: "
          f"异常原因 {int(df['异常原因'].isna().sum())}, 进线渠道 {int(df['进线渠道'].isna().sum())}。")
    print(f"[R5] {tag}: 不删任何重复行/异常行（行数保持 {total}）。理由：a3_02 [8] 已证附件1 重复为"
          f"25 组成对行、无主键不可判错；附件2 有唯一运单号且 a3_01 [6] 证无重复。")

    # ---- R6 主键
    if has_id:
        assert df["运单号"].is_unique and df["运单号"].is_monotonic_increasing
        print(f"[R6] {tag}: 主键=运单号（唯一且严格递增，1..{df['运单号'].max()}），保持原行序。")
    else:
        df.insert(0, "行ID", np.arange(1, total + 1, dtype="int64"))
        print(f"[R6] {tag}: 无业务主键（a3_01 [6]），增设稳定行键 行ID=1..{total}（原行序，供 Q1 标注挂接）。")
    return df


print("\n--- 附件1 清洗 ---")
d1 = load_and_strip(F1_P, "附件1")
assert list(d1.columns) == SHARED_FEATS + ["实际赔付金额"], f"附件1 列序漂移: {list(d1.columns)}"
d1 = clean(d1, "附件1", has_id=False, has_target=True)

print("\n--- 附件2 清洗 ---")
d2 = load_and_strip(F2_P, "附件2")
assert list(d2.columns) == ["运单号"] + SHARED_FEATS, f"附件2 列序漂移: {list(d2.columns)}"
d2 = clean(d2, "附件2", has_id=True, has_target=False)

# ---------------------------------------------------------------- 列序整理与落盘
front1 = ["行ID"] + SHARED_FEATS + ["实际赔付金额"]
tail1 = [c for c in d1.columns if c not in front1]
d1 = d1[front1 + tail1]
front2 = ["运单号"] + SHARED_FEATS
tail2 = [c for c in d2.columns if c not in front2]
d2 = d2[front2 + tail2]

p1 = OUT / "附件1_clean.csv"
p2 = OUT / "附件2_clean.csv"
d1.to_csv(p1, index=False, encoding="utf-8-sig")
d2.to_csv(p2, index=False, encoding="utf-8-sig")

print("\n--- 落盘核验 ---")
print(f"附件1_clean.csv: 行={len(d1)} 列={len(d1.columns)}  -> {p1}")
print(f"附件2_clean.csv: 行={len(d2)} 列={len(d2.columns)}  -> {p2}")
assert len(d2) == 2792, "附件2_clean 行数与原始运单数不一致！"
print(f"硬约束核验：附件2_clean 行数 {len(d2)} == 附件2 原始运单数 2792 : PASS")
assert len(d1) == 11167
print(f"硬约束核验：附件1_clean 行数 {len(d1)} == 勘察基准 11167 : PASS")
print("附件1_clean 列清单:", list(d1.columns))
print("附件2_clean 列清单:", list(d2.columns))
print("两表列口径差异：附件1 多 [实际赔付金额] 与 行ID（无运单号）；附件2 有 运单号；"
      "哨兵标记列两表同名同口径（附件1 4 个 / 附件2 4 个）。")
r1 = pd.read_csv(p1)
r2 = pd.read_csv(p2)
print(f"回读核验：附件1_clean {r1.shape}，附件2_clean {r2.shape}；回读 dtypes 数值列解析正常="
      f"{bool(r1['实际赔付金额'].notna().all() and r2['索赔金额'].notna().all())}")
print("\nA3-03 清洗结束。")
