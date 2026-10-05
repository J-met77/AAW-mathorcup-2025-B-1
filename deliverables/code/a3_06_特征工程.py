# -*- coding: utf-8 -*-
"""
A3-06 特征工程（盲测 Run-2，工作区 agent_workspace_B2）
职责：在 a3_03 基础清洗表上追加衍生特征，覆写最终版
      output/tables/附件1_clean.csv、output/tables/附件2_clean.csv。
      每个衍生列在日志中注明【定义+理由（题面条款/数据事实）+ 覆盖范围】。
      不设标注阈值、不做建模。
复现方式：依次运行 a3_03_清洗.py → a3_06_特征工程.py 即可从 data/ 原始 xlsx 一键重建两份 clean.csv。
输入：output/tables/附件1_clean.csv, output/tables/附件2_clean.csv（a3_03 基础清洗版）
输出：同名 clean.csv（最终版）+ output/logs/a3_06_特征工程.log
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_06_特征工程.log"
OUT = WS / "output" / "tables"
LOG.parent.mkdir(parents=True, exist_ok=True)


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
np.random.seed(20251004)  # 本脚本无随机操作（频率映射为确定性统计）
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print("=" * 100)
print("A3-06 特征工程 | pandas", pd.__version__, "| 种子 20251004（无随机操作，登记用）")
print("=" * 100)

d1 = pd.read_csv(OUT / "附件1_clean.csv")
d2 = pd.read_csv(OUT / "附件2_clean.csv")
n1, n2 = len(d1), len(d2)
print(f"[0] 读入基础清洗表：附件1 {d1.shape}，附件2 {d2.shape}")
WALL = 427864.0  # 配送超时时长全局硬墙（两表同值，a3_02 [2] 证实）


def log_feat(name: str, tables: str, definition: str, reason: str):
    print(f"[F] {name}（{tables}）| 定义: {definition} | 理由: {reason}")


def signed_log(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def add_common(d: pd.DataFrame, tag: str, freq_maps: dict) -> pd.DataFrame:
    # F1 时长有符号对数变换（重尾压缩；a3_02 [2]：负支~17%+墙体堆叠 64.6% 的双峰结构）
    d["配送超时时长_sl"] = signed_log(d["配送超时时长"])
    d["妥投到进线时长_sl"] = signed_log(d["妥投到进线时长"])
    log_feat("配送超时时长_sl / 妥投到进线时长_sl", tag,
             "sign(x)·ln(1+|x|)",
             "a3_02 [2]：两时长列重尾（妥投到进线 max≈9.98e8 ms）且配送超时呈负支+427864 硬墙双峰；"
             "有符号对数保留方向信息并压缩量纲（通用变换手法，非本题资料）")
    # F2 配送超时结构标记
    d["配送超时时长_触墙"] = (d["配送超时时长"] == WALL).astype("int64")
    d["配送超时时长_为负"] = (d["配送超时时长"] < 0).astype("int64")
    log_feat("配送超时时长_触墙 / _为负", tag,
             f"(== {WALL:.0f}) 与 (<0) 的 0/1 标记",
             "a3_02 [2] 密度剖面：64.6% 质量堆叠于硬墙 [427500,427864]，16.6% 为负支；"
             "截断结构信息需显式暴露给下游")
    # F3 发单量对数（跨 11 个数量级：min -1 已哨兵化，max 9.9e10；a3_01 [10]）
    d["始发网点发单量_log10"] = np.log10(1 + d["始发网点发单量"].clip(lower=0))
    d["目的网点发单量_log10"] = np.log10(1 + d["目的网点发单量"].clip(lower=0))
    log_feat("始发/目的网点发单量_log10", tag, "log10(1+x)，x<0 已在 a3_03 哨兵化为 NaN 后取 0 下限",
             "a3_01 [10]：发单量跨多个数量级（附件1 目的网点 33~9.9e10），对数尺度稳定")
    # F4 缺失指示列（缺失即信息；T11：异常原因缺失组与非缺失组赔付中位 185.3 vs 217.7）
    d["异常原因_缺失"] = d["异常原因"].isna().astype("int64")
    d["进线渠道_缺失"] = d["进线渠道"].isna().astype("int64")
    log_feat("异常原因_缺失 / 进线渠道_缺失", tag, "isna 的 0/1 指示",
             "a3_01 [8] 缺失统计 + a3_04 T11：缺失组与非缺失组目标行为存在差异，缺失本身携带信息")
    # F5 双端网点风险复合
    r = d[["始发网点万单理赔率", "目的网点万单理赔率"]]
    d["网点万单理赔率_均值"] = r.mean(axis=1)   # NaN 感知：双 NaN 才 NaN
    log_feat("网点万单理赔率_均值", tag, "始发/目的万单理赔率（哨兵清洗后）的行内均值",
             "题面 L63/L66：双端网点理赔率均为风险语境；复合提供单一风险水平概览（行内均值，NaN 感知）")
    # F6 频率编码（基于附件1 统计，两表同口径；unseen→0）
    for c in ["寄件人id", "收件人id", "始发城市", "目的城市"]:
        m = freq_maps[c]
        d[c + "_频率"] = d[c].map(m).fillna(0.0).astype("float64")
    pair_m = freq_maps["城市对"]
    key = list(zip(d["始发城市"], d["目的城市"]))
    d["城市对_频率"] = pd.Series(key).map(pair_m).fillna(0.0).astype("float64").values
    log_feat("寄件人id/收件人id/始发城市/目的城市/城市对 _频率", tag,
             "该取值在附件1 中的出现占比（计数/11167）；附件2 未见取值=0",
             "a3_05 [3]：高基数（寄件人 2936 个、收件人 3899 个），不可 one-hot；"
             "附件2 冷启动规模 6.27%/8.20%，unseen→0 为确定性回退，不臆造数值")
    # F7 疑似平台号标记（a3_02 [7]：邻域 [299900,300100] 内仅 300000 孤立出现）
    d["寄件人id_疑似平台号"] = (d["寄件人id"] == 300000).astype("int64")
    d["收件人id_疑似平台号"] = (d["收件人id"] == 300000).astype("int64")
    log_feat("寄件人id/收件人id _疑似平台号", tag, "(==300000) 的 0/1 标记",
             "a3_02 [7]：300000 在两表高频（收件人 8.97%/8.95%）且其 ±100 邻域无任何其他取值，"
             "判定为系统/平台账号而非自然序取值")
    return d


# 附件1 统计的频率映射（含哨兵/缺失无关；频率基于清洗后全部行）
freq_maps = {}
for c in ["寄件人id", "收件人id", "始发城市", "目的城市"]:
    freq_maps[c] = (d1[c].value_counts() / n1)
pair_cnt = d1.groupby(["始发城市", "目的城市"]).size()
freq_maps["城市对"] = (pair_cnt / n1)

d1 = add_common(d1, "附件1", freq_maps)
d2 = add_common(d2, "附件2", freq_maps)

# ---------- 附件1 专有：目标相关衍生（题面 L19/L33 的分析对象；附件2 无目标列，不衍生） ----------
d1["索赔差额"] = d1["实际赔付金额"] - d1["索赔金额"]
log_feat("索赔差额", "附件1", "实际赔付金额 - 索赔金额",
         "题面 L19 定义式（D09 仲裁 D1 口径）；Q1 指定分析对象，附件中无同名物理列（a3_01 [2]）")
d1["赔付索赔比"] = d1["实际赔付金额"] / d1["索赔金额"]
log_feat("赔付索赔比", "附件1", "实际赔付金额 / 索赔金额",
         "a3_04 T4：差额的绝对量纲随索赔额增长，比值提供尺度无关视图（C6 密集程度对比素材）")
d1["相对超额"] = (d1["索赔金额"] - d1["实际赔付金额"]) / d1["索赔金额"]
log_feat("相对超额", "附件1", "(索赔金额-实际赔付金额)/索赔金额",
         "a3_04 T7：相对超额∈(0,1)（全表差额均为负），密度结构更可分")
d1["实际赔付金额_log10"] = np.log10(d1["实际赔付金额"])
d1["索赔金额_log10"] = np.log10(d1["索赔金额"])
log_feat("实际赔付金额_log10 / 索赔金额_log10", "附件1", "log10 变换",
         "a3_04 T1/T2：两金额列右偏（skew≈1.9/2.0），log 空间近线性分布（Q2 目标变换与差额尺度统一的备选输入）")
d1["保价索赔比"] = d1["保价金额"] / d1["索赔金额"]
d2["保价索赔比"] = d2["保价金额"] / d2["索赔金额"]
log_feat("保价索赔比", "附件1+附件2", "保价金额 / 索赔金额（保价 NaN→NaN）",
         "附注 L83：保价为赔付上限语境；a3_04 T9 证实 86.72% 索赔超保价，该比率暴露保价相对索赔的水平")

# ---------- 列序整理与落盘 ----------
# 保持 a3_03 原列序，衍生列按生成顺序附加在尾部
base1 = ["行ID", "线路类型", "是否c2c", "是否生鲜妥投及时", "保价金额", "始发城市", "目的城市",
         "寄件人id", "收件人id", "寄件是否内部", "配送超时时长", "异常原因", "进线渠道",
         "妥投到进线时长", "索赔金额", "商品类型", "新旧程度", "寄件B/C", "进线人身份",
         "始发网点发单量", "始发网点万单理赔率", "始发网点赔付比例",
         "目的网点发单量", "目的网点万单理赔率", "目的网点赔付比例", "实际赔付金额",
         "保价金额_哨兵负值", "始发网点万单理赔率_哨兵负值", "目的网点万单理赔率_哨兵负值", "始发网点发单量_哨兵负值"]
base2 = ["运单号"] + [c for c in base1 if c not in ("行ID", "实际赔付金额")]  # 附件2 首列=运单号；无行ID 与目标列
assert base2[0] == "运单号" and len(base2) == 1 + len(base1) - 2
d1 = d1[base1 + [c for c in d1.columns if c not in base1]]
d2 = d2[base2 + [c for c in d2.columns if c not in base2]]

p1, p2 = OUT / "附件1_clean.csv", OUT / "附件2_clean.csv"
d1.to_csv(p1, index=False, encoding="utf-8-sig")
d2.to_csv(p2, index=False, encoding="utf-8-sig")

print("\n--- 落盘核验 ---")
r1 = pd.read_csv(p1)
r2 = pd.read_csv(p2)
assert len(r1) == n1 == 11167 and len(r2) == n2 == 2792
print(f"附件1_clean.csv 最终版: {r1.shape} -> {p1}")
print(f"附件2_clean.csv 最终版: {r2.shape} -> {p2}")
print("硬约束核验：附件2_clean 行数 2792 == 附件2 原始运单数 : PASS")
print("硬约束核验：附件1_clean 行数 11167 == 勘察基准 : PASS")
dd = (r1["索赔差额"] - (r1["实际赔付金额"] - r1["索赔金额"])).abs()
assert float(dd.max()) < 1e-6  # 容差仅吸收 read_csv 浮点回读的 1-ulp 噪声（实测 max 9.1e-13，见日志）
assert ((r1["相对超额"] > 0) & (r1["相对超额"] < 1)).all()
assert (r2["配送超时时长_触墙"] == (r2["配送超时时长"] == 427864).astype(int)).all()
print(f"一致性抽核：索赔差额定义式回算最大偏差={dd.max():.3g}（read_csv 浮点回读 1-ulp 噪声，非数据误差）；"
      "相对超额∈(0,1)；触墙标记回算一致 : PASS")
print("附件1_clean 衍生列:", [c for c in r1.columns if c not in base1])
print("附件2_clean 衍生列:", [c for c in r2.columns if c not in base2])
print("两表列口径差异：附件1 专有 行ID/实际赔付金额/索赔差额/赔付索赔比/相对超额/实际赔付金额_log10；"
      "附件2 首列=运单号（与 Result.xlsx 同序）；其余列两表同名同序同口径。")
print(f"列序核验：附件2 首列={d2.columns[0]}（应为 运单号），附件1 首列={d1.columns[0]}（应为 行ID）")
assert d2.columns[0] == "运单号" and d1.columns[0] == "行ID"
print("\nA3-06 特征工程结束。复现路径：a3_03_清洗.py → a3_06_特征工程.py（依次从 data/ 原始文件重建）。")
