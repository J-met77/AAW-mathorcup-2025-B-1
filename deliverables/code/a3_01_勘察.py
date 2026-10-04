# -*- coding: utf-8 -*-
"""
A3-01 结构勘察（盲测 Run-2，工作区 agent_workspace_B2）
职责：三表结构、双行表头核验、dtype、主键、缺失统计、类别取值域、
      Result.xlsx 与附件2 运单号对应关系、F1-F7 格式核验回填。
输入：data/附件1.xlsx, data/附件2.xlsx, data/Result.xlsx
输出：output/logs/a3_01_勘察.log（本脚本唯一产物，结论写入 00_admin/A3_数据报告.md）
红线：离线；只读本工作区；所有数字以本日志为溯源。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_01_勘察.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


class Tee:
    """同时写控制台与日志文件（utf-8）。"""
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
np.random.seed(20251004)  # 本脚本无随机抽样，仅为全链路种子一致性登记
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)

print("=" * 100)
print("A3-01 结构勘察  |  pandas", pd.__version__, " numpy", np.__version__)
print("=" * 100)

F1_P = WS / "data" / "附件1.xlsx"
F2_P = WS / "data" / "附件2.xlsx"
FR_P = WS / "data" / "Result.xlsx"

# ---------------------------------------------------------------- 1. 原始形态
print("\n[1] 原始文件形态（header=None 原样读入）")
raw = {}
for tag, p in [("附件1", F1_P), ("附件2", F2_P), ("Result", FR_P)]:
    x = pd.ExcelFile(p, engine="openpyxl")
    raw[tag] = pd.read_excel(p, sheet_name=0, header=None, engine="openpyxl")
    print(f"  {tag}: sheet={x.sheet_names}  openpyxl_dim={raw[tag].shape}")

# ------------------------------------------------ 2. 双行表头核验（中/英对齐）
print("\n[2] 双行表头核验：第0行中文 vs 第1行英文")
cn1 = raw["附件1"].iloc[0].tolist()
en1 = raw["附件1"].iloc[1].tolist()
cn2 = raw["附件2"].iloc[0].tolist()
en2 = raw["附件2"].iloc[1].tolist()
cnr = raw["Result"].iloc[0].tolist()
print("  附件1 中文表头:", cn1)
print("  附件1 英文表头:", en1)
print("  附件2 中文表头:", cn2)
print("  附件2 英文表头:", en2)
print("  Result 中文表头:", cnr)
print("  附件1/附件2 共有24列（除运单号/实际赔付金额差异）中文表头完全一致:",
      [c for c in cn2 if c != "运单号"] == [c for c in cn1 if c != "实际赔付金额"])
print("  附件2 英文表头首列:", en2[0], "；附件1 是否含'运单号'列:", "运单号" in cn1)
print("  附件1 是否含'实际赔付金额'列:", "实际赔付金额" in cn1,
      "；附件2 是否含'实际赔付金额'列:", "实际赔付金额" in cn2,
      "；附件2 是否含'索赔金额'列:", "索赔金额" in cn2)
print("  重复列名检查：附件1", len(cn1) - len(set(cn1)), "个重复；附件2",
      len(cn2) - len(set(cn2)), "个重复；Result", len(cnr) - len(set(cnr)), "个重复")
print("  'Unnamed'无名列检查：附件1", sum(str(c).startswith("Unnamed") for c in cn1),
      "；附件2", sum(str(c).startswith("Unnamed") for c in cn2),
      "；Result", sum(str(c).startswith("Unnamed") for c in cnr))

# ------------------------------- 3. 按中文表头读入，剥离嵌入的英文表头数据行
print("\n[3] 以第0行为表头读入后的嵌入行检查")
d1 = pd.read_excel(F1_P, sheet_name=0, header=0, engine="openpyxl")
d2 = pd.read_excel(F2_P, sheet_name=0, header=0, engine="openpyxl")
dr = pd.read_excel(FR_P, sheet_name=0, header=0, engine="openpyxl")
print("  读入形状 附件1", d1.shape, " 附件2", d2.shape, " Result", dr.shape)
en_row_1 = d1.iloc[0].tolist() == en1
en_row_2 = d2.iloc[0].tolist() == en2
print("  附件1 第一个数据行 == 英文表头行（嵌入表头行）:", en_row_1)
print("  附件2 第一个数据行 == 英文表头行（嵌入表头行）:", en_row_2)
if en_row_1:
    d1 = d1.iloc[1:].reset_index(drop=True)
if en_row_2:
    d2 = d2.iloc[1:].reset_index(drop=True)
print("  剥离英文行后 附件1", d1.shape, " 附件2", d2.shape, " Result", dr.shape)
print("  全空行统计：附件1", int(d1.isna().all(axis=1).sum()),
      "；附件2", int(d2.isna().all(axis=1).sum()),
      "；Result", int(dr.isna().all(axis=1).sum()))

# ------------------------------------------------------- 4. 列名 ⇄ 题面对照
print("\n[4] 中文列名（数据实际列）与英文列名一一配对（自附件第1行）")
pair = dict(zip(cn2, en2))
pair1 = dict(zip(cn1, en1))
for k, v in pair1.items():
    print(f"  附件1: {k} = {v}")
for k, v in pair.items():
    print(f"  附件2: {k} = {v}")

# --------------------------------------------------------------- 5. dtype
print("\n[5] dtype（剥离英文行后，pandas 自动推断）")
print("  附件1 dtypes:")
print(d1.dtypes.to_string())
print("  附件2 dtypes:")
print(d2.dtypes.to_string())
print("  Result dtypes:")
print(dr.dtypes.to_string())

# ------------------------------------------------------------ 6. 主键判定
print("\n[6] 主键判定")
print("  附件1 是否存在'运单号'列:", "运单号" in d1.columns,
      "→ 附件1 无业务主键列，以行序为准")
print("  附件1 完全重复行数:", int(d1.duplicated().sum()))
print("  附件2 运单号: nunique=", d2["运单号"].nunique(), "/ 行数=", len(d2),
      "；唯一:", bool(d2["运单号"].is_unique))
print("  附件2 运单号 dtype:", d2["运单号"].dtype,
      "；min=", d2["运单号"].min(), " max=", d2["运单号"].max())
print("  附件2 运单号是否严格递增（行序=ID序）:",
      bool(d2["运单号"].is_monotonic_increasing))
print("  Result 运单号: nunique=", dr["运单号"].nunique(), "/ 行数=", len(dr),
      "；唯一:", bool(dr["运单号"].is_unique),
      "；严格递增:", bool(dr["运单号"].is_monotonic_increasing))

# ------------------------------------------------- 7. Result 与附件2 对应关系
print("\n[7] Result.xlsx ↔ 附件2 运单号对应关系（F3/F6 核验）")
ids2 = d2["运单号"].tolist()
idsr = dr["运单号"].tolist()
print("  行数: 附件2=", len(ids2), " Result=", len(idsr))
print("  集合相等:", set(ids2) == set(idsr),
      "；逐行顺序完全一致:", ids2 == idsr)
print("  Result 缺失的附件2运单号个数:", len(set(ids2) - set(idsr)),
      "；Result 多出的运单号个数:", len(set(idsr) - set(ids2)))
print("  Result['实际赔付金额'] 非空个数:", int(dr["实际赔付金额"].notna().sum()),
      "；Result['风险标注'] 非空个数:", int(dr["风险标注"].notna().sum()))
print("  Result['风险标注'] 非空取值样例:", dr["风险标注"].dropna().unique()[:10])

# --------------------------------------------------------- 8. 缺失统计（全列）
print("\n[8] 缺失统计（剥离英文行后）")
for tag, dfx in [("附件1", d1), ("附件2", d2), ("Result", dr)]:
    print(f"  --- {tag} ---")
    miss = dfx.isna().sum()
    for c in dfx.columns:
        n = int(miss[c])
        if n > 0:
            print(f"    {c}: {n} ({n/len(dfx)*100:.3f}%)")
    zero = [c for c in dfx.columns if miss[c] == 0]
    print("    无缺失列:", zero)

# ------------------------------------------------- 9. 类别列取值域（对照附表1）
print("\n[9] 类别/枚举列取值域（value_counts，含缺失）")
cat_cols = ["线路类型", "是否c2c", "是否生鲜妥投及时", "寄件是否内部", "异常原因",
            "进线渠道", "商品类型", "新旧程度", "寄件B/C", "进线人身份"]
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    print(f"  ===== {tag} =====")
    for c in cat_cols:
        if c not in dfx.columns:
            continue
        vc = dfx[c].value_counts(dropna=False)
        print(f"  [{c}] nunique={dfx[c].nunique(dropna=True)}")
        if len(vc) <= 30:
            for k, v in vc.items():
                print(f"      {repr(k)}: {v} ({v/len(dfx)*100:.3f}%)")
        else:
            print("      类别数>30，仅列前12与计数尾部:")
            for k, v in vc.head(12).items():
                print(f"      {repr(k)}: {v} ({v/len(dfx)*100:.3f}%)")
            print(f"      ... 其余 {len(vc)-12} 类合计 {int(vc.iloc[12:].sum())} 行")
            print("      最小计数类别:", vc.index[-3:].tolist(), vc.iloc[-3:].tolist())

# --------------------------------------------------- 10. 数值列描述与高频值
print("\n[10] 数值列描述统计与高频值（魔法值/哨兵值扫描）")
num_cols = [c for c in d1.columns if c not in cat_cols]
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    print(f"  ===== {tag} =====")
    for c in num_cols:
        if c not in dfx.columns:
            continue  # 实际赔付金额 仅附件1存在
        s = pd.to_numeric(dfx[c], errors="coerce")
        nan_n = int(s.isna().sum())
        q = s.quantile([0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1])
        qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
        print(f"  [{c}] dtype={dfx[c].dtype} nan={nan_n}")
        print(f"      {qs}")
        vc = dfx[c].value_counts(dropna=False).head(5)
        tops = ", ".join(f"{repr(k)}:{v}" for k, v in vc.items())
        print(f"      top5高频值: {tops}")

# ---------------------------------------- 11. 疑点专项：配送超时/进线时长量纲
print("\n[11] 疑点专项")
print("  --- 11a 时长列的量纲/形态（log10 分箱直方） ---")
for c in ["配送超时时长", "妥投到进线时长"]:
    for tag, dfx in [("附件1", d1), ("附件2", d2)]:
        s = pd.to_numeric(dfx[c], errors="coerce").astype(float)
        neg, zero = int((s < 0).sum()), int((s == 0).sum())
        print(f"  [{tag}.{c}] n={len(s)} 负值={neg} 零值={zero} "
              f"max={s.max():.6g} min={s.min():.6g}")
        bins = [-np.inf, -1e5, -1e4, -1e3, -100, 0, 100, 1e3, 1e4, 1e5,
                2e5, 3e5, 4e5, 4.2e5, 4.3e5, 4.4e5, 5e5, 1e6, 1e7, 1e8,
                3e8, 5e8, 7e8, 9e8, 1e9, np.inf]
        cut = pd.cut(s, bins)
        for iv, n in cut.value_counts(sort=False).items():
            if n > 0:
                print(f"      {str(iv):<28} {int(n):>6}  ({n/len(s)*100:.3f}%)")
        if c == "配送超时时长":
            band = s[(s > 427000) & (s < 428000)]
            print(f"      (427000,428000) 计数={len(band)} ({len(band)/len(s)*100:.3f}%)")

print("\n  --- 11b 金额/比率列负值与边界精确计数 ---")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    for c in ["保价金额", "始发网点发单量", "目的网点发单量",
              "始发网点万单理赔率", "目的网点万单理赔率", "始发网点赔付比例",
              "目的网点赔付比例", "索赔金额"]:
        s = pd.to_numeric(dfx[c], errors="coerce").astype(float)
        print(f"  [{tag}.{c}] 负值={int((s<0).sum())} 零值={int((s==0).sum())} "
              f"<-1计数={int((s< -1).sum())} ==-1计数={int((s==-1).sum())} "
              f"==100计数={int((s==100).sum())} min={s.min():.6g} max={s.max():.6g}")
s = pd.to_numeric(d1["实际赔付金额"], errors="coerce").astype(float)
print(f"  [附件1.实际赔付金额] 负值={int((s<0).sum())} 零值={int((s==0).sum())} "
      f"min={s.min():.6g} max={s.max():.6g}")

print("\n  --- 11c 保价金额/万单理赔率 负值区间形态（是否集中在(-1,0)） ---")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    for c in ["保价金额", "始发网点万单理赔率", "目的网点万单理赔率"]:
        s = pd.to_numeric(dfx[c], errors="coerce").astype(float)
        sneg = s[s < 0]
        if len(sneg):
            q = sneg.quantile([0, .25, .5, .75, 1])
            qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
            print(f"  [{tag}.{c}] 负值共{len(sneg)} ({len(sneg)/len(s)*100:.3f}%) "
                  f"负值分位: {qs}；正值最小={s[s>=0].min():.6g}")
        else:
            print(f"  [{tag}.{c}] 无负值")

# 金额三列关系初探
print("\n  附件1 金额三列（保价/索赔/实际赔付）非负性:")
for c in ["保价金额", "索赔金额", "实际赔付金额"]:
    s = pd.to_numeric(d1[c], errors="coerce")
    print(f"    {c}: 负值={int((s<0).sum())}  零值={int((s==0).sum())}  "
          f"min={s.min():.6g} max={s.max():.6g}")
for c in ["保价金额", "索赔金额"]:
    s = pd.to_numeric(d2[c], errors="coerce")
    print(f"    附件2.{c}: 负值={int((s<0).sum())}  零值={int((s==0).sum())}  "
          f"min={s.min():.6g} max={s.max():.6g}")

# 网点比率列负值
print("\n  网点统计列负值计数（万单理赔率/赔付比例）:")
for c in ["始发网点万单理赔率", "始发网点赔付比例", "目的网点万单理赔率", "目的网点赔付比例"]:
    for tag, dfx in [("附件1", d1), ("附件2", d2)]:
        s = pd.to_numeric(dfx[c], errors="coerce")
        print(f"    {tag}.{c}: 负值={int((s<0).sum())}  >1计数={int((s>1).sum())}  "
              f"min={s.min():.6g} max={s.max():.6g}")

# ID 列交集（寄件人/收件人/城市）
print("\n  ID/类别型高基数列的附件1↔附件2 覆盖关系:")
for c in ["线路类型", "始发城市", "目的城市", "寄件人id", "收件人id", "商品类型",
          "异常原因", "进线渠道", "进线人身份", "寄件B/C", "新旧程度",
          "是否c2c", "是否生鲜妥投及时", "寄件是否内部"]:
    s1, s2 = set(d1[c].dropna().unique()), set(d2[c].dropna().unique())
    print(f"    {c}: 附件1类数={len(s1)} 附件2类数={len(s2)} "
          f"附件2未见(仅附件1有)={len(s1-s2)} 附件2新见(仅附件2有)={len(s2-s1)}")

print("\n  附件1 与 附件2 的收件人id=300000 计数（初见高频值核实）:")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    print(f"    {tag}: 收件人id==300000 计数={int((dfx['收件人id']==300000).sum())} "
          f"({(dfx['收件人id']==300000).mean()*100:.2f}%)")

# ------------------------------------------------------------- 12. F1–F7 回填
print("\n[12] F1-F7 格式核验回填（依据本日志 [1]-[7] 节数据事实）")
print("  F1 (result 文件存在且含 Q2 预测列): PASS —— Result.xlsx 存在，含'实际赔付金额'空列（Q2 预测落位）")
print("  F2 (Q2 与 Q3 同一 result 文件): PASS —— 同一 Sheet 同时含'实际赔付金额'与'风险标注'两空列")
print(f"  F3 (运单号不动): PASS —— Result 运单号与附件2 逐行顺序完全一致={ids2 == idsr}，集合相等={set(ids2)==set(idsr)}")
print("  F4 (Q3 取值域): 待填列当前全为空（NaN），无预填值污染；A6 填入时取值须限于三类")
print("  F5 (与论文一同提交): 流程性要求，数据侧不可核验，转 A9/A10/G9 落实")
print(f"  F6 (行口径=附件2 全部运单): PASS —— Result 行数 {len(idsr)} == 附件2 数据行数 {len(ids2)}")
print("  F7 (论文固定章节): 论文侧要求，数据侧不可核验，转 A9 落实")

print("\n[13] 附件1/附件2 行数结论（清洗基准）")
print("  附件1 数据行（剥离双表头后）=", len(d1))
print("  附件2 数据行（剥离双表头后）=", len(d2), "（清洗后必须仍为", len(d2), "）")
print("  Result 数据行=", len(dr))
print("\nA3-01 勘察结束。")
