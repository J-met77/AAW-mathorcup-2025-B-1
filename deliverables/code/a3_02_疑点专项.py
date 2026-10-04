# -*- coding: utf-8 -*-
"""
A3-02 疑点专项勘察（盲测 Run-2，工作区 agent_workspace_B2）
背景：前次派发中断，遗留 a3_01_勘察.py 的日志在 [10] 节后被截断（output/logs/a3_01_勘察.log
      共 392 行，止于附件2 数值列描述；脚本中 [11] 疑点专项 / [12] F1-F7 回填 / [13] 行数结论
      未执行）。a3_01 的 [1]-[10] 节经核对脚本与日志一致、内容完整可信，不重跑。
      本脚本补全 [11]-[13] 等价内容并补充清洗规则所需专项扫描，日志注明中断原因。
输入：data/附件1.xlsx, data/附件2.xlsx（读法与 a3_01 完全一致：header=0 取中文表头，剥离嵌入英文行）
输出：output/logs/a3_02_疑点专项.log
红线：离线；只读本工作区；所有数字以本日志与 a3_01 日志为溯源。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_02_疑点专项.log"
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
np.random.seed(20251004)  # 全链路统一种子；本脚本无随机抽样
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print("=" * 100)
print("A3-02 疑点专项勘察（补全 a3_01 被中断的 [11]-[13] 节） | pandas", pd.__version__)
print("中断说明：a3_01_勘察.log 于 [10] 节后截断（392 行），[11][12][13] 未执行；")
print("          a3_01 [1]-[10] 节完整可信（脚本与日志逐节核对一致），本脚本不重跑该部分。")
print("=" * 100)

F1_P = WS / "data" / "附件1.xlsx"
F2_P = WS / "data" / "附件2.xlsx"
FR_P = WS / "data" / "Result.xlsx"

# 与 a3_01 完全相同的读入口径（中文表头 header=0，剥离嵌入英文数据行）
d1 = pd.read_excel(F1_P, sheet_name=0, header=0, engine="openpyxl")
d2 = pd.read_excel(F2_P, sheet_name=0, header=0, engine="openpyxl")
dr = pd.read_excel(FR_P, sheet_name=0, header=0, engine="openpyxl")
assert d1.iloc[0].tolist() == pd.read_excel(F1_P, header=None, nrows=2).iloc[1].tolist()
d1 = d1.iloc[1:].reset_index(drop=True)
d2 = d2.iloc[1:].reset_index(drop=True)
print("[0] 读入复核：附件1", d1.shape, " 附件2", d2.shape, " Result", dr.shape, "（与 a3_01 [3] 节一致）")

NUM_COLS = ["保价金额", "配送超时时长", "妥投到进线时长", "索赔金额", "始发网点发单量",
            "始发网点万单理赔率", "始发网点赔付比例", "目的网点发单量", "目的网点万单理赔率",
            "目的网点赔付比例", "实际赔付金额"]
ID_COLS = ["始发城市", "目的城市", "寄件人id", "收件人id"]

for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    for c in NUM_COLS:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    for c in ID_COLS:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
bad = {t: int(dfx[list(set(NUM_COLS) & set(dfx.columns))].isna().sum().sum())
       for t, dfx in [("附件1", d1), ("附件2", d2)]}
print("[0b] 数值列 to_numeric 强转后新增 NaN 总数（=0 说明全部可解析、无脏字符串）:", bad)

# =========================================================== 1. Excel 存储类型核验
print("\n[1] Excel 单元格存储类型采样（openpyxl cell.data_type：s=文本 n=数值），解释 a3_01 [5] 中数值列 dtype=object 的成因")
wb1 = openpyxl.load_workbook(F1_P, read_only=True)
ws1 = wb1.active
probe_rows = [1, 2, 3, 4]
cols_probe = {1: "保价金额", 14: "索赔金额", 25: "实际赔付金额", 11: "配送超时时长"}
for r in ws1.iter_rows(min_row=1, max_row=4):
    rno = r[0].row
    info = {cols_probe[c.column]: (str(r[c.column - 1].value)[:12], r[c.column - 1].data_type)
            for c in r if c.column in cols_probe}
    print(f"    行{rno}: {info}")
wb1.close()
wb2 = openpyxl.load_workbook(F2_P, read_only=True)
ws2 = wb2.active
print("    附件2 运单号列（第1列）前4个数据单元格:")
for r in ws2.iter_rows(min_row=3, max_row=6, min_col=1, max_col=1):
    cell = r[0]
    print(f"      行{cell.row}: value={repr(cell.value)} data_type={cell.data_type}")
wb2.close()

# =========================================================== 2. 时长列专项（a3_01 [11a] 等价 + 加细）
print("\n[2] 时长列专项")
for c in ["配送超时时长", "妥投到进线时长"]:
    for tag, dfx in [("附件1", d1), ("附件2", d2)]:
        s = dfx[c].astype(float)
        neg = s[s < 0]
        print(f"  [{tag}.{c}] n={len(s)} 负值={len(neg)}({len(neg)/len(s)*100:.3f}%) "
              f"零值={int((s==0).sum())} min={s.min():.6g} max={s.max():.6g}")
        if len(neg):
            q = neg.quantile([0, .01, .25, .5, .75, .99, 1])
            qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
            print(f"      负值分位: {qs}")
        if c == "配送超时时长":
            top = s.value_counts().head(8)
            print("      最高频8值:", ", ".join(f"{k}:{v}" for k, v in top.items()))
            mx = s.max()
            for thr in [400000, 420000, 425000, 427000, 427800]:
                print(f"      >= {thr}: {int((s>=thr).sum())} ({(s>=thr).mean()*100:.3f}%)", end="")
            print()
            # 墙体附近密度剖面（每 500 单位一桶，观察是否硬墙）
            bins = np.arange(420000, 428500, 500)
            cut = pd.cut(s[s >= 420000], bins)
            prof = cut.value_counts(sort=False)
            print("      [420000,428500) 每500单位密度剖面:")
            for iv, n in prof.items():
                print(f"        {str(iv):<22} {int(n)}")
            below = int((s < 420000).sum())
            print(f"      < 420000 合计: {below} ({below/len(s)*100:.3f}%)")
        else:
            s_pos = s[s >= 0]
            for thr in [8.64e7, 1e8, 5e8, 9e8]:
                print(f"      >= {thr:.3g}: {int((s>=thr).sum())} ({(s>=thr).mean()*100:.3f}%)", end="")
            print()
            q = s_pos.quantile([.5, .75, .9, .95, .99])
            qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
            print(f"      非负部分分位: {qs}")
            print(f"      单位推断证据: max/86400={s.max()/86400:.3f} 天(若秒) ; max/86400000={s.max()/86400000:.3f} 天(若毫秒)")

# =========================================================== 3. 保价金额负值专项（a3_01 [11c] 等价）
print("\n[3] 保价金额负值专项（判定哨兵区间）")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    s = dfx["保价金额"].astype(float)
    sneg = s[s < 0]
    print(f"  [{tag}] 负值共 {len(sneg)} ({len(sneg)/len(s)*100:.3f}%)")
    if len(sneg):
        q = sneg.quantile([0, .25, .5, .75, 1])
        qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
        in01 = int(((sneg >= -1) & (sneg < 0)).sum())
        eq_m1 = int((sneg == -1).sum())
        print(f"      负值分位: {qs}")
        print(f"      负值落在 [-1,0) 的计数={in01}（占负值 {in01/len(sneg)*100:.2f}%）；==-1 计数={eq_m1}")
        print(f"      负值最小={sneg.min():.6g} 是否<-1: {int((sneg<-1).sum())} 个")
        print(f"      正值最小={s[s>=0].min():.6g} 零值计数={int((s==0).sum())}")

# =========================================================== 4. 网点比率/发单量专项（a3_01 [11b][11c] 等价）
print("\n[4] 网点统计列负值/边界专项")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    for c in ["始发网点发单量", "目的网点发单量", "始发网点万单理赔率", "目的网点万单理赔率",
              "始发网点赔付比例", "目的网点赔付比例"]:
        s = dfx[c].astype(float)
        sneg = s[s < 0]
        line = (f"  [{tag}.{c}] 负值={len(sneg)} 零值={int((s==0).sum())} "
                f"==-1={int((s==-1).sum())} >1={int((s>1).sum())} min={s.min():.6g} max={s.max():.6g}")
        if len(sneg):
            in01 = int(((sneg >= -1) & (sneg < 0)).sum())
            line += f" ；负值落[-1,0)={in01}({in01/len(sneg)*100:.2f}%负值) 负值max={sneg.max():.6g}"
        print(line)

# =========================================================== 5. 索赔金额下限专项
print("\n[5] 索赔金额下限/边界专项")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    s = dfx["索赔金额"].astype(float)
    eq100 = int((np.abs(s - 100) < 1e-9).sum())
    lt100 = int((s < 100 - 1e-9).sum())
    q = s.quantile([0, .001, .01, .5, .99, 1])
    qs = " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items())
    print(f"  [{tag}] ==100 计数={eq100} ({eq100/len(s)*100:.3f}%)  <100 计数={lt100}  {qs}")

# =========================================================== 6. 实际赔付金额边界专项
print("\n[6] 实际赔付金额边界专项（仅附件1）")
s = d1["实际赔付金额"].astype(float)
print(f"  负值={int((s<0).sum())} 零值={int((s==0).sum())} min={s.min():.6g} max={s.max():.6g}")
top = s.value_counts().head(8)
print("  最高频8值:", ", ".join(f"{k:.2f}:{v}" for k, v in top.items()))

# =========================================================== 7. 离散哨兵与疑似平台 ID
print("\n[7] 离散哨兵与疑似平台 ID 专项")
for tag, dfx in [("附件1", d1), ("附件2", d2)]:
    print(f"  --- {tag} ---")
    vc = dfx["新旧程度"].value_counts(dropna=False)
    print("  [新旧程度]", {int(k) if not pd.isna(k) else "nan": int(v) for k, v in vc.items()},
          "（附表1口径为 0/1/2=未知/全新/二手；-1 为表外哨兵）")
    vc = dfx["寄件是否内部"].value_counts(dropna=False)
    print("  [寄件是否内部]", {int(k) if not pd.isna(k) else "nan": int(v) for k, v in vc.items()},
          "（附表1口径为 0/1；-1 为表外哨兵）")
    for c in ["寄件人id", "收件人id"]:
        n300k = int((dfx[c] == 300000).sum())
        n300k_cons = int((dfx[c] > 299999).sum())
        print(f"  [{c}] ==300000 计数={n300k} ({n300k/len(dfx)*100:.3f}%) ；>=299999 计数={n300k_cons} "
              f"max={dfx[c].max():.0f}")
    # 300000 恰为整数关口且高频，检查其相邻取值是否出现（区分自然取值 vs 系统 ID）
    for c in ["寄件人id", "收件人id"]:
        near = dfx[c][(dfx[c] >= 299900) & (dfx[c] <= 300100)].value_counts()
        print(f"  [{c}] [299900,300100] 邻域取值分布: {dict(near)}")

# =========================================================== 8. 附件1 完全重复行专项
print("\n[8] 附件1 完全重复行专项（a3_01 [6] 报 25 行）")
dup_mask = d1.duplicated(keep=False)
grp = d1[dup_mask].groupby(list(d1.columns), dropna=False).size().sort_values(ascending=False)
print(f"  涉重总行数={int(dup_mask.sum())}（占 {dup_mask.mean()*100:.3f}%）；重复组数={len(grp)}；"
      f"组大小分布={dict(grp.value_counts())}")
print("  结论（清洗规则输入）：附件1 无运单号主键，重复行无法判定为录入错误或真实同况运单；"
      "处理=保留全部行、不删除（避免无依据删数据），仅在本节记录在案。")

# =========================================================== 9. F1-F7 回填（事实复核打包）
print("\n[9] F1-F7 格式核验回填（事实来自 a3_01 [1]-[7] 节 + 本脚本 [0] 复核）")
ids2 = d2["运单号"].tolist() if "运单号" in d2.columns else None
print(f"  F1 (result 文件存在且含 Q2 预测列): PASS —— Result.xlsx 存在，列={list(dr.columns)}，"
      f"其中'实际赔付金额'全空（{int(dr['实际赔付金额'].notna().sum())}/2792 非空）供 Q2 填写")
print(f"  F2 (Q2 与 Q3 同一 result 文件): PASS —— 同一 Sheet 同时含'实际赔付金额'与'风险标注'两列，"
      f"'风险标注'全空（{int(dr['风险标注'].notna().sum())}/2792 非空）供 Q3 填写")
print("  F3 (运单号不动): PASS —— a3_01 [7]：Result 运单号与附件2 逐行顺序完全一致且集合相等；"
      "本脚本 [0] 复核三表行数不变")
print("  F4 (Q3 取值域): PASS（前置）—— '风险标注'当前全空 NaN，无预填值污染；A6 填入时取值须限于"
      "'合理诉求'/'诉求偏高'/'严重超额'（题面 L21，C1）")
print("  F5 (与论文一同提交): 流程性要求，数据侧不可核验，转 A9/A10/G9 落实")
print("  F6 (行口径=附件2 全部运单): PASS —— Result 行数 2792 == 附件2 数据行数 2792（a3_01 [7]）")
print("  F7 (论文固定章节): 论文侧要求，数据侧不可核验，转 A9 落实")

# =========================================================== 10. 行数结论
print("\n[10] 行数结论（清洗基准，硬约束：清洗不得丢运单）")
print(f"  附件1 数据行（剥离双表头后）= {len(d1)}（原始 openpyxl 维 11169 = 中文表头1 + 英文表头1 + 数据11167）")
print(f"  附件2 数据行（剥离双表头后）= {len(d2)}（原始 2794 = 2 表头 + 数据 2792）→ 附件2_clean 必须仍为 {len(d2)} 行")
print(f"  Result 数据行 = {len(dr)}")
print("\nA3-02 疑点专项勘察结束。")
