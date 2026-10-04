# -*- coding: utf-8 -*-
"""
A3-04 目标关系勘察（盲测 Run-2，工作区 agent_workspace_B2）
职责：只陈述数据事实，不做建模、不设标注阈值。围绕题面 C1-C6 相关维度，对附件1 的
      实际赔付金额（Q2 目标）与索赔差额（=实际赔付金额-索赔金额，题面 L19 定义式，D09 仲裁口径）
      给出分布、相关性、分箱均值、联合结构等事实，逐条编号 T1, T2, ... 供 A4 引用。
输入：output/tables/附件1_clean.csv（a3_03 产物；衍生差额在本脚本内按 L19 定义式计算）
输出：output/logs/a3_04_目标关系.log
红线：离线；只读本工作区；所有数字以本日志为溯源。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_04_目标关系.log"
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
np.random.seed(20251004)  # 本脚本无随机抽样（分位/相关均为确定性计算）
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 200)

print("=" * 100)
print("A3-04 目标关系勘察 | pandas", pd.__version__, "| scipy", __import__("scipy").__version__)
print("口径声明：索赔差额 = 实际赔付金额 - 索赔金额（题面 L19 定义式；D09 仲裁 D1：'理赔差额'与'索赔差额'同一量，")
print("          附件1 无名为'理赔差额/索赔差额'的物理列，须按定义式衍生——a3_01 [2][4] 列名核查为据）。")
print("=" * 100)

d1 = pd.read_csv(WS / "output" / "tables" / "附件1_clean.csv")
print(f"[0] 读入 附件1_clean.csv {d1.shape}（a3_03 产物）")
n = len(d1)
pay = d1["实际赔付金额"].astype(float)
claim = d1["索赔金额"].astype(float)
diff = pay - claim                      # 索赔差额（L19）
ratio = pay / claim                     # 赔付/索赔比
rel_excess = (claim - pay) / claim      # 相对超额 = (索赔-赔付)/索赔，尺度无关
d1["索赔差额"] = diff
d1["赔付索赔比"] = ratio
d1["相对超额"] = rel_excess

QS = [0, .001, .01, .05, .10, .25, .50, .75, .90, .95, .99, .999, 1]


def qline(s: pd.Series, name: str):
    q = s.quantile(QS)
    print(f"  [{name}] " + " ".join(f"{k*100:g}%={v:.4g}" for k, v in q.items()))


def numline(s: pd.Series, name: str):
    print(f"  [{name}] mean={s.mean():.4g} std={s.std():.4g} skew={s.skew():.4g} "
          f"nunique={s.nunique()}")


# ---------------------------------------------------------------- T1 目标分布
print("\n[T1] 实际赔付金额（Q2 目标变量）分布")
numline(pay, "实际赔付金额")
qline(pay, "实际赔付金额")
lg = np.log10(pay)
qline(lg, "log10(实际赔付金额)")
print(f"  事实：log10 空间 P1~P99 为 [{lg.quantile(.01):.3f}, {lg.quantile(.99):.3f}]，"
      f"跨度 {lg.quantile(.99)-lg.quantile(.01):.2f} 个数量级；原空间右偏（skew 见上）。")
b = pd.cut(pay, [0, 50, 100, 200, 400, 800, 1600, 3200, np.inf])
vc = b.value_counts(sort=False)
print("  原空间分箱计数:", {str(k): int(v) for k, v in vc.items()})

print("\n[T2] 索赔金额分布")
numline(claim, "索赔金额")
qline(claim, "索赔金额")
b2 = pd.cut(claim, [0, 100.001, 200, 400, 800, 1600, 3200, 6400, np.inf])
print("  分箱计数:", {str(k): int(v) for k, v in b2.value_counts(sort=False).items()})

# ---------------------------------------------------------------- T3 索赔差额
print("\n[T3] 索赔差额 = 实际赔付金额 - 索赔金额（L19 定义式衍生）")
numline(diff, "索赔差额")
qline(diff, "索赔差额")
npos, nzero, nneg = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
print(f"  符号结构：差额>0（实际赔付高于索赔）={npos} ({npos/n*100:.3f}%)；==0={nzero} ({nzero/n*100:.3f}%)；"
      f"<0（索赔高于实际赔付，即超额索赔）={nneg} ({nneg/n*100:.3f}%)")
print(f"  极值：min={diff.min():.4g}（该单 索赔={claim[diff.idxmin()]:.4g}, 赔付={pay[diff.idxmin()]:.4g}）；"
      f"max={diff.max():.4g}（该单 索赔={claim[diff.idxmax()]:.4g}, 赔付={pay[diff.idxmax()]:.4g}）")

# ---------------------------------------------------------------- T4 赔付/索赔比
print("\n[T4] 赔付/索赔比（尺度无关视图）")
numline(ratio, "赔付/索赔比")
qline(ratio, "赔付/索赔比")
eq1 = int(np.isclose(ratio, 1.0, rtol=1e-9).sum())
gt1 = int((ratio > 1 + 1e-9).sum())
lt1 = int((ratio < 1 - 1e-9).sum())
print(f"  比值 ==1（全额赔付）={eq1} ({eq1/n*100:.3f}%)；>1（赔付超索赔）={gt1} ({gt1/n*100:.3f}%)；"
      f"<1（赔付低于索赔）={lt1} ({lt1/n*100:.3f}%)")
b3 = pd.cut(ratio, [-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-9, 1 + 1e-9, np.inf])
print("  比值分箱计数:", {str(k): int(v) for k, v in b3.value_counts(sort=False).items()})

# ---------------------------------------------------------------- T5 相关性
print("\n[T5] 与 实际赔付金额 的相关系数（Pearson / Spearman，两两完备行）")
num_cols = ["索赔金额", "保价金额", "配送超时时长", "妥投到进线时长",
            "始发网点发单量", "始发网点万单理赔率", "始发网点赔付比例",
            "目的网点发单量", "目的网点万单理赔率", "目的网点赔付比例",
            "线路类型", "是否c2c", "是否生鲜妥投及时", "寄件是否内部", "新旧程度",
            "始发城市", "目的城市", "寄件人id", "收件人id"]
for c in num_cols:
    sub = d1[[c, "实际赔付金额"]].dropna()
    x = sub[c].astype(float)
    pr, pp = stats.pearsonr(x, sub["实际赔付金额"])
    sr, sp = stats.spearmanr(x, sub["实际赔付金额"])
    print(f"  {c:<14} Pearson={pr:+.4f} (p={pp:.2g})  Spearman={sr:+.4f} (p={sp:.2g})  n={len(sub)}")

# ---------------------------------------------------------------- T6 分箱均值
print("\n[T6] 分箱均值（C3/C4 相关原始事实：差额随赔付水平的走向）")
dq = pd.qcut(claim, 10, duplicates="drop")
tmp = pd.DataFrame({"单数": d1.groupby(dq, observed=True).size(),
                    "索赔均值": claim.groupby(dq, observed=True).mean(),
                    "赔付均值": pay.groupby(dq, observed=True).mean(),
                    "赔付中位": pay.groupby(dq, observed=True).median(),
                    "差额均值": diff.groupby(dq, observed=True).mean(),
                    "比值均值": ratio.groupby(dq, observed=True).mean(),
                    "相对超额均值": rel_excess.groupby(dq, observed=True).mean(),
                    "赔付>索赔占比": (ratio > 1).groupby(dq, observed=True).mean()})
print(tmp.to_string(float_format=lambda v: f"{v:.4g}"))

print("\n[T6b] 按 实际赔付金额 十分位的比值/差额走向（C4 单调性检验素材）")
pq = pd.qcut(pay, 10, duplicates="drop")
tmp2 = pd.DataFrame({"单数": d1.groupby(pq, observed=True).size(),
                     "赔付中位": pay.groupby(pq, observed=True).median(),
                     "差额均值": diff.groupby(pq, observed=True).mean(),
                     "差额中位": diff.groupby(pq, observed=True).median(),
                     "比值均值": ratio.groupby(pq, observed=True).mean(),
                     "比值中位": ratio.groupby(pq, observed=True).median(),
                     "相对超额中位": rel_excess.groupby(pq, observed=True).median()})
print(tmp2.to_string(float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- T7 相对超额分布
print("\n[T7] 相对超额 = (索赔金额-实际赔付金额)/索赔金额 的分布（C6 密集程度对比的原始事实）")
numline(rel_excess, "相对超额")
qline(rel_excess, "相对超额")
b4 = pd.cut(rel_excess, [-np.inf, -0.5, -0.2, -0.05, 0, 1e-9, 0.5, 0.8, 0.95, np.inf])
print("  相对超额分箱计数:", {str(k): int(v) for k, v in b4.value_counts(sort=False).items()})
cum = [(t, float((rel_excess <= t).mean())) for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]]
print("  相对超额<=t 的累计占比: " + ", ".join(f"t={t}: {v*100:.2f}%" for t, v in cum))

# ---------------------------------------------------------------- T8 联合结构
print("\n[T8] 实际赔付金额(行,十分位) × 相对超额(列,五分位) 单元格占比% —— C3~C6 联合结构原始事实")
rq = pd.qcut(rel_excess, 5, duplicates="drop")
mat = pd.crosstab(pq, rq, normalize="all") * 100
print(mat.to_string(float_format=lambda v: f"{v:.2f}"))
print("  （读法：若超额索赔运单集中于低赔付×高相对超额格、且同类格内相对超额取值窄，即对应题面"
      "'赔付越高需更高差额才判偏高/超额'与'同类差额应尽可能接近/密集程度有别'的数据基础——"
      "此处仅给占比矩阵，不设任何标注边界。）")

# ---------------------------------------------------------------- T9 保价关系
print("\n[T9] 保价金额与索赔/赔付的关系（附注 L83：保价为赔付上限语境）")
valid = d1["保价金额"].notna()
bv = d1.loc[valid, "保价金额"].astype(float)
cv = claim[valid]
pv = pay[valid]
print(f"  有效保价行数={int(valid.sum())}（哨兵剔除后，a3_03 R3）")
print(f"  索赔金额 > 保价金额 的行数={int((cv > bv).sum())} ({(cv > bv).mean()*100:.2f}% of 有效保价行)")
print(f"  实际赔付 > 保价金额 的行数={int((pv > bv).sum())} ({(pv > bv).mean()*100:.2f}% of 有效保价行)"
      f" —— 超上限赔付的事实规模")
pr, _ = stats.pearsonr(bv, pv)
sr, _ = stats.spearmanr(bv, pv)
print(f"  保价金额 与 实际赔付金额：Pearson={pr:+.4f} Spearman={sr:+.4f}")
pr2, _ = stats.pearsonr(bv, cv)
sr2, _ = stats.spearmanr(bv, cv)
print(f"  保价金额 与 索赔金额：Pearson={pr2:+.4f} Spearman={sr2:+.4f}")

# ---------------------------------------------------------------- T10 类别分组事实
print("\n[T10] 类别列分组的 目标均值/相对超额均值（原始分组事实，不做检验结论）")
for c in ["异常原因", "线路类型", "是否c2c", "新旧程度", "寄件B/C", "进线人身份", "商品类型"]:
    g = d1.groupby(c, observed=True).agg(单数=("实际赔付金额", "size"),
                                         赔付均值=("实际赔付金额", "mean"),
                                         赔付中位=("实际赔付金额", "median"),
                                         相对超额中位=("相对超额", "median"))
    g = g.sort_values("单数", ascending=False)
    show = g.head(8) if len(g) > 8 else g
    print(f"  --- {c}（{len(g)} 类；按单数取前 {len(show)}）---")
    print(show.to_string(float_format=lambda v: f"{v:.4g}"))

print("\n[T11] 缺失指示与目标的关系（'缺失即信息'的证据尺度，仅分组均值）")
for c, flag in [("异常原因", None), ("进线渠道", None)]:
    m = d1[c].isna()
    a = rel_excess[m]
    b = rel_excess[~m]
    print(f"  {c}: 缺失组 n={int(m.sum())} 相对超额中位={a.median():.4g}；非缺失组 n={int((~m).sum())} "
          f"相对超额中位={b.median():.4g}；赔付中位 缺失组={pay[m].median():.4g} vs 非缺失组={pay[~m].median():.4g}")

print("\nA3-04 目标关系勘察结束。事实编号 T1~T11。")
