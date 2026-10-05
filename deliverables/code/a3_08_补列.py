# -*- coding: utf-8 -*-
"""
A3-08 补列（盲测 Run-2，工作区 agent_workspace_B2；对应 A4 决策 D21 的数据侧收尾）
职责：为 output/tables/附件2_clean.csv 增补衍生列 索赔金额_log10（附件1_clean 已有同名列）。
  公式与 a3_06 完全一致：索赔金额_log10 = np.log10(索赔金额)
  （a3_06 中附件1 即此式；索赔金额两表均 >=100，a3_02 [5]，log10 无定义域问题）。
  列位置：插入到 保价索赔比（末列）之前，使 ["运单号"] + setdiff(附件1列, 附件1专有列)
  与 附件2_new 列序逐列一致（含顺序）。
实现方式：文本级插入（逐行 rsplit 末字段后插入新字段），其余字段字节原样保留——
  规避 pandas 读入-回写的浮点 1-ulp 解析噪声（实测回环 md5 漂移，见日志 [1]）。
  幂等：目标列已存在时跳过插入，仍执行全部核验后原样退出（不重写文件）。
校验：①插入轮：其余 46 列字节级保留证明（逐行删新增字段还原旧行）+ pandas 全表逐格严格相等；
      ②行数仍 2792；③shape/列名全对比与列序对齐；④新列公式回验 + 与附件1 同名列口径一致。
输入：output/tables/附件2_clean.csv（a3_06 最终版，md5 5708251b…）
输出：同名文件（2792×47）+ output/logs/a3_08_补列.log
红线：离线；只读本工作区；不碰 STATE.md 与他人文档。
"""
import sys
import csv
import io
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B2")
LOG = WS / "output" / "logs" / "a3_08_补列.log"
P2 = WS / "output" / "tables" / "附件2_clean.csv"
P1 = WS / "output" / "tables" / "附件1_clean.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)


class Tee:
    """日志追加模式：多次运行分段留痕（各段以分隔线+时间戳开始）。"""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
        self.so = sys.stdout

    def write(self, s):
        self.so.write(s)
        self.f.write(s)

    def flush(self):
        self.so.flush()
        self.f.flush()


sys.stdout = Tee(LOG)
np.random.seed(20251004)  # 无随机操作
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


print("=" * 100)
print("A3-08 补列 运行段开始 |", pd.Timestamp.now().isoformat(timespec="seconds"), "| pandas", pd.__version__)
print("公式（与 a3_06 附件1 完全一致）：索赔金额_log10 = np.log10(索赔金额)；插入位置=末列 保价索赔比 之前")
print("读取口径：本脚本所有 read_csv 一律 encoding='utf-8-sig'（clean.csv 由 a3_03/a3_06 以 utf-8-sig 落盘，")
print("          带BOM；若用默认 utf-8 读取，首列名会带 \\ufeff 前缀——此使用注意已记入数据报告 §7。）")
print("=" * 100)

md5_old = md5(P2)
raw = P2.read_bytes()
body = raw.split(b"\n")[:-1]  # 去掉末尾空片段
print(f"[0] 当前文件 md5={md5_old}；行片段数={len(body)}（含表头）；行尾=CRLF")
hdr_fields = next(csv.reader([body[0].decode("utf-8")]))

inserted_now = False
if "索赔金额_log10" not in hdr_fields:
    # ---------------- 文本级插入 ----------------
    i_claim = hdr_fields.index("索赔金额")
    print(f"[1] 旧表 {len(hdr_fields)} 列；索赔金额 字段序号={i_claim}；末字段={hdr_fields[-1]}")
    # 表头：只把末字段前的 "," 换成 ",索赔金额_log10,"，其后字节（含行尾 \r）原样保留
    idx_last_sep = body[0].index(b"," + hdr_fields[-1].encode("utf-8"))
    new_lines = [body[0][:idx_last_sep] + ",索赔金额_log10,".encode("utf-8") + body[0][idx_last_sep + 1:]]
    n_bad_quote = 0
    for ln in body[1:]:
        head, last = ln.rsplit(b",", 1)
        if b'"' in last:
            n_bad_quote += 1
        fields = next(csv.reader([head.decode("utf-8")]))
        v = float(fields[i_claim])
        nv = repr(float(np.log10(np.float64(v))))
        new_lines.append(head + b"," + nv.encode("utf-8") + b"," + last)
    assert n_bad_quote == 0, "末字段含引号，文本插入不安全"
    assert len(new_lines) == len(body)
    P2.write_bytes(b"\n".join(new_lines) + b"\n")
    inserted_now = True
    print(f"[2] 文本插入完成：新行片段数={len(new_lines)}；末字段引号检查 PASS（{n_bad_quote} 个异常）")
    print(f"[3] 落盘：新 md5={md5(P2)}（旧 {md5_old}）")

    # ---- 校验 A：字节级——逐行"删除倒数第2字段后还原为原行"
    ok_bytes = True
    for old_ln, new_ln in zip(body, new_lines):
        parts = new_ln.rsplit(b",", 2)
        if parts[0] + b"," + parts[2] != old_ln:
            ok_bytes = False
            break
    print(f"[4] 字节级保留证明：全部 {len(body)} 行删除新增字段后与原行逐字节一致：{'PASS' if ok_bytes else 'FAIL'}")
    assert ok_bytes
else:
    print(f"[幂等] 目标列已存在（字段序号={hdr_fields.index('索赔金额_log10')}），跳过插入，仅核验。")

# ---- 校验 B：pandas 全表逐格严格相等 + 行数/shape/列序
d_new = pd.read_csv(P2, encoding="utf-8-sig")
r1 = pd.read_csv(P1, encoding="utf-8-sig")
assert len(d_new) == 2792, "行数不等于 2792！"
print(f"[5] 行数校验：{len(d_new)} == 2792 PASS；shape {d_new.shape}")

if inserted_now:
    d_old = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert len(d_old) == 2792
    all_eq = True
    for c in d_old.columns:
        if not d_old[c].equals(d_new[c]):
            all_eq = False
            print(f"    不等列：{c}")
    print(f"[6] 其余 46 列逐格严格相等（pandas .equals；旧列字节同源，判据为精确相等）：{'PASS' if all_eq else 'FAIL'}")
    assert all_eq

feats1_specific = ["行ID", "实际赔付金额", "索赔差额", "赔付索赔比", "相对超额", "实际赔付金额_log10"]
seq1 = ["运单号"] + [c for c in r1.columns if c not in feats1_specific]
print(f"[7] 列序对齐校验：['运单号'] + setdiff(附件1列, 附件1专有{feats1_specific}) == 附件2_new 列序 ："
      f"{seq1 == list(d_new.columns)}")
assert seq1 == list(d_new.columns)
print("    附件2_new 列序（尾 8 列）:", list(d_new.columns[-8:]))

# ---- 校验 C：新列公式回验 + 与附件1 同名列口径一致 + 定义域
dev = float((d_new["索赔金额_log10"] - np.log10(d_new["索赔金额"].astype(float))).abs().max())
dev1 = float((r1["索赔金额_log10"] - np.log10(r1["索赔金额"].astype(float))).abs().max())
print(f"[8] 新列公式回验：附件2 max 偏差={dev:.3g}（阈值 1e-12）；附件1 同名列回验 max 偏差={dev1:.3g}")
assert dev <= 1e-12 and dev1 <= 1e-12
assert d_new["索赔金额_log10"].notna().all() and d_new["索赔金额"].min() >= 100
print("    定义域核验：索赔金额 min>=100（a3_02 [5]），全列无 NaN PASS")
idx_show = [0, 1, 2, int(d_new["索赔金额"].idxmin()), int(d_new["索赔金额"].idxmax())]
print("    抽样（首 3 行 + 索赔金额 最小/最大行）：")
print(d_new.loc[idx_show, ["运单号", "索赔金额", "索赔金额_log10"]].to_string(float_format=lambda v: f"{v:.6g}"))

print(f"\n[9] 变更摘要：附件2_clean.csv 2792×46 -> {d_new.shape[0]}×{d_new.shape[1]}，"
      "新增 索赔金额_log10（位置 45，保价索赔比 之前）；其余 46 列字节级原样保留、数值逐格严格相等；行数 2792 不变。")
print("A3-08 补列结束。")
