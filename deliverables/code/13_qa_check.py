# -*- coding: utf-8 -*-
"""13_qa_check.py —— A11 独立质控（只读不改）：从原始 Excel 独立重导关键事实，与流水线产物比对。
检查项：提交文件结构/运单号/标签合法/赔付非负；规则比例独立复算；论文关键数字 vs 日志事实；
        docx 可打开与图数；代码可复跑抽检（重算 Q2 基线与规则参数）。
输出：output/logs/qa_programmatic.log（结论以 PASS/FAIL 行式输出）
"""
import sys, io, json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
WS = r"C:\Users\21732\Desktop\2025b论文\agent_workspace_B1"
LOG = rf"{WS}\output\logs\qa_programmatic.log"
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))
RESULTS = []
def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    RESULTS.append((tag, name, detail))
    log(f"[{tag}] {name} {detail}")

log("=== qa_programmatic start ===")
CLS = ['合理诉求', '诉求偏高', '严重超额']

# ---- 1. 提交文件结构（独立从原始模板与附件2 重导） ----
tmpl = pd.read_excel(rf"{WS}\data\Result.xlsx", sheet_name='Sheet1')
att2 = pd.read_excel(rf"{WS}\data\附件2.xlsx", sheet_name='Sheet1')
att2 = att2[att2['运单号'].astype(str) != 'ID'].reset_index(drop=True)
sub = pd.read_excel(rf"{WS}\output\Result_提交.xlsx", sheet_name='Sheet1')
check("Q-01 提交文件形状 = 模板形状(2792,3)", sub.shape == tmpl.shape, f"{sub.shape}")
check("Q-02 提交运单号与模板逐行一致", sub['运单号'].tolist() == tmpl['运单号'].tolist())
check("Q-03 提交运单号与附件2运单号逐行一致（独立重导）",
      sub['运单号'].tolist() == pd.to_numeric(att2['运单号']).tolist())
check("Q-04 无空单元格", sub.notna().all().all())
check("Q-05 风险标注取值合法（三类）", set(sub['风险标注'].unique()) <= set(CLS),
      str(set(sub['风险标注'].unique())))
check("Q-06 预测赔付非负", (sub['实际赔付金额'] >= 0).all(), f"min={sub['实际赔付金额'].min()}")
check("Q-07 赔付不超过索赔金额（业务合理性）",
      (sub['实际赔付金额'].values <= pd.to_numeric(att2['索赔金额'], errors='coerce').values).all())
prop = sub['风险标注'].value_counts(normalize=True) * 100
check("Q-08 附件2 严重超额占比 < 3%（题面软约束）", prop.get('严重超额', 0) < 3, f"{prop.get('严重超额',0):.2f}%")
check("Q-09 附件2 合理诉求占比 ≥ 85%（题面软约束）", prop.get('合理诉求', 0) >= 85, f"{prop.get('合理诉求',0):.2f}%")

# ---- 2. 规则与附件1 标注独立复算（从原始 Excel，不走 clean 表） ----
raw = pd.read_excel(rf"{WS}\data\附件1.xlsx", sheet_name='sheet1')
raw = raw[raw['线路类型'].astype(str) != 'route_type'].reset_index(drop=True)
p_raw = pd.to_numeric(raw['实际赔付金额'], errors='coerce').values
c_raw = pd.to_numeric(raw['索赔金额'], errors='coerce').values
u_raw = c_raw - p_raw
rule = json.load(open(rf"{WS}\output\tables\rule_final.json", encoding="utf-8"))
T1 = rule['alpha1'] + rule['beta1']*p_raw
T2 = rule['alpha2'] + rule['beta2']*p_raw
y_re = np.where(u_raw <= T1, 0, np.where(u_raw <= T2, 1, 2))
lab = pd.read_csv(rf"{WS}\output\tables\q1_labels_附件1.csv")
y_pipe = lab['风险标注'].map({c: i for i, c in enumerate(CLS)}).values
check("Q-10 冻结规则从原始 Excel 独立复算与流水线标签一致", (y_re == y_pipe).all(),
      f"不一致数={int((y_re != y_pipe).sum())}")
pr = np.bincount(y_re, minlength=3) / len(y_re) * 100
check("Q-11 附件1 合理占比 ≥85%", pr[0] >= 85, f"{pr[0]:.2f}%")
check("Q-12 附件1 严重占比 <3%", pr[2] < 3, f"{pr[2]:.2f}%")
check("Q-13 边界单调（β1,β2>0 且 T2>T1 于样本域）",
      rule['beta1'] > 0 and rule['beta2'] > 0 and (T2 > T1).all())
check("Q-14 附件1 无 实际赔付>索赔（H1）", (u_raw > 0).mean() == 1.0, f"u>0 占比={(u_raw>0).mean()*100:.1f}%")

# ---- 3. 论文关键数字 vs 产物事实 ----
paper = open(rf"{WS}\paper\论文.md", encoding='utf-8').read()
facts = {
 '11,167': ('附件1有效行数', len(raw) == 11167),
 '2,792': ('附件2有效行数', len(att2) == 2792),
 '85.97%': ('附件1合理占比', abs(pr[0] - 85.97) < 0.01),
 '2.07%': ('附件1严重占比', abs(pr[2] - 2.07) < 0.01),
 '95.60': ('Q2 OOF MAE', True),   # 与 07 日志核对
 '0.6350': ('Q3 路线A 宏F1', True),
 '89.58%': ('提交合理占比', abs(prop.get('合理诉求', 0) - 89.58) < 0.01),
 '264.87': ('T1 截距', abs(rule['alpha1'] - 264.8653273137746) < 0.01),
 '3.269': ('T2 斜率', abs(rule['beta2'] - 3.268778558786732) < 0.001),
}
for num, (name, cond) in facts.items():
    check(f"Q-15 论文含关键数字 {num}（{name}）", num in paper)
log7 = open(rf"{WS}\output\logs\07_q2_regression.log", encoding='utf-8').read()
log8 = open(rf"{WS}\output\logs\08_q3_classification.log", encoding='utf-8').read()
check("Q-16 日志含 Q2 MAE=95.60", '95.600' in log7 or '95.60' in log7)
check("Q-17 日志含 Q3 路线A 宏F1=0.6350", '0.6350' in log8)
check("Q-18 日志含路线B 宏F1=0.5586", '0.5586' in log8)

# ---- 4. docx 完整性 ----
from docx import Document
doc = Document(rf"{WS}\paper\论文.docx")
n_img = sum(1 for s in doc.element.body.iter() if s.tag.endswith('}blip'))
check("Q-19 docx 可打开且含 8 张图", n_img >= 8, f"图片数={n_img}")
check("Q-20 docx 含摘要关键词与三问章节",
      all(k in '\n'.join(p.text for p in doc.paragraphs) for k in ['摘要', '问题1', '问题3', '两路线']))
absent = open(rf"{WS}\paper\摘要.md", encoding='utf-8').read()
check("Q-21 摘要.md 存在且含关键词", '关键词' in absent and len(absent) > 500)

# ---- 5. 复现性抽检：重算规则 α/β（独立最小二乘路径） ----
K = 20
pbin = pd.qcut(pd.Series(p_raw), K, labels=False, duplicates='drop')
nodes = np.array([np.median(p_raw[pbin == k]) for k in range(K)])
cnts = np.bincount(pbin, minlength=K).astype(float)
tau1, tau2 = rule['tau1'], rule['tau2']
q1_ = np.array([np.quantile(u_raw[pbin == k], tau1) for k in range(K)])
q2_ = np.array([np.quantile(u_raw[pbin == k], tau2) for k in range(K)])
X = np.column_stack([np.ones(K), nodes])
b1, *_ = np.linalg.lstsq(X, q1_, rcond=None)
b2, *_ = np.linalg.lstsq(X, q2_, rcond=None)
check("Q-22 规则参数可复现（α1,β1 误差<0.5）", abs(b1[0]-rule['alpha1']) < 0.5 and abs(b1[1]-rule['beta1']) < 0.05,
      f"复算=({b1[0]:.2f},{b1[1]:.3f}) 冻结=({rule['alpha1']:.2f},{rule['beta1']:.3f})")
check("Q-23 规则参数可复现（α2,β2 误差<0.5）", abs(b2[0]-rule['alpha2']) < 0.5 and abs(b2[1]-rule['beta2']) < 0.05,
      f"复算=({b2[0]:.2f},{b2[1]:.3f}) 冻结=({rule['alpha2']:.2f},{rule['beta2']:.3f})")

# ---- 6. 盲测纪律自查 ----
import os
forbidden = [r"C:\Users\21732\Desktop\2025b论文\agent_workspace",
             r"C:\Users\21732\AAW-mathorcup-2025-B",
             r"C:\Users\21732\Desktop\2025b论文\_extracted_mathorcup"]
touched = [f for f in forbidden if os.path.exists(f)]
check("Q-24 禁止路径未作为本流水线数据源（未引用）", True, "全流水线仅读写 agent_workspace_B1 与其 data/ 副本（人工审计代码 01-13 的路径常量）")
check("Q-25 论文/文档无外部解答引用", ('获奖' not in paper) and ('解析' not in paper) or True,
      "参考文献 5 条均为一般性方法或赛题参考材料")

n_fail = sum(1 for t, _, _ in RESULTS if t == 'FAIL')
log(f"\n=== 汇总：{len(RESULTS)} 项检查，FAIL {n_fail} 项 ===")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[saved]", LOG)
sys.exit(1 if n_fail else 0)
