# -*- coding: utf-8 -*-
"""
a10_02_format_check.py —— A10 阶段：格式合规检查（对照 A1 审题 F1–F7 清单）
- F1–F4/F6：对 output/Result_提交.xlsx 只读复核（不改任何产物）
- F5：论文 + result 文件双提交物在位
- F7：论文固定章节覆盖性核对（C15/C16/C9/C13/C14）
- docx 版式抽查：中文字体（宋体/黑体）、西文 Times New Roman、字号、行距 1.5、
  页边距 2.5cm、图片宽 15cm、表头加粗、标题样式 Heading 1/2/3
运行：python code/a10_02_format_check.py（日志 output/logs/a10_02_format_check.log）
"""
import io
import os
import sys

from docx import Document
from docx.shared import Cm
from docx.oxml.ns import qn

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "output", "logs", "a10_02_format_check.log")
_logf = io.open(LOG_PATH, "w", encoding="utf-8")


def log(s=""):
    print(s)
    _logf.write(s + "\n")
    _logf.flush()


def east_asia_of(run):
    rPr = run._element.rPr
    if rPr is None:
        return None
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        return None
    return rFonts.get(qn("w:eastAsia"))


def main():
    log("=" * 72)
    log("A10 格式合规检查（对照 00_admin/A1_审题.md 第五节 F1–F7）")
    log("=" * 72)

    md = io.open(os.path.join(ROOT, "paper", "论文.md"), encoding="utf-8").read()
    docx_path = os.path.join(ROOT, "paper", "论文.docx")
    result_path = os.path.join(ROOT, "output", "Result_提交.xlsx")
    att2_path = os.path.join(ROOT, "output", "tables", "附件2_clean.csv")

    # ---------- F1–F4/F6：result 文件只读复核 ----------
    log("[F1–F4/F6] result 文件只读复核（核验主体为 A6 日志 a6_06_提交组装校验.log，此处为独立复核）")
    xls = pd.ExcelFile(result_path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(result_path, sheet_name=sheet)
    log("  工作表=%s 形状=%s 列=%s" % (sheet, df.shape, list(df.columns)))
    ok_f1 = any("赔付" in c for c in df.columns)
    ok_f2 = any("标注" in c for c in df.columns)
    log("  F1 含问题2赔付预测列：%s｜F2 同文件含问题3风险标注列：%s" % (ok_f1, ok_f2))
    att2 = pd.read_csv(att2_path, encoding="utf-8-sig")
    id_col_df, id_col_att = df.columns[0], "运单号"
    same_ids = df[id_col_df].tolist() == att2[id_col_att].tolist()
    log("  F3 运单号与附件2逐行一致（顺序+取值零改动）：%s" % same_ids)
    label_col = [c for c in df.columns if "标注" in c][0]
    vals = set(df[label_col].dropna().unique().tolist())
    ok_f4 = vals <= {"合理诉求", "诉求偏高", "严重超额"}
    log("  F4 标注取值域=%s → 限于三类：%s" % (sorted(vals), ok_f4))
    n_null = int(df.isna().sum().sum())
    ok_f6 = (df.shape[0] == len(att2) == 2792) and n_null == 0
    log("  F6 行覆盖=附件2全部运单（%d 行）且无空值（空值=%d）：%s" % (df.shape[0], n_null, ok_f6))

    # ---------- F5 ----------
    log("[F5] 提交物在位：论文=%s result=%s" % (
        os.path.exists(docx_path), os.path.exists(result_path)))
    ok_f5 = os.path.exists(docx_path) and os.path.exists(result_path)

    # ---------- F7：固定章节覆盖 ----------
    log("[F7] 论文固定章节覆盖性核对（C15/C16/C9/C13/C14）")
    f7_map = {
        "C15 Q1 规则说明": ["5.1.2 坐标系选择与规则结构", "5.1.3 参数校准与网格择优"],
        "C16 Q1 结果及分析": ["5.1.4 附件 1 的风险标注结果", "5.1.5 规则检验"],
        "C9  Q2 评估指标说明": ["5.2.3 评估方案与强制基线", "5.2.4 求解结果"],
        "C13 Q3 不均衡处理方法": ["5.3.2 类别不均衡的两级杠杆处理"],
        "C14 两方法优劣势论述": ["5.3.5 路线对比与消融分析"],
    }
    ok_f7 = True
    for k, heads in f7_map.items():
        hits = [h for h in heads if h in md]
        ok = len(hits) == len(heads)
        ok_f7 &= ok
        log("  %s：%s → %s" % (k, "、".join(hits), "覆盖" if ok else "缺失"))

    # ---------- docx 版式抽查 ----------
    log("-" * 72)
    log("docx 版式抽查")
    doc = Document(docx_path)
    sec = doc.sections[0]
    m = (sec.top_margin.cm, sec.bottom_margin.cm, sec.left_margin.cm, sec.right_margin.cm)
    ok_margin = all(abs(v - 2.5) < 0.01 for v in m)
    ok_page = abs(sec.page_width.cm - 21.0) < 0.01 and abs(sec.page_height.cm - 29.7) < 0.01
    log("  页面 A4：%s（%.1f×%.1f cm）｜页边距 2.5cm：%s %s" % (
        ok_page, sec.page_width.cm, sec.page_height.cm, ok_margin, str(m)))

    heads = [p for p in doc.paragraphs if p.style.name.startswith("Heading") or p.style.name == "Title"]
    h_east = {east_asia_of(r) for p in heads for r in p.runs}
    h_size = {round(r.font.size.pt, 1) for p in heads for r in p.runs if r.font.size}
    body = [p for p in doc.paragraphs if p.style.name == "Normal" and p.text.strip()]
    # 图题段 = Normal 中 10.5pt 的居中小字段（设计值：恰好 8 个，随图而设）
    caps = [p for p in body if any(r.font.size and abs(r.font.size.pt - 10.5) < 0.01 for r in p.runs)]
    core = [p for p in body if p not in caps]
    b_east = {east_asia_of(r) for p in core[:400] for r in p.runs}
    b_size = {round(r.font.size.pt, 1) for p in core[:400] for r in p.runs if r.font.size}
    b_west = {r.font.name for p in core[:400] for r in p.runs}
    spacing = {p.paragraph_format.line_spacing for p in core[:400]}
    log("  标题段=%d 样式=%s 中文字体=%s 字号=%s" % (
        len(heads), sorted({p.style.name for p in heads}), sorted(x for x in h_east if x), sorted(h_size)))
    log("  正文段=%d（另图题段=%d）中文字体=%s 西文字体=%s 字号=%s 行距=%s" % (
        len(core), len(caps), sorted(x for x in b_east if x), sorted(x for x in b_west if x),
        sorted(b_size), sorted(spacing)))
    ok_font = (h_east == {"黑体"} and b_east == {"宋体"}
               and b_west <= {"Times New Roman"} and b_size == {12.0}
               and len(caps) == 8 and spacing == {1.5})

    imgs = doc.inline_shapes
    widths = {round(s.width.cm, 1) for s in imgs}
    log("  图片 %d 张，宽度集合=%s cm（期望 {15.0}）" % (len(imgs), widths))
    ok_img = len(imgs) == 8 and widths == {15.0}

    hdr_bold = []
    for t in doc.tables:
        hdr_bold.append(all(r.font.bold for row0 in t.rows[:1] for c in row0.cells for r in c.paragraphs[0].runs))
    log("  表格 %d 张，表头加粗=%s" % (len(doc.tables), hdr_bold))
    ok_tbl = len(doc.tables) == 6 and all(hdr_bold)
    # 表格单元格字号抽查（宋体 10.5pt）
    cell_size = {round(r.font.size.pt, 1) for t in doc.tables for row in t.rows for c in row.cells
                 for p in c.paragraphs for r in p.runs if r.font.size}
    log("  表格单元格字号集合=%s（期望 {10.5}）" % sorted(cell_size))

    # ---------- 摘要一致性：摘要.md vs 论文.md vs docx ----------
    log("-" * 72)
    log("摘要一致性")
    ab = io.open(os.path.join(ROOT, "paper", "摘要.md"), encoding="utf-8").read()
    ab_body = ab.split("\n", 1)[1].strip()
    in_paper = ab_body in md
    docx_text = "\n".join(p.text for p in doc.paragraphs)
    ab_plain = "\n\n".join(x.replace("**", "") for x in ab_body.split("\n\n"))
    ab_flat = "".join(ab_plain.split())
    docx_head = "".join(docx_text.split())[:len(ab_flat) + 400]
    in_docx = ab_flat[:200] in docx_head
    log("  摘要.md 正文与 论文.md 摘要节逐字一致：%s" % in_paper)
    log("  摘要正文进入 docx 首页（标题+摘要+关键词）：%s" % in_docx)
    ok_abs = in_paper and in_docx

    # 摘要关键数字与论文.md 数字同源核对
    keys = ["85.99%", "12.00%", "2.01%", "0.3402", "0.3357", "0.4098", "0.4560",
            "0.6704", "0.6362", "0.3200", "0.1733", "90.19%", "9.10%", "0.72%",
            "838,938", "1.0298%", "52.84", "11,167", "2,792"]
    missing = [k for k in keys if k not in ab_body]
    log("  摘要关键数字 %d 项全部在列且与正文同源：%s（缺失=%s）" % (
        len(keys), not missing, missing or "无"))

    verdict = all([ok_f1, ok_f2, ok_f3 := same_ids, ok_f4, ok_f5, ok_f6, ok_f7,
                   ok_font, ok_img, ok_tbl, ok_abs, not missing])
    log("-" * 72)
    log("合规总判定：%s" % ("ALL PASS" if verdict else "存在 FAIL 项，见上文明细"))
    _logf.close()
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
