# -*- coding: utf-8 -*-
"""
a10_01_make_docx.py —— A10 阶段：paper/论文.md -> paper/论文.docx（python-docx 自写转制）
- 标题层级：md H1->Title、H2->Heading 1、H3->Heading 2、H4->Heading 3（黑体，标题 16/15/14/12pt）
- 正文：宋体 12pt（小四）、西文 Times New Roman、行距 1.5、首行缩进 2 字符；页边距 2.5cm；A4
- 表格：md 全部表格 -> docx Table Grid，表头加粗，转义竖线还原为 |
- 图片：8 张按引用位置嵌入，居中，宽 15cm，图题居中 10.5pt
- 公式：Unicode 符号体系按纯文本保留，不做公式对象转换
- 自检五项写入 output/logs/a10_01_make_docx.log：LaTeX 残片 / 图片数 / 表格数 / 段落数 / 回读验证
运行：python code/a10_01_make_docx.py
"""
import io
import os
import re
import sys

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_MD = os.path.join(ROOT, "paper", "论文.md")
PAPER_DOCX = os.path.join(ROOT, "paper", "论文.docx")
LOG_PATH = os.path.join(ROOT, "output", "logs", "a10_01_make_docx.log")

_logf = io.open(LOG_PATH, "w", encoding="utf-8")


def log(s=""):
    print(s)
    _logf.write(s + "\n")
    _logf.flush()


SONG = "宋体"
HEI = "黑体"
WEST = "Times New Roman"


def set_run_font(run, east=SONG, west=WEST, size=12.0, bold=False, italic=False):
    run.font.name = west
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), west)
    rFonts.set(qn("w:hAnsi"), west)
    rFonts.set(qn("w:eastAsia"), east)


def add_inline_runs(par, text, east=SONG, size=12.0, base_bold=False):
    """解析行内 **加粗** 片段为多个 run。"""
    parts = re.split(r"\*\*", text)
    for i, seg in enumerate(parts):
        if not seg:
            continue
        run = par.add_run(seg)
        set_run_font(run, east=east, size=size, bold=base_bold or (i % 2 == 1))


def add_body_par(doc, text, indent=True, align=None, size=12.0, bullet=False):
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if bullet:
        text = "• " + text
        pf.left_indent = Cm(0.74)
    elif indent:
        pf.first_line_indent = Pt(size * 2)  # 首行缩进 2 字符
    if align is not None:
        par.alignment = align
    add_inline_runs(par, text, size=size)
    return par


def add_heading(doc, text, level):
    """level: 0=论文主标题, 1/2/3=Heading 1/2/3。md 层级映射：H2->1, H3->2, H4->3。"""
    sizes = {0: 16.0, 1: 15.0, 2: 14.0, 3: 12.0}
    par = doc.add_paragraph(style="Heading %d" % level if level > 0 else "Title")
    pf = par.paragraph_format
    pf.line_spacing = 1.5
    if level == 0:
        pf.space_before = Pt(6)
        pf.space_after = Pt(12)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif level == 1:
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
    else:
        pf.space_before = Pt(6)
        pf.space_after = Pt(3)
    run = par.add_run(text)
    set_run_font(run, east=HEI, size=sizes[level], bold=True)
    return par


def unescape_cell(t):
    return t.replace("\\|", "|").strip()


def parse_md(text):
    """把论文.md 解析为块序列：('h', level, text) / ('p', text) / ('quote', text)
    / ('img', caption, relpath) / ('table', rows) / ('hr',) / ('caption', text)。"""
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue
        if re.match(r"^-{3,}$", s):
            blocks.append(("hr",))
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)\s*$", s)
        if m:
            blocks.append(("img", m.group(1).strip(), m.group(2).strip()))
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [unescape_cell(c) for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        if s.startswith(">"):
            blocks.append(("quote", s[1:].strip()))
            i += 1
            continue
        if re.match(r"^\*\*(.+)\*\*$", s):
            blocks.append(("caption", re.match(r"^\*\*(.+)\*\*$", s).group(1).strip()))
            i += 1
            continue
        if re.match(r"^-\s+", s):
            blocks.append(("p", re.sub(r"^-\s+", "", s), True))
            i += 1
            continue
        blocks.append(("p", s, False))
        i += 1
    return blocks


def count_md_tables(md_text):
    """与解析器同口径统计 md 表格块数。"""
    return sum(1 for b in parse_md(md_text) if b[0] == "table")


def build_doc(blocks):
    doc = Document()
    # 页面：A4、页边距 2.5cm
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    # Normal 样式兜底
    normal = doc.styles["Normal"]
    normal.font.name = WEST
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    stats = {"paras": 0, "tables": 0, "images": 0, "headings": 0}
    title_seen = False
    for blk in blocks:
        kind = blk[0]
        if kind == "h":
            lvl, text = blk[1], blk[2]
            if lvl == 1:
                add_heading(doc, text, 0)  # 论文主标题
                title_seen = True
            else:
                add_heading(doc, text, lvl - 1)  # H2->Heading1, H3->2, H4->3
            stats["headings"] += 1
            stats["paras"] += 1  # 标题段计入正文段总数（与回读 doc.paragraphs 口径一致）
        elif kind == "p":
            text, bullet = blk[1], blk[2]
            # 副标题（紧跟主标题的圆括号行）居中、无缩进
            if title_seen and stats["paras"] == 0 and text.startswith("（2025"):
                add_body_par(doc, text, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                add_body_par(doc, text, bullet=bullet)
            stats["paras"] += 1
        elif kind == "quote":
            par = doc.add_paragraph()
            pf = par.paragraph_format
            pf.line_spacing = 1.5
            pf.left_indent = Cm(0.74)
            add_inline_runs(par, blk[1])
            stats["paras"] += 1
        elif kind == "caption":
            par = doc.add_paragraph()
            pf = par.paragraph_format
            pf.line_spacing = 1.5
            pf.space_before = Pt(6)
            pf.space_after = Pt(3)
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline_runs(par, blk[1], size=12.0, base_bold=True)
            stats["paras"] += 1
        elif kind == "img":
            caption, relpath = blk[1], blk[2]
            path = os.path.normpath(os.path.join(ROOT, "paper", relpath))
            if not os.path.exists(path):
                raise FileNotFoundError("figure missing: %s" % path)
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_before = Pt(6)
            run = par.add_run()
            run.add_picture(path, width=Cm(15.0))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(6)
            cap.paragraph_format.line_spacing = 1.0
            r = cap.add_run(caption)
            set_run_font(r, size=10.5)
            stats["images"] += 1
            stats["paras"] += 2
        elif kind == "table":
            rows = blk[1]
            # 剔除分隔行
            body_rows = [r for r in rows if not all(re.match(r"^:?-{2,}:?$", c or "-") for c in r)]
            ncols = max(len(r) for r in body_rows)
            table = doc.add_table(rows=len(body_rows), cols=ncols)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True
            for ri, row in enumerate(body_rows):
                for ci in range(ncols):
                    cell = table.cell(ri, ci)
                    cell.paragraphs[0].paragraph_format.line_spacing = 1.0
                    txt = row[ci] if ci < len(row) else ""
                    add_inline_runs(cell.paragraphs[0], txt, size=10.5, base_bold=(ri == 0))
            stats["tables"] += 1
        elif kind == "hr":
            continue
    return doc, stats


LATEX_PATTERNS = [
    ("dollar $", chr(36)),
    ("\\begin", chr(92) + "begin"),
    ("\\frac", chr(92) + "frac"),
    ("\\le(命令)", chr(92) + "le"),
    ("\\ge(命令)", chr(92) + "ge"),
    ("\\sum", chr(92) + "sum"),
    ("\\int", chr(92) + "int"),
    ("\\times(命令)", chr(92) + "times"),
    ("\\(或\\[定界符", chr(92) + "("),
    ("\\[定界符", chr(92) + "["),
    ("_{下标花括号", "_{"),
    ("反斜杠命令通配", None),  # 用正则 r"\\[a-zA-Z]+" 单独扫
]


def scan_latex(texts):
    hits = []
    joined = "\n".join(texts)
    for name, p in LATEX_PATTERNS:
        if p is None:
            continue
        if p in joined:
            for i, t in enumerate(texts):
                if p in t:
                    hits.append((name, i, t[:60]))
    m = re.finditer(chr(92) + chr(92) + "[a-zA-Z]+", joined)
    for mm in m:
        frag = mm.group(0)
        if frag not in (chr(92) + "|",):  # 表格转义竖线合法
            hits.append(("反斜杠命令通配", -1, frag))
    return hits


def main():
    log("=" * 72)
    log("A10 md->docx 转制  |  python-docx  |  源：%s" % PAPER_MD)
    log("=" * 72)
    md = io.open(PAPER_MD, encoding="utf-8").read()
    blocks = parse_md(md)
    kinds = {}
    for b in blocks:
        kinds[b[0]] = kinds.get(b[0], 0) + 1
    log("解析块统计：%s" % kinds)
    md_tables = kinds.get("table", 0)
    md_images = kinds.get("img", 0)
    log("md 表格块=%d（符号说明表 + 表1–表5）、md 图片引用=%d" % (md_tables, md_images))

    doc, stats = build_doc(blocks)
    doc.save(PAPER_DOCX)
    log("已保存：%s" % PAPER_DOCX)
    log("构建统计：标题=%d 段落=%d 表格=%d 图片=%d" % (
        stats["headings"], stats["paras"], stats["tables"], stats["images"]))

    # ---------- 程序化自检 ----------
    log("-" * 72)
    log("自检 ①：裸 LaTeX 残片扫描（$、\\begin、\\frac、\\le、\\ge、\\sum、\\(、_[、反斜杠命令）")
    texts = []
    for p in doc.paragraphs:
        texts.append(p.text)
    for tb in doc.tables:
        for row in tb.rows:
            for cell in row.cells:
                texts.append(cell.text)
    hits = scan_latex(texts)
    # 排除合法出现：超参名 feature_fraction 等不含反斜杠，不受影响
    if hits:
        for h in hits[:20]:
            log("  发现残片：%s | %s" % (h[0], h[2]))
        log("自检① 结论：FAIL（%d 处）" % len(hits))
        ok1 = False
    else:
        log("自检① 结论：PASS（未发现任何裸 LaTeX 残片）")
        ok1 = True

    log("自检 ②：图片嵌入数（期望 %d）" % md_images)
    n_img = len(doc.inline_shapes)
    ok2 = (n_img == md_images == 8)
    log("  inline_shapes=%d -> %s" % (n_img, "PASS" if ok2 else "FAIL"))

    log("自检 ③：表格数量与 md 一致（md=%d）" % md_tables)
    ok3 = (stats["tables"] == md_tables)
    log("  docx 表格=%d -> %s" % (stats["tables"], "PASS" if ok3 else "FAIL"))

    log("自检 ④：段落总数 > 100")
    ok4 = (stats["paras"] > 100)
    log("  段落统计=%d -> %s" % (stats["paras"], "PASS" if ok4 else "FAIL"))

    log("自检 ⑤：保存后重新打开回读验证")
    doc2 = Document(PAPER_DOCX)
    p2 = len(doc2.paragraphs)
    t2 = len(doc2.tables)
    i2 = len(doc2.inline_shapes)
    alltext = "\n".join(p.text for p in doc2.paragraphs)
    placeholders = [w for w in ("留位", "TODO", "占位", "XXX") if w in alltext]
    has_title = "基于条件分位保序边界与双口径验证的物流理赔风险识别模型" in alltext
    has_abstract = "摘要" in alltext and "关键词" in alltext
    has_kw = "风险标注；条件分位数保序；梯度提升树；类别不均衡；WAPE" in alltext
    ok5 = (p2 == stats["paras"] and t2 == md_tables and i2 == 8
           and not placeholders and has_title and has_abstract and has_kw)
    log("  回读：段落=%d 表格=%d 图片=%d 标题在=%s 摘要/关键词在=%s 占位符=%s" % (
        p2, t2, i2, has_title, has_abstract, placeholders or "无"))
    log("  自检⑤ 结论：%s" % ("PASS" if ok5 else "FAIL"))

    verdict = all([ok1, ok2, ok3, ok4, ok5])
    log("-" * 72)
    log("总判定：%s" % ("ALL PASS —— 转制完成" if verdict else "FAIL —— 需修复重跑"))
    _logf.close()
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
