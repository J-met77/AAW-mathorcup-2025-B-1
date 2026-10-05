# -*- coding: utf-8 -*-
"""
a8_00_图表公共.py —— A8 阶段公共渲染模块（全部 a8_* 绘图脚本 import 本模块）
规范依据：decisions.md D05（纯 matplotlib）/ D06（SimHei·Microsoft YaHei + 负号处理）
配色体系：全文统一（蓝=合理诉求/方式2，橙=诉求偏高/方式1，紫=严重超额；蓝橙紫为色盲友好三元组，避免红绿对立）
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 路径 ----------
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 工作区根
TBL = os.path.join(WS, "output", "tables")
FIG = os.path.join(WS, "output", "figures")
LOG = os.path.join(WS, "output", "logs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(LOG, exist_ok=True)

# ---------- 渲染规范（D06 强制：rcParams 显式设置，D05：仅 matplotlib） ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False        # 负号显示
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300                 # 300 dpi 输出
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 10.5

# ---------- 统一配色（固化常量，全文一致；蓝/橙/紫色盲友好） ----------
PAL = {
    "blue":   "#2B6CA3",  # 主：合理诉求 / 方式2 / 分层5折
    "orange": "#E6852B",  # 次：诉求偏高 / 方式1 / 行序外推
    "purple": "#7D54A5",  # 三：严重超额
    "blue_l":   "#9DC3E0",
    "orange_l": "#F5C08A",
    "purple_l": "#C2A8DD",
    "navy":   "#1B3A5C",  # 边界/强调线
    "gray":   "#7F7F7F",
    "gray_l": "#D9D9D9",
}
CLASS_COLORS = {"合理诉求": PAL["blue"], "诉求偏高": PAL["orange"], "严重超额": PAL["purple"]}

YH_FONT = "Microsoft YaHei"  # 含 ŷ/下标/数学负号等 SimHei 缺失字形时按 D06 备选字体渲染

# ---------- 日志 ----------
def get_logger(script_name: str) -> logging.Logger:
    """每个 a8_XX 脚本一个日志：output/logs/a8_XX_*.log（utf-8）"""
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(LOG, script_name + ".log"), mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger

def log_font_check(logger) -> None:
    """渲染规范核验：SimHei/微软雅黑可用性写入日志"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["SimHei", "Microsoft YaHei"]:
        logger.info("字体核验：%s 可用=%s", f, f in names)
    logger.info("rcParams.font.sans-serif=%s", plt.rcParams["font.sans-serif"])
    logger.info("rcParams.axes.unicode_minus=%s, savefig.dpi=%s",
                plt.rcParams["axes.unicode_minus"], plt.rcParams["savefig.dpi"])

def save_fig(fig, out_name: str, logger) -> str:
    """落盘 300dpi png 到 output/figures/"""
    path = os.path.join(FIG, out_name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("图已落盘：%s", path)
    return path

def read_csv(name: str):
    """统一 utf-8-sig 读取 output/tables/ 下的 csv"""
    import pandas as pd
    return pd.read_csv(os.path.join(TBL, name), encoding="utf-8-sig")
