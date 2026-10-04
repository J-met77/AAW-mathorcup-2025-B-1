# 决策记录 —— 盲测 Run-2

> 维护约定：`STATE.md` §3 由主会话统一维护；本文件是 A0 在 G0 建立的决策台账副本与扩展记录。
> D00–D04 转录自 `STATE.md` §3（提出方=主会话）；D05 起为 A0 在 G0 阶段基于自检事实新增，
> **供主会话审核后转录进 STATE.md §3**。列：编号 / 决策 / 理由 / 提出方 / 时间。

| 编号 | 决策 | 理由 | 提出方 | 时间 |
|---|---|---|---|---|
| D00 | 建立盲测工作区 agent_workspace_B2，与既往任何工作区/仓库物理隔离 | 保证盲测纯净：方法只能来自题面、数据与自身产物 | 主会话 | 转录自 STATE.md §3（G0，2026-10-04） |
| D01 | 调度模式：主会话担任总控真实派发，A0 仅承担 G0 规划与 G9 收尾，A1–A11 逐阶段独立真实派发 | 前轮曾因子代理会话缺少派发工具退化为串行扮演；本会话十二个角色均已注册可用，真实派发可兑现，且主会话可在阶段间执行门禁 | 主会话 | 转录自 STATE.md §3（G0，2026-10-04） |
| D02 | STATE.md 由主会话统一维护；子代理只写各自阶段产物，不碰台账 | 避免并行写冲突与格式漂移破坏同步脚本解析 | 主会话 | 转录自 STATE.md §3（G0，2026-10-04） |
| D03 | 同步目标为本仓库 run-2 孤立分支（main 保持前轮存档，互不覆盖） | 无新建仓库权限；孤立分支实现两轮成果物理隔离，目录仍清晰 | 主会话 | 转录自 STATE.md §3（G0，2026-10-04） |
| D04 | 全程离线：不联网检索，A2 仅引用自身已掌握的通用方法论知识并注明"通用方法论，非本题资料" | 盲测红线：禁止检索任何与本题解答相关的信息；离线纪律保证与既有运行唯一变量为调度模式 | 主会话 | 转录自 STATE.md §3（G0，2026-10-04） |
| D05 | 缺失工具的替代路径：statsmodels 与 seaborn 未安装、pandoc 不在 PATH；后续阶段一律基于可用库实现——A5/A7 的统计推断类需求用 scipy/sklearn 等价实现或由主会话决定是否安装，A8 图表纯 matplotlib，A10 的 md→docx 用 python-docx | G0 自检实测：statsmodels、seaborn import 失败，pandoc 不存在；核心库 numpy/pandas/sklearn/scipy/matplotlib/openpyxl/python-docx/lightgbm/xgboost 全部可用。溯源：`code/g0_env_check.py` + `output/logs/g0_env_check.log` | A0（G0） | 2026-10-04 |
| D06 | matplotlib 中文渲染基准字体定为 SimHei / Microsoft YaHei（备选 DengXian、KaiTi）；A8 所有图表须在 rcParams 显式设置中文字体并处理负号显示 | G0 自检实测命中 SimHei、Microsoft YaHei 等 11 个常用中文字体，中文图表无字体障碍；显式设置可避免各阶段渲染不一致。溯源：`output/logs/g0_env_check.log` 第 4 节 | A0（G0） | 2026-10-04 |
| D07 | 环境自检落位约定：脚本 `code/g0_env_check.py`、日志 `output/logs/g0_env_check.log`、报告 `00_admin/env_check.md`；此三件构成"关键数字/事实须有脚本+日志溯源"（红线第 6 条）的首个范例，供 A3/A6/A7/A11 引用执行 | G0 产出需可核查；统一 code/ 与 output/logs/ 落位便于 A11 程序化终审 | A0（G0） | 2026-10-04 |
| D08 | 本机 pandas 为 3.0.5（3.x 大版本）、numpy 2.5.3：A3/A6 编码不得沿用 pandas 1.x/2.x 已废弃 API（如 `DataFrame.append`、隐式 downcasting 等），一切兼容性以本环境实测为准 | G0 自检实测版本；3.x 与旧教程代码存在行为差异，提前立项避免 A6 复跑不一致。溯源：`output/logs/g0_env_check.log` 第 1 节 | A0（G0） | 2026-10-04 |

## G0 发现摘要（供主会话派发简报引用）

1. **数据文件开放性**：附件1.xlsx（sheet1，11169×25）、附件2.xlsx（Sheet1，2794×25）、Result.xlsx（Sheet1，2793×3）、赛题原文.md（91 行 utf-8）全部可正常打开——A3 可直接开工。行列数为 openpyxl 表维（通常含表头），精确口径由 A3 核实。
2. **影响调度的缺库事实**：statsmodels、seaborn、pandoc 缺失（见 D05），已给出替代路径，不阻塞任何阶段的派发。
3. **并行窗口提醒**：A1 通过门禁后即可同时派发 A2、A3；A6 通过门禁后即可同时派发 A7、A8（详见 `00_admin/task_board.md` 调度约束）。
