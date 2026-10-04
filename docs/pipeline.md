# 12-Agent 流水线架构与门禁体系

> 本仓库的执行主体是部署在 `~/.zcode/agents` 的 12 个角色化 Agent。主会话只负责"雇主"角色：启动 A0、收发汇报；一切建模决策由流水线内部产生。

## 角色分工

| Agent | 角色 | 职责一句话 | 关键产物 |
|---|---|---|---|
| **A0** chief-orchestrator | 总控 | 任务拆解、调度、门禁检查、冲突仲裁、进度台账 | STATE.md、task_board、decisions |
| **A1** problem-analyst | 审题 | 拆解子问题/目标/约束/评价指标，输出题意确认书 | A1_审题.md |
| **A2** literature-assumption | 文献与假设 | 检索**通用**方法论依据，建立可检验的假设体系 | A2_假设.md |
| **A3** data-engineer | 数据工程 | 读取、勘察、清洗、缺失/异常处理、特征工程 | 01_*.py、A3_数据报告.md |
| **A4** model-designer | 建模 | 定义变量、目标函数、约束与标注规则 | A4_模型设计.md |
| **A5** algorithm-solver | 算法 | 求解策略、评估方案、复杂度/收敛性分析 | A5_算法方案.md |
| **A6** implementation-coder | 实现 | 可复现代码、实验、结果文件产出 | code/*.py、Result_提交.xlsx |
| **A7** result-analyst | 结果分析 | 误差、敏感性、稳健性、业务解读 | A7_结果分析.md |
| **A8** visualization-designer | 可视化 | 论文级图表（统一风格、矢量导出） | output/figures/*.png |
| **A9** paper-writer | 论文 | 按竞赛结构撰写正文、整合图表结论 | paper/论文.md |
| **A10** abstract-polisher | 摘要润色 | 摘要、语言/格式/引用润色、docx 转制 | paper/论文.docx |
| **A11** qa-reproducer | 质控 | **只读不改**：复现、一致性、合规终审 | output/logs/qa_check.md |

## 调度拓扑

```
主会话(雇主) ──> A0 总控
                  ├─> G0 环境自检
                  ├─> A1 审题 ─> A2 假设 ─┐
                  ├─> A3 数据 ────────────┤
                  │                       v
                  ├─> A4 建模 ─> A5 算法 ─> A6 实现 ─┬─> A7 分析 ─┐
                  │                                  └─> A8 图表 ─┤
                  │                                               v
                  ├─> A9 论文 ─> A10 摘要润色 ─> A11 终审 ─> G9 交付
                  └─（A11 打回则回到对应阶段返工）
```

## 门禁规则

1. **产物门禁**：每个阶段开始前，A0 检查依赖阶段的产物文件是否存在且含"必须包含"要素，不满足则打回。
2. **盲测纪律（本题专属红线）**：任何方法/阈值/特征/模型选择只能来自赛题原文、附件数据与流水线自身产物；禁止检索本题相关解答、禁止读取既往 run 的任何工作区。理由必须写入 decisions.md。
3. **实测门禁**：论文与台账中的一切数字必须来自已运行代码的真实输出，禁止预计值冒充实测值。
4. **可复现门禁**：A11 用干净环境逐脚本复跑，核对 Result_提交.xlsx 运单号与附件2 完全一致。
5. **视觉门禁**：论文渲染后逐页检查排版、图表可读性，瑕疵打回 A0 返修。

## 进度同步

- 权威进度在 `agent_workspace_B1/STATE.md`（A0 维护）。
- `sync/sync_to_github.py` 每小时运行：解析 STATE.md → 生成 README 仪表盘 / PROGRESS 台账 / 快照 reports/ → 镜像成果 deliverables/ → git push。
- 竞赛原始数据（附件*.xlsx、Result.xlsx）与赛题原文不入仓库。
