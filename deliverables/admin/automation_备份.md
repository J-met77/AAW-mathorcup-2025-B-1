# Run-2 每小时同步自动化配置备份（2026-10-05，应用户要求暂停时导出）

> 用途：用户暂停项目期间修改子智能体权限，恢复时按此配置原样重建。
> 恢复方式：CronCreate，title/cron/prompt 照抄下方字段，recurring=true。

- **automationId（原）**：automation-06dce9ae-bdc8-4df1-b96e-c095aff2de91
- **title**：每小时同步数模盲测Run-2进度到GitHub(run-2分支)
- **cron**：`0 * * * *`（intervalUnit=hourly, interval=1）
- **recurring**：true
- **prompt**：

```text
执行数模盲测项目（2025 MathorCup 赛道B 盲测 Run-2，12-Agent 流水线真实派发模式）到 GitHub 仓库 https://github.com/J-met77/AAW-mathorcup-2025-B-1 的每小时例行同步。Run-2 专用：只推送 run-2 分支，绝不触碰 main 分支。

步骤：
1. 运行同步脚本：`python "C:\Users\21732\AAW-mathorcup-2025-B-1-run2\sync\sync_to_github.py"`（Python 3.13，工作目录任意）。
2. 脚本会自动完成：解析工作区 `C:\Users\21732\Desktop\2025b论文\agent_workspace_B2\STATE.md` 的阶段状态表 → 重新生成本地仓库（`C:\Users\21732\AAW-mathorcup-2025-B-1-run2`，固定在 run-2 分支）的 README.md 进度仪表盘与 PROGRESS.md 台账 → 镜像成果到 deliverables/（code/*.py、paper/*.md|*.docx、00_admin 管理文档、output/figures|tables|logs、output/Result_提交.xlsx；竞赛原始数据附件*.xlsx、Result.xlsx、赛题原文.md 永不入库）→ 复制 00_admin 管理文档到 docs/ → 写 reports/YYYY-MM/ 时间戳快照与 reports/LATEST.md → git commit + push 到 origin run-2 分支。
3. 若脚本输出 `[skip] 无变更，跳过提交`，直接确认"本轮无变更"即可。
4. 若脚本输出 `[error]` 或非零退出：读取报错内容诊断（常见：GitHub 502/网络抖动导致 push 失败——等 60 秒重跑一次脚本；工作区文件被占用——稍后重试）。重试仍失败时，在回复中明确报告失败原因与建议，不要删除或改写仓库已有内容，不要修改 git 凭据。
5. 全程不要向用户提问，不要改动工作区内的任何文件（脚本对工作区只读），不要动 main 分支，不要动本地目录 C:\Users\21732\AAW-mathorcup-2025-B-1（前轮克隆）与 C:\Users\21732\AAW-mathorcup-2025-B（更早轮次克隆）。最终回复只需一段话：同步时间、进度百分比、当前阶段、是否推送成功或失败原因。
```

## 暂停时刻状态（便于恢复对账）

- 本地 run-2 克隆领先远端 1 个提交（11:00 快照，62%），恢复后先补推。
- A7 结果分析：已硬停（中断点见其已落盘产物）；A8 图表：优雅待命（fig1–fig3 完成、fig4 修正代码已写未跑）。
- 恢复动作清单：①CronCreate 重建本自动化并补推；②SendMessage 复活 A7/A8 继续；③继续 A9→A10→视觉终检→A11→G9。
