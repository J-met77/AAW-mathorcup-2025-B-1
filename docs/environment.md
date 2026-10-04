# 环境与工具链

## 运行环境（2026-10-04 自检）

| 组件 | 版本 |
|---|---|
| OS | Windows 11 (10.0.26200) / Git Bash |
| Python | 3.13.14 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| xgboost | 3.4.1 |
| matplotlib | 3.11.2 |
| openpyxl | 已安装（xlsx 读写） |
| python-docx | 1.2.0（docx 生成；pandoc 不可用） |
| git | 2.55.0.windows.5（凭据已配置） |

## 目录约定

```
工作区 C:\Users\21732\Desktop\2025b论文\agent_workspace_B1\
├── STATE.md          # 唯一进度权威（A0 维护，同步脚本解析）
├── data/             # 附件1.xlsx、附件2.xlsx、Result.xlsx、赛题原文.md（不入仓库）
├── 00_admin/         # 管理文档：task_board / decisions / progress / A1-A7 阶段文档
├── code/             # 全部可复跑脚本（编号命名）
├── output/
│   ├── figures/      # 论文级图表 png
│   ├── tables/       # 结果表 csv
│   ├── logs/         # 运行日志与 QA 记录
│   └── Result_提交.xlsx  # 最终提交文件
└── paper/            # 论文.md / 论文.docx / 摘要.md

仓库   C:\Users\21732\AAW-mathorcup-2025-B-1\
├── README.md         # 自动生成的进度仪表盘
├── PROGRESS.md       # 自动生成的进度台账（含时间线）
├── reports/          # 每小时快照 + LATEST.md
├── docs/             # pipeline / environment / data-notes / decisions / task-board
├── deliverables/     # 成果镜像：code / paper / admin / figures / tables / logs / result
└── sync/             # 本同步脚本
```

## 同步脚本用法

```bash
python "C:\Users\21732\AAW-mathorcup-2025-B-1\sync\sync_to_github.py"
```

每小时 :00 由自动化任务触发；幂等，无变更自动跳过。
