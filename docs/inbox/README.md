# Inbox 实验说明

Inbox 是与正式知识库聊天分开的个人发现实验。它直接使用本地 Python 页面/CLI 与现有数据库、HTTP 客户端及模型 provider；不新增 FastAPI 路由，不属于 API-only 知识问答主链路。当前正式 Streamlit 导航只注册知识库聊天页，Inbox 页面源码未注册，不能假定存在 `/inbox` 路由。

## 当前已实现

- `fetch` 从 AIHOT 公开接口拉取已精选的 tip/paper 候选；支持 24h/7d 时间窗、分页与 dry-run。候选按外部 ID、规范化 URL、标题/摘要内容摘要去重，写入 Inbox 表，不进入 RAG 知识文档。
- `triage` 按提示词和本地读者档案调用生成模型，为候选保存结构化建议；校验失败最多重试一次，provider/网络错误不重试。每日最多 50 次物理调用，按 UTC 日期计数。
- `report` 只读输出近期筛选结果。人工决定与模型建议分别存储，模型建议不会替用户更改决定。
- 页面代码包含手动拉取、批量筛选、结果查看和人工决定，但没有由正式入口导航开放。

目前 CLI 仅实现 `fetch`、`triage`、`report`。笔记生成、笔记入库及 stats 不属于现有功能，不要描述为已经实现。

## 数据与模型边界

运行时 prompt 位于 `data/inbox/prompts/triage.md`；本地读者档案路径为 `data/inbox/profile.md`，由用户维护且不应提交。本文不复制档案正文。筛选按钮/CLI triage 才调用生成模型；拉取候选本身只访问 AIHOT。Inbox 候选、筛选结果和人工状态事件使用独立表，与知识库文档表分开。

dry-run 不代表完全没有数据库写入：现有 session factory 会创建缺失的 Inbox 表，但 dry-run 不新增候选或状态事件。筛选输出的相似文档字段仍为空；不要把它解释成“没有相似文档”。

## 代码入口

| 功能 | 入口 |
| --- | --- |
| 独立实验页面 | `app/ui/pages/inbox.py` |
| CLI 命令 | `app/inbox/cli.py`：`fetch_candidates` 与命令分发 |
| AIHOT 分页、HTTP 错误和重试 | `app/inbox/aihot.py`：`AihotClient` |
| prompt/profile 组装、模型输出校验与配额 | `app/inbox/triage.py`：`load_prompt`、`build_messages`、`triage_candidates` |
| 候选、筛选结果和状态事件持久化 | `app/inbox/store.py`：`InboxStore` |
| URL 规范化与内容摘要 | `app/inbox/urlnorm.py` |

本页只说明当前可见代码，不构成未来待办或知识库主线的功能承诺。
