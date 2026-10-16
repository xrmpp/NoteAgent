# AGENTS.md

## 沟通与工作方式

- 默认用中文回复，除非用户明确要求英文；内容简洁、具体。
- 回复中出现 Linux 命令时，说明命令的用途。
- 需求有歧义且仓库上下文支持安全默认值时，采用该默认值继续。
- `docs/AI_PROJECT_BRIEF.md` 是唯一活跃开发计划；`CONTEXT.md` 保存稳定工程事实，`docs/adr/` 保存已接受的架构决策。历史记录只作证据，不自动形成待办。
- 只执行当前纵向切片已接受事项；每个切片应贯通用户问题、实际数据、回答、Trace 与一个最小可运行验收。除非请求本身需要，不做大范围重构。

## 工程边界

- UI 放在 `app/ui/`，聊天编排放在 `app/chat/`，provider 逻辑通过 `app/llm/` 抽象，持久化细节放在 `app/storage/`，领域概念放在 `app/domain/`。
- UI 只调用 HTTP API，不装配数据库、模型或 Agent；后端服务和维护 CLI 可使用后端组件。
- 优先受控演进与单 Agent；控制流放在 Python 代码中，不藏在 prompt 里。prompt 构建、上下文选择和工具执行应显式、可核查。
- 新工具使用 structured output 加确定性 Python 执行，并在 Python 中进行明确校验；不把工具行为埋进 prompt。
- 聊天状态与 Agent run state 分开建模。
- 不把 Trace 描述为完整私有思维链；不宣称引用校验证明语义正确、杜绝幻觉或错误必然可恢复。
- 不提交 `.env` 等文件中的真实密钥；不把没有鉴权和多租户隔离的服务直接开放到不可信网络。

## 本轮验收约定

- 本轮不新增单元测试、不运行 `pytest`。
- 使用现有容器中的 `app.eval.knowledge_e2e` 验收真实知识问答、追问和 SSE；通过真实页面检查 Trace 按需加载及会话切换。
- 验收结果保存在 `.operations-evidence/knowledge-e2e.json`，并如实注明未覆盖路径；静态检查、已有历史证据或脚本入口不等同于本轮端到端通过。

## 记忆工作流

当用户明确要求记住项目规则、计划或决策时，若信息属于仓库上下文则更新 `CONTEXT.md`；若 `.codex/memories/extensions/ad_hoc/notes/` 可用，再添加 Codex memory note。
