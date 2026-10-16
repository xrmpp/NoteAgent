# NoteAgent 稳定工程上下文

## 产品与边界

- 对外名称为 NoteAgent，公开仓库为 `https://github.com/xrmpp/NoteAgent`，Python 分发名为 `note-agent`。应用包 `app`、本地目录、数据库 `ai_agent` 和已有数据卷名称保持不变；名称调整不迁移或清空用户数据。
- 公开分支为 `main`，从当前源码快照独立起步，提交使用 GitHub 隐私邮箱。本地 `master` 保留含个人邮箱的旧开发历史，不对外推送；后续发布只推送 `main`，不要使用 `git push --all` 暴露旧历史。
- 项目定位为个人知识库 Agent，面向个人笔记检索、阅读和问答。README 面向安装与使用者，不承载内部进度汇报、协作限制、验收经过或代码学习记录；开发状态放在唯一活跃计划中，不用“实验版”等笼统标签代替具体能力和使用边界。当前仍冻结新增功能。
- 项目是个人专业知识库受控问答助手。`mode="agent"` 为单 Agent 主入口，`mode="rag"` 为共享检索与证据处理循环、但关闭工具的对照模式。
- 唯一活跃计划见 [`docs/AI_PROJECT_BRIEF.md`](docs/AI_PROJECT_BRIEF.md)。本文件仅记录稳定事实；已接受的架构取舍见 `docs/adr/`。
- 早期 Text-to-SQL、审批和图查询执行链路及依赖已移除；旧 Trace 返回字段保留读取兼容。

## 服务分层与知识问答链路

- 正式 UI 位于 `app/ui/`，通过 HTTP/SSE 调用 FastAPI，不装配数据库、模型或 Agent。API lifespan 在后端装配服务；维护 CLI 仍可直接使用后端组件。
- 主要模块：`app/api/` 负责 HTTP、SSE、会话及问答；`app/chat/` 管理聊天会话；`app/agent/` 包含 `AgentService`、`KnowledgeLoop`、`EvidenceStore`、只读知识工具与引用校验；`app/rag/` 提供导入与检索评测 CLI；`app/llm/` 隔离模型和 embedding provider；`app/storage/` 持久化对话、知识、run 与 Trace。
- 每轮由 `AgentService` 建立 run、读取本会话短期历史并消费 `KnowledgeLoop` 事件；`RagService` 与仓储执行检索和原文读取；步骤、工具调用与终态结果以结构化 Trace 持久化。
- 正式知识工具仅有 `search_knowledge` 和 `read_document`，均为只读；`KnowledgeToolExecutor` 执行唯一一次参数校验与执行。工具只在本轮已发现文档范围内工作。

## 检索、证据与预算

- 当前路径是 pgvector Vector + Python `bm25s` + RRF。两路各取 10 个候选，RRF 后知识循环初始交付 Top-3；RRF 使用名次而非直接相加不同检索器的原始分数。这些参数不是经过质量评测得出的普适最优值。
- 正式知识循环调用检索时关闭 LLM rerank；配置中保留的可选重排不是该主链路的实际步骤。
- `EvidenceStore` 管理本轮证据身份、引用编号、阅读范围、覆盖情况及最多 20000 字符的已交付证据预算。`CitationValidator` 校验引用编号来源与正文列表一致性，不判定结论是否被证据语义支持。
- `KnowledgeLoop` 最多 8 次模型调用、6 次工具尝试、120 秒总预算；工具阶段结束时至少预留最后 30 秒用于收尾。追问补全若发生，计入模型调用与总预算。
- 有历史时最多读取同一会话最近 3 轮已完成问答；每条消息最多 1000 字符、总上下文最多 6000 字符。旧引用不进入本轮证据。`MAX_CONTEXT_MESSAGES` 仍是旧 ChatService 消息窗口配置，不代表 Agent 的历史窗口。

## Trace、HTTP 与运行环境

- 原生 reasoning 字段只在循环内存中保留，不作为完整私有思维链写入 Trace。Trace 是步骤、工具调用、结果和运行元数据，不代表模型内部全部推理。
- SSE 在当前 HTTP 请求中执行并传递进度；断开连接会触发收尾，不提供可靠后台队列、断线续跑或自动恢复。
- API、UI 和数据库之间无用户鉴权或多租户隔离。当前适合受控的个人环境；不同客户端可复用 HTTP API，但对外部署另需身份验证、用户隔离及网络保护。
- 基础 `docker-compose.yml` 将 Streamlit 宿主端口绑定到 `127.0.0.1`（默认 8501）；API 8000 只在 Compose 网络内暴露，PostgreSQL 5432 不发布宿主端口。本轮只修改配置，未重建或重启现有服务，不能据此声称长期运行的旧容器映射已改变。
- 被 Git 忽略的本机 Compose override 可增加本机映射和源码挂载；其中的 API `127.0.0.1:8001`、PostgreSQL `127.0.0.1:5433` 等配置仅对持有该 override 的开发环境生效，不是基础 Compose 默认值。
- UI 只读取 `APP_TITLE` 与 `AGENT_API_BASE_URL` 等轻量 UI 配置；不读取后端 `.env`。生成模型与 embedding 配置由后端持有。

## 默认语料与连续性

- `data/corpus/` 是默认初始化输入，不是临时废文档目录。默认仅在知识库文档数为 0 时导入；`AUTO_INGEST_DEMO_CORPUS=false` 可关闭。已有库不会因文件变化自动同步。
- 基础 Compose 默认将 `data/personal_notes/` 只读挂载到 API 容器的 `/knowledge`，用户不需要配置 override；挂载不自动入库，仍需显式 `ingest /knowledge`。目录只跟踪 `.gitkeep`，私人笔记由 Git 和镜像构建忽略。已有容器需按正常启动命令重建后才有新增挂载；此配置改动未在运行容器中应用。
- 知识库维护统一入口为 `python -m app.rag`：`ingest <目录>` 递归导入 Markdown/txt、`list` 查看来源、`delete <source>` 在输入 `DELETE` 后删除精确来源及块。维护 CLI 不装配聊天服务；查看和删除不调用模型、不触发示例导入。旧 `app.rag.ingest` 和 `--source-prefix` 保留兼容。
- 默认 source 为相对导入根目录的路径，顶层文件仍使用文件名；不同根目录相同相对路径属于同一来源，不做隐式迁移。文件缺失不自动删库，重命名需导入新来源并明确删除旧来源。
- 内容 hash 和索引签名均相同时跳过 embedding 和块替换；签名涵盖切块、分词、输入拼接、标题及 embedding 身份。其他情况先完成 embedding，再在文档级事务中替换旧块。旧记录在下次显式导入时补齐签名，不自动全库重导。BM25 继续依据语料指纹按需重建。
- Markdown 通过 `markdown-it-py` 识别结构；切块软目标 600、硬上限 2000 字符，标题路径及必要代码语言/表头独立于原文保存。`Chunker.split()` 仍返回正文列表，读取与引用正文保持为实际片段。旧表的 `index_signature`、`metadata_json` 由后续服务/CLI 初始化兼容补列；修改向量维度另需处理，不自动清库。
- 当用户明确要求记住规则、计划或决策时，更新本文件；若 `.codex/memories/extensions/ad_hoc/notes/` 可用，再添加 Codex memory note。
