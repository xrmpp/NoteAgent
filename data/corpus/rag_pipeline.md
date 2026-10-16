# 混合检索与评测

知识库检索组合两路候选：pgvector 向量检索面向语义相近内容；Python `bm25s` 词法检索有助于命中术语、名称和配置项。当前每路初始取 10 个候选。

系统用 Reciprocal Rank Fusion（RRF）按名次融合：每路对第 `rank` 名贡献 `1 / (60 + rank)`。这避免直接相加量纲不同的向量距离/相似度与 BM25 分数。正式知识循环在融合后初始交付 Top-3；需要时再通过只读工具补充检索或阅读原文。

`RagService` 保留可选 rerank 配置，但正式知识循环的 `search_knowledge` 调用明确关闭 LLM rerank；`.env.example` 默认 `RERANK_ENABLED=false`。不要把 rerank 写成知识问答主链路当前必经步骤。

离线评测入口 `app.rag.evaluate_retrieval` 接收非空 JSON 用例数组，每项提供 `question` 和 `expected_source`。脚本报告 Recall@3、Recall@5、MRR@5 与延迟，并记录目标来源排名。指标反映所用问题集、语料、embedding 配置与运行环境；它们不能脱离这些条件被解读为一般质量保证。
