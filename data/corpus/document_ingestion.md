# 文档导入与切片

知识文档以 `source` 作为稳定身份。显式导入相同 `source` 时，先比较内容 hash 和索引签名；均未变化则跳过 embedding 和块替换，否则先完成切块及 embedding，再在一个数据库事务中更新文档记录、删除旧 chunks、插入新 chunks。失败的文档保留旧内容，其他已成功文档不受影响。

`Chunker` 使用 Markdown 结构识别标题、段落、代码块、列表和表格。同一小节内合并短段落，代码与表格尽量完整；软目标为 600 字符、硬上限为 2000 字符。超长正文优先按句子切分并保留少量 overlap，超长代码或表格按行拆分，单行仍过长时按字符兜底。结构化切分减少语境破坏，但不保证每个片段语义完全自足。

每个片段保存原文、词法检索用的 `keyword_text`、embedding 及独立元数据。标题路径、代码语言和必要表头可参与检索输入，但不混入引用正文。PostgreSQL 使用 pgvector 向量列与 cosine HNSW 索引；BM25 按语料指纹在进程内按需重建。`hash` embedding 仅是确定性流程 stub，不代表语义向量。

维护入口只有三个操作：`python -m app.rag ingest <directory>` 递归导入 Markdown/txt，source 为相对路径；`python -m app.rag list` 查看来源；`python -m app.rag delete <source>` 在确认后删除指定文档及全部切块。目录文件缺失不自动删库，旧导入入口仍兼容。不同根目录下同一相对路径视为同一来源。

默认在空库启动时导入 `data/corpus/`，个人库可用 `AUTO_INGEST_DEMO_CORPUS=false` 关闭。已有库不自动同步文件变化，也不会自动采用新切块规则，必须显式导入。
