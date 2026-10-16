from __future__ import annotations

import logging
from collections import Counter

import streamlit as st

from app.ui.api_backend import ApiAgentBackend, ApiBackendError
from app.ui.config import get_ui_settings

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def build_api_backend(base_url: str) -> ApiAgentBackend:
    return ApiAgentBackend(base_url)


def _display_thread_name(title: str | None) -> str:
    if not title or title.strip() in {"Primary Thread", "新会话", "默认会话"}:
        return "新会话"
    return title


def _select_thread() -> None:
    st.session_state["current_thread_id"] = st.session_state["thread_picker"]
    st.session_state.pop("active_trace", None)


def main() -> None:
    settings = get_ui_settings()
    st.title(settings.app_title)
    try:
        _render_chat(build_api_backend(settings.agent_api_base_url))
    except ApiBackendError as exc:
        st.error(f"API 调用失败：{exc}")


def _render_chat(backend: ApiAgentBackend) -> None:
    info = backend.get_info()

    # 1. 确保会话列表与激活会话有效
    threads = backend.list_threads(limit=20)
    if not threads:
        initial = backend.create_thread()
        threads = [initial]

    thread_ids = [t.id for t in threads]
    thread_map = {t.id: t for t in threads}
    titles = {t.id: _display_thread_name(t.title) for t in threads}
    counts = Counter(titles.values())
    # Streamlit resolves selections by formatted text, so equal labels lose identity.
    labels = {t.id: titles[t.id] if counts[titles[t.id]] == 1 else
              f"{titles[t.id]} · {t.created_at:%m-%d %H:%M:%S}" for t in threads}
    if len(set(labels.values())) != len(labels):
        labels = {tid: f"{label} · {index + 1}" for index, (tid, label) in enumerate(labels.items())}

    if "current_thread_id" not in st.session_state or st.session_state["current_thread_id"] not in thread_map:
        st.session_state["current_thread_id"] = thread_ids[0]

    current_id = st.session_state["current_thread_id"]
    st.session_state["thread_picker"] = current_id
    active_thread = thread_map[current_id]
    trace = st.session_state.get("active_trace")
    if trace and trace["thread_id"] != current_id:
        st.session_state.pop("active_trace", None)

    # 加载当前会话消息，判断是否为空白对话（无任何用户提问）
    messages = backend.load_messages(active_thread.id)
    is_empty_dialog = not any(m.role == "user" for m in messages)

    with st.sidebar:
        st.header("知识库助手")
        mode = "agent"
        st.caption("结合本会话最近 3 轮问答理解追问；回答依据本次检索的资料。")

        # 1. 新建会话：若当前已是空白对话，点击不做任何操作（避免创建无意义的重复空会话）
        if st.button("➕ 新建会话", use_container_width=True):
            if not is_empty_dialog:
                new_thread = backend.create_thread()
                st.session_state["current_thread_id"] = new_thread.id
                st.rerun()

        st.selectbox(
            "当前会话",
            options=thread_ids,
            key="thread_picker",
            on_change=_select_thread,
            format_func=labels.__getitem__,
            label_visibility="collapsed",
        )
        col_clear, col_del = st.columns(2)
        with col_clear:
            if st.button("清空消息", type="secondary", use_container_width=True):
                backend.clear_thread(active_thread.id)
                st.session_state.pop("active_trace", None)
                st.rerun()
        with col_del:
            if st.button("删除会话", type="secondary", use_container_width=True):
                backend.delete_thread(active_thread.id)
                st.session_state.pop("active_trace", None)
                remaining = backend.list_threads(limit=1)
                if not remaining:
                    fallback = backend.create_thread()
                    st.session_state["current_thread_id"] = fallback.id
                else:
                    st.session_state["current_thread_id"] = remaining[0].id
                st.rerun()

        st.divider()
        st.header("运行信息")
        st.caption(f"Provider: `{info.provider}`")
        st.caption(f"Model: `{info.model}`")
        st.caption("Backend: `api`")
        st.metric("本会话消息数", len(messages))

    # 右侧主聊天区（复用 messages，零重复数据库查询）
    assistant_run_ids = {
        message.run_id for message in messages
        if message.role == "assistant" and message.run_id is not None
    }
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            show_trace = message.run_id is not None and (
                message.role == "assistant" or message.run_id not in assistant_run_ids
            )
            if show_trace:
                _render_trace(backend, current_id, int(message.run_id))

    prompt = st.chat_input("请输入你的问题")
    if not prompt:
        return
    st.session_state.pop("active_trace", None)

    with st.chat_message("user"):
        st.markdown(prompt)

    result = None
    try:
        with st.status("正在处理问题", expanded=False) as status:
            for event in backend.stream(active_thread.id, prompt, mode=mode):
                if event.node:
                    status.update(label=_progress_label(event))
                if event.event_type == "final":
                    result = event.result
                    state = "complete" if result and result.status == "completed" else "error"
                    label = "回答完成" if state == "complete" else "本次运行未完成"
                    if result and result.stop_reason == "clarification":
                        label = "需要补充说明"
                    status.update(label=label, state=state)
    except ApiBackendError as exc:
        st.error(f"API 调用失败：{exc}")
        return
    except Exception as exc:
        LOGGER.exception("Unexpected error while handling prompt")
        st.error(f"处理请求时发生错误：{exc}")
        return

    if result is None:
        st.error("Agent 没有返回结果。")
        return
    if result.error:
        st.error(result.error)
        return

    st.rerun()


def _toggle_trace(thread_id: int, run_id: int) -> None:
    previous = st.session_state.pop("active_trace", None)
    if not previous or previous["run_id"] != run_id:
        st.session_state["active_trace"] = {"thread_id": thread_id, "run_id": run_id}


def _render_trace(backend, thread_id: int, run_id: int) -> None:
    selected = st.session_state.get("active_trace")
    opened = selected is not None and selected["run_id"] == run_id
    st.button(f"{'收起' if opened else '查看'} Trace #{run_id}", key=f"trace_{run_id}",
              on_click=_toggle_trace, args=(thread_id, run_id))
    if not opened:
        return
    if "result" not in selected and "error" not in selected:
        try:
            selected["result"] = backend.get_run(run_id)
        except Exception:
            LOGGER.exception("Failed to load trace %s", run_id)
            selected["error"] = "加载失败，请收起后重试。"
    if selected.get("error"):
        st.caption(selected["error"])
    elif selected.get("result"):
        _render_trace_summary(selected["result"])
        if selected["result"].status == "running":
            selected.pop("result")  # Only terminal runs are immutable and reusable.
    else:
        st.caption("执行记录不存在。")


def _step_summary(step) -> str:
    latency = f"{step.latency_ms:.0f}ms" if step.latency_ms is not None else "0ms"
    output = step.output or {}
    if step.name == "knowledge_run":
        return f"任务启动 · mode={output.get('mode', 'agent')} · {latency}"
    if step.name == "knowledge_context":
        context = step.input or {}
        detail = output.get("clarification") or output.get("query") or "补全失败"
        return f"上下文补全 · {len(context.get('history_message_ids', [])) // 2} 轮 · {detail} · {latency}"
    if step.name == "knowledge_initial_retrieval":
        query = (step.input or {}).get("query") or output.get("query") or ""
        short_query = (query[:22] + "...") if len(query) > 22 else query
        return f"初始混合检索 · \"{short_query}\" · {latency}"
    if step.name == "knowledge_model":
        model_call = output.get("model_call", "?")
        decision = "调用工具" if output.get("decision") == "tools" else "生成回答"
        tools = output.get("tool_names") or []
        detail = f" -> {', '.join(tools)}" if tools else ""
        return f"模型决策 #{model_call} · {decision}{detail} · {latency}"
    if step.name == "knowledge_tool":
        tool_name = (step.input or {}).get("tool_name") or output.get("tool_name") or "工具执行"
        executed = "失败" if step.error else "成功" if output.get("executed") else "未执行"
        return f"知识工具 · {tool_name} ({executed}) · {latency}"
    if step.name == "knowledge_validation":
        valid = output.get("valid")
        status = "校验通过" if valid else ("校验失败: " + str(output.get("error", "")))
        citations = output.get("citation_ids") or []
        cite_str = f" · 引用: {', '.join(citations)}" if citations else ""
        return f"引用合规校验 · {status}{cite_str} · {latency}"
    if step.name == "knowledge_finished":
        reason = output.get("stop_reason") or "结束"
        return f"运行终态 · stop_reason={reason} · {latency}"
    return f"{step.name} · {latency}"


def _render_trace_summary(result) -> None:
    """Render the explicitly selected trace as a timeline and folded raw data."""
    finished = next((step.output for step in reversed(result.steps)
                     if step.name == "knowledge_finished" and step.output), {})
    tokens = finished.get("tokens")
    chars = finished.get("delivered_text_chars")

    total_ms = result.total_latency_ms
    caption_parts = [f"状态: `{result.status}`"]
    if total_ms is not None:
        caption_parts.append(f"总耗时: `{total_ms:.0f}ms`")
    if tokens is not None:
        caption_parts.append(f"Tokens: `{tokens}`")
    if chars:
        caption_parts.append(f"交付证据: `{chars}字`")
    st.caption(" · ".join(caption_parts))

    for step in result.steps:
        st.markdown(f"- **{step.step_order}.** {_step_summary(step)}")
        if step.error:
            st.error(step.error)

    if result.tool_calls:
        st.markdown("**工具调用记录**：")
        for call in result.tool_calls:
            st.markdown(f"  - `{call.tool_name}` (耗时 {(call.latency_ms or 0):.0f}ms)")

    # 2. 仅保留最底层的一个原始 JSON 节点备查，默认不展开
    raw_payload = {
        "steps": [{"order": s.step_order, "name": s.name, "latency_ms": s.latency_ms, "input": s.input, "output": s.output, "error": s.error} for s in result.steps],
        "tool_calls": [{"tool": c.tool_name, "arguments": c.arguments, "result": c.result, "error": c.error} for c in result.tool_calls],
    }
    st.json(raw_payload, expanded=False)


def _progress_label(event) -> str:
    output = event.payload
    if output.get("error"):
        return "本步骤未完成，正在记录结果"
    if event.node == "knowledge_context":
        return "需要补充说明" if output.get("clarification") else "问题已补全，准备检索"
    if event.node == "knowledge_initial_retrieval":
        return f"初始检索完成：{len(output.get('model_payload', {}).get('chunks', []))} 个片段"
    if event.node == "knowledge_tool":
        action = "补读" if output.get("tool_name") == "read_document" else "补充检索"
        return f"{action}完成：新增交付 {output.get('new_text_chars', 0)} 字符"
    if event.node == "knowledge_model":
        return "模型已请求工具" if output.get("decision") == "tools" else "回答已生成，正在校验引用"
    labels = {
        "knowledge_validation": "引用校验通过",
        "knowledge_finished": "处理完成，正在保存结果",
    }
    return labels.get(event.node, "正在处理问题")


if __name__ == "__main__":
    settings = get_ui_settings()
    st.set_page_config(page_title=settings.app_title, page_icon="🤖", layout="wide")
    st.navigation([st.Page(main, title="知识库助手", icon="🤖", default=True)], position="hidden").run()
