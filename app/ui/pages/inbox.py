"""A manual discovery inbox; reuse the CLI workflow without chat/RAG services."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import httpx
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.inbox.aihot import AihotClient
from app.inbox.cli import fetch_candidates
from app.inbox.models import utc_now
from app.inbox.store import InboxStore
from app.inbox.triage import DAILY_CALL_LIMIT, build_messages, load_prompt, triage_candidates
from app.llm.base import ProviderConfigurationError
from app.llm.factory import create_provider
from app.storage.database import create_session_factory


ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "data/inbox/prompts/triage.md"
PROFILE = ROOT / "data/inbox/profile.md"
USAGE = {"practice": "值得实践", "know": "仅了解", "skip": "跳过"}
STATUS = {"new": "待筛选", "triaged": "已筛选", "queued": "想试一试",
          "archived": "留作了解", "dropped": "已跳过", "started": "已开始", "ingested": "已入库"}


@st.cache_resource
def resources():
    settings = get_settings()
    store = InboxStore(create_session_factory(
        settings.database_url, embedding_dimensions=settings.embedding_dimensions,
    ))
    # Single Streamlit process only; this does not serialize separate CLI processes.
    return store, Lock()


def run_action(lock, action):
    if not lock.acquire(blocking=False):
        st.warning("另一个页面正在处理，请稍后重试。")
        return
    try:
        with st.spinner("处理中，请等待结果…"):
            message, candidate_id = action()
        st.session_state["discovery_notice"] = message
        if candidate_id is not None:
            st.session_state["pending_candidate"] = candidate_id
    except (httpx.HTTPError, SQLAlchemyError, ProviderConfigurationError, OSError, ValueError) as exc:
        # Upstream/database errors may carry URLs, credentials or request bodies.
        st.error(f"操作未完成（{type(exc).__name__}）。请检查服务或本地配置；此前已保存的数据仍保留。")
        return
    finally:
        lock.release()
    st.rerun()


def fetch(store, categories, window):
    with AihotClient() as client:
        report = fetch_candidates(store, client, categories=categories, window=window, max_pages=1)
    message = (f"收到 {report.fetched} 条，新增 {report.added} 条，重复跳过 {report.skipped} 条；"
               f"HTTP 请求 {report.requests} 次。此次未调用模型。")
    if report.stopped_categories:
        message += " 部分分类的游标已失效，已保留成功拉取的数据。"
    return message, None


def triage(store, limit, include_reason, candidate_id=None, retriage=False):
    settings = get_settings()
    batch = triage_candidates(
        store, create_provider(settings), model=settings.model_name, limit=limit,
        candidate_id=candidate_id, retriage=retriage, exclude_aihot_reason=not include_reason,
        prompt_path=PROMPT, profile_path=PROFILE,
    )
    message = f"筛选 {batch.candidates} 条，成功 {batch.succeeded} 条，失败 {batch.failed} 条；模型调用 {batch.calls} 次。"
    if batch.quota_reached:
        message += " 已到达今日额度。"
    elif not batch.candidates:
        message += " 没有可处理的候选（新候选累计失败 3 次后不再自动尝试）。"
    return message, batch.results[-1][0].id if batch.results else None


def table(rows):
    st.dataframe([{
        "编号": c.id, "标题": c.title,
        "建议": "筛选失败" if r and r.error else USAGE.get(r.usage if r else None, "待筛选"),
        "相关性": r.relevance if r else None, "质量": r.quality if r else None,
    } for c, r in rows], hide_index=True, width="stretch", height=min(260, 36 * (len(rows) + 1)),
       column_config={"标题": st.column_config.TextColumn(width="medium")})


def original_link(label, url):
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
    except ValueError:
        valid = False
    if valid:
        st.link_button(label, url)
    else:
        st.caption(f"{label}：链接不可用")


def show_result(result):
    if result is None:
        st.info("这条候选尚未筛选。可以先读摘要，再决定是否让模型判断。")
        return
    if result.error:
        st.error(result.error)
        return
    st.markdown(f"**{USAGE[result.usage]}** · 相关性 {result.relevance}/5 · 质量 {result.quality}/5")
    st.write(result.reason)
    if result.try_what:
        st.markdown("**可以尝试什么**")
        st.write(result.try_what)
    if result.est_minutes is not None:
        st.caption(f"模型估计投入：{result.est_minutes} 分钟")
    for label, value in (("前置条件", result.prerequisites_json), ("还不确定", result.unknowns_json)):
        items = json.loads(value)
        if items:
            st.write(f"{label}：" + "；".join(items))


def details(store, lock, candidate, result, include_reason, configured, remaining):
    st.divider()
    st.subheader(candidate.title)
    date = candidate.published_at or candidate.discovered_at
    st.caption(f"#{candidate.id} · {candidate.source_name} · {candidate.category} · "
               f"{date:%Y-%m-%d %H:%M} UTC · {STATUS[candidate.status]}")
    left, right = st.columns(2)
    with left:
        original_link("打开原文", candidate.url)
    with right:
        original_link("在 AIHOT 查看", candidate.aihot_url)
    st.markdown("**上游摘要**")
    st.write(candidate.summary)
    st.caption("当前筛选只读取标题、摘要等元信息；原文需要点开阅读。")
    show_result(result)
    history = store.list_triage_results(candidate.id)
    has_success = any(r.error is None and r.usage for r in history)
    eligible = bool(store.triage_candidates(candidate_id=candidate.id, retriage=has_success))
    if st.button("重新筛选这一条" if has_success else "筛选这一条",
                 disabled=not configured or not remaining or not eligible):
        run_action(lock, lambda: triage(store, 1, include_reason, candidate.id, has_success))

    with st.expander("记录我的决定"):
        st.caption("这是你的选择，与模型建议分开保存。")
        with st.form(f"decision_{candidate.id}"):
            decision = st.selectbox("我的决定", ["想试一试", "留作了解", "跳过"])
            reason = st.text_input("为什么这样决定（想试一试时必填）")
            save = st.form_submit_button("保存决定")
        if save:
            if decision == "想试一试" and not reason.strip():
                st.warning("请写下为什么想试，再保存。")
            else:
                def save_decision():
                    store.change_status(candidate.id, {"想试一试": "queued", "留作了解": "archived", "跳过": "dropped"}[decision],
                                        user_reason=reason.strip(), drop_reason=reason.strip() if decision == "跳过" else "")
                    return "已保存你的决定。", candidate.id
                run_action(lock, save_decision)
        if candidate.user_reason:
            st.write("已记录的理由：", candidate.user_reason)

    with st.expander("这条是怎么筛选的？"):
        st.caption("上游记录 → 拼接读者档案与筛选标准 → 模型 JSON → 类型校验 → 保存结果。")
        st.markdown("**1. 实际收到的上游记录**")
        st.json(json.loads(candidate.raw_json), expanded=False)
        if configured:
            system, version = load_prompt(PROMPT, PROFILE)
            st.markdown("**2. 按当前配置生成的输入预览**")
            st.caption(f"{version}；此处是当前预览，历史结果未保存完整输入，不能据此还原当时的读者档案。")
            for message in build_messages(system, candidate, exclude_aihot_reason=not include_reason):
                st.code(message.content, language="text")
        if history:
            st.markdown("**3. 实际保存的模型输出（每次尝试独立保存）**")
            chosen = st.selectbox("查看某次筛选", list(reversed(history)),
                                  format_func=lambda r: f"#{r.id} · {r.created_at:%Y-%m-%d %H:%M} UTC · {r.prompt_version}",
                                  key=f"history_{candidate.id}")
            st.caption(f"模型：{chosen.model}")
            if chosen.error:
                st.error(chosen.error)
            st.code(chosen.raw_output or "此次没有可保存的模型输出", language="json")
        st.markdown("**4. 状态变化**")
        st.dataframe([{"时间（UTC）": e.created_at.isoformat(), "从": e.from_status or "未存在", "到": e.to_status}
                      for e in store.list_status_events(candidate.id)], hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="发现 · 个人知识助手", page_icon="🔎", layout="wide")
    st.title("发现")
    st.write("拉取候选 → 按我的读者档案筛选 → 阅读原文，决定是否值得投入时间。")
    st.caption("当前来源：AIHOT 已精选的技巧（tip）与论文（paper）。")
    try:
        store, lock = resources()
        rows = store.discovery_candidates()
        used = store.daily_triage_count(utc_now())
    except SQLAlchemyError:
        st.error("暂时无法读取发现列表，请检查数据库连接。")
        st.stop()
    configured = True
    try:
        load_prompt(PROMPT, PROFILE)
    except (OSError, ValueError):
        configured = False
    remaining = max(0, DAILY_CALL_LIMIT - used)
    if notice := st.session_state.pop("discovery_notice", None):
        st.info(notice)
    if pending := st.session_state.pop("pending_candidate", None):
        st.session_state["candidate_id"] = pending
        st.session_state["discovery_view"] = "全部"

    with st.sidebar:
        st.header("发现设置")
        with st.form("fetch_candidates"):
            categories = st.multiselect("候选分类", ["tip", "paper"], default=["tip", "paper"])
            window = st.selectbox("时间范围", ["24h", "7d"])
            st.caption("每次每类最多 1 页、50 条；重复记录自动跳过。")
            fetch_clicked = st.form_submit_button("拉取候选")
        if fetch_clicked:
            if categories:
                run_action(lock, lambda: fetch(store, categories, window))
            else:
                st.warning("至少选择一个分类。")
        st.divider()
        limit = st.number_input("每批筛选条数", min_value=1, max_value=5, value=1)
        include_reason = st.checkbox("参考 AIHOT 推荐理由", value=True)
        st.caption(f"模型：{get_settings().model_name}。每条通常 1 次调用，校验失败最多再试 1 次；单次超时 60 秒。")
        if not configured:
            st.warning("缺少可读的读者档案或筛选 prompt，请检查 data/inbox 挂载。")
        if st.button("筛选下一批", type="primary", disabled=not configured or not remaining):
            run_action(lock, lambda: triage(store, int(limit), include_reason))
        with st.expander("读者档案与筛选标准"):
            st.caption("读取 data/inbox/profile.md 与 prompts/triage.md；修改文件后，下次筛选生效。")
            if configured:
                st.code(PROFILE.read_text(encoding="utf-8"), language="text")
                st.code(PROMPT.read_text(encoding="utf-8"), language="text")

    cols = st.columns(3)
    cols[0].metric("最近候选", len(rows))
    cols[1].metric("其中待筛选", sum(c.status == "new" for c, _ in rows))
    cols[2].metric("今日模型调用（UTC）", f"{used} / {DAILY_CALL_LIMIT}")
    st.caption("列表只读取最新 200 条候选；每条显示最后一次筛选结果。")
    view = st.radio("查看", ["值得实践", "仅了解", "待筛选", "全部"], horizontal=True, key="discovery_view")
    visible = [(c, r) for c, r in rows if view == "全部"
               or (view == "待筛选" and c.status == "new")
               or (r and not r.error and USAGE.get(r.usage) == view and c.status != "dropped")]
    if view in {"值得实践", "仅了解"}:
        visible.sort(key=lambda pair: (pair[1].relevance, pair[1].quality, pair[0].id), reverse=True)
    if not visible:
        st.info("当前视图没有候选。可以切换到“全部”，或先拉取、筛选一小批。")
        return
    table(visible[:3] if view == "值得实践" else visible)
    if view == "值得实践" and len(visible) > 3:
        with st.expander(f"其余 {len(visible) - 3} 条值得实践的候选"):
            table(visible[3:])
    by_id = {c.id: (c, r) for c, r in visible}
    if st.session_state.get("candidate_id") not in by_id:
        st.session_state["candidate_id"] = next(iter(by_id))
    selected = st.selectbox("展开一条候选", list(by_id),
                            format_func=lambda key: f"#{key} · {by_id[key][0].title}", key="candidate_id")
    details(store, lock, *by_id[selected], include_reason, configured, remaining)


if __name__ == "__main__":
    main()
