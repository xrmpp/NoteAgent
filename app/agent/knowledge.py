"""Compatibility entry point for the knowledge loop and its evidence ledger.

The implementations live in ``loop``, ``evidence`` and ``knowledge_prompts``.
This module keeps the older import path working (``app.agent.knowledge``).
"""
from __future__ import annotations

from app.agent.evidence import EvidenceStore, model_tool_payload
from app.agent.knowledge_prompts import KNOWLEDGE_PROMPT_VERSION, SYSTEM_PROMPT
from app.agent.loop import KnowledgeLoop


__all__ = [
    "EvidenceStore",
    "KnowledgeLoop",
    "KNOWLEDGE_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "model_tool_payload",
]
