from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True, slots=True)
class UISettings:
    app_title: str
    agent_api_base_url: str


@lru_cache(maxsize=1)
def get_ui_settings() -> UISettings:
    return UISettings(
        app_title=os.environ.get("APP_TITLE", "NoteAgent").strip() or "NoteAgent",
        agent_api_base_url=os.environ.get("AGENT_API_BASE_URL", "http://127.0.0.1:8001").strip(),
    )
