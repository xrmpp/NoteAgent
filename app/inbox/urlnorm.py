from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMETERS = {"ref", "fbclid", "gclid"}


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Expected an absolute HTTP(S) URL.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL credentials are not allowed.")
    host = parts.hostname.lower()
    if host == "twitter.com":
        host = "x.com"
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    query = urlencode([
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS
    ])
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), query, ""))


def content_hash(title: str, summary: str) -> str:
    # Keep the field boundary so different title/summary splits cannot collide.
    normalized = "\n".join(re.sub(r"\s+", " ", value).strip().casefold()
                           for value in (title, summary))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
