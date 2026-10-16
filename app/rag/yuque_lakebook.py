from __future__ import annotations

import hashlib
import html
import json
import re
import tarfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote


CONVERTER_VERSION = "1.1"
SUPPORTED_CARD_TYPES = frozenset({"checkbox", "codeblock", "hr", "image"})
VOID_TAGS = frozenset({"br", "hr", "img", "input", "meta", "source", "wbr"})


class LakebookConversionError(ValueError):
    pass


@dataclass
class HtmlNode:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    children: list[HtmlNode | str] = field(default_factory=list)


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self._stack = [self.root]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        node = HtmlNode(tag.lower(), dict(attrs))
        self._stack[-1].children.append(node)
        if node.tag not in VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._stack[-1].children.append(HtmlNode(tag.lower(), dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


@dataclass(frozen=True)
class ConversionWarning:
    kind: str
    message: str


@dataclass(frozen=True)
class ConversionResult:
    markdown: str
    card_counts: dict[str, int]
    warnings: tuple[ConversionWarning, ...]


@dataclass(frozen=True)
class YuqueDocument:
    slug: str
    title: str
    source_url: str
    created_at: str | None
    updated_at: str | None
    markdown: str
    content_sha256: str
    card_counts: dict[str, int]
    warnings: tuple[ConversionWarning, ...]


@dataclass(frozen=True)
class Lakebook:
    title: str
    path: str
    documents: tuple[YuqueDocument, ...]


@dataclass(frozen=True)
class ExportReport:
    output_dir: Path
    document_count: int
    card_counts: dict[str, int]
    warning_count: int
    manifest_path: Path
    report_path: Path


class LakeAslToMarkdown:
    def __init__(self, *, strict: bool = False, include_image_ocr: bool = True) -> None:
        self._strict = strict
        self._include_image_ocr = include_image_ocr
        self._card_counts: Counter[str] = Counter()
        self._warnings: list[ConversionWarning] = []

    def convert(self, source: str) -> ConversionResult:
        self._card_counts.clear()
        self._warnings.clear()
        parser = _TreeBuilder()
        parser.feed(source)
        parser.close()
        rendered = self._render(parser.root).strip()
        return ConversionResult(
            markdown=f"{rendered}\n" if rendered else "",
            card_counts=dict(sorted(self._card_counts.items())),
            warnings=tuple(self._warnings),
        )

    def _render(self, node: HtmlNode | str, *, list_depth: int = 0) -> str:
        if isinstance(node, str):
            return node

        tag = node.tag
        if tag in {"document", "body"}:
            return self._render_children(node, list_depth=list_depth)
        if tag in {"doctype", "meta", "script", "style"}:
            return ""
        if tag == "br":
            return "\n"
        if tag == "card":
            return self._render_card(node)
        if re.fullmatch(r"h[1-6]", tag):
            level = int(tag[1])
            content = self._clean_inline(self._render_children(node)).strip()
            return f"{'#' * level} {content}\n\n" if content else ""
        if tag == "p":
            content = self._render_children(node).strip()
            return f"{content}\n\n" if content else ""
        if tag in {"ul", "ol"}:
            return self._render_list(node, list_depth=list_depth)
        if tag == "li":
            return self._render_children(node, list_depth=list_depth)
        if tag == "blockquote":
            content = self._render_children(node).strip()
            if not content:
                return ""
            quoted = "\n".join(f"> {line}" if line else ">" for line in content.splitlines())
            return f"{quoted}\n\n"
        if tag in {"strong", "b"}:
            content = self._render_children(node).strip()
            return f"**{content}**" if content else ""
        if tag in {"em", "i"}:
            content = self._render_children(node).strip()
            return f"*{content}*" if content else ""
        if tag in {"del", "s", "strike"}:
            content = self._render_children(node).strip()
            return f"~~{content}~~" if content else ""
        if tag == "code":
            content = self._render_children(node).strip()
            if not content:
                return ""
            fence = "`" * max(1, _longest_run(content, "`") + 1)
            return f"{fence}{content}{fence}"
        if tag == "pre":
            content = self._text_content(node).strip("\n")
            return self._fenced_code(content, "text")
        if tag == "a":
            label = self._render_children(node).strip()
            href = (node.attrs.get("href") or node.attrs.get("data-href") or "").strip()
            if not href:
                return label
            return f"[{label or href}]({href})"
        if tag == "img":
            src = (node.attrs.get("src") or "").strip()
            alt = (node.attrs.get("alt") or "image").strip()
            return f"![{alt}]({src})" if src else ""
        if tag == "hr":
            return "---\n\n"
        if tag == "table":
            return self._render_table(node)
        if tag == "details":
            return self._render_details(node)
        if tag == "summary":
            content = self._clean_inline(self._render_children(node)).strip()
            return f"**{content}**\n\n" if content else ""
        if tag == "sup":
            content = self._render_children(node).strip()
            return f"<sup>{content}</sup>" if content else ""
        if tag == "sub":
            content = self._render_children(node).strip()
            return f"<sub>{content}</sub>" if content else ""
        if tag in {"div", "article", "section", "figure", "figcaption"}:
            content = self._render_children(node, list_depth=list_depth).strip()
            return f"{content}\n\n" if content else ""
        return self._render_children(node, list_depth=list_depth)

    def _render_children(self, node: HtmlNode, *, list_depth: int = 0) -> str:
        return "".join(self._render(child, list_depth=list_depth) for child in node.children)

    def _render_list(self, node: HtmlNode, *, list_depth: int) -> str:
        ordered = node.tag == "ol"
        lines: list[str] = []
        item_number = 0
        for child in node.children:
            if not isinstance(child, HtmlNode) or child.tag != "li":
                continue
            item_number += 1
            direct_children = [
                item
                for item in child.children
                if not isinstance(item, HtmlNode) or item.tag not in {"ul", "ol"}
            ]
            nested_lists = [
                item
                for item in child.children
                if isinstance(item, HtmlNode) and item.tag in {"ul", "ol"}
            ]
            body = "".join(
                self._render(item, list_depth=list_depth + 1) for item in direct_children
            ).strip()
            prefix = f"{item_number}." if ordered else "-"
            indent = "  " * list_depth
            body_lines = body.splitlines() or [""]
            lines.append(f"{indent}{prefix} {body_lines[0]}".rstrip())
            continuation_indent = " " * (len(indent) + len(prefix) + 1)
            lines.extend(
                f"{continuation_indent}{line}".rstrip() for line in body_lines[1:]
            )
            for nested in nested_lists:
                lines.extend(
                    self._render_list(nested, list_depth=list_depth + 1).rstrip().splitlines()
                )
        return "\n".join(lines) + "\n\n" if lines else ""

    def _render_table(self, node: HtmlNode) -> str:
        rows: list[list[str]] = []
        for row in _find_nodes(node, "tr"):
            cells = [
                self._clean_inline(self._render_children(cell)).strip().replace("|", "\\|")
                for cell in row.children
                if isinstance(cell, HtmlNode) and cell.tag in {"th", "td"}
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        lines = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(['---'] * width)} |",
        ]
        lines.extend(f"| {' | '.join(row)} |" for row in normalized[1:])
        return "\n".join(lines) + "\n\n"

    def _render_details(self, node: HtmlNode) -> str:
        summary = next(
            (
                child
                for child in node.children
                if isinstance(child, HtmlNode) and child.tag == "summary"
            ),
            None,
        )
        summary_text = (
            self._clean_inline(self._render_children(summary)).strip() if summary else "折叠内容"
        )
        body = "".join(
            self._render(child)
            for child in node.children
            if child is not summary
        ).strip()
        if not body:
            return f"**{summary_text}**\n\n"
        return f"**{summary_text}**\n\n{body}\n\n"

    def _render_card(self, node: HtmlNode) -> str:
        name = (node.attrs.get("name") or "unknown").strip().lower()
        self._card_counts[name] += 1
        if name not in SUPPORTED_CARD_TYPES:
            return self._unsupported_card(name, "unsupported card type")
        try:
            payload = _decode_card_value(node.attrs.get("value") or "")
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            return self._unsupported_card(name, f"card payload decode failed: {exc}")

        if name == "hr":
            return "\n\n---\n\n"
        if name == "checkbox":
            return "[x] " if payload is True else "[ ] "
        if name == "codeblock":
            if not isinstance(payload, dict):
                return self._unsupported_card(name, "codeblock payload is not an object")
            code = str(payload.get("code") or "")
            language = _safe_language(str(payload.get("mode") or "text"))
            return f"\n\n{self._fenced_code(code, language)}"
        if name == "image":
            if not isinstance(payload, dict):
                return self._unsupported_card(name, "image payload is not an object")
            return self._render_image_card(payload)
        raise AssertionError(f"Unhandled supported card: {name}")

    def _render_image_card(self, payload: dict[str, Any]) -> str:
        src = str(payload.get("src") or "").strip()
        name = str(payload.get("name") or payload.get("title") or "image").strip()
        if not src:
            return self._unsupported_card("image", "image source is empty")
        parts = [f"\n\n![{name}]({src})"]
        if self._include_image_ocr:
            ocr_text = _extract_ocr_text(payload.get("ocrLocations"))
            if ocr_text:
                quoted = "\n".join(f"> {line}" for line in _wrap_text(ocr_text, 180))
                parts.append(f"\n\n> 图片 OCR：\n{quoted}")
        parts.append("\n\n")
        return "".join(parts)

    def _unsupported_card(self, name: str, message: str) -> str:
        warning = ConversionWarning(kind=name, message=message)
        self._warnings.append(warning)
        if self._strict:
            raise LakebookConversionError(f"{name}: {message}")
        return f"\n\n> [未转换的语雀卡片：{name}；{message}]\n\n"

    @staticmethod
    def _fenced_code(code: str, language: str) -> str:
        fence = "`" * max(3, _longest_run(code, "`") + 1)
        return f"{fence}{language}\n{code.rstrip()}\n{fence}\n\n"

    def _text_content(self, node: HtmlNode) -> str:
        return "".join(
            child if isinstance(child, str) else self._text_content(child)
            for child in node.children
        )

    @staticmethod
    def _clean_inline(value: str) -> str:
        return re.sub(r"[ \t]+", " ", value).replace(" \n", "\n")


def read_lakebook(
    path: str | Path,
    *,
    strict: bool = False,
    include_image_ocr: bool = True,
) -> Lakebook:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        meta_members = [
            member for member in members if PurePosixPath(member.name).name == "$meta.json"
        ]
        if len(meta_members) != 1:
            raise LakebookConversionError(
                f"Expected exactly one $meta.json, found {len(meta_members)}."
            )
        outer_meta = _read_json_member(archive, meta_members[0])
        raw_meta = outer_meta.get("meta")
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        if not isinstance(meta, dict):
            raise LakebookConversionError("$meta.json does not contain a valid meta object.")

        book = meta.get("book") or {}
        book_path = str(book.get("path") or "").rstrip("/")
        book_title = str(book.get("name") or book.get("title") or archive_path.stem)
        docs_by_slug = {
            str(item.get("slug")): item
            for item in (meta.get("docs") or [])
            if isinstance(item, dict) and item.get("slug")
        }
        ordered_slugs = _toc_slugs(str(book.get("tocYml") or ""))
        ordered_slugs.extend(slug for slug in docs_by_slug if slug not in ordered_slugs)
        members_by_name = {PurePosixPath(member.name).name: member for member in members}

        converter = LakeAslToMarkdown(
            strict=strict,
            include_image_ocr=include_image_ocr,
        )
        documents: list[YuqueDocument] = []
        for slug in ordered_slugs:
            member = members_by_name.get(f"{slug}.json")
            if member is None:
                raise LakebookConversionError(f"Document payload missing for slug {slug}.")
            payload = _read_json_member(archive, member)
            doc = payload.get("doc")
            if not isinstance(doc, dict):
                raise LakebookConversionError(f"Invalid document payload for slug {slug}.")
            metadata = docs_by_slug.get(slug, {})
            title = str(doc.get("title") or metadata.get("title") or slug)
            source = _select_document_body(doc)
            result = converter.convert(source)
            markdown = _with_front_matter(
                title=title,
                slug=slug,
                source_url=f"{book_path}/{slug}" if book_path else slug,
                created_at=_optional_string(doc.get("created_at") or metadata.get("created_at")),
                updated_at=_optional_string(doc.get("updated_at") or metadata.get("updated_at")),
                body=result.markdown,
            )
            documents.append(
                YuqueDocument(
                    slug=slug,
                    title=title,
                    source_url=f"{book_path}/{slug}" if book_path else slug,
                    created_at=_optional_string(
                        doc.get("created_at") or metadata.get("created_at")
                    ),
                    updated_at=_optional_string(
                        doc.get("updated_at") or metadata.get("updated_at")
                    ),
                    markdown=markdown,
                    content_sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                    card_counts=result.card_counts,
                    warnings=result.warnings,
                )
            )

    return Lakebook(title=book_title, path=book_path, documents=tuple(documents))


def export_lakebook(
    path: str | Path,
    output_dir: str | Path,
    *,
    strict: bool = False,
    include_image_ocr: bool = True,
) -> ExportReport:
    archive_path = Path(path)
    destination = Path(output_dir)
    lakebook = read_lakebook(
        archive_path,
        strict=strict,
        include_image_ocr=include_image_ocr,
    )
    destination.mkdir(parents=True, exist_ok=True)

    card_counts: Counter[str] = Counter()
    warnings: list[dict[str, str]] = []
    manifest_documents: list[dict[str, Any]] = []
    existing_file_names = _existing_manifest_file_names(destination)
    used_file_names: set[str] = set()
    for document in lakebook.documents:
        file_name = _document_file_name(
            document,
            used_file_names,
            preferred=existing_file_names.get(document.slug),
        )
        _atomic_write_text(destination / file_name, document.markdown)
        card_counts.update(document.card_counts)
        warnings.extend(
            {
                "slug": document.slug,
                "kind": warning.kind,
                "message": warning.message,
            }
            for warning in document.warnings
        )
        manifest_documents.append(
            {
                "slug": document.slug,
                "title": document.title,
                "file": file_name,
                "source": document.source_url,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "content_sha256": document.content_sha256,
                "card_counts": document.card_counts,
            }
        )

    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "source_archive": archive_path.name,
        "source_archive_sha256": archive_sha256,
        "book": {"title": lakebook.title, "path": lakebook.path},
        "documents": manifest_documents,
    }
    report = {
        "converter_version": CONVERTER_VERSION,
        "document_count": len(lakebook.documents),
        "card_counts": dict(sorted(card_counts.items())),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    manifest_path = destination / "manifest.json"
    report_path = destination / "conversion-report.json"
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(report_path, report)
    return ExportReport(
        output_dir=destination,
        document_count=len(lakebook.documents),
        card_counts=dict(sorted(card_counts.items())),
        warning_count=len(warnings),
        manifest_path=manifest_path,
        report_path=report_path,
    )


def _read_json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    if member.size > 50 * 1024 * 1024:
        raise LakebookConversionError(f"Archive member is too large: {member.name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise LakebookConversionError(f"Cannot read archive member: {member.name}")
    payload = json.loads(handle.read().decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise LakebookConversionError(f"JSON member is not an object: {member.name}")
    return payload


def _select_document_body(doc: dict[str, Any]) -> str:
    for key in ("body_asl", "body_html", "body", "body_draft_asl", "body_draft"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _decode_card_value(value: str) -> Any:
    decoded = html.unescape(value)
    if decoded.startswith("data:"):
        decoded = decoded[5:]
    decoded = unquote(decoded)
    return json.loads(decoded)


def _extract_ocr_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    entries = [item for item in value if isinstance(item, dict)]
    entries.sort(key=lambda item: (float(item.get("y") or 0), float(item.get("x") or 0)))
    seen: set[str] = set()
    texts: list[str] = []
    for item in entries:
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return " ".join(texts)


def _toc_slugs(toc_yml: str) -> list[str]:
    slugs: list[str] = []
    for line in toc_yml.splitlines():
        match = re.match(r"^\s*url:\s*['\"]?([^'\"\s]+)", line)
        if match and match.group(1) not in slugs:
            slugs.append(match.group(1))
    return slugs


def _with_front_matter(
    *,
    title: str,
    slug: str,
    source_url: str,
    created_at: str | None,
    updated_at: str | None,
    body: str,
) -> str:
    lines = [
        "---",
        f"source: {json.dumps(source_url, ensure_ascii=False)}",
        f"slug: {json.dumps(slug, ensure_ascii=False)}",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"created_at: {json.dumps(created_at, ensure_ascii=False)}",
        f"updated_at: {json.dumps(updated_at, ensure_ascii=False)}",
        "---",
        "",
    ]
    stripped_body = body.strip()
    expected_heading = f"# {title}".strip()
    if not stripped_body.startswith(expected_heading):
        lines.extend([expected_heading, ""])
    if stripped_body:
        lines.append(stripped_body)
    return "\n".join(lines).rstrip() + "\n"


def _find_nodes(node: HtmlNode, tag: str) -> Iterable[HtmlNode]:
    for child in node.children:
        if not isinstance(child, HtmlNode):
            continue
        if child.tag == tag:
            yield child
        yield from _find_nodes(child, tag)


def _longest_run(value: str, character: str) -> int:
    return max((len(match.group(0)) for match in re.finditer(re.escape(character) + "+", value)), default=0)


def _safe_language(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_+.-]", "", value.strip())
    return cleaned or "text"


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip(".-")
    if not cleaned:
        raise LakebookConversionError(f"Cannot create a safe file name for slug {value!r}.")
    return cleaned


def _document_file_name(
    document: YuqueDocument,
    used_file_names: set[str],
    *,
    preferred: str | None = None,
) -> str:
    if preferred:
        preferred_path = Path(preferred)
        safe_preferred = (
            preferred_path.name == preferred
            and preferred_path.suffix.casefold() == ".md"
            and _safe_title_stem(preferred_path.stem, fallback=document.slug)
            == preferred_path.stem
        )
        if safe_preferred and preferred.casefold() not in used_file_names:
            used_file_names.add(preferred.casefold())
            return preferred

    stem = _safe_title_stem(document.title, fallback=document.slug)
    file_name = f"{stem}.md"
    normalized_name = file_name.casefold()
    if normalized_name in used_file_names:
        suffix = f"--{_safe_slug(document.slug)}"
        stem = stem[: max(1, 100 - len(suffix))].rstrip(" .-")
        file_name = f"{stem}{suffix}.md"
        normalized_name = file_name.casefold()
    if normalized_name in used_file_names:
        raise LakebookConversionError(
            f"Cannot create a unique file name for document {document.slug!r}."
        )
    used_file_names.add(normalized_name)
    return file_name


def _existing_manifest_file_names(destination: Path) -> dict[str, str]:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list):
        return {}
    return {
        str(item["slug"]): str(item["file"])
        for item in documents
        if isinstance(item, dict) and item.get("slug") and item.get("file")
    }


def _safe_title_stem(value: str, *, fallback: str) -> str:
    cleaned = unicodedata.normalize("NFKC", value).strip()
    cleaned = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "-", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(" .-")
    cleaned = cleaned[:100].rstrip(" .-")
    if not cleaned:
        cleaned = _safe_slug(fallback)
    if cleaned.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        cleaned = f"{cleaned}-note"
    return cleaned


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _wrap_text(value: str, width: int) -> list[str]:
    return [value[index : index + width] for index in range(0, len(value), width)]


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
