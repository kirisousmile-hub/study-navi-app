import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def infer_lesson_from_path(path: Path) -> str | None:
    s = str(path)
    m = re.search(r"(?:Lesson|lesson)\s*[_\- ]?\s*(\d{1,2})", s)
    if m:
        return m.group(1)
    m = re.search(r"[\\/](\d{1,2})[_\- ]", s)
    if m:
        return m.group(1)
    return None


def format_source_page(meta: dict) -> str:
    src = meta.get("source", "unknown")
    page = meta.get("page", None)
    lesson = meta.get("lesson")

    prefix = f"Lesson{lesson} / " if lesson else ""
    filename = src.split("/")[-1]

    if isinstance(page, int):
        return f"{prefix}{filename} p.{page + 1}"

    return f"{prefix}{filename}"


def unique_by_source_page(docs: List[Document], limit: int) -> List[Document]:
    seen = set()
    out = []

    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))

        if key in seen:
            continue

        seen.add(key)
        out.append(d)

        if len(out) >= limit:
            break

    return out


def count_turns(history: list) -> tuple[int, int]:
    msg_count = len(history)
    turn_count = msg_count // 2
    return turn_count, msg_count