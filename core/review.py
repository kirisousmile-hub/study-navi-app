import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from langchain_core.documents import Document

from core.utils import format_source_page


REVIEW_FILE = Path("data/user/review_cards.json")


def load_review_cards() -> list:
    if REVIEW_FILE.exists():
        try:
            return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_review_cards(cards: list):
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def compute_next_review_date(score: int) -> str:
    today = datetime.now().date()

    if score == 0:
        next_day = today + timedelta(days=1)
    elif score == 1:
        next_day = today + timedelta(days=2)
    elif score == 2:
        next_day = today + timedelta(days=7)
    else:
        next_day = today + timedelta(days=1)

    return next_day.strftime("%Y-%m-%d")


def make_review_card(topic: str, answer: str, hits: List[Document], question: str | None = None) -> dict:
    sources = [format_source_page(d.metadata) for d in hits]

    return {
        "id": str(uuid.uuid4())[:8],
        "topic": topic,
        "question": question or topic,
        "answer": answer,
        "sources": sources,
        "score": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_review_at": None,
        "next_review_date": datetime.now().strftime("%Y-%m-%d"),
    }