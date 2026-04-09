import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.utils import format_source_page


REVIEW_FILE = Path("data/user/review_cards.json")
REVIEW_INTERVAL_DAYS = {
    "表面的理解": 1,
    "部分理解": 2,
    "概念理解": 7,
    "応用理解": 14,
    0: 1,
    1: 2,
    2: 7,
}


def load_review_cards() -> list:
    if not REVIEW_FILE.exists():
        return []

    try:
        data = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
    except OSError:
        return []


def save_review_cards(cards: list) -> None:
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_next_review_date(score_or_level) -> str:
    today = datetime.now().date()
    days = REVIEW_INTERVAL_DAYS.get(score_or_level, 1)
    next_day = today + timedelta(days=days)
    return next_day.strftime("%Y-%m-%d")

def is_due(card: dict, today: str | None = None) -> bool:
    """カードが今日の復習対象かどうかを返す"""
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    next_review_date = card.get("next_review_date")
    return (next_review_date is None) or (next_review_date <= today)


def update_review_card_score(
    cards: list[dict],
    card_id: str,
    new_score,
) -> dict | None:
    """指定カードの復習スコアを更新して保存し、更新後カードを返す"""
    updated_card = None

    for i, card in enumerate(cards):
        if card.get("id") == card_id:
            card["score"] = new_score
            if isinstance(new_score, str):
                card["level"] = new_score
            card["last_review_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            card["next_review_date"] = compute_next_review_date(new_score)
            cards[i] = card
            updated_card = card
            break

    if updated_card is None:
        return None

    save_review_cards(cards)
    return updated_card


def make_review_card(
    topic: str,
    answer: str,
    hits: List[Document],
    question: str | None = None,
) -> dict:
    sources = [format_source_page(doc.metadata) for doc in hits]

    return {
        "id": str(uuid.uuid4())[:8],
        "topic": topic,
        "question": question or topic,
        "answer": answer,
        "sources": sources,
        "score": None,
        "level": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_review_at": None,
        "next_review_date": datetime.now().strftime("%Y-%m-%d"),
    }