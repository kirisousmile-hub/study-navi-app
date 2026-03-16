import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from typing import List
from langchain_core.documents import Document

from core.utils import format_source_page


# 永続ファイル
REVIEW_FILE = Path("data/user/review_cards.json")


def load_review_cards() -> list:
    """カードをJSONで永続化して復習機能を成立させる"""
    if REVIEW_FILE.exists():
        try:
            return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_review_cards(cards: list):
    REVIEW_FILE.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def compute_next_review_date(score: int) -> str:
    """最小の間隔反復（0/1/2）で次回日付を決める"""
    days = {2: 7, 1: 2, 0: 1}.get(score, 2)
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def make_review_card(topic: str, answer: str, hits: List[Document]) -> dict:
    """RAGや壁打ちまとめを復習カードとして保存する"""

    sources = [format_source_page(d.metadata) for d in hits]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "id": str(uuid.uuid4())[:8],
        "topic": topic.strip(),
        "question": topic.strip(),
        "answer": answer,
        "sources": sources,
        "score": None,
        "created_at": now,
        "last_review_at": None,
        "next_review_date": datetime.now().strftime("%Y-%m-%d"),
    }