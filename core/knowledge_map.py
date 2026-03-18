import json
from pathlib import Path

import streamlit as st


MAP_FILE = Path("knowledge_map.json")


def load_knowledge_map() -> dict:
    if not MAP_FILE.exists():
        return {}

    try:
        data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}


def find_root_weakness(profile: dict):
    if not profile:
        return None, []

    kmap = load_knowledge_map()
    weak_topics = []

    for topic, data in profile.items():
        total = data.get("total", 0)
        correct = data.get("correct", 0)
        score = correct / total if total else 0
        weak_topics.append((score, topic))

    if not weak_topics:
        return None, []

    weak_topics.sort()
    weakest = weak_topics[0][1]
    prereq = kmap.get(weakest, [])

    return weakest, prereq


def show_knowledge_map() -> None:
    kmap = load_knowledge_map()

    st.subheader("🧠 Knowledge Map")

    if not kmap:
        st.caption("Knowledge Map はまだありません。")
        return

    for topic, deps in kmap.items():
        if deps:
            st.write(f"{topic} ← {', '.join(deps)}")
        else:
            st.write(topic)