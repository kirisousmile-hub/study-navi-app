import json
import os
import streamlit as st

MAP_FILE = "knowledge_map.json"


def load_knowledge_map():

    if not os.path.exists(MAP_FILE):
        return {}

    with open(MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_root_weakness(profile):

    kmap = load_knowledge_map()

    weak_topics = []

    for topic, data in profile.items():

        total = data["total"]
        correct = data["correct"]

        score = correct / total if total else 0

        weak_topics.append((score, topic))

    weak_topics.sort()

    weakest = weak_topics[0][1]

    prereq = kmap.get(weakest, [])

    return weakest, prereq


def show_knowledge_map():

    kmap = load_knowledge_map()

    st.subheader("🧠 Knowledge Map")

    for topic, deps in kmap.items():

        if deps:
            st.write(f"{topic} ← {', '.join(deps)}")
        else:
            st.write(topic)