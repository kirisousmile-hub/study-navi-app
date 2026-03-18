import json
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.knowledge_map import find_root_weakness
from core.utils import format_source_page


PROFILE_FILE = Path("data/user/learning_profile.json")
LEARNING_LOG_FILE = Path("data/user/learning_log.json")
WEAK_POINTS_FILE = Path("data/user/weak_points.json")


def load_json_file(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    except OSError:
        return default


def save_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_llm_json(text: str, error_prefix: str = "JSON解析に失敗しました") -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and start < end:
        cleaned = cleaned[start:end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"{error_prefix}。AI出力: {cleaned}") from e

    if not isinstance(data, dict):
        raise ValueError(f"{error_prefix}。JSONがdict形式ではありません: {cleaned}")

    return data


def calc_correct_rate(total: int, correct: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def normalize_topic_label(topic: str) -> str:
    """分析表示用にトピック名を短く整える"""
    if not topic:
        return "不明"

    topic = topic.strip()
    topic = topic.splitlines()[0].strip()

    if ":" in topic:
        left, right = topic.split(":", 1)
        left = left.strip()
        right = right.strip()

        if len(right) > 12:
            right = right[:12].strip()

        return f"{left}: {right}"

    if len(topic) > 16:
        return topic[:16].strip()

    return topic


def load_learning_profile():
    return load_json_file(PROFILE_FILE, {})


def save_learning_profile(profile) -> None:
    save_json_file(PROFILE_FILE, profile)


def update_learning_profile(topic, correct):
    profile = load_learning_profile()

    if topic not in profile:
        profile[topic] = {"total": 0, "correct": 0}

    profile[topic]["total"] += 1

    if correct:
        profile[topic]["correct"] += 1

    save_learning_profile(profile)
    return profile


def load_learning_log() -> list:
    return load_json_file(LEARNING_LOG_FILE, [])


def save_learning_log(logs: list) -> None:
    save_json_file(LEARNING_LOG_FILE, logs)


def add_learning_log(entry):
    logs = load_learning_log()
    logs.append({
        "entry": entry,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_learning_log(logs)
    return logs


def load_weak_points() -> list:
    return load_json_file(WEAK_POINTS_FILE, [])


def save_weak_points(weak_points: list) -> None:
    save_json_file(WEAK_POINTS_FILE, weak_points)


def register_weak_point(topic):
    weak_points = load_weak_points()

    found = False
    for item in weak_points:
        if item.get("topic") == topic:
            item["count"] = item.get("count", 0) + 1
            found = True
            break

    if not found:
        weak_points.append({
            "topic": topic,
            "count": 1
        })

    save_weak_points(weak_points)
    return weak_points


def get_weak_topics_sorted(profile: dict):
    weak_topics = []

    for topic, data in profile.items():
        total = data.get("total", 0)
        correct = data.get("correct", 0)
        score = calc_correct_rate(total, correct)
        weak_topics.append((score, topic))

    weak_topics.sort()
    return weak_topics


def generate_self_test(topic: str, hits: List[Document], llm_creative) -> dict:
    context = "\n\n".join(
        [f"{format_source_page(d.metadata)}\n{d.page_content[:800]}" for d in hits]
    )

    prompt = f"""
以下の教材内容から理解度を確認する問題を作ってください。

# トピック
{topic}

# 教材
{context}

# 出力形式
必ずJSONのみを返してください。
説明文、前置き、```json などのコードフェンスは不要です。

{{
  "questions": ["質問1", "質問2", "質問3"],
  "answers": ["答え1", "答え2", "答え3"]
}}
"""

    result = llm_creative.invoke(prompt).content
    return parse_llm_json(result, "自己テストJSONの解析に失敗しました")


def grade_answer(
    topic: str,
    question: str,
    user_answer: str,
    reference: str,
    llm,
    update_learning_profile,
    add_learning_log,
    register_weak_point
):
    prompt = f"""
問題
{question}

模範回答
{reference}

生徒の回答
{user_answer}

【評価】
正解 / 部分正解 / 不正解

【解説】
"""

    result = llm.invoke(prompt).content
    correct = ("正解" in result) and ("不正解" not in result)

    topic_label = normalize_topic_label(topic)
    update_learning_profile(topic_label, correct)
    add_learning_log("テスト回答")

    if "不正解" in result or "部分正解" in result:
        register_weak_point(topic_label)

    return result


def generate_weak_question(load_weak_points, llm_creative):
    weak = load_weak_points()
    if not weak:
        return None

    weak_sorted = sorted(weak, key=lambda x: x["count"], reverse=True)
    topic = weak_sorted[0]["topic"]

    prompt = f"""
次のPythonトピックについて理解確認問題を1つ作ってください。

トピック
{topic}
"""
    return llm_creative.invoke(prompt).content


def generate_today_mission(load_learning_profile, llm):
    profile = load_learning_profile()
    if not profile:
        return "まず自己テストを行ってください"

    weak_topics = get_weak_topics_sorted(profile)
    focus = [t[1] for t in weak_topics[:3]]

    prompt = f"""
弱点
{focus}

今日のミッションを作ってください
"""
    return llm.invoke(prompt).content


def generate_ai_curriculum(load_learning_profile, llm):
    profile = load_learning_profile()
    if not profile:
        return "まず自己テストを行ってください"

    weakest, prereq = find_root_weakness(profile)

    prompt = f"""
弱点
{weakest}

前提知識
{prereq}

学習順序を作る
"""
    return llm.invoke(prompt).content


def explain_weakness(load_learning_profile, llm):
    profile = load_learning_profile()
    if not profile:
        return "まだ学習データがありません"

    weakest, prereq = find_root_weakness(profile)

    prompt = f"""
弱点
{weakest}

原因と対策を説明
"""
    return llm.invoke(prompt).content


def get_learning_level(load_learning_profile):
    profile = load_learning_profile()
    if not profile:
        return "Lv.0 Beginner"

    total = 0
    correct = 0

    for _, data in profile.items():
        total += data.get("total", 0)
        correct += data.get("correct", 0)

    if total == 0:
        return "Lv.0 Beginner"

    score = calc_correct_rate(total, correct)

    if score > 0.8:
        return "Lv.5 Expert"
    if score > 0.6:
        return "Lv.4 Advanced"
    if score > 0.4:
        return "Lv.3 Intermediate"
    if score > 0.2:
        return "Lv.2 Beginner+"
    return "Lv.1 Beginner"


def generate_next_question(
    db,
    embeddings,
    llm_creative,
    load_learning_profile,
    get_weak_topics_sorted,
    retrieve_hits
):
    profile = load_learning_profile()
    if not profile:
        return None

    weak_topics = get_weak_topics_sorted(profile)
    topic = weak_topics[0][1]
    hits = retrieve_hits(topic, db, embeddings, k=3)

    context = "\n".join([d.page_content[:500] for d in hits])

    prompt = f"""
次の条件で、Pythonの理解確認問題を1つ作ってください。

# トピック
{topic}

# 教材
{context}

# 条件
・短い問題
・コード理解問題
・出力は必ずJSONのみ
・説明文やコードフェンスは不要

# 出力形式
{{
  "question": "問題文",
  "reference": "模範回答",
  "explanation": "解説"
}}
"""

    result = llm_creative.invoke(prompt).content
    data = parse_llm_json(result, "次問題JSONの解析に失敗しました")
    data["topic"] = topic
    return data


def generate_drill_question(topic, llm_creative):
    prompt = f"""
Python学習者向けに
{topic} の理解度を確認する問題を1つ作ってください。

条件
・短い問題
・初心者向け
・答えは書かない
"""
    return llm_creative.invoke(prompt).content


def recommend_next_topic(load_learning_profile, llm, find_root_weakness):
    profile = load_learning_profile()
    if not profile:
        return "まだ学習データがありません"

    weakest, prereq = find_root_weakness(profile)

    prompt = f"""
Python学習コーチとして答えてください。

現在の弱点
{weakest}

前提知識
{prereq}

次に学ぶべきトピックを
1つだけ提案してください。

短く答えてください。
"""
    return llm.invoke(prompt).content


def generate_adaptive_question(
    db,
    embeddings,
    llm_creative,
    load_learning_profile,
    get_weak_topics_sorted,
    retrieve_hits
):
    profile = load_learning_profile()
    if not profile:
        return "まだ弱点データがありません"

    weak_topics = get_weak_topics_sorted(profile)
    topic = weak_topics[0][1]
    hits = retrieve_hits(topic, db, embeddings, k=3)

    context = "\n\n".join([d.page_content[:600] for d in hits])

    prompt = f"""
Pythonの次のトピックについて理解確認問題を1つ作ってください。

トピック
{topic}

教材
{context}

条件
・短い問題
・コード理解問題
"""
    return llm_creative.invoke(prompt).content