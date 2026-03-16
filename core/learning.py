import json
from typing import List
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document

from core.utils import format_source_page
from core.knowledge_map import find_root_weakness


# 外部依存（appから渡される）
# LLM
# LLM_CREATIVE
# retrieve_hits
# load_learning_profile
# update_learning_profile
# add_learning_log
# register_weak_point
# load_weak_points


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
        "questions": [
            "質問1",
            "質問2",
            "質問3"
        ],
        "answers": [
            "答え1",
            "答え2",
            "答え3"
        ]
        }}
        """

    result = llm_creative.invoke(prompt).content.strip()

    # コードフェンス除去
    result = result.replace("```json", "").replace("```", "").strip()

    # 先頭の { から末尾の } までを切り出す
    start = result.find("{")
    end = result.rfind("}")

    if start != -1 and end != -1 and start < end:
        result = result[start:end + 1]

    try:
        data = json.loads(result)
    except Exception as e:
        raise ValueError(f"自己テストJSONの解析に失敗しました。AI出力: {result}") from e

    return data


def normalize_topic_label(topic: str) -> str:
    """分析表示用にトピック名を短く整える"""

    if not topic:
        return "不明"

    topic = topic.strip()

    # 改行があれば1行目だけ使う
    topic = topic.splitlines()[0].strip()

    # 「A1: xxx」形式なら右側だけ短くする
    if ":" in topic:
        left, right = topic.split(":", 1)
        left = left.strip()
        right = right.strip()

        if len(right) > 12:
            right = right[:12].strip()

        return f"{left}: {right}"

    # 普通の文字列なら16文字まで
    if len(topic) > 16:
        return topic[:16].strip()

    return topic


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

    question = llm_creative.invoke(prompt).content

    return question


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

    total = 0
    correct = 0

    for topic, data in profile.items():

        total += data["total"]
        correct += data["correct"]

    if total == 0:
        return "Lv.0 Beginner"

    score = correct / total

    if score > 0.8:
        return "Lv.5 Expert"
    elif score > 0.6:
        return "Lv.4 Advanced"
    elif score > 0.4:
        return "Lv.3 Intermediate"
    elif score > 0.2:
        return "Lv.2 Beginner+"
    else:
        return "Lv.1 Beginner"


PROFILE_FILE = Path("data/user/learning_profile.json")


def load_learning_profile():
    """
    学習プロファイルを読み込む
    """
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    return {}


def save_learning_profile(profile):
    """
    学習プロファイルを保存
    """
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)

    PROFILE_FILE.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

LEARNING_LOG_FILE = Path("data/user/learning_log.json")
WEAK_POINTS_FILE = Path("data/user/weak_points.json")


def update_learning_profile(topic, correct):
    """
    学習プロファイルを更新する
    """
    profile = load_learning_profile()

    if topic not in profile:
        profile[topic] = {
            "total": 0,
            "correct": 0
        }

    profile[topic]["total"] += 1

    if correct:
        profile[topic]["correct"] += 1

    save_learning_profile(profile)

    return profile


def load_learning_log() -> list:
    if LEARNING_LOG_FILE.exists():
        try:
            return json.loads(LEARNING_LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_learning_log(logs: list):
    LEARNING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNING_LOG_FILE.write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def add_learning_log(entry):
    """
    学習ログを追加
    """
    logs = load_learning_log()

    logs.append({
        "entry": entry,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    save_learning_log(logs)

    return logs


def load_weak_points() -> list:
    if WEAK_POINTS_FILE.exists():
        try:
            return json.loads(WEAK_POINTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_weak_points(weak_points: list):
    WEAK_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEAK_POINTS_FILE.write_text(
        json.dumps(weak_points, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def register_weak_point(topic):
    """
    弱点を登録
    """
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
    """
    学習プロフィールから弱点トピックをスコア順で返す
    score = correct / total
    小さいほど弱い
    """

    weak_topics = []

    for topic, data in profile.items():

        total = data.get("total", 0)
        correct = data.get("correct", 0)

        if total == 0:
            score = 0
        else:
            score = correct / total

        weak_topics.append((score, topic))

    weak_topics.sort()

    return weak_topics