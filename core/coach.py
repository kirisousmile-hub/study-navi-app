import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.utils import format_source_page


WALL_MEMORY_FILE = "wall_memory.json"
TURN_LIMIT = 30


def empty_wall_memory() -> dict:
    return {"facts": [], "summaries": []}


def normalize_wall_memory(data: dict) -> dict:
    data.setdefault("facts", [])
    data.setdefault("summaries", [])
    return data


def load_wall_memory() -> dict:
    path = Path(WALL_MEMORY_FILE)

    if not path.exists():
        return empty_wall_memory()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return empty_wall_memory()
        return normalize_wall_memory(data)
    except json.JSONDecodeError:
        return empty_wall_memory()
    except OSError:
        return empty_wall_memory()


def save_wall_memory(mem: dict) -> None:
    Path(WALL_MEMORY_FILE).write_text(
        json.dumps(normalize_wall_memory(mem), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_wall_memory_item(key: str, item: dict) -> dict:
    data = load_wall_memory()
    data.setdefault(key, [])
    data[key].append(item)
    save_wall_memory(data)
    return item


def add_wall_fact(text: str) -> dict:
    fact = {
        "id": str(uuid.uuid4())[:8],
        "text": text.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return append_wall_memory_item("facts", fact)


def add_wall_summary(summary_text: str) -> dict:
    item = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "text": summary_text.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return append_wall_memory_item("summaries", item)


def build_memory_block(
    limit: int = 10,
    include_facts: bool = True,
    include_summaries: bool = True,
) -> str:
    data = load_wall_memory()
    lines = []

    if include_facts:
        facts = data.get("facts", [])[-limit:]
        if facts:
            lines.append("【長期記憶: 固定メモ】")
            for fact in facts:
                lines.append(f"- {fact['text']}")

    if include_summaries:
        summaries = data.get("summaries", [])[-limit:]
        if summaries:
            lines.append("【長期記憶: 学習要約】")
            for summary in summaries:
                lines.append(f"- {summary['text']}")

    return "\n".join(lines).strip()


def get_role_rule(mode: str) -> str:
    if mode.startswith("A"):
        return (
            "あなたは『用語理解コーチ』です。\n"
            "目的は、用語の意味・役割・具体例を、ユーザーが理解できる形に整理して伝えることです。\n"
            "必要なら短く説明してから確認してください。\n"
            "1回の返答では、意味・役割・具体例のどれか1つだけを進めてください。"
        )

    if mode.startswith("B"):
        return (
            "あなたは『設計理解コーチ』です。\n"
            "目的は、分ける理由・役割の違い・設計判断を、ユーザーが納得できる形に整理することです。\n"
            "必要なら短く説明してから確認してください。\n"
            "1回の返答では、分ける理由・分けないと困ること・境界のどれか1つだけを進めてください。"
        )

    return (
        "あなたは『コード理解コーチ』です。\n"
        "目的は、コードを1行ずつ読み、実行順・変数の変化・出力を、ユーザーが理解できる形に整理することです。\n"
        "必要なら短く説明してから確認してください。\n"
        "1回の返答では、実行順・変数・出力予想のどれか1つだけを進めてください。\n"
        "特にコード理解では、確認したい値や出力を先に完全に言い切らず、直前のヒントまでにとどめてください。\n"
        "ユーザーが『最初の値』『次の値』『出力』を考えられる段階なら、正解を先に書かずに質問してください。"
    )


def build_context_block(hits: List[Document], limit: int = 3, max_chars: int = 500) -> str:
    context = "\n\n".join(
        [f"{format_source_page(d.metadata)}\n{d.page_content[:max_chars]}" for d in hits[:limit]]
    )
    return context or "(教材根拠なし)"


def coach_reply(
    history: list,
    hits: List[Document],
    mode: str,
    llm,
    use_long_memory: bool = False,
) -> str:
    recent = history[-(TURN_LIMIT * 2):]
    context = build_context_block(hits)

    coaching_rule = (
        "次のルールを必ず守ってください。\n"
        "- 返答は合理的で簡潔にする\n"
        "- 目的は『理解を助けること』であり、雑談や過剰な励ましではない\n"
        "- まず短く受け止める\n"
        "- 次に必要最小限の説明を入れる\n"
        "- 最後に理解確認の質問を1つだけ入れる\n"
        "- 長文講義にしない\n"
        "- 1回の返答では1テーマだけ進める\n"
        "- 1回の返答で結論まで全部説明しない\n"
        "- 説明は、次の1問に答えられる分だけにとどめる\n"
        "- 直前と同じ聞き方を繰り返しすぎない\n"
        "- ユーザーが詰まっていそうなら、説明を少しだけ厚くしてよい\n"
        "- 教材根拠があるときはその内容を優先する\n"
        "- 教材にないことは断定しない\n"
        "- 箇条書きにしない\n"
        "- 2〜4文で返す\n"
        "- 最後の1文は必ず質問文にする\n"
        "- 最後の確認は『理解できましたか？』ではなく、具体的に答えられる質問にする\n"
        "- はい/いいえで終わる確認ではなく、値・役割・違い・例を答えさせる質問を優先する\n"
        "- コード理解では、最終出力を先に言い切る前に、途中の値や最初の1回を確認する\n"
        "- 口調はやわらかすぎず、冷たすぎず、落ち着いた学習コーチとして振る舞う\n"
        "- 『素晴らしいです』『完璧です』『一緒に頑張りましょう』のような大げさな表現は避ける\n"
        "- コード理解では『iは0です。その値は何ですか？』のように、答えを書いた直後に同じことを聞かない\n"
    )

    style_hint = (
        "出力スタイル:\n"
        "1. 受け止め 1文\n"
        "2. 必要なら短い説明 1〜2文\n"
        "3. 最後に、具体的に答えられる理解確認の質問 1文\n"
    )

    messages = [
        SystemMessage(content=get_role_rule(mode)),
        SystemMessage(content=coaching_rule),
        SystemMessage(content=style_hint),
    ]

    if use_long_memory:
        memory_text = build_memory_block(limit=10, include_facts=True, include_summaries=True)
        if memory_text:
            messages.append(SystemMessage(content=f"長期記憶\n{memory_text}"))

    messages.append(
        SystemMessage(
            content=(
                "教材根拠\n"
                f"{context}\n\n"
                "教材に一致する内容があれば、その範囲で説明と確認を行ってください。"
            )
        )
    )

    for message in recent:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))

    return llm.invoke(messages).content


def summarize_wall_history(history: list, hits: List[Document], llm) -> str:
    context = "\n\n".join(
        [f"- {format_source_page(d.metadata)}\n{d.page_content[:800]}" for d in hits]
    )

    prompt = f"""
以下の壁打ち履歴を、次回の続きを再開しやすい形で要約してください。

# 壁打ち履歴
{history}

# 根拠
{context}

# 出力フォーマット
【今回わかったこと】
-

【まだ曖昧なこと】
-

【次に続ける論点】
-

【覚えておくルール】
-
"""
    return llm.invoke(prompt).content


def delete_wall_fact(fact_id: str) -> bool:
    data = load_wall_memory()
    facts = data.get("facts", [])
    new_facts = [fact for fact in facts if fact.get("id") != fact_id]

    if len(new_facts) == len(facts):
        return False

    data["facts"] = new_facts
    save_wall_memory(data)
    return True


def delete_wall_summary(summary_id: str) -> bool:
    data = load_wall_memory()
    summaries = data.get("summaries", [])
    new_summaries = [summary for summary in summaries if summary.get("id") != summary_id]

    if len(new_summaries) == len(summaries):
        return False

    data["summaries"] = new_summaries
    save_wall_memory(data)
    return True