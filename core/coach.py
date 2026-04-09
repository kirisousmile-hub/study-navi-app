import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.utils import format_source_page
from core.learning import parse_llm_json


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
    if not hits:
        return "(教材根拠なし)"

    def score_doc(doc: Document) -> int:
        score = 0

        text = (doc.page_content or "").strip()
        meta_text = " ".join(
            str(v) for v in (doc.metadata or {}).values() if v is not None
        )
        full_text = f"{meta_text}\n{text}".lower()

        # 具体的な学習トピックがあるものを少し優先
        good_keywords = [
            "変数", "if", "for", "while", "関数", "引数", "戻り値",
            "list", "dict", "tuple", "set",
            "class", "import", "例外", "try", "def", "return",
            "文法", "基本", "コード", "サンプル", "例"
        ]
        for kw in good_keywords:
            if kw.lower() in full_text:
                score += 2

        # 概要・導入・広すぎる説明は少し弱める
        bad_keywords = [
            "pythonとは", "概要", "はじめに", "導入", "全体像",
            "特徴", "歴史", "目次", "まとめ", "overview"
        ]
        for kw in bad_keywords:
            if kw.lower() in full_text:
                score -= 2

        # 短すぎず長すぎない方を少し優先
        text_len = len(text)
        if 150 <= text_len <= 1200:
            score += 2
        elif text_len < 80:
            score -= 2
        elif text_len > 2500:
            score -= 1

        # コード例らしさがあるものを少し優先
        code_markers = ["def ", "for ", "if ", "print(", "return ", "try:", "except"]
        if any(marker in text for marker in code_markers):
            score += 2

        # 見出しっぽいものだけで中身が薄い場合は弱める
        if text_len < 200 and ("章" in full_text or "lesson" in full_text):
            score -= 1

        return score

    ranked_hits = sorted(hits, key=score_doc, reverse=True)

    blocks = []
    for doc in ranked_hits[:limit]:
        source = format_source_page(doc.metadata)
        body = (doc.page_content or "").strip().replace("\n\n", "\n")
        body = body[:max_chars]
        blocks.append(f"{source}\n{body}")

    return "\n\n".join(blocks) if blocks else "(教材根拠なし)"


def coach_reply(
    history: list,
    hits: List[Document],
    mode: str,
    llm,
    use_long_memory: bool = False,
    focus: Dict | None = None,
) -> str:
    recent = history[-(TURN_LIMIT * 2):]
    context = build_context_block(hits)

    coaching_rule = (
        "次のルールを必ず守ってください。\n"
        "- 返答の目的は、ユーザーを次の1歩に進めることです\n"
        "- 雑談ではなく学習前進を優先してください\n"
        "- まず短く受け止める\n"
        "- 次に必要最小限の説明を入れる\n"
        "- 最後に理解確認または次の1歩の質問を1つだけ入れる\n"
        "- 長文講義にしない\n"
        "- 1回の返答では1テーマだけ進める\n"
        "- 1回の返答で結論まで全部説明しない\n"
        "- 説明は、次の1問に答えられる分だけにとどめる\n"
        "- 直前と同じ聞き方を繰り返しすぎない\n"
        "- ユーザーが詰まっていそうなら、説明を少しだけ厚くしてよい\n"
        "- 教材根拠があるときはその内容を優先する\n"
        "- ただし、教材の中でも『初学者が次に読むのに向く内容』を優先して扱う\n"
        "- 概要だけの説明より、最初の一歩に使える内容を優先する\n"
        "- 教材にないことは断定しない\n"
        "- 箇条書きにしない\n"
        "- 2〜4文で返す\n"
        "- 最後の1文は必ず質問文にする\n"
        "- 最後の確認は『理解できましたか？』ではなく、具体的に答えられる質問にする\n"
        "- はい/いいえで終わる確認ではなく、値・役割・違い・例を答えさせる質問を優先する\n"
        "- ユーザーの入力が広すぎる場合は、そのまま広く返さず、学習テーマを1つに絞って提示する\n"
        "- ユーザーの入力が広すぎる場合は、AI側で候補を最大3つまでに絞って提示する\n"
        "- 『どのような点に興味がありますか？』『何を学びたいですか？』のような自由すぎる質問は避ける\n"
        "- 初学者には、用語一覧を広く投げず、最初に扱う1テーマを決めてから聞く\n"
        "- 初学者への最後の質問は、選択式または短く具体的に答えられる形を優先する\n"
        "- コード理解では、最終出力を先に言い切る前に、途中の値や最初の1回を確認する\n"
        "- 口調はやわらかすぎず、冷たすぎず、落ち着いた学習コーチとして振る舞う\n"
        "- 『素晴らしいです』『完璧です』『一緒に頑張りましょう』のような大げさな表現は避ける\n"
        "- コード理解では『iは0です。その値は何ですか？』のように、答えを書いた直後に同じことを聞かない\n"
        "- 最後の質問は原則として2〜3択にする\n"
        "- 選択肢は『1. ○○ 2. ○○ 3. ○○』のように短く並べる\n"
        "- 自由記述でないと成立しない場合を除き、最後を自由質問で終えない\n"
    )

    style_hint = (
        "出力スタイル:\n"
        "1. 受け止め 1文\n"
        "2. 必要なら短い説明 1〜2文\n"
        "3. 最後に、2〜3択の理解確認または次の1歩の質問 1文\n\n"
        "追加ルール:\n"
        "- ユーザーの入力が広いときは、AI側で最初の学習テーマを1つに絞る\n"
        "- その際は、必要に応じて候補を最大3つまで示してよい\n"
        "- 初学者には自由記述を求めすぎず、選びやすい聞き方にする\n"
        "- 最後の質問は原則として『1. ○○ 2. ○○ 3. ○○』のような2〜3択にする\n"
        "- 選択肢は短く、1テーマの中の次の一歩になるものだけにする\n"
        "- 教材根拠を使うときは、『今読む意味がある箇所』に寄せて説明する\n"
        "- 概要説明で終わらず、『今どこをやるか』がわかる返しにする\n"
    )

    messages = [
        SystemMessage(content=get_role_rule(mode)),
        SystemMessage(content=coaching_rule),
        SystemMessage(content=style_hint),
    ]

    if focus:
        choice_hints_text = ", ".join(focus.get("choice_hints", [])) or "なし"

        focus_hint = (
            f"今回の復習テーマ: {focus['topic_label']} ({focus['lesson']})\n"
            f"候補: {', '.join(focus['candidate_labels']) or 'なし'}\n"
            f"このテーマで優先したい切り口: {choice_hints_text}\n"
            f"必要なら補助として触れてよいPython: {', '.join(focus['support_python']) or 'なし'}\n"
            "主テーマを優先し、補助知識が主役にならないようにしてください。\n"
            "最後の質問は、可能なら『このテーマで優先したい切り口』を使って2〜3択にしてください。"
        )
        messages.append(SystemMessage(content=focus_hint))

    if use_long_memory:
        memory_text = build_memory_block(limit=10, include_facts=True, include_summaries=True)
        if memory_text:
            messages.append(SystemMessage(content=f"長期記憶\n{memory_text}"))

    messages.append(
        SystemMessage(
            content=(
                "教材根拠\n"
                f"{context}\n\n"
                "教材に一致する内容があれば、その範囲で説明と確認を行ってください。\n"
                "ただし、教材の中でも『初学者が今読むと前進しやすい部分』を優先して扱ってください。\n"
                "広い概要より、最初の1歩に使える具体部分を優先してください。"
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

COURSE_TOPIC_MAP = [
    {
        "id": "llm_basics",
        "label": "LLM基礎",
        "lesson": "Lesson 1",
        "keywords": [
            "llm", "大規模言語モデル", "生成ai", "chatgpt", "前提知識"
        ],
        "query_keywords": ["LLM", "生成AI", "ChatGPT", "前提知識"],
        "support_python": [],
    },
    {
        "id": "prompt_engineering",
        "label": "プロンプトエンジニアリング",
        "lesson": "Lesson 2-3",
        "keywords": [
            "prompt", "プロンプト", "few-shot", "zero-shot",
            "指示", "出力形式", "プロンプトエンジニアリング"
        ],
        "query_keywords": ["プロンプト", "few-shot", "出力形式", "実務活用"],
        "support_python": [],
    },
    {
        "id": "colab",
        "label": "Google Colab",
        "lesson": "Lesson 4",
        "keywords": [
            "colab", "google colaboratory", "ノートブック", "ランタイム"
        ],
        "query_keywords": ["Google Colab", "ノートブック", "環境"],
        "support_python": [],
    },
    {
        "id": "python_basics",
        "label": "Python基礎",
        "lesson": "Lesson 5",
        "keywords": [
            "python", "for", "if", "while", "関数", "引数", "戻り値",
            "list", "dict", "import", "例外", "def", "return"
        ],
        "query_keywords": ["Python", "基本文法", "関数", "for", "dict", "import"],
        "support_python": [],
    },
    {
        "id": "openai_api",
        "label": "OpenAI API",
        "lesson": "Lesson 6",
        "keywords": [
            "openai", "api", "chat completions", "client",
            "apiキー", "レスポンス", "モデル呼び出し"
        ],
        "query_keywords": ["OpenAI API", "Chat Completions", "client"],
        "support_python": ["import", "関数", "dict", "例外"],
    },
    {
        "id": "langchain_intro",
        "label": "LangChain概要",
        "lesson": "Lesson 7",
        "keywords": ["langchain", "langchainの概要", "langchainとは"],
        "query_keywords": ["LangChain", "概要"],
        "support_python": ["import", "関数"],
        "choice_hints": [
            "LangChainとは何か",
            "何ができるのか",
            "主要モジュールの全体像",
        ],
    },
    {
        "id": "langchain_models",
        "label": "LangChain Language Models",
        "lesson": "Lesson 8",
        "keywords": [
            "language model", "language models", "chat model",
            "llmchain", "langchain model"
        ],
        "query_keywords": ["LangChain", "Language Models", "Chat Model"],
        "support_python": ["import", "関数"],
    },
    {
        "id": "langchain_prompts",
        "label": "LangChain Prompts",
        "lesson": "Lesson 9",
        "keywords": [
            "prompttemplate", "prompt template", "example selector",
            "langchain prompt", "prompts"
        ],
        "query_keywords": ["LangChain", "PromptTemplate", "Example Selector"],
        "support_python": ["import", "関数", "dict"],
    },
    {
        "id": "langchain_output_parser",
        "label": "LangChain Output Parser",
        "lesson": "Lesson 10",
        "keywords": [
            "output parser", "parser", "structured output", "json output"
        ],
        "query_keywords": ["LangChain", "Output Parser", "Structured Output"],
        "support_python": ["dict", "list", "関数"],
    },
    {
        "id": "langchain_chains",
        "label": "LangChain Chains",
        "lesson": "Lesson 11",
        "keywords": ["chain", "chains", "lcEL", "runnable", "pipe"],
        "query_keywords": ["LangChain", "Chains", "Runnable"],
        "support_python": ["関数", "import", "dict"],
    },
    {
        "id": "langchain_memory",
        "label": "LangChain Memory",
        "lesson": "Lesson 12",
        "keywords": ["memory", "conversationbuffermemory", "会話履歴", "記憶"],
        "query_keywords": ["LangChain", "Memory", "会話履歴"],
        "support_python": ["dict", "list", "関数"],
    },
    {
        "id": "rag",
        "label": "RAG / Retrieval",
        "lesson": "Lesson 13",
        "keywords": [
            "rag", "retrieval", "ベクトル", "embedding",
            "ベクトル検索", "chroma", "会話履歴の記憶機能"
        ],
        "query_keywords": ["RAG", "Retrieval", "Embedding", "Chroma"],
        "support_python": ["import", "関数", "list", "dict"],
    },
    {
        "id": "agents",
        "label": "Agents",
        "lesson": "Lesson 14",
        "keywords": ["agent", "agents", "tool", "tools", "自作tool"],
        "query_keywords": ["Agents", "Tool", "Retrieval"],
        "support_python": ["import", "関数", "dict"],
    },
    {
        "id": "callbacks",
        "label": "Callbacks",
        "lesson": "Lesson 15",
        "keywords": ["callback", "callbacks", "ログ", "ストリーミング"],
        "query_keywords": ["Callbacks", "ロギング", "ストリーミング"],
        "support_python": ["関数", "import"],
    },
    {
        "id": "llamaindex",
        "label": "LlamaIndex",
        "lesson": "Lesson 17",
        "keywords": ["llamaindex", "index", "query engine"],
        "query_keywords": ["LlamaIndex", "RAG"],
        "support_python": ["import", "関数", "dict"],
    },
    {
        "id": "finetuning",
        "label": "ファインチューニング",
        "lesson": "Lesson 18",
        "keywords": ["fine tuning", "finetuning", "ファインチューニング"],
        "query_keywords": ["ファインチューニング", "学習データ"],
        "support_python": [],
    },
    {
        "id": "dify",
        "label": "Dify",
        "lesson": "Lesson 19",
        "keywords": ["dify", "workflow", "ノーコード"],
        "query_keywords": ["Dify", "Workflow"],
        "support_python": [],
    },
    {
        "id": "streamlit",
        "label": "Streamlit",
        "lesson": "Lesson 21",
        "keywords": ["streamlit", "webアプリ", "st.", "community cloud", "デプロイ"],
        "query_keywords": ["Streamlit", "Webアプリ", "デプロイ"],
        "support_python": ["import", "関数", "if"],
    },
    {
        "id": "evaluation",
        "label": "生成AIの評価と改善",
        "lesson": "Lesson 22",
        "keywords": [
            "評価", "改善", "llm as a judge", "judge",
            "評価指標", "改善策"
        ],
        "query_keywords": ["評価", "改善", "LLM as a Judge"],
        "support_python": [],
    },
    {
        "id": "final_app",
        "label": "最終課題 / オリジナルLLMアプリ開発",
        "lesson": "Lesson 23-25",
        "keywords": [
            "最終課題", "オリジナルアプリ", "llmアプリ", "社内情報特化型",
            "aiエージェント", "英会話アプリ"
        ],
        "query_keywords": ["LLMアプリ", "最終課題", "オリジナルアプリ"],
        "support_python": ["import", "関数", "dict", "list"],
    },
    
]


def map_to_course_topic(user_msg: str) -> Dict:
    text = (user_msg or "").strip().lower()

    if not text:
        return {
            "topic_id": "unknown",
            "topic_label": "未特定",
            "lesson": "",
            "adjusted_query": "",
            "support_python": [],
            "candidate_labels": [],
        }

    scored = []
    for topic in COURSE_TOPIC_MAP:
        score = 0
        for kw in topic["keywords"]:
            kw_l = kw.lower()
            if kw_l in text:
                # 長いキーワードほど少し強くする
                score += 3 if len(kw_l) >= 6 else 2

        # topic名そのものが入る場合を少し強化
        if topic["label"].lower() in text:
            score += 3

        if score > 0:
            scored.append((score, topic))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return {
            "topic_id": "general_review",
            "topic_label": "コース全体の復習相談",
            "lesson": "",
            "adjusted_query": user_msg,
            "support_python": [],
            "candidate_labels": [],
        }

    best_topic = scored[0][1]
    candidates = [t["label"] for _, t in scored[:3]]

    adjusted_query_parts = [best_topic["label"], best_topic["lesson"]]
    adjusted_query_parts.extend(best_topic["query_keywords"])

    adjusted_query = " ".join([p for p in adjusted_query_parts if p])

    return {
        "topic_id": best_topic["id"],
        "topic_label": best_topic["label"],
        "lesson": best_topic["lesson"],
        "adjusted_query": adjusted_query,
        "support_python": best_topic["support_python"],
        "candidate_labels": candidates,
        "choice_hints": best_topic.get("choice_hints", []),
    }


def build_wall_excellent_card(summary: str, hits: list, llm) -> dict:
    context = "\n\n".join(
        [f"{format_source_page(d.metadata)}\n{d.page_content[:500]}" for d in hits[:2]]
    ) if hits else ""

    prompt = f"""
あなたは学習アプリの復習カード作成アシスタントです。
次の壁打ちまとめから、「復習価値が最も高い1枚の優カード」を作ってください。

# 目的
- あとで見返したときに、一番重要な学びを短時間で思い出せること
- まとめ全文ではなく、最重要ポイントを1枚に圧縮すること

# 壁打ちまとめ
{summary}

# 参照候補
{context}

# 条件
- 出力は必ずJSONのみ
- 説明文、前置き、コードフェンスは禁止
- topic は短く具体的に
- question は「何を思い出すカードか」が分かる問いにする
- answer は短すぎず、復習に使える密度にする

# 出力形式
{{
  "topic": "短い題名",
  "question": "復習用の問い",
  "answer": "復習用の答え"
}}
"""
    result = llm.invoke(prompt).content
    data = parse_llm_json(result, "優カードJSONの解析に失敗しました")

    return {
        "topic": data.get("topic", "壁打ちの重要ポイント"),
        "question": data.get("question", "この壁打ちの最重要ポイントは？"),
        "answer": data.get("answer", summary),
    }