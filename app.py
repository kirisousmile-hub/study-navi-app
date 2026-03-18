############################################
# IMPORTS
############################################
import os
from dotenv import load_dotenv

# LangChain telemetry OFF
os.environ["ANONYMIZED_TELEMETRY"] = "1"
os.environ["CHROMA_TELEMETRY"] = "0"

# Standard library
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

# Third-party libraries
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Study Navi",
    layout="wide"
)

# LLM / LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# Local modules
from core.knowledge_map import (
    find_root_weakness,
    show_knowledge_map,
)

from core.utils import (
    count_turns,
    format_sources,
)

from core.review import (
    load_review_cards,
    save_review_cards,
    make_review_card,
    is_due,
    update_review_card_score,
)

from core.learning import (
    load_learning_profile,
    generate_self_test,
    grade_answer,
    generate_weak_question,
    generate_today_mission,
    generate_ai_curriculum,
    explain_weakness,
    get_learning_level,
    get_weak_topics_sorted,
    update_learning_profile,
    add_learning_log,
    register_weak_point,
    load_weak_points,
    generate_next_question,
    generate_drill_question,
    recommend_next_topic,
)

from core.coach import (
    load_wall_memory,
    save_wall_memory,
    add_wall_fact,
    add_wall_summary,
    build_memory_block,
    summarize_wall_history,
    coach_reply,
    delete_wall_fact,
    delete_wall_summary,
)

from core.vector_db import (
    ensure_dirs,
    file_fingerprint,
    get_main_dir,
    get_registry_dir,
    load_registry,
    is_file_indexed,
    mark_file_indexed,
    get_db,
    has_main_index,
    build_or_update_vectorstore,
)

from core.rag import (
    retrieve_hits,
    answer_with_rag,
)

from core.materials import (
    save_uploaded_files,
    collect_local_files,
    load_one_file,
    split_docs,
)

from core.analytics import (
    show_weak_heatmap,
    show_learning_dashboard,
)
############################################
# ENVIRONMENT / MODEL SETTINGS
############################################

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------------------
# LLM instances
# ------------------------------------------

LLM = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

LLM_CREATIVE = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.4,
    api_key=OPENAI_API_KEY
)
LEARNING_DEPS = {
    "llm": LLM,
    "load_learning_profile": load_learning_profile,
    "update_learning_profile": update_learning_profile,
    "add_learning_log": add_learning_log,
    "register_weak_point": register_weak_point,
}

# ------------------------------------------
# Embeddings / DB
# ------------------------------------------

EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
db = get_db(EMBEDDINGS)

############################################
# CONSTANTS / SESSION KEYS
############################################
# Streamlitは上から順に実行されるため、定数とキーを先にまとめる
APP_TITLE = "Study Navi"

# ディレクトリ
TMP_UPLOAD_DIR = "tmp_uploads"         # # なぜ：アップロードファイルを一時保存してLoaderに渡すため

# session_state keys
WALL_KEY = "wall_history"          # # なぜ：壁打ちチャット履歴
WALL_SUMMARY_KEY = "wall_summary"  # # なぜ：壁打ちまとめ
WALL_HITS_KEY = "wall_hits"        # # なぜ：壁打ちの検索ヒット（根拠候補）

# 壁打ち履歴の最大ターン
TURN_LIMIT = 30                    # # なぜ：LLMに渡す履歴が長すぎるとコスト/遅延/脱線が増えるため
# 自動要約を走らせるターン数
AUTO_SUMMARY_TURN = 12


############################################
# HELPER FUNCTIONS
############################################

# session_state の初期化
def init_session_state():
    if WALL_KEY not in st.session_state:
        st.session_state[WALL_KEY] = []

    if WALL_SUMMARY_KEY not in st.session_state:
        st.session_state[WALL_SUMMARY_KEY] = None

    if WALL_HITS_KEY not in st.session_state:
        st.session_state[WALL_HITS_KEY] = []

    if "drill_question" not in st.session_state:
        st.session_state["drill_question"] = None


# 学習タブ用の問題生成
def build_next_question():
    return generate_next_question(
        db=db,
        embeddings=EMBEDDINGS,
        llm_creative=LLM_CREATIVE,
        load_learning_profile=load_learning_profile,
        get_weak_topics_sorted=get_weak_topics_sorted,
        retrieve_hits=retrieve_hits,
    )


# クイック確認ドリルの対象トピック決定
def get_drill_topic() -> str:
    profile = load_learning_profile()

    if profile:
        weak_topics = get_weak_topics_sorted(profile)
        return weak_topics[0][1]

    return "基礎プログラミング"


# 復習カードの採点更新
def handle_review_score(cards: list[dict], card_id: str, new_score: int):
    updated = update_review_card_score(cards, card_id, new_score)
    if updated:
        st.success(f"更新しました：score={new_score} / next={updated['next_review_date']}")
        st.rerun()
    else:
        st.error("カード更新に失敗しました")


# インデックスと一時保存領域の完全初期化
def reset_indexes_and_tmp():
    shutil.rmtree(get_main_dir(), ignore_errors=True)
    shutil.rmtree(get_registry_dir(), ignore_errors=True)
    shutil.rmtree(TMP_UPLOAD_DIR, ignore_errors=True)
    Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    for key_name in ["retriever", "db"]:
        st.session_state.pop(key_name, None)


# インデックス対象とスキップ対象の仕分け
def split_target_and_skipped_paths(candidate_paths: list[Path]) -> tuple[list[Path], list[str]]:
    target_paths: list[Path] = []
    skipped: list[str] = []

    for p in candidate_paths:
        fp = file_fingerprint(p)
        if is_file_indexed(fp):
            skipped.append(p.name)
            continue
        target_paths.append(p)

    return target_paths, skipped


# パス一覧から教材ファイルを読み込む
def load_documents_from_paths(paths: list[Path]) -> list[Document]:
    raw_docs: list[Document] = []
    for p in paths:
        raw_docs.extend(load_one_file(p))
    return raw_docs


# インデックス登録済みファイルを記録する
def mark_indexed_paths(paths: list[Path]) -> None:
    for p in paths:
        mark_file_indexed(file_fingerprint(p), p)


def show_skipped_files(skipped: list[str]) -> None:
    if skipped:
        st.info("スキップ: " + ", ".join(skipped[:20]) + ("..." if len(skipped) > 20 else ""))


def build_index_result_message(
    saved_paths: list[Path],
    local_paths: list[Path],
    target_paths: list[Path],
    skipped: list[str],
    chunk_count: int,
) -> str:
    return (
        f"追加完了：アップロード {len(saved_paths)}ファイル / "
        f"ローカル {len(local_paths)}ファイル / "
        f"対象 {len(target_paths)}ファイル / "
        f"スキップ {len(skipped)}ファイル / "
        f"{chunk_count}チャンク"
    )

def show_error(prefix: str, e: Exception) -> None:
    st.error(f"{prefix}: {e}")
    st.code(traceback.format_exc())


def get_db_count() -> int:
    try:
        return get_db(EMBEDDINGS)._collection.count()
    except Exception:
        return 0


def get_registry_count() -> int | None:
    try:
        reg = load_registry()
        return len(reg)
    except Exception:
        return None

def render_self_test_block(test: dict) -> None:
    st.subheader("自己テスト")

    for i, q in enumerate(test["questions"]):
        st.markdown(f"### Q{i+1}. {q}")
        user_key = f"user_answer_{i}"
        user_answer = st.text_area(
            "あなたの回答",
            key=user_key
        )

        if st.button(f"採点する Q{i+1}", key=f"grade_{i}"):
            result = grade_answer(
                q,   # topic
                q,   # question
                user_answer,
                test["answers"][i],
                LEARNING_DEPS["llm"],
                LEARNING_DEPS["update_learning_profile"],
                LEARNING_DEPS["add_learning_log"],
                LEARNING_DEPS["register_weak_point"]
            )
            st.markdown("### AI採点")
            st.write(result)

def render_quick_drill_block() -> None:
    st.subheader("🧠 クイック確認ドリル")
    st.caption("弱点トピックを短い問題で素早く確認する、軽めの反復モードです。")

    if st.button("クイック問題を作成"):
        topic = get_drill_topic()
        st.session_state["drill_question"] = generate_drill_question(
            topic,
            LLM_CREATIVE
        )

    if st.session_state["drill_question"]:
        st.write("### クイック問題")
        st.write(st.session_state["drill_question"])

        answer = st.text_area("あなたの回答")

        if st.button("回答を採点", key="drill_grade_btn"):
            grading_prompt = f"""
次の問題の回答を採点してください。

問題
{st.session_state["drill_question"]}

回答
{answer}

出力
・正解かどうか
・改善ポイント
"""

            result = LLM.invoke(grading_prompt).content

            st.write("### AI採点")
            st.write(result)

        if st.button("次の問題を作成", key="drill_next_btn"):
            topic = get_drill_topic()
            st.session_state["drill_question"] = generate_drill_question(
                topic,
                LLM_CREATIVE
            )
            st.rerun()

def render_last_answer_block(last_q, last_answer, last_hits) -> None:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("回答")
        st.write(last_answer)

        if st.button("📝 直前の回答をカード化"):
            try:
                cards = load_review_cards()
                card = make_review_card(
                    topic=last_q,
                    answer=last_answer,
                    hits=last_hits
                )
                cards.append(card)
                save_review_cards(cards)
                st.success(f"カード化しました！ id={card['id']}")
                st.rerun()
            except Exception as e:
                show_error("カード化に失敗", e)

        if st.button("🧠 自己テストを作成"):
            try:
                test = generate_self_test(last_q, last_hits, LLM_CREATIVE)
                st.session_state["self_test"] = test
                st.rerun()
            except Exception as e:
                show_error("自己テスト作成に失敗しました", e)

        test = st.session_state.get("self_test")
        if test:
            render_self_test_block(test)

    with right:
        st.subheader("参照（sources）")
        st.markdown(format_sources(last_hits))

def inject_dark_css() -> None:
    st.markdown("""
    <style>
    /* 全体背景 */
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main,
    section.main > div,
    div.block-container {
        background-color: #1f232a !important;
        color: #e8e8e8 !important;
    }

    /* Streamlit上部 */
    header[data-testid="stHeader"] {
        background-color: #1f232a !important;
    }

    div[data-testid="stToolbar"] {
        background-color: #1f232a !important;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #252a33 !important;
    }

    /* 見出し・本文 */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #e8e8e8 !important;
    }

    /* 入力欄 */
    input, textarea {
        background-color: #2b313c !important;
        color: #f2f2f2 !important;
    }

    /* selectbox / number_input */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #2b313c !important;
        color: #f2f2f2 !important;
    }

    /* ファイルアップローダー */
    div[data-testid="stFileUploader"] {
        background-color: #252a33 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #252a33 !important;
        color: #e8e8e8 !important;
    }

    /* ボタン */
    .stButton > button {
        background-color: #313846 !important;
        color: #f2f2f2 !important;
        border: 1px solid #4b5563 !important;
    }

    .stButton > button:hover {
        background-color: #3b4354 !important;
        border-color: #6b7280 !important;
    }

    /* タブ固定 */
    div[data-baseweb="tab-list"] {
        position: sticky !important;
        top: 0 !important;
        z-index: 99999 !important;
        background-color: #1f232a !important;
        border-bottom: 1px solid #3a404c !important;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }

    button[data-baseweb="tab"] {
        color: #e8e8e8 !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #8ab4f8 !important;
    }

    /* expander */
    div[data-testid="stExpander"] {
        background-color: #252a33 !important;
        border-radius: 8px !important;
    }

    /* 区切り線 */
    hr {
        border-color: #3a404c !important;
    }
    </style>
    """, unsafe_allow_html=True)

def inject_light_css() -> None:
    st.markdown("""
    <style>
    /* 全体背景 */
    .stApp,
    div[data-testid="stAppViewContainer"],
    section.main,
    section.main > div,
    div.block-container {
        background-color: #f7f7f8 !important;
        color: #222222 !important;
    }

    /* Streamlit上部 */
    header[data-testid="stHeader"] {
        background-color: #f7f7f8 !important;
    }

    div[data-testid="stToolbar"] {
        background-color: #f7f7f8 !important;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #eef1f4 !important;
    }

    /* 見出し・本文 */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #222222 !important;
    }

    /* 入力欄 */
    input, textarea {
        background-color: #ffffff !important;
        color: #222222 !important;
    }

    /* selectbox / number_input */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #222222 !important;
    }

    /* ファイルアップローダー */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        color: #222222 !important;
    }

    /* ボタン */
    .stButton > button {
        background-color: #ffffff !important;
        color: #222222 !important;
        border: 1px solid #c7ccd1 !important;
    }

    .stButton > button:hover {
        background-color: #f1f3f5 !important;
        border-color: #aeb6bf !important;
    }

    /* タブ固定 */
    div[data-baseweb="tab-list"] {
        position: sticky !important;
        top: 0 !important;
        z-index: 99999 !important;
        background-color: #f7f7f8 !important;
        border-bottom: 1px solid #d7dbe0 !important;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }

                
    button[data-baseweb="tab"] {
        color: #333333 !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111111 !important;
        border-bottom: 2px solid #4f8cff !important;
    }

    /* expander */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* 区切り線 */
    hr {
        border-color: #d7dbe0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_scroll_to_top_button() -> None:
    components.html(
        """
        <div style="position: fixed; right: 24px; bottom: 24px; z-index: 99999;">
            <button
                onclick="window.parent.scrollTo({top: 0, behavior: 'smooth'});"
                style="
                    background: #4f8cff;
                    color: white;
                    border: none;
                    border-radius: 999px;
                    padding: 10px 14px;
                    font-size: 14px;
                    font-weight: bold;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                "
            >
                ↑ 上へ
            </button>
        </div>
        """,
        height=0,
    )

############################################
# APP INITIALIZATION
############################################

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY が設定されていません。.env を確認してください。")

ensure_dirs()
init_session_state()

############################################
# SIDEBAR
############################################
with st.sidebar:
    st.subheader("🔧 メンテ")
    theme_mode = st.radio(
        "表示テーマ",
        ["ライト", "ダーク"],
        index=1,
        horizontal=True
    )
    db_count = get_db_count()
    st.caption(f"📦 現在のインデックス: {db_count} チャンク")

    reg_count = get_registry_count()
    st.caption(f"🧾 登録済みファイル: {reg_count if reg_count is not None else '-'} 件")

    if st.button("🧨 完全初期化（DB + registry + tmp）"):
        reset_indexes_and_tmp()
        st.success("DB/registry/tmp を完全初期化しました。")
        st.rerun()

    st.divider()
    st.subheader("データ投入")

    uploaded = st.file_uploader(
        "PDF/DOCX/TXT/CSVをアップロード",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    k = st.number_input("検索件数 k", min_value=2, max_value=10, value=4, step=1)

if theme_mode == "ライト":
    inject_light_css()
else:
    inject_dark_css()

st.title(APP_TITLE)
st.caption("アップロードしたあなたのメモ/提出物を元に、根拠つきで回答し、次の一手を3つ提案します。")

render_scroll_to_top_button()
############################################
# TABS
############################################
tabs = st.tabs([
    "📚 学習",
    "🧠 壁打ち",
    "📝 復習",
    "📊 分析",
    "📂 教材管理"
])

tab_learn, tab_coach, tab_review, tab_analytics, tab_material = tabs

############################################
# TAB: LEARNING
############################################
with tab_learn:
    st.header("📚 学習")

    # ------------------------------------------
    # Question with RAG
    # ------------------------------------------
    st.divider()
    st.subheader("質問する")

    only_textbook = st.checkbox("教材（lectures_pdf）だけ検索する", value=True)
    lesson_filter = st.text_input(
        "Lessonフィルター（例: 13）",
        value=""
    )
    question = st.text_input("質問（例：Chroma永続化のしくみを自分の言葉で説明したい）", value="")

    ask_disabled = (not question.strip()) or (not has_main_index(db))

    if st.button("質問する", disabled=ask_disabled):
        try:
            with st.spinner("検索＆回答中..."):
                ans, hits = answer_with_rag(
                    question=question,
                    db=db,
                    embeddings=EMBEDDINGS,
                    llm=LLM,
                    k=int(k),
                    only_textbook=only_textbook,
                    lesson_filter=lesson_filter or None,
                )

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = ans
            st.session_state["last_hits"] = hits
        except Exception as e:
            show_error("回答に失敗", e)

    last_q = st.session_state.get("last_question")
    last_answer = st.session_state.get("last_answer")
    last_hits = st.session_state.get("last_hits")

    if last_answer:
        render_last_answer_block(last_q, last_answer, last_hits)

    # 壁打ち履歴の見える化（RAGの下に置くのが分かりやすい）
    turns, msgs = count_turns(st.session_state[WALL_KEY])
    st.caption(f"壁打ち履歴: {turns} / {TURN_LIMIT} ターン（メッセージ {msgs}件）")

    st.divider()
    st.subheader("AIトレーニング")

    if st.button("弱点トレーニングを開始"):
        q = generate_weak_question(load_weak_points, LLM_CREATIVE)

        if q:
            st.session_state["duo_question"] = q
        else:
            st.warning("まだ弱点がありません")

    duo_q = st.session_state.get("duo_question")

    if duo_q:
        st.subheader("AIトレーニング問題")

        st.write(duo_q)

        duo_answer = st.text_area("あなたの回答", key="duo_answer")

        if st.button("回答を採点", key="duo_grade_btn"):
            result = grade_answer(
                duo_q,   # topic
                duo_q,   # question
                duo_answer,
                "Pythonの正しい説明",
                LLM,
                update_learning_profile,
                add_learning_log,
                register_weak_point
            )
            st.write(result)

    st.divider()

    # 今日のミッション
    st.subheader("🎯 今日のミッション")

    if st.button("今日のミッションを作成"):
        mission = generate_today_mission(
            LEARNING_DEPS["load_learning_profile"],
            LEARNING_DEPS["llm"]
        )

        st.session_state["mission"] = mission

    if "mission" in st.session_state:
        st.write(st.session_state["mission"])

    st.divider()

    # AIカリキュラム
    st.markdown("### 🤖 AIカリキュラム")
    # ------------------------------------------
    # Next topic recommendation
    # ------------------------------------------

    st.markdown("### 🧭 次に学ぶトピック")

    if st.button("次に学ぶ内容を提案"):
        topic = recommend_next_topic(
            load_learning_profile=load_learning_profile,
            llm=LLM,
            find_root_weakness=find_root_weakness,
        )

        st.success(topic)

    if st.button("今日の学習メニューを作成"):
        curriculum = generate_ai_curriculum(load_learning_profile, LLM)
        st.session_state["ai_curriculum"] = curriculum

    curriculum = st.session_state.get("ai_curriculum")

    if curriculum:
        st.write(curriculum)

    st.subheader("🚀 深掘り学習モード")
    st.caption("教材をもとに問題・模範回答・解説を生成して、じっくり理解を深めます。")

    if st.button("深掘り学習を開始"):
        st.session_state["loop_question"] = build_next_question()

    loop_data = st.session_state.get("loop_question")

    if loop_data:
        st.write("### 深掘り問題")
        st.write(loop_data["question"])

        loop_answer = st.text_area("あなたの回答", key="loop_answer")

        if st.button("回答を採点", key="loop_grade_btn"):
            result = grade_answer(
                loop_data["topic"],
                loop_data["question"],
                loop_answer,
                loop_data["reference"],
                LLM,
                update_learning_profile,
                add_learning_log,
                register_weak_point
            )

            st.write("### AI採点")
            st.write(result)

            st.markdown("### 模範回答")
            st.write(loop_data["reference"])

            st.markdown("### 解説")
            st.write(loop_data["explanation"])

        if st.button("次の問題を作成", key="loop_next_btn"):
            st.session_state["loop_question"] = build_next_question()
            st.rerun()

    # ------------------------------------------
    # Quick drill mode
    # ------------------------------------------
    render_quick_drill_block()

############################################
# TAB: COACH
############################################
with tab_coach:
    st.header("🧠 壁打ち")

    colL, colR = st.columns([2, 1])

    with colR:
        wall_mode = st.selectbox(
            "学習フェーズ",
            ["A: 用語理解", "B: 設計理解", "C: コード理解"],
            index=0
        )
        wall_only_textbook = st.checkbox(
            "教材だけ参照",
            value=True,
            key="wall_only_textbook"
        )
        wall_k = st.number_input(
            "壁打ち検索k",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            key="wall_k"
        )
        use_long_memory = st.checkbox(
            "長期記憶を使う",
            value=False,
            key="use_long_memory"
        )

        st.divider()
        st.subheader("🧠 覚えさせるメモ（永続）")
        st.caption("保存済みメモ（直近）")
        st.caption("壁打ち履歴は一時保存、長期記憶と復習カードはファイル保存です。")
        st.text_area(
            "saved_memory",
            build_memory_block(limit=10, include_facts=True, include_summaries=True),
            height=180,
            disabled=True
        )

        st.divider()
        st.subheader("🗂 保存済みメモを個別削除")

        mem_data = load_wall_memory()

        facts = mem_data.get("facts", [])
        summaries = mem_data.get("summaries", [])

        if facts:
            st.caption("固定メモ")
            fact_options = {
                f"{f['id']} | {f['text'][:40]}": f["id"]
                for f in facts
            }
            selected_fact_label = st.selectbox(
                "削除する固定メモ",
                options=list(fact_options.keys()),
                key="delete_fact_select"
            )

            if st.button("固定メモを削除", key="delete_fact_btn"):
                ok = delete_wall_fact(fact_options[selected_fact_label])
                if ok:
                    st.success("固定メモを削除しました")
                    st.rerun()
                else:
                    st.warning("削除対象が見つかりませんでした")

        if summaries:
            st.caption("学習要約")
            summary_options = {
                f"{s['id']} | {s['text'][:40].replace(chr(10), ' ')}": s["id"]
                for s in summaries
            }
            selected_summary_label = st.selectbox(
                "削除する学習要約",
                options=list(summary_options.keys()),
                key="delete_summary_select"
            )

            if st.button("学習要約を削除", key="delete_summary_btn"):
                ok = delete_wall_summary(summary_options[selected_summary_label])
                if ok:
                    st.success("学習要約を削除しました")
                    st.rerun()
                else:
                    st.warning("削除対象が見つかりませんでした")

        mem_text = st.text_input(
            "覚えてほしいこと（例：合言葉はリンゴ）",
            key="mem_text"
        )

        colm1, colm2 = st.columns(2)
        with colm1:
            if st.button("➕ メモを保存", key="save_mem_btn"):
                if mem_text.strip():
                    f = add_wall_fact(mem_text)
                    st.success(f"保存しました id={f['id']}")
                    st.rerun()
                else:
                    st.warning("空です")

        with colm2:
            if st.button("🗑 メモ全消し（危険）", key="clear_mem_btn"):
                save_wall_memory({"facts": [], "summaries": []})
                st.warning("全メモを削除しました")
                st.rerun()

    with colL:
        st.caption("あなたが喋る → 根拠を差し込む → 質問で掘る、の順で進めます。")

        chat_area = st.container(height=600, border=True)

        with chat_area:
            for m in st.session_state[WALL_KEY]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

        user_msg = st.chat_input(
            "いま何を復習したい？（例：for文 / 関数 / 例外 / import / 合言葉確認）"
        )

        if user_msg:
            st.session_state[WALL_KEY].append(
                {"role": "user", "content": user_msg}
            )

            hits = retrieve_hits(
                user_msg,
                db,
                EMBEDDINGS,
                k=int(wall_k),
                only_textbook=wall_only_textbook
            )
            st.session_state[WALL_HITS_KEY] = hits

            assistant_msg = coach_reply(
                st.session_state[WALL_KEY],
                hits,
                wall_mode,
                LLM,
                use_long_memory=use_long_memory
            )

            st.session_state[WALL_KEY].append(
                {"role": "assistant", "content": assistant_msg}
            )

            turns = len(st.session_state[WALL_KEY]) // 2

            if turns >= AUTO_SUMMARY_TURN:
                summary_text = summarize_wall_history(
                    st.session_state[WALL_KEY],
                    hits,
                    LLM
                )

                add_wall_summary(summary_text)

                st.session_state[WALL_SUMMARY_KEY] = summary_text

                st.session_state[WALL_KEY] = [
                    {
                        "role": "assistant",
                        "content": "ここまでの壁打ち内容を自動要約して長期記憶に保存しました。続きをどうぞ。"
                    }
                ]

            st.rerun()

        hits = st.session_state.get(WALL_HITS_KEY, [])
        if hits:
            with st.expander("参照（sources）"):
                st.markdown(format_sources(hits))

    st.divider()
    colA, colB = st.columns(2)

    with colA:
        if st.button("🧾 壁打ちまとめを作成"):
            hist = st.session_state[WALL_KEY]
            hits = st.session_state.get(WALL_HITS_KEY, [])

            summary_text = summarize_wall_history(
                hist,
                hits,
                LLM
            )

            st.session_state[WALL_SUMMARY_KEY] = summary_text
            add_wall_summary(summary_text)
            st.rerun()

    with colB:
        if st.button("🗑 壁打ちをリセット"):
            st.session_state[WALL_KEY] = []
            st.session_state.pop(WALL_SUMMARY_KEY, None)
            st.session_state.pop(WALL_HITS_KEY, None)
            st.rerun()

    summary = st.session_state.get(WALL_SUMMARY_KEY)
    if summary:
        st.subheader("まとめ")
        st.write(summary)
        if st.button("📝 壁打ちまとめをカード化"):
            cards = load_review_cards()
            card = make_review_card(
                topic="壁打ちまとめ",
                answer=summary,
                hits=st.session_state.get(WALL_HITS_KEY, []),
            )
            cards.append(card)
            save_review_cards(cards)
            st.success(f"カード化しました！ id={card['id']}")
            st.rerun()

    st.caption(f"壁打ち履歴: {len(st.session_state[WALL_KEY])} メッセージ")

############################################
# TAB: REVIEW
############################################
with tab_review:
    st.header("📝 復習")
    # ------------------------------------------
    # Review cards
    # ------------------------------------------

    cards = load_review_cards()
    if not cards:
        st.info("まだカードがありません。まずは学習タブで質問 → 回答 → 『📝 直前の回答をカード化』を押してください。")
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        only_due = st.checkbox("今日の復習だけ表示する（next_review_date <= 今日）", value=True)

        filtered = [c for c in cards if (is_due(c, today) if only_due else True)]
        st.write(f"カード数: {len(cards)} / 表示: {len(filtered)}（今日={today}）")

        sort_key = st.selectbox("並び順", ["next_review_dateが古い順", "作成日時が新しい順"], index=0)
        if sort_key == "next_review_dateが古い順":
            filtered.sort(key=lambda c: c.get("next_review_date") or "0000-00-00")
        else:
            filtered.sort(key=lambda c: c.get("created_at") or "", reverse=True)

        options = [
            f"{c.get('id','????')} | {c.get('next_review_date','-')} | {c.get('topic','(no topic)')[:50]}"
            for c in filtered
        ]
        sel = st.selectbox("カードを選択", options, index=0)
        sel_id = sel.split("|")[0].strip()
        card = next((c for c in cards if c.get("id") == sel_id), None)

        if not card:
            st.error("カードが見つかりませんでした（データ不整合の可能性）。")
        else:
            st.subheader("問題（自分で答えてから開く）")
            st.write(card.get("question") or card.get("topic"))

            show_answer = st.checkbox("答えを表示する", value=False)
            if show_answer:
                st.subheader("答え（保存された回答）")
                st.write(card.get("answer", ""))

                st.subheader("参照（sources）")
                srcs = card.get("sources", [])
                if srcs:
                    st.markdown("\n".join([f"- {s}" for s in srcs]))
                else:
                    st.write("(なし)")

            st.divider()
            st.subheader("採点して次回復習日を更新")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("0 😵 無理（明日）"):
                    handle_review_score(cards, card["id"], 0)

            with col2:
                if st.button("1 🤔 微妙（2日後）"):
                    handle_review_score(cards, card["id"], 1)

            with col3:
                if st.button("2 ✅ できた（1週間後）"):
                    handle_review_score(cards, card["id"], 2)

            st.divider()
            st.subheader("管理")
            if st.button("このカードを削除（危険）"):
                cards2 = [c for c in cards if c.get("id") != card.get("id")]
                save_review_cards(cards2)
                st.warning("削除しました。")
                st.rerun()

            st.divider()
            st.subheader("弱点トピック")

            weak = load_weak_points()

            if weak:
                weak_sorted = sorted(weak, key=lambda x: x["count"], reverse=True)

                for w in weak_sorted[:5]:
                    st.write(f"{w['topic']} （{w['count']}回ミス）")

            else:
                st.caption("まだ弱点はありません")

############################################
# TAB: ANALYTICS
############################################
with tab_analytics:
    st.header("📊 分析")

    show_learning_dashboard(load_learning_profile)
    show_weak_heatmap(load_learning_profile, get_weak_topics_sorted)

    show_knowledge_map()

    st.subheader("🧠 弱点分析")

    if st.button("弱点の原因を分析"):
        explanation = explain_weakness(
            LEARNING_DEPS["load_learning_profile"],
            LEARNING_DEPS["llm"]
        )
        st.session_state["weak_explain"] = explanation

    explanation = st.session_state.get("weak_explain")

    if explanation:
        st.write(explanation)

    st.divider()

    st.subheader("🏆 学習レベル")

    level = get_learning_level(load_learning_profile)

    st.write(level)

    st.subheader("🧑‍🏫 AI学習アドバイス")

    if st.button("学習状況を分析"):
        profile = load_learning_profile()

        if not profile:
            st.info("まだ学習データがありません")
        else:
            summary = []

            for topic, data in profile.items():
                total = data["total"]
                correct = data["correct"]
                score = correct / total if total else 0
                summary.append(f"{topic}:{round(score*100)}%")

            prompt = f"""
Python学習コーチとして
次の学習データを分析してください。

{summary}

・現在の理解度
・弱点
・次の学習アドバイス

を短く説明してください。
"""
            advice = LLM.invoke(prompt).content
            st.write(advice)

############################################
# TAB: MATERIALS
############################################
with tab_material:
    st.header("📂 教材管理")
    # ------------------------------------------
    # Build or update index
    # ------------------------------------------
    local_paths = collect_local_files()

    if st.button("インデックスを作成（追加）"):
        try:
            saved_paths: List[Path] = []
            if uploaded:
                saved_paths = save_uploaded_files(uploaded)

            local_paths = collect_local_files()

            candidate_paths: List[Path] = [*saved_paths, *local_paths]

            target_paths, skipped = split_target_and_skipped_paths(candidate_paths)

            raw_docs = load_documents_from_paths(target_paths)

            chunks = split_docs(raw_docs)

            if len(chunks) == 0:
                st.success(
                    build_index_result_message(
                        saved_paths=saved_paths,
                        local_paths=local_paths,
                        target_paths=target_paths,
                        skipped=skipped,
                        chunk_count=0,
                    )
                )
                show_skipped_files(skipped)
            else:
                before = get_db(EMBEDDINGS)._collection.count()

                build_or_update_vectorstore(chunks, EMBEDDINGS)

                after = get_db(EMBEDDINGS)._collection.count()
                st.info(f"Chroma count: {before} -> {after} (+{after - before})")

                mark_indexed_paths(target_paths)

                st.success(
                    build_index_result_message(
                        saved_paths=saved_paths,
                        local_paths=local_paths,
                        target_paths=target_paths,
                        skipped=skipped,
                        chunk_count=len(chunks),
                    )
                )

                show_skipped_files(skipped)

        except Exception as e:
            show_error("インデックス作成に失敗", e)