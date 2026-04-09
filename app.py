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
    format_source_page,
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
    parse_llm_json,
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
    map_to_course_topic,
    build_wall_excellent_card,
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
    rerank_docs,
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

from core.ui_blocks import (
    render_scroll_to_top_button,
    render_sources_block,
    render_last_answer_block,
    render_self_test_block,
    render_quick_drill_block,
    render_deep_learning_block,
    render_mission_curriculum_block,
    render_analytics_tab,
    render_review_tab,
    render_material_tab,
    render_weak_training_block,
    render_coach_sidebar_block,
    render_coach_summary_actions_block,
    render_coach_chat_block,
    render_rag_question_block,
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
#汎用 helper
# session_state の初期化
def init_session_state():
    # 壁打ちチャット履歴
    if WALL_KEY not in st.session_state:
        st.session_state[WALL_KEY] = []

    # 壁打ち要約
    if WALL_SUMMARY_KEY not in st.session_state:
        st.session_state[WALL_SUMMARY_KEY] = None

    # 壁打ち参照ヒット
    if WALL_HITS_KEY not in st.session_state:
        st.session_state[WALL_HITS_KEY] = []

    # クイックドリル
    if "drill_item" not in st.session_state:
        st.session_state["drill_item"] = None

    if "drill_result" not in st.session_state:
        st.session_state["drill_result"] = None

    # AIトレーニング
    if "duo_item" not in st.session_state:
        st.session_state["duo_item"] = None

    if "duo_result" not in st.session_state:
        st.session_state["duo_result"] = None

    # 深掘り学習やループ採点
    if "loop_question" not in st.session_state:
        st.session_state["loop_question"] = None

    if "loop_grade_result" not in st.session_state:
        st.session_state["loop_grade_result"] = None


    # セルフテスト
    if "self_test" not in st.session_state:
        st.session_state["self_test"] = None

    if "self_test_results" not in st.session_state:
        st.session_state["self_test_results"] = {}

    # ミッション / カリキュラム
    if "mission" not in st.session_state:
        st.session_state["mission"] = None

    if "ai_curriculum" not in st.session_state:
        st.session_state["ai_curriculum"] = None

    # 分析
    if "weak_explain" not in st.session_state:
        st.session_state["weak_explain"] = None

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
def handle_review_score(cards: list[dict], card_id: str, new_score):
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


#見た目用
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

/* コードブロック */
pre, code {
    background-color: #2b313c !important;
    color: #f2f2f2 !important;
}

pre {
    border: 1px solid #4b5563 !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    overflow-x: auto !important;
}

/* Streamlitのコード表示 */
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCode"] pre {
    background-color: #2b313c !important;
    color: #f2f2f2 !important;
}

/* code内文字 */
div[data-testid="stCodeBlock"] code,
div[data-testid="stCode"] code,
pre code {
    color: #f2f2f2 !important;
    background-color: transparent !important;
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
    st.subheader("💾 保存先")
    st.caption("セッション履歴: session_state（一時）")
    st.caption("長期メモ: data/user/wall_memory.json")
    st.caption("復習カード: data/user/review_cards.json")
    st.caption("学習プロフィール: data/user/learning_profile.json")
    st.caption("弱点記録: data/user/weak_points.json")

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

    render_rag_question_block(
        k,
        db,
        EMBEDDINGS,
        LLM,
        LLM_CREATIVE,
        WALL_KEY,
        TURN_LIMIT,
        has_main_index,
        answer_with_rag,
        generate_self_test,
        render_last_answer_block,
        render_sources_block,
        format_source_page,
        render_self_test_block,
        LEARNING_DEPS,
        grade_answer,
        count_turns,
        show_error,
    )
    render_weak_training_block(
            generate_weak_question,
            grade_answer,
            LLM,
            LLM_CREATIVE,
            db,
            EMBEDDINGS,
            load_weak_points,
            update_learning_profile,
            add_learning_log,
            register_weak_point,
            retrieve_hits,
        )
    render_mission_curriculum_block(
            LEARNING_DEPS["load_learning_profile"],
            LEARNING_DEPS["llm"],
            generate_today_mission,
            recommend_next_topic,
            generate_ai_curriculum,
            find_root_weakness,
        )
    render_deep_learning_block(
            build_next_question,
            grade_answer,
            LLM,
            update_learning_profile,
            add_learning_log,
            register_weak_point,
        )
    render_quick_drill_block(
            get_drill_topic,
            generate_drill_question,
            grade_answer,
            LLM_CREATIVE,
            update_learning_profile,
            add_learning_log,
            register_weak_point,
        )

############################################
# TAB: COACH
############################################
with tab_coach:
    st.header("🧠 壁打ち")

    colL, colR = st.columns([2, 1])

    with colR:
        coach_sidebar = render_coach_sidebar_block(
            build_memory_block,
            load_wall_memory,
            delete_wall_fact,
            delete_wall_summary,
            add_wall_fact,
            save_wall_memory,
        )

    wall_mode = coach_sidebar["wall_mode"]
    wall_only_textbook = coach_sidebar["wall_only_textbook"]
    wall_k = coach_sidebar["wall_k"]
    use_long_memory = coach_sidebar["use_long_memory"]

    with colL:
        render_coach_chat_block(
            coach_sidebar["wall_mode"],
            coach_sidebar["wall_only_textbook"],
            coach_sidebar["wall_k"],
            coach_sidebar["use_long_memory"],
            db,
            EMBEDDINGS,
            LLM,
            AUTO_SUMMARY_TURN,
            WALL_KEY,
            WALL_HITS_KEY,
            WALL_SUMMARY_KEY,
            map_to_course_topic,
            retrieve_hits,
            rerank_docs,
            coach_reply,
            summarize_wall_history,
            add_wall_summary,
            render_sources_block,
            format_source_page,
        )

    render_coach_summary_actions_block(
        WALL_KEY,
        WALL_SUMMARY_KEY,
        WALL_HITS_KEY,
        summarize_wall_history,
        add_wall_summary,
        load_review_cards,
        save_review_cards,
        make_review_card,
        build_wall_excellent_card,
        LLM,
        show_error,
    )

############################################
# TAB: REVIEW
############################################
with tab_review:
    render_review_tab(
        load_review_cards,
        save_review_cards,
        is_due,
        handle_review_score,
        load_weak_points,
        datetime.now,
    )
############################################
# TAB: ANALYTICS
############################################
with tab_analytics:
    render_analytics_tab(
        LEARNING_DEPS["load_learning_profile"],
        LEARNING_DEPS["llm"],
        show_learning_dashboard,
        show_weak_heatmap,
        get_weak_topics_sorted,
        show_knowledge_map,
        explain_weakness,
        get_learning_level,
    )

############################################
# TAB: MATERIALS
############################################
with tab_material:
    render_material_tab(
        uploaded,
        collect_local_files,
        save_uploaded_files,
        EMBEDDINGS,
        get_db,
        build_or_update_vectorstore,
        split_target_and_skipped_paths,
        load_documents_from_paths,
        split_docs,
        build_index_result_message,
        show_skipped_files,
        mark_indexed_paths,
        show_error,
    )