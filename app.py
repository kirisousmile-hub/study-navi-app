############################################
# 0) IMPORTS / ENV
############################################
import os

# # なぜ：LangChain/Chromaの匿名テレメトリを無効化して、挙動差や不要な送信を避ける
os.environ["ANONYMIZED_TELEMETRY"] = "1"
os.environ["CHROMA_TELEMETRY"] = "0"

import shutil
import hashlib
import traceback
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# # なぜ：docx loaderの環境差で落ちやすいので python-docx で自前読み込みするため
from docx import Document as DocxDocument


############################################
# 1) CONSTANTS / KEYS
############################################
# # なぜ：Streamlitは上から順に実行。キー/定数は「どこでも参照できる」ように最上段で固定
APP_TITLE = "学習ナビ（最小版）"

# ディレクトリ
PERSIST_DIR = "vectorstore_main"       # # なぜ：Chromaの本体DB保存先（run_idで分割）
REGISTRY_DIR = "vectorstore_registry"  # # なぜ：重複登録を防ぐための「登録済み台帳」保存先（run_idで分割）
TMP_UPLOAD_DIR = "tmp_uploads"         # # なぜ：アップロードファイルを一時保存してLoaderに渡すため

# ローカル教材フォルダ（自動取り込み）
LECTURES_PDF_DIR = Path("data/lectures_pdf")  # # なぜ：教材PDF（textbook扱い）
NOTES_DIR = Path("data/notes")                # # なぜ：自作メモ（notes扱い）

# 永続ファイル
REVIEW_FILE = Path("review_cards.json")       # # なぜ：カード復習の永続保存
WALL_MEMORY_FILE = Path("wall_memory.json")   # # なぜ：壁打ちで「覚えさせる」永続メモ

# session_state keys
RUN_ID_KEY = "run_id"              # # なぜ：DB領域をrun単位で分けて「完全初期化」を簡単にする
WALL_KEY = "wall_history"          # # なぜ：壁打ちチャット履歴
WALL_SUMMARY_KEY = "wall_summary"  # # なぜ：壁打ちまとめ
WALL_HITS_KEY = "wall_hits"        # # なぜ：壁打ちの検索ヒット（根拠候補）

# 壁打ち履歴の最大ターン
TURN_LIMIT = 30                    # # なぜ：LLMに渡す履歴が長すぎるとコスト/遅延/脱線が増えるため


############################################
# 2) UTILITIES (small)
############################################
def ensure_dirs():
    """# なぜ：起動時に必要な保存先が無いと失敗するので、先に作る"""
    Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(get_main_dir()).mkdir(parents=True, exist_ok=True)
    Path(get_registry_dir()).mkdir(parents=True, exist_ok=True)

def clear_tmp_uploads():
    """# なぜ：アップロードの差し替え時に古いファイルが残ると混乱するため"""
    if Path(TMP_UPLOAD_DIR).exists():
        shutil.rmtree(TMP_UPLOAD_DIR, ignore_errors=True)
    Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

def file_fingerprint(path: Path) -> str:
    """# なぜ：ファイル名ではなく「中身」で重複判定する（同一内容なら同一hash）"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def infer_lesson_from_path(path: Path) -> str | None:
    """# なぜ：教材PDF名/パスからLesson番号を推定して、根拠表示に使う"""
    s = str(path)
    m = re.search(r"(?:Lesson|lesson)\s*[_\- ]?\s*(\d{1,2})", s)
    if m:
        return m.group(1)
    m = re.search(r"[\\/](\d{1,2})[_\- ]", s)
    if m:
        return m.group(1)
    return None

def format_source_page(meta: dict) -> str:
    """# なぜ：根拠表示を『Lesson / ファイル名 p.X』の形に統一する"""
    src = meta.get("source", "unknown")
    page = meta.get("page", None)
    lesson = meta.get("lesson")
    prefix = f"Lesson{lesson} / " if lesson else ""
    filename = src.split("/")[-1]
    if isinstance(page, int):
        return f"{prefix}{filename} p.{page + 1}"
    return f"{prefix}{filename}"

def unique_by_source_page(docs: List[Document], limit: int) -> List[Document]:
    """# なぜ：同じPDF同じページが何回も出るのを防いで、根拠の多様性を上げる"""
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= limit:
            break
    return out

def format_sources(docs: List[Document]) -> str:
    """# なぜ：sources表示の重複を除いて読みやすくする"""
    seen = set()
    lines = []
    for d in docs:
        label = format_source_page(d.metadata)
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"- {label}")
    return "\n".join(lines) if lines else "- (なし)"

def save_uploaded_files(uploaded_files) -> List[Path]:
    """# なぜ：StreamlitのUploadedFileはそのままLoaderに渡せないので一旦保存する"""
    ensure_dirs()
    clear_tmp_uploads()
    saved_paths: List[Path] = []
    for uf in uploaded_files:
        p = Path(TMP_UPLOAD_DIR) / uf.name
        with open(p, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(p)
    return saved_paths

def collect_local_files() -> List[Path]:
    """# なぜ：教材/メモをフォルダに置くだけで自動投入できるようにする"""
    paths: List[Path] = []
    if LECTURES_PDF_DIR.exists():
        paths.extend(sorted(LECTURES_PDF_DIR.glob("*.pdf")))
    if NOTES_DIR.exists():
        paths.extend(sorted(NOTES_DIR.glob("*.txt")))
        paths.extend(sorted(NOTES_DIR.glob("*.md")))
    return paths

def count_turns(history: list) -> tuple[int, int]:
    """# なぜ：壁打ちの『30ターン制限』をUIで見える化する"""
    msg_count = len(history)
    turn_count = msg_count // 2
    return turn_count, msg_count


############################################
# 3) LOAD / SPLIT / IDS
############################################
def load_docx(path: Path) -> List[Document]:
    """# なぜ：LangChainのdocx loaderは環境差で落ちることがあるため自前で読む"""
    doc = DocxDocument(str(path))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [Document(page_content=text, metadata={"source": path.name, "path": str(path), "page": 1})]

def load_one_file(path: Path) -> List[Document]:
    """# なぜ：各形式の読み込み＋metadata付与を1箇所に集約して事故を減らす"""
    ext = path.suffix.lower()

    lesson = infer_lesson_from_path(path)
    is_lecture_pdf = (LECTURES_PDF_DIR in path.parents) and (ext == ".pdf")

    base_meta = {
        "path": str(path),
        "lesson": lesson,
        "category": "textbook" if is_lecture_pdf else "notes",
    }

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = f"lectures/{path.name}"
            d.metadata.update(base_meta)
        return docs

    if ext == ".txt":
        for enc in ("utf-8", "utf-8-sig", "cp932"):
            try:
                loader = TextLoader(str(path), encoding=enc)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = f"notes/{path.name}"
                    d.metadata.update(base_meta)
                return docs
            except Exception:
                continue
        raise ValueError(f"TXTの読み込みに失敗しました: {path.name}")

    # ※必要ならここに .csv .docx を追加
    raise ValueError(f"未対応形式: {ext}")

def split_docs(docs: List[Document]) -> List[Document]:
    """# なぜ：PDFをそのまま入れると長すぎて検索精度が落ちるのでchunk化する"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = c.metadata.get("source", "unknown")
        if "page" not in c.metadata:
            c.metadata["page"] = None
    return chunks

def generate_chunk_id(doc: Document) -> str:
    """# なぜ：Chromaに同一IDが入ると例外や重複が起きるので、内容ベースで安定IDを作る"""
    src = doc.metadata.get("source", "")
    path = doc.metadata.get("path", "")
    lesson = str(doc.metadata.get("lesson", ""))
    page = str(doc.metadata.get("page", ""))

    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    key = f"{lesson}|{src}|{page}|{path}"
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{key_hash}_{content_hash}"


############################################
# 4) DB / REGISTRY
############################################
def get_run_id() -> str:
    """# なぜ：DB領域をrun単位で分け、完全初期化を簡単にする"""
    rid = st.session_state.get(RUN_ID_KEY)
    if rid:
        return rid
    rid = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    st.session_state[RUN_ID_KEY] = rid
    return rid

def get_main_dir() -> str:
    # run_idを使わず、常に同じDBを使う
    return str(Path(PERSIST_DIR) / "default")

def get_registry_dir() -> str:
    # registryも常に同じ台帳を使う
    return str(Path(REGISTRY_DIR) / "default")

def get_registry_file() -> Path:
    """# なぜ：重複登録を防ぐ台帳ファイルの保存先"""
    p = Path(get_registry_dir()) / "file_registry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_registry() -> dict:
    f = get_registry_file()
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}

def save_registry(data: dict):
    f = get_registry_file()
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def is_file_indexed(fp: str) -> bool:
    return fp in load_registry()

def mark_file_indexed(fp: str, path: Path):
    registry = load_registry()
    registry[fp] = str(path)
    save_registry(registry)

def get_db() -> Chroma:
    """# なぜ：検索/壁打ちのたびに同じDBを開くための入口"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=get_main_dir(), embedding_function=embeddings)

def has_main_index() -> bool:
    """# なぜ：DBが空のときに『質問する』を無効化するため"""
    try:
        return get_db()._collection.count() > 0
    except Exception:
        return False

def build_or_update_vectorstore(all_docs: List[Document]) -> Chroma:
    """既存DBに追加しつつ、同一IDは追加しない（重複防止）"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=get_main_dir(), embedding_function=embeddings)

    if not all_docs:
        return db

    ids = [generate_chunk_id(doc) for doc in all_docs]
    if not ids:
        return db

    # 既に入っているIDを除外
    existing_ids = set()
    try:
        got = db._collection.get(ids=ids, include=[])
        existing_ids = set(got.get("ids", []) or [])
    except Exception:
        existing_ids = set()

    new_docs: List[Document] = []
    new_ids: List[str] = []
    for doc, _id in zip(all_docs, ids):
        if _id in existing_ids:
            continue
        new_docs.append(doc)
        new_ids.append(_id)

    if not new_docs:
        return db

    db.add_documents(new_docs, ids=new_ids)

    try:
        db.persist()
    except Exception:
        pass

    return db


############################################
# 5) RAG (answer_with_rag)
############################################
def answer_with_rag(question: str, k: int = 4, only_textbook: bool = False) -> Tuple[str, List[Document]]:
    """# なぜ：『質問→根拠→次の一手』を最小構成で安定させる"""
    db = get_db()

    search_kwargs = {"k": int(k) * 3}
    if only_textbook:
        search_kwargs["filter"] = {"category": "textbook"}

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    raw_hits = retriever.get_relevant_documents(question)
    hits = unique_by_source_page(raw_hits, int(k))

    context = "\n\n".join(
        [f"[{i}] {format_source_page(d.metadata)}\n{d.page_content}" for i, d in enumerate(hits, start=1)]
    )

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY)

    prompt = f"""あなたは「学習ナビ」です。
以下の「参照コンテキスト」だけに基づいて回答してください。
推測で断定しない。分からなければ「不明」と言い、確認手順を提案する。

# ユーザーの質問
{question}

# 参照コンテキスト
{context}

# 出力フォーマット（必ず守る）
【結論】
- （1〜3行）

【根拠（参照した資料の要点）】
- （必ず番号[1]などを交えて）
- （可能なら必ず「PDF名 p.X」を含める）

【次の一手（最短3つ）】
1.
2.
3.
"""
    return llm.invoke(prompt).content, hits


############################################
# 6) REVIEW CARDS
############################################
def load_review_cards() -> list:
    """# なぜ：カードをJSONで永続化して復習機能を成立させる"""
    if REVIEW_FILE.exists():
        try:
            return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_review_cards(cards: list):
    REVIEW_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

def compute_next_review_date(score: int) -> str:
    """# なぜ：最小の間隔反復（0/1/2）で次回日付を決める"""
    days = {2: 7, 1: 2, 0: 1}.get(score, 2)
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

def make_review_card(topic: str, answer: str, hits: List[Document]) -> dict:
    """# なぜ：RAGや壁打ちまとめを『復習カード』として再利用できる形にする"""
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


############################################
# 7) COACH (wall)
############################################
def load_wall_memory() -> dict:
    """# なぜ：壁打ちで『覚えさせる情報』をセッションを超えて保持する"""
    if WALL_MEMORY_FILE.exists():
        try:
            return json.loads(WALL_MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"facts": []}
    return {"facts": []}

def save_wall_memory(mem: dict):
    WALL_MEMORY_FILE.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")

def add_wall_fact(text: str) -> dict:
    """# なぜ：ユーザー指定の『覚えてほしいこと』を永続化する"""
    mem = load_wall_memory()
    fact = {
        "id": str(uuid.uuid4())[:8],
        "text": text.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mem.setdefault("facts", [])
    mem["facts"].append(fact)
    save_wall_memory(mem)
    return fact

def build_memory_block(limit: int = 30) -> str:
    """# なぜ：LLMに渡す『保存メモ』をコンパクトにする"""
    facts = load_wall_memory().get("facts", [])[-limit:]
    if not facts:
        return "(保存メモなし)"
    return "\n".join([f"- {f['text']}" for f in facts])

def retrieve_hits(query: str, k: int = 5, only_textbook: bool = True) -> List[Document]:
    """# なぜ：壁打ち中に根拠候補（教材）を引くため"""
    db = get_db()
    search_kwargs = {"k": int(k) * 3}
    if only_textbook:
        search_kwargs["filter"] = {"category": "textbook"}
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    raw_hits = retriever.get_relevant_documents(query)
    return unique_by_source_page(raw_hits, int(k))

def coach_reply(history: list, hits: List[Document], mode: str) -> str:
    """# なぜ：答えを与えすぎず、質問で理解を深める壁打ちコーチ"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY)
    sources = "\n".join([f"- {format_source_page(d.metadata)}" for d in hits[:3]]) or "- (なし)"

    if mode.startswith("A"):
        role_rule = """あなたはPython復習コーチ。短い一言 + 質問3つ + 必要なら根拠候補（最大3つ）。"""
    elif mode.startswith("C"):
        role_rule = """あなたはPythonコード読解コーチ。短い一言 + 質問3つ + 必要なら根拠候補（最大3つ）。"""
    else:
        role_rule = """あなたは設計コーチ。短い一言 + 質問3つ + 必要なら根拠候補（最大3つ）。"""

    messages = [SystemMessage(content=f"""{role_rule}
共通ルール：
- 長文解説は禁止（ユーザーが「解説して」と言った時だけ）
- 保存メモ（永続）は前提にして良い
- 根拠候補は sources にある範囲だけ
""")]

    messages.append(SystemMessage(content=f"【保存メモ（永続）】\n{build_memory_block(limit=30)}"))

    recent = history[-(TURN_LIMIT * 2):]
    for m in recent:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            messages.append(AIMessage(content=m.get("content", "")))

    messages.append(SystemMessage(content=f"根拠候補（sources）:\n{sources}"))
    return llm.invoke(messages).content


############################################
# 8) UI
############################################
# # なぜ：Streamlitの設定は最初に1回だけ（UI表示の前提）
st.set_page_config(page_title=APP_TITLE, layout="wide")

# # なぜ：環境変数を読み込んでAPIキーとモデル名を決める
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.title(APP_TITLE)
st.caption("アップロードしたあなたのメモ/提出物を元に、根拠つきで回答し、次の一手を3つ提案します。")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY が設定されていません。.env を確認してください。")

# 初期ディレクトリ作成
ensure_dirs()

# 壁打ち履歴（初期化）
if WALL_KEY not in st.session_state:
    st.session_state[WALL_KEY] = []

# =========================
# Sidebar：メンテ＆データ投入
# =========================
with st.sidebar:
    st.subheader("🔧 メンテ")

    # ---- 現在のDB状況を表示（押しすぎ防止）----
    try:
        db_count = get_db()._collection.count()
    except Exception:
        db_count = 0

    st.caption(f"📦 現在のインデックス: {db_count} チャンク")

    # registry登録済みファイル数
    try:
        reg = load_registry()
        st.caption(f"🧾 登録済みファイル: {len(reg)} 件")
    except Exception:
        st.caption("🧾 登録済みファイル: -")

    if st.session_state.get("just_reset", False):
        st.success("完全初期化しました。PDFを1つ入れて再検証してください。")
        st.session_state["just_reset"] = False

    if st.button("🧨 完全初期化（DB + registry + tmp）"):
        shutil.rmtree(get_main_dir(), ignore_errors=True)
        shutil.rmtree(get_registry_dir(), ignore_errors=True)
        shutil.rmtree(TMP_UPLOAD_DIR, ignore_errors=True)
        Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

        for k in ["retriever", "db"]:
            st.session_state.pop(k, None)

        st.success("DB/registry/tmp を完全初期化しました。")
        st.rerun()

    st.divider()
    st.header("① データ投入")

    uploaded = st.file_uploader(
        "PDF/DOCX/TXT/CSVをアップロード",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        k = st.number_input("検索件数 k", min_value=2, max_value=10, value=4, step=1)
    with col_b:
        st.write("")

# =========================
# ① インデックス作成
# =========================
local_paths = collect_local_files()

if st.button("インデックス作成（追加）"):
    try:
        saved_paths: List[Path] = []
        if uploaded:
            saved_paths = save_uploaded_files(uploaded)

        local_paths = collect_local_files()

        candidate_paths: List[Path] = []
        candidate_paths.extend(saved_paths)
        candidate_paths.extend(local_paths)

        target_paths: List[Path] = []
        skipped: List[str] = []

        for p in candidate_paths:
            fp = file_fingerprint(p)
            if is_file_indexed(fp):
                skipped.append(p.name)
                continue
            target_paths.append(p)

        raw_docs: List[Document] = []
        for p in target_paths:
            raw_docs.extend(load_one_file(p))

        chunks = split_docs(raw_docs)

        if len(chunks) == 0:
            st.success(
                f"追加完了：アップロード {len(saved_paths)}ファイル / "
                f"ローカル {len(local_paths)}ファイル / "
                f"対象 {len(target_paths)}ファイル / "
                f"スキップ {len(skipped)}ファイル / "
                f"0チャンク"
            )
            if skipped:
                st.info("スキップ: " + ", ".join(skipped[:20]) + ("..." if len(skipped) > 20 else ""))
        else:
            before = get_db()._collection.count()

            build_or_update_vectorstore(chunks)

            after = get_db()._collection.count()
            st.info(f"Chroma count: {before} -> {after} (+{after - before})")

            for p in target_paths:
                mark_file_indexed(file_fingerprint(p), p)

            st.success(
                f"追加完了：アップロード {len(saved_paths)}ファイル / "
                f"ローカル {len(local_paths)}ファイル / "
                f"対象 {len(target_paths)}ファイル / "
                f"スキップ {len(skipped)}ファイル / "
                f"{len(chunks)}チャンク"
            )

            if skipped:
                st.info("スキップ: " + ", ".join(skipped[:20]) + ("..." if len(skipped) > 20 else ""))

    except Exception as e:
        st.error(f"失敗: {e}")
        st.code(traceback.format_exc())

# =========================
# ② 質問する（RAG）
# =========================
st.divider()
st.header("② 質問する")

only_textbook = st.checkbox("教材（lectures_pdf）だけ検索する", value=True)
question = st.text_input("質問（例：Chroma永続化のしくみを自分の言葉で説明したい）", value="")

ask_disabled = (not question.strip()) or (not has_main_index())

if st.button("質問する", disabled=ask_disabled):
    try:
        with st.spinner("検索＆回答中..."):
            ans, hits = answer_with_rag(question, k=int(k), only_textbook=only_textbook)

        st.session_state["last_question"] = question
        st.session_state["last_answer"] = ans
        st.session_state["last_hits"] = hits
    except Exception as e:
        st.error(f"回答に失敗: {e}")
        st.code(traceback.format_exc())

last_q = st.session_state.get("last_question")
last_answer = st.session_state.get("last_answer")
last_hits = st.session_state.get("last_hits")

if last_answer:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("回答")
        st.write(last_answer)

        if st.button("📝（直前の回答）この質問をカード化"):
            try:
                cards = load_review_cards()
                card = make_review_card(topic=last_q, answer=last_answer, hits=last_hits)
                cards.append(card)
                save_review_cards(cards)
                st.success(f"カード化しました！ id={card['id']}")
                st.rerun()
            except Exception as e:
                st.error(f"カード化に失敗: {e}")
                st.code(traceback.format_exc())

    with right:
        st.subheader("参照（sources）")
        st.markdown(format_sources(last_hits))

# 壁打ち履歴の見える化（RAGの下に置くのが分かりやすい）
turns, msgs = count_turns(st.session_state[WALL_KEY])
st.caption(f"壁打ち履歴: {turns} / {TURN_LIMIT} ターン（メッセージ {msgs}件）")

# =========================
# ③ 復習する
# =========================
st.divider()
st.header("③ 復習する")

cards = load_review_cards()
if not cards:
    st.info("まだカードがありません。まずは②で質問 → 回答 → 『📝この質問をカード化』を押してください。")
else:
    today = datetime.now().strftime("%Y-%m-%d")
    only_due = st.checkbox("今日の復習だけ表示する（next_review_date <= 今日）", value=True)

    def is_due(card: dict) -> bool:
        d = card.get("next_review_date")
        return (d is None) or (d <= today)

    filtered = [c for c in cards if (is_due(c) if only_due else True)]
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

        def update_score(new_score: int):
            card["score"] = new_score
            card["last_review_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            card["next_review_date"] = compute_next_review_date(new_score)

            for i, c in enumerate(cards):
                if c.get("id") == card.get("id"):
                    cards[i] = card
                    break

            save_review_cards(cards)
            st.success(f"更新しました：score={new_score} / next={card['next_review_date']}")
            st.rerun()

        with col1:
            if st.button("0 😵 無理（明日）"):
                update_score(0)
        with col2:
            if st.button("1 🤔 微妙（2日後）"):
                update_score(1)
        with col3:
            if st.button("2 ✅ できた（1週間後）"):
                update_score(2)

        st.divider()
        st.subheader("管理")
        if st.button("このカードを削除（危険）"):
            cards2 = [c for c in cards if c.get("id") != card.get("id")]
            save_review_cards(cards2)
            st.warning("削除しました。")
            st.rerun()

# =========================
# ④ 壁打ちする（Coach）
# =========================
st.divider()
st.header("④ 壁打ちする（Coach）")

colL, colR = st.columns([2, 1])

with colR:
    wall_mode = st.selectbox("学習フェーズ", ["A: 用語理解", "C: コード理解", "B: 設計理解"], index=0)
    wall_only_textbook = st.checkbox("教材だけ参照", value=True, key="wall_only_textbook")
    wall_k = st.number_input("壁打ち検索k", min_value=2, max_value=8, value=4, step=1, key="wall_k")

    st.divider()
    st.subheader("🧠 覚えさせるメモ（永続）")
    st.caption("保存済みメモ（直近）")
    st.text_area("saved_memory", build_memory_block(limit=10), height=140, disabled=True)

    mem_text = st.text_input("覚えてほしいこと（例：合言葉はリンゴ）", key="mem_text")

    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("➕ メモを保存", key="save_mem_btn"):
            if mem_text.strip():
                f = add_wall_fact(mem_text)
                st.success(f"保存しました id={f['id']}")
                st.session_state["mem_text"] = ""
                st.rerun()
            else:
                st.warning("空です")

    with colm2:
        if st.button("🗑 メモ全消し（危険）", key="clear_mem_btn"):
            save_wall_memory({"facts": []})
            st.warning("全メモを削除しました")
            st.rerun()

with colL:
    st.caption("あなたが喋る → 根拠を差し込む → 質問で掘る、の順で進めます。")

    # ===== チャット履歴エリア（スクロール枠）=====
    chat_area = st.container(height=600, border=True)

    with chat_area:
        for m in st.session_state[WALL_KEY]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

    # ===== 入力欄は外に置く（重要）=====
    user_msg = st.chat_input(
        "いま何を復習したい？（例：for文 / 関数 / 例外 / import / 合言葉確認）"
    )

    if user_msg:
        st.session_state[WALL_KEY].append(
            {"role": "user", "content": user_msg}
        )

        hits = retrieve_hits(
            user_msg,
            k=int(wall_k),
            only_textbook=wall_only_textbook
        )
        st.session_state[WALL_HITS_KEY] = hits

        assistant_msg = coach_reply(
            st.session_state[WALL_KEY],
            hits,
            wall_mode
        )

        st.session_state[WALL_KEY].append(
            {"role": "assistant", "content": assistant_msg}
        )

        st.rerun()

    # ===== 根拠表示 =====
    hits = st.session_state.get(WALL_HITS_KEY, [])
    if hits:
        with st.expander("参照（sources）"):
            st.markdown(format_sources(hits))

# --- まとめ・カード化 ---
st.divider()
colA, colB = st.columns(2)

with colA:
    if st.button("🧾 この壁打ちをまとめる"):
        hist = st.session_state[WALL_KEY]
        hits = st.session_state.get(WALL_HITS_KEY, [])
        context = "\n\n".join([f"- {format_source_page(d.metadata)}\n{d.page_content[:800]}" for d in hits])

        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY)
        prompt = f"""以下の壁打ちログを、復習に使える形でまとめてください。
解説しすぎず、再現できる形にする。

# 壁打ちログ
{hist}

# 根拠
{context}

# 出力フォーマット
【結論（自分の言葉風に要約）】
-

【根拠（参照点）】
-

【次の一手（最短3つ）】
1.
2.
3.
"""
        st.session_state[WALL_SUMMARY_KEY] = llm.invoke(prompt).content
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
    if st.button("📝 この壁打ちまとめをカード化"):
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