import hashlib
import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


PERSIST_DIR = "vectorstore_main"
REGISTRY_DIR = "vectorstore_registry"
TMP_UPLOAD_DIR = "tmp_uploads"
DEFAULT_NAMESPACE = "default"


def file_fingerprint(path: Path) -> str:
    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def normalize_metadata(metadata: dict) -> dict:
    clean_meta = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_meta[key] = value
        else:
            clean_meta[key] = str(value)

    return clean_meta


def make_clean_document(doc: Document) -> Document:
    return Document(
        page_content=doc.page_content,
        metadata=normalize_metadata(doc.metadata),
    )


def generate_chunk_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    path = doc.metadata.get("path", "")
    lesson = str(doc.metadata.get("lesson", ""))
    page = str(doc.metadata.get("page", ""))

    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    key = f"{lesson}|{source}|{page}|{path}"
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]

    return f"{key_hash}_{content_hash}"


def get_main_dir() -> str:
    return str(Path(PERSIST_DIR) / DEFAULT_NAMESPACE)


def get_registry_dir() -> str:
    return str(Path(REGISTRY_DIR) / DEFAULT_NAMESPACE)


def ensure_dirs() -> None:
    """必要な保存先ディレクトリを先に作る"""
    Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(get_main_dir()).mkdir(parents=True, exist_ok=True)
    Path(get_registry_dir()).mkdir(parents=True, exist_ok=True)


def get_registry_file() -> Path:
    path = Path(get_registry_dir()) / "file_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_registry() -> dict:
    registry_file = get_registry_file()

    if not registry_file.exists():
        return {}

    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}


def save_registry(data: dict) -> None:
    registry_file = get_registry_file()
    registry_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_file_indexed(fp: str) -> bool:
    return fp in load_registry()


def mark_file_indexed(fp: str, path: Path) -> None:
    registry = load_registry()
    registry[fp] = str(path)
    save_registry(registry)


def create_chroma_db(embeddings) -> Chroma:
    return Chroma(
        persist_directory=get_main_dir(),
        embedding_function=embeddings,
    )


def get_db(embeddings) -> Chroma:
    return create_chroma_db(embeddings)


def has_main_index(db) -> bool:
    try:
        return db._collection.count() > 0
    except Exception:
        return False


def build_or_update_vectorstore(
    all_docs: List[Document],
    embeddings,
) -> Chroma:
    db = create_chroma_db(embeddings)

    if not all_docs:
        return db

    clean_docs = [make_clean_document(doc) for doc in all_docs]
    ids = [generate_chunk_id(doc) for doc in clean_docs]

    existing_ids = set()

    try:
        got = db._collection.get(ids=ids, include=[])
        existing_ids = set(got.get("ids", []) or [])
    except Exception:
        existing_ids = set()

    new_docs = []
    new_ids = []

    for doc, doc_id in zip(clean_docs, ids):
        if doc_id in existing_ids:
            continue
        new_docs.append(doc)
        new_ids.append(doc_id)

    if not new_docs:
        return db

    batch_size = 100

    for i in range(0, len(new_docs), batch_size):
        batch_docs = new_docs[i:i + batch_size]
        batch_ids = new_ids[i:i + batch_size]
        db.add_documents(batch_docs, ids=batch_ids)

    try:
        db.persist()
    except Exception:
        # persist非対応環境でも add_documents 結果は利用できるため握りつぶす
        pass

    return db