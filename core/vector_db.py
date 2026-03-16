import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


PERSIST_DIR = "vectorstore_main"
REGISTRY_DIR = "vectorstore_registry"


def file_fingerprint(path: Path) -> str:

    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def generate_chunk_id(doc: Document) -> str:

    src = doc.metadata.get("source", "")
    path = doc.metadata.get("path", "")
    lesson = str(doc.metadata.get("lesson", ""))
    page = str(doc.metadata.get("page", ""))

    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    key = f"{lesson}|{src}|{page}|{path}"
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]

    return f"{key_hash}_{content_hash}"


def get_main_dir() -> str:

    return str(Path(PERSIST_DIR) / "default")


def get_registry_dir() -> str:

    return str(Path(REGISTRY_DIR) / "default")


def get_registry_file() -> Path:

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

    f.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def is_file_indexed(fp: str) -> bool:

    return fp in load_registry()


def mark_file_indexed(fp: str, path: Path):

    registry = load_registry()

    registry[fp] = str(path)

    save_registry(registry)


def get_db(embeddings) -> Chroma:

    return Chroma(
        persist_directory=get_main_dir(),
        embedding_function=embeddings
    )


def has_main_index(db) -> bool:

    try:
        return db._collection.count() > 0
    except Exception:
        return False


def build_or_update_vectorstore(
    all_docs: List[Document],
    embeddings
) -> Chroma:

    db = Chroma(
        persist_directory=get_main_dir(),
        embedding_function=embeddings
    )

    if not all_docs:
        return db

    ids = [generate_chunk_id(doc) for doc in all_docs]

    if not ids:
        return db

    existing_ids = set()

    try:
        got = db._collection.get(ids=ids, include=[])
        existing_ids = set(got.get("ids", []) or [])

    except Exception:
        existing_ids = set()

    new_docs = []
    new_ids = []

    for doc, _id in zip(all_docs, ids):

        if _id in existing_ids:
            continue

        new_docs.append(doc)
        new_ids.append(_id)

    if not new_docs:
        return db

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(new_docs)

    for doc in split_docs:

        clean_meta = {}

        for k, v in doc.metadata.items():

            if v is None:
                continue

            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        doc.metadata = clean_meta

    ids = [str(uuid.uuid4()) for _ in split_docs]

    BATCH_SIZE = 100

    for i in range(0, len(split_docs), BATCH_SIZE):

        batch_docs = split_docs[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        db.add_documents(batch_docs, ids=batch_ids)

    try:
        db.persist()
    except Exception:
        pass

    return db