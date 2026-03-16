from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.utils import infer_lesson_from_path


LECTURES_PDF_DIR = Path("data/lectures_pdf")
NOTES_DIR = Path("data/notes")
TMP_UPLOAD_DIR = "tmp_uploads"


def load_docx(path: Path) -> List[Document]:
    doc = DocxDocument(str(path))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "path": str(path),
                "page": 1,
            },
        )
    ]


def save_uploaded_files(uploaded_files) -> List[Path]:
    Path(TMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for uf in uploaded_files:
        p = Path(TMP_UPLOAD_DIR) / uf.name
        with open(p, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(p)

    return saved_paths


def collect_local_files() -> List[Path]:
    paths: List[Path] = []

    if LECTURES_PDF_DIR.exists():
        paths.extend(sorted(LECTURES_PDF_DIR.glob("*.pdf")))

    if NOTES_DIR.exists():
        paths.extend(sorted(NOTES_DIR.glob("*.txt")))
        paths.extend(sorted(NOTES_DIR.glob("*.md")))
        paths.extend(sorted(NOTES_DIR.glob("*.docx")))
        paths.extend(sorted(NOTES_DIR.glob("*.csv")))

    return paths


def load_one_file(path: Path) -> List[Document]:
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
            d.metadata["source"] = f"lectures/{path.name}" if is_lecture_pdf else f"uploads/{path.name}"
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

    if ext == ".md":
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
        raise ValueError(f"MDの読み込みに失敗しました: {path.name}")

    if ext == ".docx":
        docs = load_docx(path)
        for d in docs:
            d.metadata["source"] = f"notes/{path.name}"
            d.metadata.update(base_meta)
        return docs

    if ext == ".csv":
        loader = CSVLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = f"notes/{path.name}"
            d.metadata.update(base_meta)
        return docs

    raise ValueError(f"未対応形式: {ext}")


def split_docs(docs: List[Document]) -> List[Document]:
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