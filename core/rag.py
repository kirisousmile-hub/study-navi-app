from typing import List, Tuple, Optional

from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from core.utils import format_source_page, unique_by_source_page


def prepare_documents(docs: List[Document]) -> List[Document]:
    cleaned = []

    for d in docs:
        meta = {}

        for k, v in d.metadata.items():
            if v is None:
                continue

            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)

        d.metadata = meta
        cleaned.append(d)

    return cleaned


def rerank_docs(question: str, docs: List[Document], embeddings, top_k: int = 4) -> List[Document]:
    if not docs:
        return []

    q_emb = embeddings.embed_query(question)
    texts = [d.page_content[:800] for d in docs]
    doc_embs = embeddings.embed_documents(texts)

    scores = cosine_similarity([q_emb], doc_embs)[0]
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[:top_k]]


def retrieve_hits(
    query: str,
    db,
    embeddings,
    k: int = 5,
    only_textbook: bool = True,
    lesson_filter: Optional[str] = None,
) -> List[Document]:
    search_kwargs = {"k": int(k) * 3}

    where = None

    if only_textbook and lesson_filter:
        where = {
            "$and": [
                {"category": "textbook"},
                {"lesson": lesson_filter}
            ]
        }
    elif only_textbook:
        where = {"category": "textbook"}
    elif lesson_filter:
        where = {"lesson": lesson_filter}

    if where:
        search_kwargs["filter"] = where

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    raw_hits = retriever.invoke(query)

    reranked = rerank_docs(query, raw_hits, embeddings, top_k=int(k))
    return unique_by_source_page(reranked, int(k))


def answer_with_rag(
    question: str,
    db,
    embeddings,
    llm,
    k: int = 4,
    only_textbook: bool = False,
    lesson_filter: Optional[str] = None
) -> Tuple[str, List[Document]]:
    hits = retrieve_hits(
        query=question,
        db=db,
        embeddings=embeddings,
        k=k,
        only_textbook=only_textbook,
        lesson_filter=lesson_filter,
    )

    context = "\n\n".join(
        [f"[{i}] {format_source_page(d.metadata)}\n{d.page_content}" for i, d in enumerate(hits, start=1)]
    )

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