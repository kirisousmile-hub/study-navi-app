from typing import List, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from core.utils import format_source_page, unique_by_source_page


def normalize_doc_metadata(docs: List[Document]) -> List[Document]:
    normalized_docs: List[Document] = []

    for doc in docs:
        meta = {}

        for key, value in doc.metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
            else:
                meta[key] = str(value)

        normalized_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=meta,
            )
        )

    return normalized_docs


def build_search_filter(
    only_textbook: bool,
    lesson_filter: Optional[str],
) -> Optional[dict]:
    if only_textbook and lesson_filter:
        return {
            "$and": [
                {"category": "textbook"},
                {"lesson": lesson_filter},
            ]
        }

    if only_textbook:
        return {"category": "textbook"}

    if lesson_filter:
        return {"lesson": lesson_filter}

    return None


def rerank_docs(
    question: str,
    docs: List[Document],
    embeddings,
    top_k: int = 4,
) -> List[Document]:
    if not docs:
        return []

    query_embedding = embeddings.embed_query(question)
    texts = [doc.page_content[:800] for doc in docs]
    doc_embeddings = embeddings.embed_documents(texts)

    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]


def retrieve_hits(
    query: str,
    db,
    embeddings,
    k: int = 5,
    only_textbook: bool = True,
    lesson_filter: Optional[str] = None,
) -> List[Document]:
    search_kwargs = {"k": int(k) * 3}

    search_filter = build_search_filter(
        only_textbook=only_textbook,
        lesson_filter=lesson_filter,
    )
    if search_filter:
        search_kwargs["filter"] = search_filter

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    raw_hits = retriever.invoke(query)

    reranked = rerank_docs(query, raw_hits, embeddings, top_k=int(k))
    unique_hits = unique_by_source_page(reranked, int(k))

    return normalize_doc_metadata(unique_hits)


def answer_with_rag(
    question: str,
    db,
    embeddings,
    llm,
    k: int = 4,
    only_textbook: bool = False,
    lesson_filter: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    hits = retrieve_hits(
        query=question,
        db=db,
        embeddings=embeddings,
        k=k,
        only_textbook=only_textbook,
        lesson_filter=lesson_filter,
    )

    if not hits:
        return (
            "【結論】\n"
            "- 該当する教材が見つかりませんでした。\n\n"
            "【根拠（参照した資料の要点）】\n"
            "- 参照候補が0件でした。\n\n"
            "【次の一手（最短3つ）】\n"
            "1. 質問文を短くして再検索する\n"
            "2. Lessonフィルターを外して再検索する\n"
            "3. 教材管理タブでインデックス作成状況を確認する"
        ), []

    context = "\n\n".join(
        [
            f"[{i}] {format_source_page(doc.metadata)}\n{doc.page_content}"
            for i, doc in enumerate(hits, start=1)
        ]
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