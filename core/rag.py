from typing import List
from langchain_core.documents import Document


def prepare_documents(docs: List[Document]) -> List[Document]:
    """
    ドキュメントの前処理
    """
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