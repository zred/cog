"""Simple in-memory structures for conversation and vector storage."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleConversationMemory:
    """Circular conversation buffer."""

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.messages: List[str] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(f"Human: {message}")
        self.messages = self.messages[-self.k :]

    def add_ai_message(self, message: str) -> None:
        self.messages.append(f"AI: {message}")
        self.messages = self.messages[-self.k :]


class SimpleDoc:
    """Wrapper for stored text."""

    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class SimpleVectorStore:
    """Very small in-memory vector store using cosine similarity."""

    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.embedding_model = embedding_model
        self.docs: List[Tuple[np.ndarray, SimpleDoc]] = []

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        for text in texts:
            vec = self.embedding_model.encode(text)
            self.docs.append((vec, SimpleDoc(text)))

    def similarity_search(self, query: str, k: int = 4) -> List[SimpleDoc]:
        if not self.docs:
            return []
        qvec = self.embedding_model.encode(query)
        scores = [np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec)) for vec, _ in self.docs]
        topk = np.argsort(scores)[::-1][:k]
        return [self.docs[i][1] for i in topk]
