import os
import re
from typing import Any, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

GPT_KEY = ""
BASE_URL = ""


class BaseEmbedding:
    """Base interface for embedding backends."""

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError

    def similarity(self, text_a: str, text_b: str) -> float:
        vec_a, vec_b = self.encode([text_a, text_b])

        def dot(x, y):
            return sum(i * j for i, j in zip(x, y))

        def norm(x):
            return dot(x, x) ** 0.5

        denominator = norm(vec_a) * norm(vec_b)
        if denominator == 0:
            return 0.0
        return dot(vec_a, vec_b) / denominator


class ClosedSourceEmbedding(BaseEmbedding):
    """Embedding wrapper around hosted OpenAI-style APIs."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=GPT_KEY, base_url=BASE_URL)
        self.model = model

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


class LocalEmbedding(BaseEmbedding):
    """Embedding wrapper for local transformer checkpoints."""

    def __init__(self, model_path: str, device: Optional[str] = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=kwargs.get("max_length", 512),
        ).to(self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()


def EmbeddingFactory(embed_type: str, **kwargs) -> BaseEmbedding:
    if embed_type == "closed":
        return ClosedSourceEmbedding(model=kwargs.get("model", "text-embedding-3-small"))
    if embed_type == "local":
        return LocalEmbedding(
            model_path=kwargs.get("model_path"),
            device=kwargs.get("device", "cuda"),
        )
    raise ValueError(f"Unsupported embed_type: {embed_type}")


class Retriever:
    def __init__(
        self,
        source: List[str],
        embed_model: Optional[BaseEmbedding] = None,
        k_bm25: int = 20,
        k_rerank: int = 5,
        embed_type: Optional[str] = None,
        **embed_kwargs,
    ):
        """
        :param source: List of passages to retrieve from.
        :param embed_model: Optional embedding model exposing .encode(List[str]).
        :param k_bm25: Number of BM25 candidates.
        :param k_rerank: Number of results kept after reranking.
        :param embed_type: Convenience flag to construct an embedding backend in-place.
        """
        self.source = source
        if embed_model is not None:
            self.embed_model = embed_model
        elif embed_type:
            self.embed_model = EmbeddingFactory(embed_type, **embed_kwargs)
        else:
            self.embed_model = None
        self.k_bm25 = k_bm25
        self.k_rerank = k_rerank
        self.max_tokens = 50 * 1024

        self.tokenized_memory = [self._tokenize(x) for x in self.source]
        self.bm25 = BM25Okapi(self.tokenized_memory) if self.tokenized_memory else None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    def _top_k_indices(self, scores, k: int):
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    def _bm25_search(self, query: str) -> List[int]:
        if not self.bm25:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return self._top_k_indices(scores, self.k_bm25)

    def _embedding_search(self, query: str, candidate_indices: List[int]) -> List[int]:
        if not candidate_indices or not self.embed_model:
            return candidate_indices[: self.k_rerank]
        query_emb = self.embed_model.encode([query])[0]
        candidate_texts = [self.source[i] for i in candidate_indices]
        candidate_embs = self.embed_model.encode(candidate_texts)
        sims = np.dot(candidate_embs, query_emb) / (
            np.linalg.norm(candidate_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        rerank_idx = np.argsort(sims)[::-1][: self.k_rerank]
        return [candidate_indices[i] for i in rerank_idx]

    def retrieve(self, query: str) -> List[int]:
        bm25_indices = self._bm25_search(query)
        rerank_indices = (
            self._embedding_search(query, bm25_indices)
            if self.embed_model
            else bm25_indices[: self.k_rerank]
        )
        return self._truncate_by_tokens(rerank_indices)

    def similarity(self, text_a: str, text_b: str) -> float:
        if not self.embed_model:
            raise ValueError("Retriever does not have an embedding backend configured.")
        return self.embed_model.similarity(text_a, text_b)

    def _truncate_by_tokens(self, indices: List[int]) -> List[int]:
        if not indices or not self.tokenized_memory:
            return indices
        total_tokens = 0
        kept_indices: List[int] = []
        for idx in indices:
            tokens = self.tokenized_memory[idx] if idx < len(self.tokenized_memory) else []
            token_len = len(tokens)
            if total_tokens + token_len > self.max_tokens:
                break
            kept_indices.append(idx)
            total_tokens += token_len
        return kept_indices
