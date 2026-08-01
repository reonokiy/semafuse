"""Retrieval interfaces and clean-room hybrid retrieval implementation."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, Sequence

from .types import Message


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass(frozen=True, slots=True)
class RankedMessage:
    message: Message
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    ranked: tuple[RankedMessage, ...]
    latency_s: float


class Ranker(Protocol):
    def rank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with deterministic tie-breaking.

    Duplicate ids within one ranking contribute only at their first rank.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")
    actual_weights = tuple(weights or (1.0,) * len(rankings))
    if len(actual_weights) != len(rankings):
        raise ValueError("RRF weights must match ranking count")
    if any(not math.isfinite(weight) or weight < 0 for weight in actual_weights):
        raise ValueError("RRF weights must be finite and non-negative")

    scores: defaultdict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, actual_weights, strict=True):
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += weight / (k + rank)
            best_rank[item_id] = min(rank, best_rank.get(item_id, rank))
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


class BM25Ranker:
    """Small dependency-free Okapi BM25 ranker for messages."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def rank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]:
        if limit < 0:
            raise ValueError("rank limit must be non-negative")
        if not messages or limit == 0:
            return []
        documents = [self._tokens(message.retrieval_text) for message in messages]
        query_tokens = self._tokens(query)
        average_length = sum(map(len, documents)) / len(documents) or 1.0
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        scores: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            frequencies = Counter(document)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                n = document_frequency[token]
                idf = math.log(1.0 + (len(documents) - n + 0.5) / (n + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(document) / average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedMessage(messages[index], score, rank)
            for rank, (index, score) in enumerate(scores[:limit], start=1)
        ]


class RankBM25Ranker:
    """Optional adapter for the ``rank_bm25`` package named by paper v2."""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def rank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'retrieval' extra for rank_bm25") from error
        if limit < 0:
            raise ValueError("rank limit must be non-negative")
        if not messages or limit == 0:
            return []
        model = BM25Okapi([self._tokens(message.retrieval_text) for message in messages])
        scores = model.get_scores(self._tokens(query))
        ordered = sorted(
            enumerate(float(score) for score in scores),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RankedMessage(messages[index], score, rank)
            for rank, (index, score) in enumerate(ordered[:limit], start=1)
        ]


class SentenceTransformerRanker:
    """Optional local dense ranker; importing it does not load a model."""

    def __init__(self, model_name: str, *, query_prompt: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'retrieval' extra for dense retrieval") from error
        self._model = SentenceTransformer(model_name)
        self._query_prompt = query_prompt

    def rank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]:
        import numpy as np  # sentence-transformers already requires numpy

        query_text = f"{self._query_prompt}{query}" if self._query_prompt else query
        embeddings = self._model.encode(
            [query_text, *(message.retrieval_text for message in messages)],
            normalize_embeddings=True,
        )
        scores = np.asarray(embeddings[1:]) @ np.asarray(embeddings[0])
        ordered = sorted(
            enumerate(scores.tolist()), key=lambda item: (-item[1], item[0])
        )
        return [
            RankedMessage(messages[index], float(score), rank)
            for rank, (index, score) in enumerate(ordered[:limit], start=1)
        ]


class CrossEncoderReranker:
    """Optional local cross-encoder reranker (e.g. BGE-Reranker-v2-M3)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'retrieval' extra for reranking") from error
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, messages: Sequence[Message], limit: int
    ) -> list[RankedMessage]:
        values = self._model.predict(
            [(query, message.retrieval_text) for message in messages]
        )
        ordered = sorted(
            enumerate(float(value) for value in values),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RankedMessage(messages[index], score, rank)
            for rank, (index, score) in enumerate(ordered[:limit], start=1)
        ]


class HybridRetriever:
    """Dense + sparse retrieval, RRF, then a cross-encoder reranker."""

    def __init__(
        self,
        *,
        dense: Ranker,
        sparse: Ranker,
        reranker: Reranker | None,
        channel_limit: int = 100,
        fused_limit: int = 100,
        rrf_k: int = 60,
    ) -> None:
        if channel_limit <= 0 or fused_limit <= 0:
            raise ValueError("retrieval limits must be positive")
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.channel_limit = channel_limit
        self.fused_limit = fused_limit
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, messages: Sequence[Message], *, limit: int = 50
    ) -> RetrievalResult:
        if limit <= 0:
            return RetrievalResult((), 0.0)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lazymem-retrieve") as pool:
            dense_future = pool.submit(self.dense.rank, query, messages, self.channel_limit)
            sparse_future = pool.submit(self.sparse.rank, query, messages, self.channel_limit)
            dense = dense_future.result()
            sparse = sparse_future.result()
        fused = reciprocal_rank_fusion(
            [[item.message.id for item in dense], [item.message.id for item in sparse]],
            k=self.rrf_k,
        )[: self.fused_limit]
        by_id = {message.id: message for message in messages}
        candidates = [by_id[item_id] for item_id, _ in fused]
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, candidates, limit)
        else:
            ranked = [
                RankedMessage(by_id[item_id], score, rank)
                for rank, (item_id, score) in enumerate(fused[:limit], start=1)
            ]
        return RetrievalResult(tuple(ranked), time.perf_counter() - started)
