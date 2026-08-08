"""LMEB/MTEB 2.3.0 retrieval metrics, including LMEB's fixed R_cap."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .model import HarnessError, RankingEntry

CUTOFFS = (1, 5, 10, 25, 50)


def _orders(entries: Sequence[RankingEntry]) -> tuple[list[RankingEntry], list[RankingEntry]]:
    if len({entry.doc_id for entry in entries}) != len(entries):
        raise HarnessError("a ranking contains duplicate document ids")
    if any(not math.isfinite(entry.score) for entry in entries):
        raise HarnessError("ranking scores must be finite")
    # trec_eval breaks equal-score ties by reverse lexicographic docno. MTEB's
    # Python MRR and LMEB's local R_cap instead retain insertion order on ties.
    trec = sorted(entries, key=lambda entry: entry.doc_id, reverse=True)
    trec.sort(key=lambda entry: entry.score, reverse=True)
    stable = sorted(entries, key=lambda entry: entry.score, reverse=True)
    return trec, stable


def score_query(
    qrels: Mapping[str, int],
    entries: Sequence[RankingEntry],
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[str, float | None]:
    trec, stable = _orders(entries)
    relevant = {doc_id for doc_id, relevance in qrels.items() if relevance >= 1}
    ideal_gains = sorted((value for value in qrels.values() if value > 0), reverse=True)
    scores: dict[str, float | None] = {}
    for cutoff in cutoffs:
        top = trec[:cutoff]
        hits = 0
        ap_sum = 0.0
        dcg = 0.0
        for rank, entry in enumerate(top, 1):
            relevance = qrels.get(entry.doc_id, 0)
            if relevance >= 1:
                hits += 1
                ap_sum += hits / rank
            if relevance > 0:
                dcg += relevance / math.log2(rank + 1)
        idcg = sum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(ideal_gains[:cutoff], 1)
        )
        rr = 0.0
        for rank, entry in enumerate(stable[:cutoff], 1):
            if entry.doc_id in relevant:
                rr = 1.0 / rank
                break
        stable_hits = sum(entry.doc_id in relevant for entry in stable[:cutoff])
        scores[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0
        scores[f"map_at_{cutoff}"] = ap_sum / len(relevant) if relevant else 0.0
        scores[f"recall_at_{cutoff}"] = hits / len(relevant) if relevant else 0.0
        scores[f"precision_at_{cutoff}"] = hits / cutoff
        scores[f"mrr_at_{cutoff}"] = rr
        scores[f"R_cap_at_{cutoff}"] = (
            stable_hits / min(len(relevant), cutoff) if relevant else None
        )
    return scores


def evaluate_rankings(
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[RankingEntry]],
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[str, object]:
    if tuple(cutoffs) != CUTOFFS:
        raise HarnessError(f"LMEB cutoffs must be {list(CUTOFFS)}")
    missing = set(qrels) - set(rankings)
    extra = set(rankings) - set(qrels)
    if missing or extra:
        raise HarnessError(
            f"ranking/query mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    per_query = {query_id: score_query(qrels[query_id], rankings[query_id]) for query_id in qrels}
    macro: dict[str, float | None] = {}
    for metric in next(iter(per_query.values())):
        values = [query[metric] for query in per_query.values() if query[metric] is not None]
        if not values:
            macro[metric] = None
        elif metric.startswith("mrr_"):
            macro[metric] = sum(values) / len(values)  # MTEB does not round MRR.
        else:
            macro[metric] = round(sum(values) / len(values), 5)
    return {"macro": macro, "per_query": per_query}
