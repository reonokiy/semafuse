from __future__ import annotations

from .model import HarnessError


# These strings reproduce the RAG-agent prompt branch at the pinned upstream
# implementation revision. The reference repository is MIT licensed.
MEMORY_TEMPLATES = {
    "ruler_qa": (
        "Dialogue between User and Assistant {timestamp}\\n<User> The following "
        "context is the documents I have read: \n{context}\n <Assistant> I have "
        "learned the documents and I will answer the question you ask."
    ),
    "eventqa": (
        "Dialogue between User and Assistant {timestamp}\\n<User> The following "
        "context is the book excerpt: \n{context}\n <Assistant> I have read the "
        "book excerpt and I will answer the question you ask."
    ),
    "factconsolidation": (
        "Dialogue between User and Assistant {timestamp} \\n<User> The following "
        "context is the facts I have learned: \n{context}\n <Assistant> I have "
        "learned the facts and I will answer the question you ask."
    ),
}


QUERY_TEMPLATES = {
    "ruler_qa": (
        "Answer the question based on the memorized documents. Only give me the "
        "answer and do not output any other words. \n\n Now Answer the Question: "
        "{question}"
    ),
    "eventqa": (
        "Based on the context you memorized, complete the task below:\n\n"
        "{question}\n\n The event that happens next is:"
    ),
    "factconsolidation": (
        "Pretend you are a knowledge management system. Each fact in the knowledge "
        "pool is provided with a serial number at the beginning, and the newer fact "
        "has larger serial number. \n You need to solve the conflicts of facts in "
        "the knowledge pool by finding the newest fact with larger serial number. "
        "You need to answer a question based on this rule. You should give a very "
        "concise answer without saying other words for the question **only** from "
        "the knowledge pool you have memorized rather than the real facts in real "
        "world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on "
        "the provided Knowledge Pool, what is the name of the current president of "
        "Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the "
        "provided Knowledge Pool, {question} \nAnswer:"
    ),
}


def memory_message(family: str, context: str, timestamp: str) -> str:
    try:
        return MEMORY_TEMPLATES[family].format(context=context, timestamp=timestamp)
    except KeyError as error:
        raise HarnessError(f"unsupported prompt family: {family}") from error


def query_message(family: str, question: str) -> str:
    try:
        return QUERY_TEMPLATES[family].format(question=question)
    except KeyError as error:
        raise HarnessError(f"unsupported prompt family: {family}") from error


def official_bm25_query(paper_query: str) -> str:
    """Reproduce the pinned reference implementation's BM25 query extraction."""

    import re

    for pattern in (
        r"Now Answer the Question:\s*(.*)",
        r"Here is the conversation:\s*(.*)",
    ):
        match = re.search(pattern, paper_query, re.DOTALL)
        if match:
            return "".join(match.groups())
    return paper_query
