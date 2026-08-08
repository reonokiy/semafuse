from __future__ import annotations

from collections.abc import Callable

from .model import HarnessError


SentenceSplitter = Callable[[str], list[str]]
TokenCounter = Callable[[str], int]


def official_sentence_chunks(
    text: str,
    chunk_size: int,
    *,
    sentence_splitter: SentenceSplitter | None = None,
    token_counter: TokenCounter | None = None,
) -> list[str]:
    """Mirror the official NLTK + tiktoken sentence-packing algorithm.

    Dependency injection keeps fixtures deterministic; a real paper-track run uses
    NLTK ``sent_tokenize`` and the GPT-4o-mini tiktoken encoding.
    """

    if chunk_size <= 0:
        raise HarnessError("chunk_size must be positive")
    if not isinstance(text, str) or not text.strip():
        raise HarnessError("context must be a non-empty string")

    if sentence_splitter is None or token_counter is None:
        try:
            import nltk
            import tiktoken
        except ImportError as error:  # pragma: no cover - exercised by real runs
            raise HarnessError(
                "paper-track chunking requires the nltk and tiktoken packages"
            ) from error

        nltk.download("punkt", quiet=True)
        # NLTK 3.10 stores the tokenizer tables separately. Downloading the
        # resource does not change the upstream sent_tokenize algorithm.
        nltk.download("punkt_tab", quiet=True)
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        sentence_splitter = sentence_splitter or nltk.sent_tokenize
        token_counter = token_counter or (
            lambda sentence: len(
                encoding.encode(sentence, allowed_special={"<|endoftext|>"})
            )
        )

    sentences = sentence_splitter(text)
    if not sentences:
        raise HarnessError("sentence splitter returned no sentences")

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = token_counter(sentence)
        if sentence_tokens < 0:
            raise HarnessError("token counter returned a negative value")
        if current_tokens + sentence_tokens > chunk_size:
            # This intentionally matches the upstream function, including its
            # empty first chunk for an oversized first sentence.
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks
