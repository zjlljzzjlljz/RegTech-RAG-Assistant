"""Deterministic sparse tokenizer for cross-language (ZH/EN/HK-mixed) retrieval.

Design principles
-----------------
1. **No Python built-in hash()** — it is randomised across Python processes
   (PYTHONHASHSEED).  Token→ID uses blake2b, which is deterministic everywhere.
2. **Single shared implementation** — this module is the only place that builds
   sparse vectors; both ingest (BGEM3EmbeddingClient) and query (hybrid_search)
   code call tokenize_and_weight() from here.
3. **Language-aware tokenisation**
   * English / ASCII words          → word-boundary split, lowercased
   * Chinese characters (CJK U+4E00–U+9FFF) → character bigrams
   * Other CJK extensions (U+3000–U+303F, U+FF00–U+FFEF) → character bigrams
   * Mixed tokens like "CDD要求"      → split at the ASCII/CJK boundary, each
                                      part tokenised independently
4. **ID space** — blake2b digest truncated to 4 bytes → 0..2³²−1, then
   modulo SPARSE_VOCAB_SIZE so the resulting index fits in Milvus sparse
   UInt8 / UInt32 fields without reshaping the collection schema.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Final

# Milvus SPARSE_INVERTED_INDEX + COSINE/IP metric combinations accept
# uint8, uint16, uint32, float32 indices.  Keep well inside uint32 range.
SPARSE_VOCAB_SIZE: Final[int] = 500_000

# Unicode ranges used for Chinese / CJK character detection
_CJK_CHAR: re.Pattern[str] = re.compile(
    r"[一-鿿　-〿＀-￯]"
)


# ---------------------------------------------------------------------------
# Deterministic blake2b digest (persona=False, so no per-process randomisation)
# ---------------------------------------------------------------------------

def blake2b_digest(token: str) -> int:
    """Deterministic blake2b-32 digest of a UTF-8 token.

    Returns an integer in [0, SPARSE_VOCAB_SIZE) so it can be used directly
    as a Milvus sparse-vector index.
    """
    digest_bytes: bytes = hashlib.blake2b(
        token.encode("utf-8"),
        digest_size=4,  # 32 bits → max 4 294 967 295
    ).digest()
    # Unpack 4 bytes as unsigned 32-bit integer
    value: int = int.from_bytes(digest_bytes, byteorder="little")
    return value % SPARSE_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Core tokenisation
# ---------------------------------------------------------------------------

def _is_cjk(char: str) -> bool:
    """Return True if char is a CJK character or CJK punctuation/symbol."""
    return bool(_CJK_CHAR.match(char))


def _token_weight(weight: float, position: int, total: int) -> float:
    """Logarithmic position weighting — tokens earlier in the text score higher."""
    if total <= 0:
        return weight
    return weight * (1.0 / (1.0 + 0.1 * position))


def tokenize_and_weight(text: str) -> dict[int, float]:
    """Tokenise *text* and return {sparse_index: log-weight} for Milvus.

    Handles three scenarios:
    1. Pure ASCII text  → split on non-word boundaries, lower-case
    2. Pure CJK text   → character bigrams
    3. Mixed           → split into runs of ASCII vs CJK, tokenise each run

    The result is deterministic (blake2b, no Python hash()) and identical
    between the ingest side (BGEM3EmbeddingClient) and the query side.
    """
    if not text:
        return {}

    tokens: list[str] = _split_into_runs(text)
    if not tokens:
        return {}

    token_weights: dict[int, float] = {}
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if not token_lower:
            continue

        tid = blake2b_digest(token_lower)
        weight = _token_weight(1.0, i, len(tokens))
        token_weights[tid] = token_weights.get(tid, 0.0) + weight

    return token_weights


def _split_into_runs(text: str) -> list[str]:
    """Split *text* into runs of consecutive ASCII vs CJK characters.

    Example:
        "CDD要求"  → ["CDD", "要", "求"]
        "Hello世界" → ["hello", "世", "界"]
        "CDD screening" → ["cdd", "screening"]
    """
    if not text:
        return []

    # Fast path: pure ASCII (English / abbreviations)
    if text.isascii():
        return [t for t in re.split(r"\W+", text) if t]

    # Fast path: pure CJK
    if all(_is_cjk(ch) for ch in text):
        return _cjk_bigrams(text)

    # Mixed: partition into contiguous ASCII and CJK runs
    runs: list[str] = []
    current_run: list[str] = []
    current_is_ascii: bool | None = None

    for ch in text:
        is_ascii = unicodedata.category(ch) in ("Ll", "Lu", "Nd", "Pc", "Po", "Zs") or ch in ".,:/()-_"

        if current_is_ascii is None:
            current_is_ascii = is_ascii
        elif is_ascii != current_is_ascii:
            # Boundary between ASCII and CJK (or punctuation)
            if current_run:
                runs.append("".join(current_run))
            current_run = []
            current_is_ascii = is_ascii

        current_run.append(ch)

    if current_run:
        runs.append("".join(current_run))

    # Tokenise each run
    result: list[str] = []
    for run in runs:
        if run.isascii():
            # ASCII run: word-level split
            for word in re.split(r"\W+", run):
                if word:
                    result.append(word.lower())
        elif all(_is_cjk(ch) for ch in run):
            # CJK run: bigrams
            result.extend(_cjk_bigrams(run))
        else:
            # Mixed run (ASCII + CJK chars): split at ASCII/CJK boundary
            sub_runs = _split_ascii_cjk(run)
            for sub in sub_runs:
                if sub.isascii():
                    for w in re.split(r"\W+", sub):
                        if w:
                            result.append(w.lower())
                elif all(_is_cjk(ch) for ch in sub):
                    result.extend(_cjk_bigrams(sub))
                else:
                    # Fallback: treat as a single token
                    result.append(sub.lower())

    return result


def _cjk_bigrams(text: str) -> list[str]:
    """Return character bigrams for a CJK string.

    "客户尽职审查" → ["客户", "户尽", "尽职", "职审", "审查"]
    Single-char strings return a single-element list.
    """
    if not text:
        return []
    if len(text) == 1:
        return [text]
    return [text[i : i + 2] for i in range(len(text) - 1)]


def _split_ascii_cjk(text: str) -> list[str]:
    """Split a mixed ASCII/CJK string at each ASCII↔CJK boundary."""
    runs: list[str] = []
    current: list[str] = []
    prev_is_cjk: bool | None = None

    for ch in text:
        is_cjk = _is_cjk(ch)
        if prev_is_cjk is None:
            prev_is_cjk = is_cjk
        elif is_cjk != prev_is_cjk:
            runs.append("".join(current))
            current = []
            prev_is_cjk = is_cjk
        current.append(ch)

    if current:
        runs.append("".join(current))

    return runs


__all__ = [
    "SPARSE_VOCAB_SIZE",
    "blake2b_digest",
    "tokenize_and_weight",
]
