"""BM25 keyword search over the indexed chunk text.

Tokenisation handles the text this project holds: identifiers (`approvalRequired`, `max_tool_rounds`) are
indexed whole and as their parts, and Chinese is indexed as single characters plus overlapping pairs, because
there are no spaces to split on.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

K1 = 1.5
B = 0.75
WORD_RE = re.compile(r"[0-9A-Za-z_À-ɏ]+|[㐀-鿿]+")
CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from how i in is it of on or that the this to was were what when "
    "where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        if "㐀" <= word[0] <= "鿿":
            tokens.extend(word)
            tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            tokens.extend(latin_tokens(word))
    return tokens


def latin_tokens(word: str) -> list[str]:
    whole = word.lower()
    pieces = [whole]
    for part in word.split("_"):
        for match in CAMEL_RE.finditer(part):
            piece = match.group(0).lower()
            if piece not in pieces:
                pieces.append(piece)
    return [piece for piece in pieces if piece not in STOPWORDS and (len(piece) > 1 or piece.isdigit())]


@dataclass
class Bm25Index:
    doc_length: list[int]
    average_length: float
    postings: dict[str, list[tuple[int, int]]]
    idf: dict[str, float]

    @classmethod
    def build(cls, texts: list[str]) -> Bm25Index:
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for position, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((position, frequency))
        total = len(texts)
        idf = {term: math.log(1 + (total - len(items) + 0.5) / (len(items) + 0.5)) for term, items in postings.items()}
        return cls(lengths, (sum(lengths) / total) if total and sum(lengths) else 1.0, postings, idf)

    def search(self, query: str, top_k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """The best `top_k` chunk positions for the query, as (position, score), best first."""

        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            items = self.postings.get(term)
            if not items:
                continue
            weight = self.idf[term]
            for position, frequency in items:
                if allowed is not None and position not in allowed:
                    continue
                length_norm = 1 - B + B * self.doc_length[position] / self.average_length
                scores[position] = scores.get(position, 0.0) + weight * frequency * (K1 + 1) / (frequency + K1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]
