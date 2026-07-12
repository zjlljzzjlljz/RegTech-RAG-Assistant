from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")
_HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*\s+|[A-Z][A-Z\s/&:-]{4,})")


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    parent_id: str | None
    text: str
    chunk_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    def __init__(
        self,
        *,
        parent_tokens: int = 1500,
        child_tokens: int = 400,
        overlap_tokens: int = 200,
        version: str = "semantic-v2",
    ) -> None:
        if child_tokens <= 0 or parent_tokens < child_tokens:
            raise ValueError("parent_tokens must be >= child_tokens > 0")
        if overlap_tokens < 0 or overlap_tokens >= child_tokens:
            raise ValueError("overlap_tokens must be in [0, child_tokens)")
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens
        self.overlap_tokens = overlap_tokens
        self.version = version

    def split_page(self, *, document_id: str, page_number: int, text: str) -> list[SemanticChunk]:
        sections = self._sections(text)
        output: list[SemanticChunk] = []
        for section_index, (section_path, section_text) in enumerate(sections):
            parent_windows = self._windows(self._sentences(section_text), self.parent_tokens, 0)
            for parent_index, parent_text in enumerate(parent_windows):
                stable_path = f"page-{page_number}/{section_path}"
                parent_id = self._stable_id(document_id, stable_path, "parent", parent_text)
                common = {
                    "document_id": document_id,
                    "page": page_number,
                    "section_path": section_path,
                    "chunk_version": self.version,
                    "content_hash": self._content_hash(parent_text),
                    "legacy_chunk_id": f"page-{page_number}-parent-{section_index + parent_index}",
                }
                output.append(SemanticChunk(parent_id, None, parent_text, "parent", common))
                for child_index, child_text in enumerate(
                    self._windows(self._sentences(parent_text), self.child_tokens, self.overlap_tokens)
                ):
                    child_id = self._stable_id(document_id, stable_path, "child", child_text)
                    metadata = {
                        **common,
                        "content_hash": self._content_hash(child_text),
                        "parent_id": parent_id,
                        "legacy_chunk_id": f"{common['legacy_chunk_id']}-child-{child_index}",
                    }
                    output.append(SemanticChunk(child_id, parent_id, child_text, "child", metadata))
        return output

    def _sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        heading = "page"
        body: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if body:
                    body.append("")
                continue
            if len(line) <= 120 and _HEADING_PATTERN.match(line):
                if any(part.strip() for part in body):
                    sections.append((heading, "\n".join(body).strip()))
                heading = line
                body = []
            else:
                body.append(line)
        if any(part.strip() for part in body):
            sections.append((heading, "\n".join(body).strip()))
        return sections or [("page", text.strip())]

    def _sentences(self, text: str) -> list[str]:
        return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]

    def _windows(self, sentences: list[str], limit: int, overlap: int) -> list[str]:
        if not sentences:
            return []
        windows: list[str] = []
        start = 0
        while start < len(sentences):
            end = start
            token_count = 0
            while end < len(sentences):
                sentence_tokens = self._token_count(sentences[end])
                if end > start and token_count + sentence_tokens > limit:
                    break
                token_count += sentence_tokens
                end += 1
                if token_count >= limit:
                    break
            if end == start:
                end += 1
            windows.append(" ".join(sentences[start:end]))
            if end >= len(sentences):
                break
            if overlap == 0:
                start = end
                continue
            retained = 0
            next_start = end
            while next_start > start and retained < overlap:
                next_start -= 1
                retained += self._token_count(sentences[next_start])
            start = max(start + 1, next_start)
        return windows

    def _token_count(self, text: str) -> int:
        return max(1, len(_TOKEN_PATTERN.findall(text)))

    def _stable_id(self, document_id: str, section_path: str, kind: str, text: str) -> str:
        material = "\x1f".join((document_id, section_path, kind, self._normalize(text)))
        return f"{kind}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode("utf-8")).hexdigest()

    def _normalize(self, text: str) -> str:
        return " ".join(text.split()).strip()


__all__ = ["SemanticChunk", "SemanticChunker"]
