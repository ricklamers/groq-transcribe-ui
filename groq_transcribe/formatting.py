from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from .groq_stt import ChunkTranscript, TimestampItem


@dataclass
class TimedText:
    start: float
    end: float
    text: str


def _shift(items: List[TimestampItem], offset: float) -> List[TimestampItem]:
    out: List[TimestampItem] = []
    for it in items:
        out.append(TimestampItem(start=it.start + offset, end=it.end + offset, text=it.text))
    return out


def merge_to_global(
    chunks: List[ChunkTranscript],
    source: Literal["segment", "word", "none"] = "segment",
) -> List[TimedText]:
    """
    Convert relative chunk timestamps to global times using chunk offsets.
    If source is "none", fall back to a single TimedText with merged text.
    """
    merged: List[TimedText] = []
    if source == "none":
        text = " ".join(c.text.strip() for c in chunks if c.text)
        merged.append(TimedText(start=0.0, end=0.0, text=text))
        return merged

    for ch in chunks:
        offset = ch.chunk.start_sec
        items = ch.segments if source == "segment" else ch.words
        for it in _shift(items, offset):
            merged.append(TimedText(start=it.start, end=it.end, text=it.text))

    merged.sort(key=lambda t: t.start)
    return merged


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def format_literal(items: List[TimedText]) -> str:
    parts: List[str] = []
    for it in items:
        parts.append(f"[{_fmt_time(it.start)}]\n{it.text.strip()}\n")
    return "\n".join(parts).strip() + "\n"


def format_grouped(items: List[TimedText], max_chars: int = 100) -> str:
    if not items:
        return ""
    groups: List[str] = []
    buf: List[str] = []
    group_start = items[0].start

    def flush():
        if not buf:
            return
        groups.append(f"[{_fmt_time(group_start)}]\n" + " ".join(buf).strip() + "\n")

    cur_len = 0
    for it in items:
        text = it.text.strip()
        if not text:
            continue
        if not buf:
            group_start = it.start
            buf.append(text)
            cur_len = len(text)
        else:
            if cur_len + 1 + len(text) > max_chars:
                flush()
                buf = [text]
                cur_len = len(text)
                group_start = it.start
            else:
                buf.append(text)
                cur_len += 1 + len(text)
    flush()
    return "\n".join(groups).strip() + "\n"
