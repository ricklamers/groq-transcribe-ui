from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from groq import Groq

from .config import get_api_key
from .audio_chunker import AudioChunk


@dataclass
class TimestampItem:
    start: float
    end: float
    text: str


@dataclass
class ChunkTranscript:
    chunk: AudioChunk
    text: str
    segments: List[TimestampItem]
    words: List[TimestampItem]
    error: Optional[str] = None


DEFAULT_STT_MODEL = "whisper-large-v3"


def _get_client() -> Groq:
    api_key = get_api_key()
    if api_key:
        os.environ.setdefault("GROQ_API_KEY", api_key)
    # Groq() picks up GROQ_API_KEY from env
    return Groq()


def _as_dict(obj: Any) -> Dict[str, Any]:
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        # Fallback try: attributes
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    except Exception:
        return {}


def _parse_timestamps(raw: Dict[str, Any]) -> tuple[List[TimestampItem], List[TimestampItem]]:
    segments: List[TimestampItem] = []
    words: List[TimestampItem] = []

    raw_segments = raw.get("segments") or []
    # Some SDKs put words at top-level, or inside segments
    raw_words_top = raw.get("words") or []

    for seg in raw_segments:
        try:
            segments.append(
                TimestampItem(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")),
                )
            )
            # nested words if present
            seg_words = seg.get("words") or []
            for w in seg_words:
                words.append(
                    TimestampItem(
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        text=str(w.get("word", w.get("text", ""))),
                    )
                )
        except Exception:
            continue

    for w in raw_words_top:
        try:
            words.append(
                TimestampItem(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    text=str(w.get("word", w.get("text", ""))),
                )
            )
        except Exception:
            continue

    return segments, words


def _transcribe_single(
    client: Groq,
    chunk: AudioChunk,
    language: Optional[str],
    model: str,
    response_format: str = "verbose_json",
    timestamp_granularities: Optional[List[str]] = None,
) -> ChunkTranscript:
    try:
        with open(chunk.path, "rb") as f:
            kwargs: Dict[str, Any] = dict(
                file=(str(chunk.path.name), f.read()),
                model=model,
                language=language,
                response_format=response_format,
                temperature=0.0,
            )
            if timestamp_granularities:
                kwargs["timestamp_granularities"] = timestamp_granularities
            resp = client.audio.transcriptions.create(**kwargs)
        # Extract text and timestamps
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp["text"]
            except Exception:
                text = ""
        raw = _as_dict(resp)
        segments, words = _parse_timestamps(raw)
        return ChunkTranscript(chunk=chunk, text=text or "", segments=segments, words=words)
    except Exception as exc:
        return ChunkTranscript(chunk=chunk, text="", segments=[], words=[], error=str(exc))


def transcribe_chunks(
    chunks: List[AudioChunk],
    language: Optional[str] = None,
    model: str = DEFAULT_STT_MODEL,
    parallelism: int = 4,
    progress_cb: Optional[callable] = None,
    timestamp_granularities: Optional[List[str]] = None,
) -> Tuple[List[ChunkTranscript], List[ChunkTranscript]]:
    """
    Transcribe chunks in parallel. Returns (successes, failures).
    progress_cb, if provided, is called with (done_count, total_count).
    """
    if not chunks:
        return [], []

    client = _get_client()

    total = len(chunks)
    done = 0

    successes: List[ChunkTranscript] = []
    failures: List[ChunkTranscript] = []

    def _wrap(chunk: AudioChunk) -> ChunkTranscript:
        return _transcribe_single(client, chunk, language, model, "verbose_json", timestamp_granularities)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
        futures = [ex.submit(_wrap, ch) for ch in chunks]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result.error:
                failures.append(result)
            else:
                successes.append(result)
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass

    # Keep successes in original chronological order
    successes.sort(key=lambda r: r.chunk.start_sec)
    return successes, failures
