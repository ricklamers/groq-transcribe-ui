from __future__ import annotations

from typing import List, Optional, Callable

from groq import Groq

from .groq_stt import ChunkTranscript
from .config import get_api_key
import os
import difflib


DEFAULT_STITCH_MODEL = "openai/gpt-oss-20b"


# Enable verbose stdout logging only when DEBUG=true is set in the environment
DEBUG = os.getenv("DEBUG", "").lower() == "true"


BOUNDARY_PROMPT = (
    "You are merging two adjacent transcript fragments that overlap slightly at a chunk boundary.\n"
    "Rules:\n"
    "- Do NOT summarize or paraphrase.\n"
    "- Keep wording exactly as-is except to remove duplicated or partial words across the boundary.\n"
    "- Do NOT reorder content.\n"
    "- Preserve language, spelling, punctuation, and casing.\n"
    "- Return the full boundary text that replaces both inputs end-to-start (no ellipses).\n"
    "- The result must NOT be empty.\n"
)


def _client() -> Groq:
    api_key = get_api_key()
    if api_key:
        os.environ.setdefault("GROQ_API_KEY", api_key)
    return Groq()


def _exact_overlap(prev_tail: str, next_head: str) -> int:
    max_len = min(len(prev_tail), len(next_head))
    for k in range(max_len, 0, -1):
        if prev_tail[-k:] == next_head[:k]:
            return k
    return 0


def _fuzzy_overlap(prev_tail: str, next_head: str) -> tuple[int, int, int]:
    """
    Use difflib to find a longest boundary match. Returns (a_idx, b_idx, size).
    We only accept matches that touch near the boundary: a near end of prev_tail, b near start of next_head.
    """
    matcher = difflib.SequenceMatcher(a=prev_tail, b=next_head, autojunk=False)
    match = matcher.find_longest_match(0, len(prev_tail), 0, len(next_head))
    if match.size <= 0:
        return -1, -1, 0
    # Heuristics: overlap must be at least 12 chars and within 200-char windows around the boundary
    if match.size >= 12 and match.a >= len(prev_tail) - 200 and match.b <= 200:
        return match.a, match.b, match.size
    return -1, -1, 0


def _smart_overlap_merge(prev_tail: str, next_head: str) -> tuple[str, dict]:
    # 1) Exact suffix/prefix overlap
    exact = _exact_overlap(prev_tail, next_head)
    if exact > 0:
        merged = prev_tail + next_head[exact:]
        return merged, {"method": "exact", "overlap": exact}

    # 2) Fuzzy longest match near boundary
    a_idx, b_idx, size = _fuzzy_overlap(prev_tail, next_head)
    if size > 0:
        merged = prev_tail + next_head[b_idx + size :]
        return merged, {"method": "fuzzy", "a_idx": a_idx, "b_idx": b_idx, "overlap": size}

    # 3) No overlap: stitch with single space if needed
    joiner = "" if (not prev_tail or prev_tail.endswith((" ", "\n")) or next_head.startswith( (" ", ",", ".", ";", ":", "?", "!"))) else " "
    merged = prev_tail + joiner + next_head
    return merged, {"method": "concat", "overlap": 0}


def _llm_merge_boundary(
    client: Groq,
    prev_tail: str,
    next_head: str,
    language_hint: Optional[str],
    model: str,
    max_tokens: int,
    boundary_index: int,
    log_cb: Optional[Callable[[str], None]] = None,
) -> str:
    # Build prompts
    system_content = BOUNDARY_PROMPT
    user_content = (
        (f"Language: {language_hint}\n" if language_hint else "")
        + "Previous tail:\n" + prev_tail + "\n\n"
        + "Next head:\n" + next_head + "\n\n"
        + "Task: Return the full merged boundary text that replaces both inputs end-to-start, without ellipses or notes."
    )

    # Log raw prompt to stdout and optional UI log
    if DEBUG:
        print(
            f"[LLM attempt] Boundary {boundary_index}: model={model} temp=1.0 reasoning=low max_completion_tokens={max_tokens}\n"
            f"SYSTEM:\n{system_content}\nUSER:\n{user_content}"
        )
    if log_cb:
        try:
            log_cb(f"Boundary {boundary_index}: attempting LLM merge with model={model}, temp=1.0, reasoning=low")
        except Exception:
            pass

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=1.0,
        top_p=1,
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
        stream=False,
    )

    content = completion.choices[0].message.content or ""
    # Log raw response to stdout and optional UI log
    if DEBUG:
        print(f"[LLM response] Boundary {boundary_index}:\n{content}")
    if log_cb:
        try:
            if content.strip():
                log_cb(f"Boundary {boundary_index}: LLM returned text ({len(content)} chars)")
            else:
                log_cb(f"Boundary {boundary_index}: LLM returned empty text")
        except Exception:
            pass

    return content.strip()


def stitch_transcripts(
    parts: List[ChunkTranscript],
    language_hint: Optional[str] = None,
    model: str = DEFAULT_STITCH_MODEL,
    max_tokens: int = 32000,
    log_cb: Optional[Callable[[str], None]] = None,
    llm_strategy: str = "concat_only",  # 'never' | 'concat_only' | 'always'
) -> str:
    if not parts:
        return ""

    # Start with the first chunk verbatim
    result = parts[0].text or ""
    if len(parts) == 1:
        return result

    client = _client()

    WINDOW_CHARS = 400  # boundary window size for LLM and fallback

    def log(line: str) -> None:
        if log_cb:
            try:
                log_cb(line)
            except Exception:
                pass

    for i in range(1, len(parts)):
        current = parts[i].text or ""
        if not current:
            continue

        prev_tail_full = result
        prev_tail = result[-WINDOW_CHARS:]
        next_head = current[:WINDOW_CHARS]

        prev_end = parts[i - 1].chunk.end_sec
        curr_start = parts[i].chunk.start_sec

        preview_prev = prev_tail[-120:]
        preview_next = next_head[:120]

        # Always compute smart deterministic merge for baseline and diagnostics
        merged_boundary, diag = _smart_overlap_merge(prev_tail, next_head)
        method = f"SMART_{diag.get('method')}"

        # Decide whether to try LLM
        should_try_llm = (
            llm_strategy == "always" or (llm_strategy == "concat_only" and diag.get("method") == "concat")
        )

        if should_try_llm and language_hint is not None:
            try:
                llm = _llm_merge_boundary(
                    client, prev_tail, next_head, language_hint, model, max_tokens, i, log_cb
                )
                if llm:
                    merged_boundary = llm
                    method = "LLM"
                else:
                    log(f"Boundary {i}: LLM returned empty, keeping smart {diag.get('method')}.")
            except Exception as exc:
                log(f"Boundary {i}: LLM error, keeping smart {diag.get('method')}. Error={exc}")
        elif should_try_llm and language_hint is None:
            log(f"Boundary {i}: LLM requested by strategy '{llm_strategy}' but language hint is missing; keeping smart {diag.get('method')}.")

        # Recompose: everything before prev_tail + merged_boundary + everything after next_head
        new_result = prev_tail_full[: max(0, len(prev_tail_full) - len(prev_tail))] + merged_boundary + current[len(next_head) :]

        merged_preview = merged_boundary[:160]
        extra = ""
        if diag.get("method") == "exact":
            extra = f" overlap={diag.get('overlap')}"
        elif diag.get("method") == "fuzzy":
            extra = f" a_idx={diag.get('a_idx')} b_idx={diag.get('b_idx')} overlap={diag.get('overlap')}"
        log(
            f"Boundary {i}: prev_end={prev_end:.3f}s curr_start={curr_start:.3f}s method={method}{extra}\n"
            f"  Prev tail: \"{preview_prev}\"\n"
            f"  Next head: \"{preview_next}\"\n"
            f"  Merged:    \"{merged_preview}\""
        )

        result = new_result

    return result
