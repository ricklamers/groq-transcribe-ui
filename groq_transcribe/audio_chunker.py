from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
import concurrent.futures


@dataclass
class AudioChunk:
    path: Path
    start_sec: float
    end_sec: float


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def _probe_duration_seconds(input_path: str | os.PathLike) -> float:
    # Use ffprobe to get duration in seconds
    try:
        proc = _run_cmd([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ])
        return float(proc.stdout.strip())
    except Exception as exc:
        raise RuntimeError(f"Failed to probe audio duration: {exc}")


def _export_flac_segment(input_path: str | os.PathLike, start: float, end: float, out_path: Path) -> None:
    # Trim and transcode with ffmpeg to 16kHz mono FLAC
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError("Invalid segment duration")
    args = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "flac",
        str(out_path),
    ]
    try:
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for segment {start}-{end}s: {exc.stderr.decode('utf-8', 'ignore')}")


def _extract_full_audio_flac(input_path: str | os.PathLike, out_path: Path) -> None:
    # Extract audio track as 16kHz mono FLAC from containers like MP4
    args = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",  # drop video
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "flac",
        str(out_path),
    ]
    try:
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to extract audio: {exc.stderr.decode('utf-8', 'ignore')}")


def chunk_audio(
    input_path: str | os.PathLike,
    target_chunk_mb: float = 10.0,
    overlap_sec: float = 1.0,
    parallelism: int = 4,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[AudioChunk]:
    """
    Create overlapping FLAC chunks using ffmpeg, normalized to 16kHz mono.
    Chunk size targets ~target_chunk_mb based on a conservative PCM estimate.
    Exports are run in parallel with the provided parallelism.
    """
    # If the input is an MP4, extract audio once to avoid decoding video for every chunk
    source_path = Path(input_path)
    tmp_src_dir: Optional[Path] = None
    tmp_src_file: Optional[Path] = None
    try:
        if source_path.suffix.lower() == ".mp4":
            tmp_src_dir = Path(tempfile.mkdtemp(prefix="groq_src_"))
            tmp_src_file = tmp_src_dir / "source.flac"
            _extract_full_audio_flac(source_path, tmp_src_file)
            source_path = tmp_src_file

        duration = _probe_duration_seconds(source_path)

        # Estimate bytes/second for 16kHz mono 16-bit PCM
        bytes_per_second = 16000 * 2 * 1
        target_bytes = int(target_chunk_mb * 1024 * 1024 * 0.9)
        min_chunk_sec = 10.0

        chunk_sec = max(target_bytes / bytes_per_second, min_chunk_sec)
        chunk_sec = float(chunk_sec)
        overlap_sec = max(0.0, float(overlap_sec))

        tmp_dir = Path(tempfile.mkdtemp(prefix="groq_chunks_"))

        # Precompute segment boundaries
        segments: list[tuple[int, float, float, Path]] = []
        idx = 0
        start = 0.0
        while start < duration:
            end = min(start + chunk_sec, duration)
            out_path = tmp_dir / f"chunk_{idx:05d}.flac"
            segments.append((idx, start, end, out_path))
            if end >= duration:
                break
            step = (end - start) - overlap_sec
            start = max(0.0, start + max(step, 0.1))
            idx += 1

        total = len(segments)
        done = 0

        results: list[Optional[AudioChunk]] = [None] * total

        def _task(entry: tuple[int, float, float, Path]) -> tuple[int, Optional[AudioChunk], Optional[str]]:
            i, s, e, outp = entry
            try:
                _export_flac_segment(source_path, s, e, outp)
                return i, AudioChunk(path=outp, start_sec=s, end_sec=e), None
            except Exception as exc:
                return i, None, str(exc)

        max_workers = max(1, int(parallelism))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_task, seg) for seg in segments]
            for fut in concurrent.futures.as_completed(futs):
                i, chunk, err = fut.result()
                if err:
                    # Cancel remaining and raise
                    for f in futs:
                        f.cancel()
                    raise RuntimeError(f"Chunk export failed: {err}")
                results[i] = chunk
                done += 1
                if progress_cb:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass

        # All should be filled
        ordered: List[AudioChunk] = [c for c in results if c is not None]
        return ordered
    finally:
        # Best-effort cleanup of temporary extracted audio source
        if tmp_src_file is not None:
            try:
                if tmp_src_file.exists():
                    tmp_src_file.unlink()
            except Exception:
                pass
        if tmp_src_dir is not None:
            try:
                tmp_src_dir.rmdir()
            except Exception:
                pass
