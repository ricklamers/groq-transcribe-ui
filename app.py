from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from typing import Optional, List, Dict, Any

from PySide6.QtCore import QObject, QThread, Signal, Slot, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QSplitter,
    QDoubleSpinBox,
)

from groq_transcribe.config import get_api_key, set_api_key
from groq_transcribe.audio_chunker import chunk_audio
from groq_transcribe.groq_stt import transcribe_chunks, DEFAULT_STT_MODEL, ChunkTranscript
from groq_transcribe.stitcher import stitch_transcripts
from groq_transcribe.formatting import merge_to_global, format_literal, format_grouped, TimedText


class Worker(QObject):
    chunk_progress = Signal(int, int)  # done, total
    transcribe_progress = Signal(int, int)
    status = Signal(str)
    log = Signal(str)
    finished = Signal(object)  # dict with keys: plain:str, timed:List[TimedText]|None, ts_source:str
    failed = Signal(str)

    def __init__(self, path: str, language: Optional[str], stt_model: str, target_chunk_mb: float, chunk_parallel: int, transcribe_parallel: int, ts_source: str, group_output: bool, max_chars: int, llm_strategy: str):
        super().__init__()
        self.path = path
        self.language = language
        self.stt_model = stt_model
        self.target_chunk_mb = max(0.5, float(target_chunk_mb))
        self.chunk_parallel = max(1, chunk_parallel)
        self.transcribe_parallel = max(1, transcribe_parallel)
        self.ts_source = ts_source  # 'none' | 'segment' | 'word'
        self.group_output = group_output
        self.max_chars = max(20, max_chars)
        self.llm_strategy = llm_strategy

    @Slot()
    def run(self):
        try:
            start_wall = time.perf_counter()
            self.status.emit("Chunking audio...")
            chunks = chunk_audio(
                self.path,
                target_chunk_mb=self.target_chunk_mb,
                overlap_sec=1.0,
                parallelism=self.chunk_parallel,
                progress_cb=lambda d, t: self.chunk_progress.emit(d, t),
            )
            total = len(chunks)
            if total == 0:
                raise RuntimeError("No chunks produced from audio.")

            self.status.emit(f"Transcribing {total} chunks in parallel...")

            def _on_progress(done: int, total_count: int):
                self.transcribe_progress.emit(done, total_count)

            ts_grans: Optional[List[str]] = None
            if self.ts_source in ("segment", "word"):
                ts_grans = ["word", "segment"]

            successes, failures = transcribe_chunks(
                chunks=chunks,
                language=self.language,
                model=self.stt_model,
                parallelism=self.transcribe_parallel,
                progress_cb=_on_progress,
                timestamp_granularities=ts_grans,
            )

            if failures:
                self.status.emit(f"Warning: {len(failures)} chunks failed. Proceeding to stitch.")

            # Stitch only for overlap cleanup, not summarization
            self.status.emit("Merging chunk boundaries...")
            stitched_text = stitch_transcripts(
                successes,
                language_hint=self.language,
                log_cb=lambda m: self.log.emit(m),
                llm_strategy=self.llm_strategy,
            )

            timed: Optional[List[TimedText]] = None
            if self.ts_source in ("segment", "word"):
                timed = merge_to_global(successes, source=self.ts_source)

            end_wall = time.perf_counter()
            wall_seconds = max(0.0, end_wall - start_wall)
            audio_duration_seconds = float(chunks[-1].end_sec) if chunks else 0.0

            payload: Dict[str, Any] = {
                "plain": stitched_text,
                "timed": timed,
                "ts_source": self.ts_source,
                "wall_seconds": wall_seconds,
                "audio_duration_seconds": audio_duration_seconds,
            }
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Groq Transcribe")

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        # API key
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("GROQ_API_KEY:"))
        self.api_edit = QLineEdit()
        self.api_edit.setEchoMode(QLineEdit.Password)
        self.api_edit.setToolTip("Your Groq API key. Stored locally when you click Save Key.")
        existing = get_api_key() or ""
        self.api_edit.setText(existing)
        api_layout.addWidget(self.api_edit)
        self.save_key_btn = QPushButton("Save Key")
        self.save_key_btn.setToolTip("Save the API key to local config")
        self.save_key_btn.clicked.connect(self.on_save_key)
        api_layout.addWidget(self.save_key_btn)
        config_layout.addLayout(api_layout)

        # Language
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language (e.g. en, nl, fr) optional:"))
        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("Leave blank for auto-detect")
        self.lang_edit.setToolTip("Language hint for the model. Leave blank to auto-detect.")
        self.lang_edit.setText("nl")
        lang_layout.addWidget(self.lang_edit)
        config_layout.addLayout(lang_layout)

        # STT Model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("STT Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["whisper-large-v3", "whisper-large-v3-turbo"])
        self.model_combo.setCurrentText("whisper-large-v3")
        self.model_combo.setToolTip("Speech-to-text model used for transcription.")
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)

        # Parallelism for chunking
        chunk_par_layout = QHBoxLayout()
        chunk_par_layout.addWidget(QLabel("Chunk parallelism:"))
        self.chunk_parallel_spin = QSpinBox()
        self.chunk_parallel_spin.setMinimum(1)
        self.chunk_parallel_spin.setMaximum(16)
        self.chunk_parallel_spin.setValue(8)
        self.chunk_parallel_spin.setToolTip("How many chunks are prepared in parallel while pre-processing.")
        chunk_par_layout.addWidget(self.chunk_parallel_spin)
        config_layout.addLayout(chunk_par_layout)

        # Chunk size
        chunk_size_layout = QHBoxLayout()
        chunk_size_layout.addWidget(QLabel("Target chunk size (MB):"))
        self.chunk_size_spin = QDoubleSpinBox()
        self.chunk_size_spin.setDecimals(1)
        self.chunk_size_spin.setSingleStep(0.5)
        self.chunk_size_spin.setMinimum(0.5)
        self.chunk_size_spin.setMaximum(200.0)
        self.chunk_size_spin.setValue(10.0)
        self.chunk_size_spin.setToolTip("Approximate size per chunk to increase parallelism (smaller = more chunks).")
        chunk_size_layout.addWidget(self.chunk_size_spin)
        config_layout.addLayout(chunk_size_layout)

        # Parallelism for transcription
        tr_par_layout = QHBoxLayout()
        tr_par_layout.addWidget(QLabel("Transcribe parallelism:"))
        self.transcribe_parallel_spin = QSpinBox()
        self.transcribe_parallel_spin.setMinimum(1)
        self.transcribe_parallel_spin.setMaximum(16)
        self.transcribe_parallel_spin.setValue(8)
        self.transcribe_parallel_spin.setToolTip("Number of audio chunks transcribed simultaneously.")
        tr_par_layout.addWidget(self.transcribe_parallel_spin)
        config_layout.addLayout(tr_par_layout)

        # Timestamp options
        ts_layout = QHBoxLayout()
        ts_layout.addWidget(QLabel("Timestamps:"))
        self.ts_combo = QComboBox()
        self.ts_combo.addItems(["none", "segment", "word"])
        self.ts_combo.setCurrentText("segment")
        self.ts_combo.setToolTip("Include timestamps per segment or per word, or disable.")
        ts_layout.addWidget(self.ts_combo)

        self.group_checkbox = QCheckBox("Group natural breaks (max chars)")
        self.group_checkbox.setChecked(True)
        self.group_checkbox.setToolTip("Group transcript into readable paragraphs by natural pauses.")
        ts_layout.addWidget(self.group_checkbox)

        self.max_chars_spin = QSpinBox()
        self.max_chars_spin.setMinimum(40)
        self.max_chars_spin.setMaximum(400)
        self.max_chars_spin.setValue(100)
        self.max_chars_spin.setToolTip("Upper bound of characters per grouped paragraph.")
        ts_layout.addWidget(self.max_chars_spin)

        config_layout.addLayout(ts_layout)

        # LLM strategy
        llm_layout = QHBoxLayout()
        llm_layout.addWidget(QLabel("Boundary LLM strategy:"))
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(["concat_only", "always", "never"])
        self.llm_combo.setCurrentText("concat_only")
        self.llm_combo.setToolTip("How chunk boundaries are merged: LLM assist always/never or concatenate only.")
        llm_layout.addWidget(self.llm_combo)
        config_layout.addLayout(llm_layout)

        layout.addWidget(config_group)

        # Splitter for output and logs
        splitter = QSplitter()
        # Output pane
        out_container = QWidget()
        out_layout = QVBoxLayout(out_container)
        out_layout.addWidget(QLabel("Transcript Output"))
        actions_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy transcript to clipboard (Cmd/Ctrl+C)")
        self.copy_btn.clicked.connect(self.on_copy_output)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setToolTip("Save transcript to a .txt file")
        self.save_btn.clicked.connect(self.on_save_output)
        actions_layout.addWidget(self.copy_btn)
        actions_layout.addWidget(self.save_btn)
        actions_layout.addStretch(1)
        out_layout.addLayout(actions_layout)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        out_layout.addWidget(self.output)
        splitter.addWidget(out_container)

        # Logs pane
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.addWidget(QLabel("Merging Logs"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_view)
        splitter.addWidget(log_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # File picker
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)
        self.file_btn = QPushButton("Choose Audio File…")
        self.file_btn.setToolTip("Open an audio file to transcribe (Cmd/Ctrl+O)")
        self.file_btn.clicked.connect(self.on_choose_file)
        file_layout.addWidget(self.file_btn)
        layout.addLayout(file_layout)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.start_btn = QPushButton("Transcribe")
        self.start_btn.setToolTip("Start transcription (Cmd/Ctrl+Enter)")
        self.start_btn.clicked.connect(self.on_start)
        ctrl_layout.addWidget(self.start_btn)
        layout.addLayout(ctrl_layout)

        # Progress and status
        self.chunk_progress = QProgressBar()
        self.chunk_progress.setRange(0, 100)
        self.chunk_progress.setTextVisible(True)
        layout.addWidget(QLabel("Chunking Progress"))
        layout.addWidget(self.chunk_progress)

        self.transcribe_progress = QProgressBar()
        self.transcribe_progress.setRange(0, 100)
        self.transcribe_progress.setTextVisible(True)
        layout.addWidget(QLabel("Transcription Progress"))
        layout.addWidget(self.transcribe_progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.input_path: Optional[str] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None

        # Cache for dynamic formatting after completion
        self.last_timed: Optional[List[TimedText]] = None
        self.last_plain: str = ""
        self.last_ts_source: str = "none"

        # Dynamic formatting signals
        self.group_checkbox.toggled.connect(self.on_format_options_changed)
        self.max_chars_spin.valueChanged.connect(self.on_format_options_changed)

        # Shortcuts
        self.sc_open = QShortcut(QKeySequence(QKeySequence.StandardKey.Open), self)
        self.sc_open.activated.connect(self.on_choose_file)
        self.sc_copy = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self)
        self.sc_copy.activated.connect(self.on_copy_output)
        self.sc_start = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.sc_start.activated.connect(self.on_start)

        # Settings
        self.settings = QSettings("groq-transcribe", "GroqTranscribe")
        self.load_settings()

    @Slot()
    def on_format_options_changed(self, *args):
        if self.last_ts_source in ("segment", "word") and self.last_timed is not None:
            group_output = self.group_checkbox.isChecked()
            max_chars = self.max_chars_spin.value()
            if group_output:
                rendered = format_grouped(self.last_timed, max_chars=max_chars)
            else:
                rendered = format_literal(self.last_timed)
            self.output.setPlainText(rendered)
        else:
            if self.last_plain:
                self.output.setPlainText(self.last_plain)

    def set_inputs_enabled(self, enabled: bool):
        widgets = [
            self.api_edit,
            self.save_key_btn,
            self.lang_edit,
            self.model_combo,
            self.chunk_parallel_spin,
            self.chunk_size_spin,
            self.transcribe_parallel_spin,
            self.ts_combo,
            self.group_checkbox,
            self.max_chars_spin,
            self.llm_combo,
            self.file_btn,
            self.start_btn,
            self.copy_btn,
            self.save_btn,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def load_settings(self):
        try:
            self.lang_edit.setText(self.settings.value("language", self.lang_edit.text()))
            self.model_combo.setCurrentText(self.settings.value("model", self.model_combo.currentText()))
            self.chunk_parallel_spin.setValue(int(self.settings.value("chunk_parallel", self.chunk_parallel_spin.value())))
            self.chunk_size_spin.setValue(float(self.settings.value("target_chunk_mb", self.chunk_size_spin.value())))
            self.transcribe_parallel_spin.setValue(int(self.settings.value("transcribe_parallel", self.transcribe_parallel_spin.value())))
            self.ts_combo.setCurrentText(self.settings.value("ts_source", self.ts_combo.currentText()))
            self.group_checkbox.setChecked(self.settings.value("group_output", self.group_checkbox.isChecked(), type=bool))
            self.max_chars_spin.setValue(int(self.settings.value("max_chars", self.max_chars_spin.value())))
            self.llm_combo.setCurrentText(self.settings.value("llm_strategy", self.llm_combo.currentText()))
        except Exception:
            pass

    def save_settings(self):
        self.settings.setValue("language", self.lang_edit.text())
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("chunk_parallel", self.chunk_parallel_spin.value())
        self.settings.setValue("target_chunk_mb", self.chunk_size_spin.value())
        self.settings.setValue("transcribe_parallel", self.transcribe_parallel_spin.value())
        self.settings.setValue("ts_source", self.ts_combo.currentText())
        self.settings.setValue("group_output", self.group_checkbox.isChecked())
        self.settings.setValue("max_chars", self.max_chars_spin.value())
        self.settings.setValue("llm_strategy", self.llm_combo.currentText())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    @Slot()
    def on_save_key(self):
        key = self.api_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "API Key", "Please enter an API key.")
            return
        set_api_key(key)
        QMessageBox.information(self, "API Key", "Saved.")

    @Slot()
    def on_choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            str(Path.home()),
            "Audio Files (*.flac *.mp3 *.mp4 *.mpeg *.mpga *.m4a *.ogg *.wav *.webm)"
        )
        if path:
            self.input_path = path
            self.file_label.setText(Path(path).name)

    @Slot()
    def on_start(self):
        key = self.api_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "Missing API Key", "Please paste your GROQ_API_KEY and click Save Key.")
            return
        set_api_key(key)

        if not self.input_path:
            QMessageBox.warning(self, "No File", "Please choose an audio file.")
            return

        self.output.clear()
        self.log_view.clear()
        self.chunk_progress.setValue(0)
        self.transcribe_progress.setValue(0)
        self.status_label.setText("Starting...")

        language = self.lang_edit.text().strip() or None
        stt_model = self.model_combo.currentText()
        chunk_parallel = self.chunk_parallel_spin.value()
        target_chunk_mb = float(self.chunk_size_spin.value())
        transcribe_parallel = self.transcribe_parallel_spin.value()
        ts_source = self.ts_combo.currentText()
        group_output = self.group_checkbox.isChecked()
        max_chars = self.max_chars_spin.value()
        llm_strategy = self.llm_combo.currentText()

        # Reset caches
        self.last_timed = None
        self.last_plain = ""
        self.last_ts_source = ts_source

        self.thread = QThread()
        self.worker = Worker(self.input_path, language, stt_model, target_chunk_mb, chunk_parallel, transcribe_parallel, ts_source, group_output, max_chars, llm_strategy)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.chunk_progress.connect(self.on_chunk_progress)
        self.worker.transcribe_progress.connect(self.on_transcribe_progress)
        self.worker.status.connect(self.on_status)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.set_inputs_enabled(False)

    @Slot(int, int)
    def on_chunk_progress(self, done: int, total: int):
        if total <= 0:
            self.chunk_progress.setValue(0)
        else:
            pct = int((done / total) * 100)
            self.chunk_progress.setValue(min(max(pct, 0), 100))
        self.status_label.setText(f"Chunked {done}/{total}")

    @Slot(int, int)
    def on_transcribe_progress(self, done: int, total: int):
        if total <= 0:
            self.transcribe_progress.setValue(0)
        else:
            pct = int((done / total) * 100)
            self.transcribe_progress.setValue(min(max(pct, 0), 100))
        self.status_label.setText(f"Transcribed {done}/{total}")

    @Slot(str)
    def on_status(self, msg: str):
        self.status_label.setText(msg)

    @Slot(str)
    def on_log(self, line: str):
        self.log_view.append(line)

    @Slot(object)
    def on_finished(self, payload: object):
        try:
            data: Dict[str, Any] = payload  # type: ignore
            self.last_plain = data.get("plain") or ""
            self.last_timed = data.get("timed")
            self.last_ts_source = data.get("ts_source") or "none"
            wall_seconds = float(data.get("wall_seconds") or 0.0)
            audio_seconds = float(data.get("audio_duration_seconds") or 0.0)
            if wall_seconds > 0.0 and audio_seconds > 0.0:
                rtf = audio_seconds / wall_seconds if wall_seconds > 0 else 0.0
                self.status_label.setText(f"Done in {wall_seconds:.2f}s • RTF {rtf:.1f}x")
            else:
                self.status_label.setText("Done")
        except Exception:
            self.last_plain = str(payload)
            self.last_timed = None
            self.last_ts_source = "none"
            self.status_label.setText("Done")
        self.on_format_options_changed()
        self.set_inputs_enabled(True)

    @Slot(str)
    def on_failed(self, err: str):
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Transcription Failed", err)
        self.set_inputs_enabled(True)

    @Slot()
    def on_copy_output(self):
        self.output.selectAll()
        self.output.copy()
        cursor = self.output.textCursor()
        cursor.clearSelection()
        self.output.setTextCursor(cursor)

    @Slot()
    def on_save_output(self):
        text = self.output.toPlainText()
        if not text:
            QMessageBox.information(self, "Save Transcript", "There is no transcript to save yet.")
            return
        default_name = "transcript.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", str(Path.home() / default_name), "Text Files (*.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as exc:
                QMessageBox.critical(self, "Save Failed", str(exc))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 860)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
