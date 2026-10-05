"""
zoya_live.py

Standalone speech module for Zoya.

Provides:
    record_start()   — begin capturing mic audio
    record_stop()    — stop capturing, transcribe, return text
    speak(text)      — synthesize with piper and play via aplay

Run directly for a test window:

    python zoya_live.py

Hold Ctrl+Shift+Z inside the window, speak, release.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import wave


# ============================================================
# CONFIG
# ============================================================

PIPER_BIN = os.path.expanduser("~/.local/bin/piper")
PIPER_MODEL = os.path.expanduser(
    "~/.local/share/piper/en_US-lessac-medium.onnx"
)

ARECORD_BIN = "arecord"
APLAY_BIN = "aplay"
PIPER_BIN_FALLBACK = "piper"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FORMAT = "S16_LE"

ARECORD_DEVICE = "pulse"

WHISPER_MODEL = "base.en"


# ============================================================
# LAZY WHISPER
# ============================================================

_whisper_model = None
_whisper_lock = threading.Lock()


def _load_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model

        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
        return _whisper_model


# ============================================================
# RECORDING
# ============================================================

class _Recorder:
    """
    Records mic audio to a temp WAV file.

    Runs arecord in its own process group so we can signal the
    whole group cleanly.
    """

    def __init__(self):
        self.proc = None
        self.wav_path = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.proc is not None:
                return

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                self.wav_path = tmp.name

            self.proc = subprocess.Popen(
                [
                    ARECORD_BIN,
                    "-q",
                    "-D", ARECORD_DEVICE,
                    "-f", SAMPLE_FORMAT,
                    "-r", str(SAMPLE_RATE),
                    "-c", str(CHANNELS),
                    self.wav_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

    def stop(self):
        with self.lock:
            if self.proc is None:
                return b""

            proc = self.proc
            self.proc = None

            pgid = None
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGINT)
            except Exception:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGTERM)
                    else:
                        proc.terminate()
                except Exception:
                    pass

                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        if pgid is not None:
                            os.killpg(pgid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass

            wav_path = self.wav_path
            self.wav_path = None

        if not wav_path:
            return b""

        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
        except Exception:
            frames = b""
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

        return frames


_recorder = _Recorder()


def record_start():
    """Begin capturing audio from the mic."""
    _recorder.start()


def record_stop():
    """Stop capturing and transcribe. Returns text (may be empty)."""
    pcm = _recorder.stop()

    if len(pcm) < 1000:
        return ""

    model = _load_whisper()

    import numpy as np

    audio = (
        np.frombuffer(pcm, dtype=np.int16)
        .astype(np.float32) / 32768.0
    )

    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    return " ".join(
        seg.text.strip() for seg in segments
    ).strip()


# ============================================================
# SPEAKING
# ============================================================

def _piper_binary():
    if os.path.isfile(PIPER_BIN) and os.access(PIPER_BIN, os.X_OK):
        return PIPER_BIN
    found = shutil.which(PIPER_BIN_FALLBACK)
    if found:
        return found
    raise RuntimeError(
        f"piper not found at {PIPER_BIN} or on PATH"
    )


def speak(text):
    """Synthesize `text` with piper and play it with aplay."""
    if not text or not text.strip():
        return

    if not os.path.isfile(PIPER_MODEL):
        raise RuntimeError(
            f"piper voice model not found at {PIPER_MODEL}"
        )

    piper = _piper_binary()

    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as tmp:
        wav_path = tmp.name

    try:
        p1 = subprocess.run(
            [
                piper,
                "--model", PIPER_MODEL,
                "--output_file", wav_path,
            ],
            input=text,
            text=True,
            capture_output=True,
            timeout=30,
        )

        if p1.returncode != 0:
            raise RuntimeError(
                f"piper failed: {p1.stderr.strip()}"
            )

        p2 = subprocess.run(
            [APLAY_BIN, "-q", wav_path],
            capture_output=True,
            timeout=30,
        )

        if p2.returncode != 0:
            raise RuntimeError(
                f"aplay failed: {p2.stderr.decode(errors='replace')}"
            )

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def speak_async(text):
    """Fire-and-forget speak on a background thread."""
    threading.Thread(
        target=speak,
        args=(text,),
        daemon=True,
    ).start()


# ============================================================
# TEST WINDOW
# ============================================================

def _run_test_window():

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QLabel,
            QTextEdit,
        )
    except ImportError as e:
        print(
            f"PySide6 import failed: {e}",
            file=sys.stderr,
        )
        return

    class LiveWindow(QMainWindow):

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Zoya Live — test")
            self.resize(700, 480)

            self.recording = False
            self._press_active = False

            root = QWidget()
            self.setCentralWidget(root)

            layout = QVBoxLayout(root)

            title = QLabel(
                "Hold Ctrl+Shift+Z to speak, release to send"
            )
            title.setFont(QFont("Sans", 12, QFont.Bold))
            layout.addWidget(title)

            self.status = QLabel("ready")
            self.status.setStyleSheet("color: #888;")
            layout.addWidget(self.status)

            self.output = QTextEdit()
            self.output.setReadOnly(True)
            self.output.setFont(QFont("Monospace", 11))
            self.output.setStyleSheet(
                """
                QTextEdit {
                    background: #111;
                    color: #eee;
                    border: 1px solid #333;
                    padding: 10px;
                }
                """
            )
            layout.addWidget(self.output, 1)

            self.installEventFilter(self)

            # Warm the whisper model on a background thread so
            # the first recording does not feel sluggish.
            self.status.setText("warming up speech model…")

            def _warm():
                try:
                    _load_whisper()
                    self.status.setText("ready")
                except Exception as e:
                    self.status.setText(
                        f"model load failed: {e}"
                    )

            threading.Thread(
                target=_warm,
                daemon=True,
            ).start()

        # ------------------------------------------------
        # Key handling
        # ------------------------------------------------

        def eventFilter(self, obj, event):

            if event.type() == event.Type.KeyPress:

                if event.isAutoRepeat():
                    return False

                if self._is_trigger(event):
                    if not self._press_active:
                        self._press_active = True
                        self._start_recording()
                    return True

            elif event.type() == event.Type.KeyRelease:

                if event.isAutoRepeat():
                    return False

                if self._press_active:
                    key = event.key()
                    if key in (
                        Qt.Key_Z,
                        Qt.Key_Control,
                        Qt.Key_Shift,
                    ):
                        self._press_active = False
                        self._stop_recording()
                        return True

            return super().eventFilter(obj, event)

        def _is_trigger(self, event):
            mods = event.modifiers()
            key = event.key()
            return (
                key == Qt.Key_Z
                and (mods & Qt.ControlModifier)
                and (mods & Qt.ShiftModifier)
            )

        def _start_recording(self):
            try:
                record_start()
            except Exception as e:
                self.status.setText(f"record start failed: {e}")
                return

            self.recording = True
            self.status.setText("🎤 recording — release keys to stop")
            self.output.append("[recording...]")

        def _stop_recording(self):
            if not self.recording:
                return

            self.recording = False
            self.status.setText("transcribing…")

            try:
                text = record_stop()
            except Exception as e:
                self.status.setText(f"transcribe failed: {e}")
                return

            if not text:
                self.output.append("[no speech detected]")
                self.status.setText("ready")
                return

            self.output.append(f"you said: {text}")
            self.status.setText("speaking…")

            speak_async(text)

            self.status.setText("ready")

    app = QApplication(sys.argv)
    w = LiveWindow()
    w.show()
    sys.exit(app.exec())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "speak":
        speak(" ".join(sys.argv[2:]))
    else:
        _run_test_window()