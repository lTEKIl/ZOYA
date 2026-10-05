import os
import sys
import time
import subprocess
import threading


# ============================================================
# DISPLAY ENV BOOTSTRAP
#
# Runs BEFORE "import ask_llm" — importing ask_llm spawns the
# three agent subprocesses at module scope, and those children
# snapshot the parent's env at spawn time. If we haven't set
# DISPLAY/WAYLAND_DISPLAY/DBUS/XDG before that, ghostty can't
# open a window.
# ============================================================

def _bootstrap_display_env():
    uid = os.getuid()

    if not os.environ.get("XDG_RUNTIME_DIR"):
        candidate = f"/run/user/{uid}"
        if os.path.isdir(candidate):
            os.environ["XDG_RUNTIME_DIR"] = candidate

    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = f"/run/user/{uid}/bus"
        if os.path.exists(bus):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"

    if not os.environ.get("WAYLAND_DISPLAY"):
        for name in ("wayland-0", "wayland-1", "wayland-2"):
            if os.path.exists(f"/run/user/{uid}/{name}"):
                os.environ["WAYLAND_DISPLAY"] = name
                break

    if not os.environ.get("DISPLAY"):
        import glob as _glob
        candidates = sorted(_glob.glob("/tmp/.X11-unix/X*"))
        if candidates:
            n = os.path.basename(candidates[0])[1:]
            os.environ["DISPLAY"] = f":{n}"

    if not os.environ.get("XDG_SESSION_TYPE"):
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["XDG_SESSION_TYPE"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["XDG_SESSION_TYPE"] = "x11"

    path_parts = (os.environ.get("PATH") or "").split(os.pathsep)
    for extra in ("/snap/bin", "/usr/local/bin", "/usr/bin", "/bin"):
        if extra not in path_parts:
            path_parts.append(extra)
    os.environ["PATH"] = os.pathsep.join(p for p in path_parts if p)

    print(
        "[BOOTSTRAP] "
        f"DISPLAY={os.environ.get('DISPLAY', '<unset>')} "
        f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '<unset>')} "
        f"DBUS_SESSION_BUS_ADDRESS={os.environ.get('DBUS_SESSION_BUS_ADDRESS', '<unset>')} "
        f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR', '<unset>')}",
        file=sys.stderr,
        flush=True,
    )


_bootstrap_display_env()


# ============================================================
# NOW import ask_llm + zoya_live
# ============================================================

import ask_llm
import zoya_live

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QScrollArea,
)


# ============================================================
# CONFIG
# ============================================================

OPS_LOG = os.path.abspath("runs/infra_ops.log")


# ============================================================
# OPS TERMINAL
# ============================================================

def ensure_ops_log():
    os.makedirs(os.path.dirname(OPS_LOG), exist_ok=True)
    if not os.path.exists(OPS_LOG):
        open(OPS_LOG, "a").close()


def start_ops_terminal():
    ensure_ops_log()
    script = os.path.abspath("ops_terminal.py")
    terminals = [
        ["ghostty", "-e", "python", script],
        ["gnome-terminal", "--", "python", script],
        ["x-terminal-emulator", "-e", "python", script],
    ]
    for command in terminals:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            continue
    return False


# ============================================================
# STREAMING WORKER
# ============================================================

class StreamWorker(QThread):

    chunk = Signal(str)
    finished_response = Signal(str)
    error = Signal(str)

    def __init__(self, question, chat_id):
        super().__init__()
        self.question = question
        self.chat_id = chat_id

    def run(self):
        full_response = ""
        try:
            for chunk in ask_llm.ask_groq_stream(
                self.question,
                self.chat_id,
            ):
                if chunk is None:
                    continue
                chunk = str(chunk)
                full_response += chunk
                self.chunk.emit(chunk)
            self.finished_response.emit(full_response)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# SPEECH WORKER (STT)
# ============================================================

class TranscribeWorker(QThread):

    finished_text = Signal(str)
    error = Signal(str)

    def run(self):
        try:
            text = zoya_live.record_stop()
            self.finished_text.emit(text or "")
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# CHAT MESSAGE WIDGET
# ============================================================

class MessageWidget(QFrame):

    def __init__(self, role, text):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self._raw_text = str(text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 12)
        layout.setSpacing(5)

        label = QLabel("You" if role == "user" else "Zoya")
        label.setFont(QFont("Sans", 10, QFont.Bold))

        if role == "user":
            label.setStyleSheet("QLabel { color: #ffffff; }")
        else:
            label.setStyleSheet("QLabel { color: #aaaaaa; }")

        layout.addWidget(label)

        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setFont(QFont("Sans", 11))
        text_box.setFrameShape(QFrame.NoFrame)
        text_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        text_box.document().setDefaultStyleSheet(
            """
            code {
                background: #111111;
                padding: 2px 4px;
                font-family: monospace;
                color: #ffcc66;
            }
            pre {
                background: #111111;
                padding: 8px;
                font-family: monospace;
                color: #dddddd;
            }
            a {
                color: #66aaff;
                text-decoration: none;
            }
            """
        )

        text_box.setStyleSheet(
            """
            QTextEdit {
                background: transparent;
                color: #eeeeee;
                border: none;
                padding: 0px;
                selection-background-color: #444444;
                selection-color: #ffffff;
            }
            """
        )

        text_box.setMarkdown(self._raw_text)
        self.text_box = text_box
        layout.addWidget(text_box)

        self.update_height()
        text_box.textChanged.connect(self.update_height)

    def update_height(self):
        document = self.text_box.document()
        height = int(document.size().height()) + 12
        self.text_box.setFixedHeight(max(35, height))

    def set_text(self, text):
        self._raw_text = str(text)
        self.text_box.setMarkdown(self._raw_text)
        self.update_height()


# ============================================================
# CHAT WINDOW
# ============================================================

class ZoyaWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.chat_id = "default"
        self.worker = None
        self.current_zoya_widget = None
        self.current_zoya_raw = ""

        # Voice state
        self._voice_press_active = False
        self._voice_recording = False
        self._voice_reply_pending = False
        self.transcribe_worker = None

        self.setWindowTitle("ZOYA")
        self.resize(1400, 850)

        # Enable Ctrl+Shift+Z event handling.
        self.installEventFilter(self)

        self.build_ui()
        self.load_history()
        self.load_chat(self.chat_id)

        self.input_box.setFocus()

        # Warm the speech model in the background.
        threading.Thread(
            target=zoya_live._load_whisper,
            daemon=True,
        ).start()

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(
            "QWidget { background: #050505; color: #eeeeee; }"
        )

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 18, 18, 18)

        brand = QLabel("ZOYA\n\nLOCAL AI OPS")
        brand.setFont(QFont("Sans", 16, QFont.Bold))
        brand.setStyleSheet(
            """
            QLabel {
                color: white;
                padding-bottom: 15px;
                border-bottom: 1px solid #202020;
            }
            """
        )
        sidebar_layout.addWidget(brand)

        status = QLabel(
            "● SYSTEM ONLINE\n\n"
            "PLANNER       READY\n"
            "INFRA         READY\n"
            "INVESTIGATOR  READY\n"
            "VERIFIER      READY"
        )
        status.setStyleSheet(
            """
            QLabel {
                color: #777777;
                padding: 12px 0px;
                border-bottom: 1px solid #202020;
            }
            """
        )
        sidebar_layout.addWidget(status)

        new_chat = QPushButton("+ NEW CHAT")
        new_chat.setCursor(Qt.PointingHandCursor)
        new_chat.clicked.connect(self.new_chat)
        new_chat.setStyleSheet(
            """
            QPushButton {
                background: white;
                color: black;
                border: none;
                padding: 9px;
                margin-top: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background: #dddddd; }
            """
        )
        sidebar_layout.addWidget(new_chat)

        history_label = QLabel("CHAT HISTORY")
        history_label.setStyleSheet(
            """
            QLabel {
                color: #666666;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 5px;
            }
            """
        )
        sidebar_layout.addWidget(history_label)

        self.history = QListWidget()
        self.history.setStyleSheet(
            """
            QListWidget {
                background: #050505;
                border: none;
                color: #999999;
                outline: none;
            }
            QListWidget::item {
                background: #080808;
                padding: 10px;
                margin-bottom: 4px;
                border: none;
            }
            QListWidget::item:selected {
                background: #181818;
                color: white;
            }
            QListWidget::item:hover {
                background: #141414;
                color: white;
            }
            """
        )
        self.history.itemClicked.connect(self.history_clicked)
        sidebar_layout.addWidget(self.history)

        root_layout.addWidget(sidebar)

        # ----------------------------------------------------
        # MAIN AREA
        # ----------------------------------------------------

        main = QWidget()
        main.setStyleSheet(
            "QWidget { background: #000000; color: #eeeeee; }"
        )

        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.chat = QWidget()
        self.chat_layout = QVBoxLayout(self.chat)
        self.chat_layout.setContentsMargins(55, 25, 55, 25)
        self.chat_layout.setSpacing(0)
        self.chat_layout.addStretch()

        self.chat_scroll = QWidget()
        scroll_layout = QVBoxLayout(self.chat_scroll)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(self.chat)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.chat_scroll)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea { background: #000000; border: none; }
            QScrollBar:vertical { background: #000000; width: 10px; }
            QScrollBar::handle:vertical {
                background: #292929;
                min-height: 30px;
            }
            """
        )

        main_layout.addWidget(self.scroll_area, 1)

        input_area = QWidget()
        input_area.setFixedHeight(80)
        input_area.setStyleSheet(
            """
            QWidget {
                background: #050505;
                border-top: 1px solid #1c1c1c;
            }
            """
        )

        input_layout = QHBoxLayout(input_area)
        input_layout.setContentsMargins(25, 18, 25, 18)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText(
            "Ask Zoya...  (hold Ctrl+Shift+Z to speak)"
        )
        self.input_box.setStyleSheet(
            """
            QLineEdit {
                background: #000000;
                color: white;
                border: 1px solid #333333;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus { border: 1px solid #666666; }
            """
        )
        self.input_box.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_box)

        self.mic_button = QPushButton("🎤")
        self.mic_button.setFixedWidth(60)
        self.mic_button.setToolTip(
            "Hold Ctrl+Shift+Z (or hold this button) to speak"
        )
        self.mic_button.setStyleSheet(
            """
            QPushButton {
                background: #1a1a1a;
                color: white;
                border: 1px solid #333333;
                padding: 9px;
                font-size: 16px;
            }
            QPushButton:hover { background: #2a2a2a; }
            QPushButton:pressed { background: #ff4444; }
            """
        )
        # Mouse press/release to record.
        self.mic_button.pressed.connect(self._on_mic_pressed)
        self.mic_button.released.connect(self._on_mic_released)
        input_layout.addWidget(self.mic_button)

        send_button = QPushButton("SEND")
        send_button.setFixedWidth(90)
        send_button.setStyleSheet(
            """
            QPushButton {
                background: white;
                color: black;
                border: none;
                padding: 9px;
                font-weight: bold;
            }
            QPushButton:hover { background: #dddddd; }
            """
        )
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)

        main_layout.addWidget(input_area)

        self.status_bar = QLabel(
            "READY  •  LOCAL AGENTS  •  MCP TOOLS  •  A2A"
        )
        self.status_bar.setFixedHeight(28)
        self.status_bar.setStyleSheet(
            """
            QLabel {
                background: #050505;
                color: #666666;
                padding-left: 15px;
                border-top: 1px solid #111111;
            }
            """
        )
        main_layout.addWidget(self.status_bar)

        root_layout.addWidget(main, 1)

    # ========================================================
    # KEYBOARD: Ctrl+Shift+Z hold-to-record
    # ========================================================

    def eventFilter(self, obj, event):

        if event.type() == event.Type.KeyPress:

            if event.isAutoRepeat():
                return False

            if self._is_voice_trigger(event):
                if not self._voice_press_active:
                    self._voice_press_active = True
                    self._start_voice_recording()
                return True

        elif event.type() == event.Type.KeyRelease:

            if event.isAutoRepeat():
                return False

            if self._voice_press_active:
                key = event.key()
                if key in (
                    Qt.Key_Z,
                    Qt.Key_Control,
                    Qt.Key_Shift,
                ):
                    self._voice_press_active = False
                    self._stop_voice_recording()
                    return True

        return super().eventFilter(obj, event)

    def _is_voice_trigger(self, event):
        mods = event.modifiers()
        key = event.key()
        return (
            key == Qt.Key_Z
            and (mods & Qt.ControlModifier)
            and (mods & Qt.ShiftModifier)
        )

    # ========================================================
    # MIC BUTTON
    # ========================================================

    def _on_mic_pressed(self):
        if self._voice_recording:
            return
        self._start_voice_recording()

    def _on_mic_released(self):
        if not self._voice_recording:
            return
        self._stop_voice_recording()

    # ========================================================
    # VOICE RECORDING
    # ========================================================

    def _start_voice_recording(self):
        if self._voice_recording:
            return

        try:
            zoya_live.record_start()
        except Exception as e:
            self.status_bar.setText(f"MIC ERROR: {e}")
            return

        self._voice_recording = True
        self.status_bar.setText(
            "🎤 RECORDING — release keys to send"
        )

    def _stop_voice_recording(self):
        if not self._voice_recording:
            return

        self._voice_recording = False
        self.status_bar.setText("TRANSCRIBING…")

        self.transcribe_worker = TranscribeWorker()
        self.transcribe_worker.finished_text.connect(
            self._on_transcribed
        )
        self.transcribe_worker.error.connect(
            self._on_transcribe_error
        )
        self.transcribe_worker.start()

    def _on_transcribed(self, text):
        self.transcribe_worker = None

        text = (text or "").strip()

        if not text:
            self.status_bar.setText(
                "READY  •  NO SPEECH DETECTED"
            )
            return

        # Put it in the input box and send.
        self.input_box.setText(text)

        # Mark the upcoming reply as voice-mode so we speak it.
        self._voice_reply_pending = True

        self.status_bar.setText("READY  •  SENDING VOICE INPUT")
        self.send_message()

    def _on_transcribe_error(self, msg):
        self.transcribe_worker = None
        self.status_bar.setText(f"STT ERROR: {msg}")

    # ========================================================
    # CHAT HELPERS
    # ========================================================

    def clear_chat(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.chat_layout.addStretch()

    def add_message(self, role, text):
        if self.chat_layout.count():
            item = self.chat_layout.takeAt(
                self.chat_layout.count() - 1
            )
            if item.spacerItem() is None:
                self.chat_layout.addItem(item)

        widget = MessageWidget(role, text)
        self.chat_layout.addWidget(widget)
        self.chat_layout.addStretch()

        self.scroll_to_bottom()
        return widget

    def scroll_to_bottom(self):
        QApplication.processEvents()
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ========================================================
    # LOAD CHAT
    # ========================================================

    def load_chat(self, chat_id):

        self.chat_id = str(chat_id)
        self.clear_chat()

        try:
            messages = ask_llm.get_messages(self.chat_id)
        except Exception as e:
            self.status_bar.setText(f"ERROR LOADING CHAT: {e}")
            return

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                self.add_message("user", content)
            elif role == "assistant":
                self.add_message("assistant", content)

        self.update_history_selection()
        self.scroll_to_bottom()

    # ========================================================
    # HISTORY
    # ========================================================

    def load_history(self):

        self.history.clear()

        try:
            chats = ask_llm.list_chats()
        except Exception:
            chats = []

        if self.chat_id not in chats:
            chats.insert(0, self.chat_id)

        chats = list(dict.fromkeys(str(x) for x in chats))

        for chat_id in chats:
            title = self.get_chat_title(chat_id)
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, chat_id)
            self.history.addItem(item)

        self.update_history_selection()

    def get_chat_title(self, chat_id):
        try:
            messages = ask_llm.get_messages(chat_id)
        except Exception:
            messages = []

        for message in messages:
            if message.get("role") == "user":
                text = str(message.get("content", "")).strip()
                if text:
                    if len(text) > 30:
                        return text[:27] + "..."
                    return text

        if str(chat_id) == "default":
            return "Default Chat"

        return "Chat " + str(chat_id)[:12]

    def update_history_selection(self):
        for i in range(self.history.count()):
            item = self.history.item(i)
            if str(item.data(Qt.UserRole)) == str(self.chat_id):
                self.history.setCurrentItem(item)
                return

    def history_clicked(self, item):
        chat_id = item.data(Qt.UserRole)
        self.load_chat(chat_id)
        self.status_bar.setText("READY  •  CHAT LOADED")
        self.input_box.setFocus()

    # ========================================================
    # NEW CHAT
    # ========================================================

    def new_chat(self):
        self.chat_id = str(time.time())
        self.clear_chat()
        self.load_history()
        self.status_bar.setText("READY  •  NEW CHAT")
        self.input_box.setFocus()

    # ========================================================
    # SEND
    # ========================================================

    def send_message(self):

        question = self.input_box.text().strip()

        if not question:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.input_box.clear()
        self.add_message("user", question)

        self.status_bar.setText(
            "PROCESSING  •  PLANNER → AGENT → MCP"
        )

        self.current_zoya_raw = ""
        self.current_zoya_widget = self.add_message(
            "assistant",
            "",
        )

        self.worker = StreamWorker(
            question,
            self.chat_id,
        )

        self.worker.chunk.connect(self.receive_chunk)
        self.worker.finished_response.connect(self.response_finished)
        self.worker.error.connect(self.response_error)

        self.worker.start()

    # ========================================================
    # STREAM
    # ========================================================

    def receive_chunk(self, chunk):

        if self.current_zoya_widget is None:
            return

        self.current_zoya_raw += str(chunk)

        self.current_zoya_widget.set_text(
            self.current_zoya_raw
        )

        self.scroll_to_bottom()

    def response_finished(self, response):

        if self.current_zoya_widget:
            self.current_zoya_widget.set_text(response)

        self.status_bar.setText("READY  •  RESPONSE COMPLETE")

        self.load_history()

        # If this reply was triggered by voice, speak the first
        # line only.
        if self._voice_reply_pending:
            self._voice_reply_pending = False
            spoken = self._first_line(response)
            if spoken:
                threading.Thread(
                    target=zoya_live.speak,
                    args=(spoken,),
                    daemon=True,
                ).start()

        self.worker = None
        self.current_zoya_widget = None
        self.current_zoya_raw = ""
        self.input_box.setFocus()

    @staticmethod
    def _first_line(text):
        """
        Extract a speakable summary from the response.

        - Skip leading markdown / code fences.
        - Take the first non-empty line.
        - Strip surrounding markdown bold/italic markers.
        - Cap at 400 chars.
        """
        if not text:
            return ""

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip code fences / horizontal rules
            if stripped.startswith("```") or stripped.startswith("---"):
                continue
            # Strip markdown emphasis wrappers
            cleaned = stripped
            if cleaned.startswith("**") and cleaned.endswith("**"):
                cleaned = cleaned[2:-2]
            elif cleaned.startswith("*") and cleaned.endswith("*"):
                cleaned = cleaned[1:-1]
            elif cleaned.startswith("`") and cleaned.endswith("`"):
                cleaned = cleaned[1:-1]

            return cleaned[:400]

        return ""

    def response_error(self, error):

        if self.current_zoya_widget:
            self.current_zoya_widget.set_text(
                f"Agent error: {error}"
            )

        self.status_bar.setText("ERROR")

        self.worker = None
        self.current_zoya_widget = None
        self.current_zoya_raw = ""
        self._voice_reply_pending = False
        self.input_box.setFocus()

    # ========================================================
    # DELETE CHAT
    # ========================================================

    def delete_chat(self, chat_id):

        reply = QMessageBox.question(
            self,
            "Delete chat",
            "Delete this conversation?\n\n"
            "This removes the saved conversation.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            data = ask_llm._load()
            key = str(chat_id)

            if key in data:
                del data[key]
                ask_llm._save(data)

            if str(self.chat_id) == key:
                self.chat_id = "default"
                self.load_chat(self.chat_id)

            self.load_history()
            self.status_bar.setText("READY  •  CHAT DELETED")

        except Exception as e:
            QMessageBox.critical(self, "Delete error", str(e))

    # ========================================================
    # RIGHT-CLICK DELETE
    # ========================================================

    def contextMenuEvent(self, event):

        item = self.history.itemAt(
            self.history.mapFromGlobal(event.globalPos())
        )

        if item is None:
            return

        chat_id = item.data(Qt.UserRole)

        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        delete_action = menu.addAction("Delete chat")

        selected = menu.exec(event.globalPos())

        if selected == delete_action:
            self.delete_chat(chat_id)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    ensure_ops_log()
    start_ops_terminal()

    app = QApplication(sys.argv)
    app.setApplicationName("ZOYA")

    window = ZoyaWindow()
    window.show()

    exit_code = app.exec()

    try:
        ask_llm.close()
    except Exception:
        pass

    sys.exit(exit_code)