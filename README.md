# Zoya

A local AI ops assistant for Ubuntu. Ask questions about your machine,
and let it take actions in a live terminal you watch and approve.

Built as a multi-agent system:

- **Planner** — routes each request
- **Infra** — read-only system inspection
- **Investigator** — forensics, filesystem, web research
- **Verifier** — the only agent that mutates state, via a real
  terminal window with a Y/N approval prompt on dangerous commands

Everything runs locally. Only the LLM calls leave the machine (Groq).

---

## Architecture
Zoya (Qt UI) → Planner → Infra / Investigator / Verifier
│
▼
real terminal (bash)

approval prompt


Each specialist runs as a subprocess, talks A2A over stdio, and spawns
its own MCP tool server on demand. See `shell_host.py` for the exact
list of commands that trigger the approval prompt.

---

## Requirements

- Ubuntu 22.04+
- Python 3.11+
- A Groq API key (free tier)
- `arecord`, `aplay` (usually preinstalled)
- `ghostty`, `gnome-terminal` or `x-terminal-emulator`

Optional, for voice:

- [Piper](https://github.com/rhasspy/piper) for text-to-speech
- Whisper `base.en` (auto-downloads on first use)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/zoya.git
cd zoya

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."
python main.py


# Piper voice
mkdir -p ~/.local/share/piper
cd ~/.local/share/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd -

Usage
Type a message. Some examples:
how much RAM is free?
list files in my Downloads folder
why is mysqld running?
open spotify
delete /tmp/scratch.txt

When the Verifier needs to run a command, it opens a real terminal.
Dangerous commands show a Y/N prompt there. You approve or deny.

Voice

The first line of the reply is
spoken back.

Desktop launcher

cp zoya.sh.example zoya.sh
chmod +x zoya.sh
# edit zoya.sh and put your key

Create ~/.local/share/applications/zoya.desktop

[Desktop Entry]
Type=Application
Name=Zoya
Exec=/absolute/path/to/zoya.sh
Path=/absolute/path/to/zoya
Icon=utilities-terminal
Terminal=false
Categories=Development;

update-desktop-database ~/.local/share/applications/

Press Super, search "Zoya".

Project layout

main.py                  Qt UI, chat, voice
ask_llm.py               Planner + A2A client
a2a_stdio.py             Local JSON-RPC transport

infra_agent.py           Read-only system agent
investigator_agent.py    Deeper forensics + web agent
verifier_agent.py        Action agent

custom_tools.py          Infra tools
investigator_tools.py    Investigator tools
verifier_tools.py        Verifier tools (terminal sessions)

tool_server.py           Generic MCP tool server
shell_host.py            Persistent terminal host
ops_terminal.py          Live ops log viewer
zoya_live.py             Voice in/out
approval_store.py        Session permission cache

record.sh                Mic capture helper
zoya.sh.example          Launcher template
tests/                   Diagnostic scripts

Safety
The Verifier is the only mutating agent. It runs commands inside a
real terminal window on your desktop. Dangerous commands are gated by
a Y/N prompt right there. Nothing executes silently. Close the
terminal and the next request opens a new one.


Known limitations
Voice is English-only.

Groq free tier has strict daily token limits.

Linux only.

License
MIT
