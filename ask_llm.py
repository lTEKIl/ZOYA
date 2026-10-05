import asyncio
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from groq import Groq

from a2a.types import (
    AgentCard,
    Message,
    SendMessageRequest,
)

from a2a_stdio import StdioTransport


# ============================================================
# CONFIG
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

MODEL = "openai/gpt-oss-120b"

groq = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = "chat_history.json"

KEEP = 8
MAX_MSGS = 16

OPS_LOG = os.path.abspath("runs/infra_ops.log")


# ============================================================
# ZOYA SYSTEM PROMPT
# ============================================================

SYSTEM = """
You are Zoya, the user's AI assistant.

You are the single assistant the user interacts with.

You may internally use specialist agents such as:

- Infra/Ops Agent
- Investigator/RAG Agent
- Verifier/Action Agent

These agents are INTERNAL implementation details.

NEVER expose internal orchestration to the user.

Do NOT say things like:

- "I'm delegating this to the Infra/Ops Agent."
- "I'll ask the Infra Agent."
- "I'll forward this to the Infra Agent."
- "The Infra Agent will..."
- "The Investigator will..."
- "The specialist agent..."
- "Delegating..."
- "The agent is investigating..."
- "I have delegated your request."

Instead, silently perform the required internal work and answer
the user naturally.

The user should experience Zoya as one assistant.

If a specialist returns information, use that information as
internal evidence and answer the original user question directly.

Do not invent information that is not present in the specialist
result.

If an internal investigation fails, explain the failure naturally
without exposing unnecessary internal architecture.

For machine/system questions, use the available investigation
results rather than guessing.

For normal conversational or general knowledge questions, answer
directly without unnecessary delegation.

When multiple internal investigation results are provided, combine
them carefully and do not assume that one result proves something
that another result does not establish.

If you are asked a machine-specific question (paths, processes,
files, system state) and NO internal investigation results are
provided, do NOT guess. Say you were unable to determine the
answer and that the internal services may be unavailable.

ACTION RESULTS:

When an internal verifier result is present, it is the AUTHORITATIVE
record of what actually happened on the machine. Report its outcome
truthfully. If the verifier says a command failed, was denied,
timed out, or produced an error, do NOT claim the action succeeded.

Never invent an execution result.

============================================================
SPOKEN SUMMARY RULE
============================================================

Every response MUST begin with a single-line summary that
stands on its own. Then a blank line. Then the full answer.

The first line is spoken aloud when voice mode is on. It must
contain the actual answer, not a preamble.

Format:

    <few-line/s summary>

    <full answer, formatted however is most useful>

Example (good):

    There are 42 files in your Downloads folder. (its oke if it is 2-3 lines... completly absed on the expression )

    Here is the list:
    - file1.pdf
    - file2.pdf
    ...

Example (bad — do NOT do this):

    Sure! Let me help you with that. Here are the files in
    your Downloads folder: ...
    (no blank line, no standalone summary line)

Rules:
- The first or first few  line/s must be a complete sentence.
- The first or first few  line/s must contain the answer.
- The first or first few  line/s must NOT contain markdown, code, or a list.
- A blank line must follow the first or first few  line/s.
"""


# ============================================================
# CHAT HISTORY
# ============================================================

def _load():
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# INTERNAL PLANNER EVENT LOG
# ============================================================

_PLANNER_EVENTS = {}
_PLANNER_EVENTS_LOCK = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _create_request():
    request_id = str(uuid.uuid4())

    with _PLANNER_EVENTS_LOCK:
        _PLANNER_EVENTS[request_id] = []

    return request_id


def _add_planner_event(request_id, event_type, **data):
    event = {
        "timestamp": _now(),
        "request_id": request_id,
        "event_type": event_type,
        **data,
    }

    with _PLANNER_EVENTS_LOCK:
        _PLANNER_EVENTS.setdefault(request_id, []).append(event)

    return event


def get_planner_events(request_id):
    with _PLANNER_EVENTS_LOCK:
        return list(_PLANNER_EVENTS.get(request_id, []))


# ============================================================
# SHARED FILE OPS LOG
# ============================================================

def _ops_log(message):
    try:
        os.makedirs(os.path.dirname(OPS_LOG), exist_ok=True)

        timestamp = datetime.now().strftime("%H:%M:%S")

        with open(OPS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()

    except Exception:
        pass


# ============================================================
# BASE A2A AGENT CLIENT
# ============================================================

class AgentClient:

    def __init__(self, agent_name, script_name):

        self.agent_name = agent_name
        self.script_name = script_name

        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self.thread.start()

        self.transport = None
        self._ready = threading.Event()

        self._init_future = asyncio.run_coroutine_threadsafe(
            self._initialize(),
            self.loop,
        )
        self._init_future.result()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _initialize(self):

        self.transport = StdioTransport(
            agent_card=AgentCard(),
            command=[sys.executable, self.script_name],
        )

        await self.transport._ensure_process()
        self._ready.set()

        _ops_log(
            f"{self.agent_name.upper()} A2A transport initialized."
        )

    async def _send_async(self, task, request_id):

        _add_planner_event(
            request_id,
            "agent_request",
            agent=self.agent_name,
            task=task,
        )

        _ops_log(
            f"▶ PLANNER → {self.agent_name.upper()}: {task}"
        )

        message = Message()
        message.message_id = str(uuid.uuid4())
        message.role = 1

        part = message.parts.add()
        part.text = task

        request = SendMessageRequest()
        request.message.CopyFrom(message)

        _add_planner_event(
            request_id,
            "a2a_send",
            agent=self.agent_name,
            message_id=message.message_id,
        )

        try:
            response = await self.transport.send_message(request)

        except Exception as e:
            _add_planner_event(
                request_id,
                "error",
                agent=self.agent_name,
                message=str(e),
            )

            _ops_log(
                f"✗ {self.agent_name.upper()} ERROR: {e}"
            )

            raise

        texts = [
            part.text
            for part in response.message.parts
            if part.text
        ]

        result = "\n".join(texts).strip()

        _add_planner_event(
            request_id,
            "agent_response",
            agent=self.agent_name,
            result=result,
        )

        _ops_log(
            f"✓ {self.agent_name.upper()} RESPONSE RECEIVED"
        )

        return result

    def send(self, task, request_id):

        future = asyncio.run_coroutine_threadsafe(
            self._send_async(task, request_id),
            self.loop,
        )
        return future.result()

    async def _close_async(self):
        if self.transport:
            await self.transport.close()

    def close(self):

        if not self.loop.is_running():
            return

        future = asyncio.run_coroutine_threadsafe(
            self._close_async(),
            self.loop,
        )
        future.result()

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


# ============================================================
# AGENT CLIENTS
# ============================================================

infra_agent = AgentClient(
    agent_name="infra",
    script_name="infra_agent.py",
)

investigator_agent = AgentClient(
    agent_name="investigator",
    script_name="investigator_agent.py",
)

verifier_agent = AgentClient(
    agent_name="verifier",
    script_name="verifier_agent.py",
)


# ============================================================
# PLANNER JSON SCHEMA
# ============================================================

PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "delegate": {"type": "boolean"},
        "agents": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "infra",
                    "investigator",
                    "verifier",
                ],
            },
        },
        "tasks": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "infra": {"type": ["string", "null"]},
                "investigator": {"type": ["string", "null"]},
                "verifier": {"type": ["string", "null"]},
            },
            "required": [
                "infra",
                "investigator",
                "verifier",
            ],
        },
    },
    "required": [
        "delegate",
        "agents",
        "tasks",
    ],
}


# ============================================================
# INTERNAL PLANNER / ROUTER
# ============================================================

def _route_question(question, chat):

    routing_prompt = """
You are the INTERNAL PLANNER of Zoya.

Your ONLY job is to route the user's CURRENT request to the
appropriate internal specialist.

You do NOT answer the user.

You do NOT investigate anything yourself.

You do NOT perform web searches.

You do NOT call tools.

You ONLY create a routing plan.

============================================================
AVAILABLE INTERNAL AGENTS
============================================================

1. infra

Use "infra" for current machine operational information such as:

- CPU
- memory
- disk
- system performance
- running processes
- process usage
- process trees
- network connections
- listening ports
- services
- applications
- system status
- battery
- GPU
- temperature
- audio
- recent system logs

Infra is the operational/system-observation specialist.

------------------------------------------------------------

2. investigator

Use "investigator" for deeper investigation such as:

- filesystem investigation
- suspicious files
- forensic analysis
- binary analysis
- process internals
- executable locations
- open files
- process environment
- memory maps
- persistence
- cron
- systemd timers
- startup entries
- permissions
- SUID/SGID
- sudo configuration
- users
- security configuration
- packages
- installed software
- kernel information
- kernel logs
- network investigation
- DNS
- routes
- interfaces
- Git repositories
- configuration
- read-only Linux commands
- searching files
- searching text
- external research
- web searches
- current information from the public internet
- news
- announcements
- release dates
- current software versions
- current public information
- deeper root-cause investigation

IMPORTANT:

The Investigator has the web-search capability.

Therefore, ANY request requiring external/current web information
must be routed to "investigator".

------------------------------------------------------------

3. verifier

Use "verifier" for ANY request to CHANGE the state of the local
machine:

- kill / terminate / stop a process
- start / restart a process
- start / stop / restart / enable / disable a service
- delete / move / copy / rename a file or directory
- create a directory or file
- write text to a file
- change file permissions or ownership
- install / remove / upgrade a package
- add / remove a firewall rule
- enable / disable a network interface
- add / replace / delete a route
- reboot / shutdown / suspend the system
- execute any shell command
- run anything with sudo

Also use "verifier" for:

- Questions about the current working directory, current user,
  or the state of a live shell: "where am I", "what is pwd",
  "which directory am I in", "whoami", "what is my shell"
- Anything that requires running a command in a terminal

Do NOT route "where am I" or "what is pwd" to infra.
Infra observes machine state read-only. Only the verifier has
a live shell that can answer shell-state questions.

The verifier has a real terminal session. It executes commands
there. The user sees them live and can approve or deny each one.

IMPORTANT ROUTING RULES:

- Do NOT route read-only investigation to the verifier.
- Do NOT route web research to the verifier.
- Do NOT route normal conversation to the verifier.
- For "why is X running" or "what is X" questions, route to
  investigator (they can inspect process internals).
- For "kill X" or "stop X" or "restart X", route to verifier.
- For "kill the process using the most CPU", route to infra
  first to identify the PID, then verifier to act on it.

When creating a verifier task, describe the ACTION you want in
plain language. Do not invent tool names or argument names.
Examples:

User: "open spotify"
Route:
    agents = ["verifier"]
    tasks.verifier =
        "Launch the Spotify application."

User: "kill the chrome process eating my CPU"
Route:
    agents = ["infra", "verifier"]
    tasks.infra =
        "Find the Chrome process with the highest CPU usage.
         Report the PID."
    tasks.verifier =
        "Kill the Chrome process reported by infra."

User: "restart docker"
Route:
    agents = ["verifier"]
    tasks.verifier =
        "Restart the docker service."

User: "create hello.py in documents with print('hello world')"
Route:
    agents = ["verifier"]
    tasks.verifier =
        "Create a file named hello.py in the user's Documents
         directory containing the text: print('hello world')"

============================================================
WEB / EXTERNAL RESEARCH
============================================================

The Planner itself has NO web access.

NEVER attempt to search the web.

Instead, if the user asks for web research, create an
"investigator" task describing exactly what needs to be searched.

============================================================
WHEN TO USE MULTIPLE AGENTS
============================================================

Use both "infra" and "investigator" when the request benefits
from current operational state AND deeper investigation.

Use both "infra" and "verifier" when the user wants a mutation
on a target that must first be identified (a PID, a service
name, an absolute path).

============================================================
CONVERSATION CONTEXT
============================================================

The user's current message may be a continuation or confirmation.

Examples:

Assistant: "Would you like me to check VS Code's CPU and memory usage?"
User: "yes"
→ The user accepted the previous request.

Likewise: yes, yeah, do it, go ahead, please, check it, do that,
sure, try now — may be confirmations of a pending request.

Use recent conversation context to determine what the confirmation
refers to.

============================================================
IMPORTANT ROUTING RULES
============================================================

Do not perform the investigation yourself.

Only create the investigation tasks.

Tasks must be concrete and useful.

Do not expose internal planning language to the user.

If the user message is a fragment, a single word, a greeting,
or has no clear actionable intent, set delegate=false and
return the empty plan.

Return ONLY the structured routing object requested by the schema.
"""

    messages = [
        {"role": "system", "content": routing_prompt}
    ]

    messages.extend(chat.get("messages", [])[-KEEP:])
    messages.append({"role": "user", "content": question})

    messages.append(
        {
            "role": "system",
            "content": (
                "Respond with ONLY a JSON object matching the schema. "
                "Do not add any prose. Do not apologize. Do not refuse. "
                "If the user message is unclear, empty, a greeting, or "
                "a fragment, set delegate=false and return:\n"
                '{"delegate": false, "agents": [], '
                '"tasks": {"infra": null, "investigator": null, '
                '"verifier": null}}'
            ),
        }
    )

    try:
        response = groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "zoya_routing_plan",
                    "strict": True,
                    "schema": PLANNER_SCHEMA,
                },
            },
        )
    except Exception as e:
        _ops_log(f"⚠ PLANNER SCHEMA FAILURE: {e}")
        return {
            "delegate": False,
            "agents": [],
            "tasks": {
                "infra": None,
                "investigator": None,
                "verifier": None,
            },
        }

    raw = response.choices[0].message.content.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        _ops_log(f"⚠ PLANNER JSON PARSE FAILURE: {e}")
        return {
            "delegate": False,
            "agents": [],
            "tasks": {
                "infra": None,
                "investigator": None,
                "verifier": None,
            },
        }

    if not isinstance(plan, dict):
        raise ValueError("Planner returned a non-object routing plan.")

    allowed_agents = {"infra", "investigator", "verifier"}

    requested_agents = plan.get("agents", [])
    if not isinstance(requested_agents, list):
        requested_agents = []

    agents = [a for a in requested_agents if a in allowed_agents]

    tasks = plan.get("tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}

    infra_task = tasks.get("infra")
    investigator_task = tasks.get("investigator")
    verifier_task = tasks.get("verifier")

    if not isinstance(infra_task, str):
        infra_task = None
    if not isinstance(investigator_task, str):
        investigator_task = None
    if not isinstance(verifier_task, str):
        verifier_task = None

    final_agents = []
    if "infra" in agents and infra_task:
        final_agents.append("infra")
    if "investigator" in agents and investigator_task:
        final_agents.append("investigator")
    if "verifier" in agents and verifier_task:
        final_agents.append("verifier")

    return {
        "delegate": bool(final_agents),
        "agents": final_agents,
        "tasks": {
            "infra":
                infra_task if "infra" in final_agents else None,
            "investigator":
                investigator_task if "investigator" in final_agents else None,
            "verifier":
                verifier_task if "verifier" in final_agents else None,
        },
    }


# ============================================================
# STREAM FINAL ANSWER
# ============================================================

def _stream_final_answer(messages, temperature=0.2):

    stream = groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for event in stream:

        if not event.choices:
            continue

        delta = event.choices[0].delta
        if not delta:
            continue

        content = delta.content
        if content:
            time.sleep(0.03)
            yield content


# ============================================================
# MAIN ASK FUNCTION - STREAMING
# ============================================================

def ask_groq_stream(
    question,
    chat_id,
    system=SYSTEM,
):
    data = _load()

    chat = data.setdefault(
        str(chat_id),
        {"summary": "", "messages": []},
    )

    request_id = _create_request()

    _add_planner_event(request_id, "planner_request", question=question)

    _ops_log("=" * 70)
    _ops_log(f"▶ PLANNER REQUEST: {question}")

    try:
        plan = _route_question(question, chat)
    except Exception as e:
        _add_planner_event(request_id, "planner_error", message=str(e))
        _ops_log(f"✗ PLANNER ERROR: {e}")

        answer = "I encountered an error while processing the request: " + str(e)

        chat["messages"].append({"role": "user", "content": question})
        chat["messages"].append({"role": "assistant", "content": answer})
        chat["messages"] = chat["messages"][-MAX_MSGS:]

        _save(data)
        yield answer
        return

    _add_planner_event(request_id, "planner_plan", plan=plan)
    _ops_log(f"→ PLANNER PLAN: {json.dumps(plan, ensure_ascii=False)}")

    # ========================================================
    # DIRECT ANSWER
    # ========================================================

    if not plan["delegate"]:

        messages = [{"role": "system", "content": system}]
        messages.extend(chat.get("messages", [])[-KEEP:])
        messages.append({"role": "user", "content": question})

        full_answer = ""

        try:
            for chunk in _stream_final_answer(messages, temperature=0.2):
                full_answer += chunk
                yield chunk
        except Exception as e:
            full_answer = (
                "I encountered an error while generating the response: "
                + str(e)
            )
            yield full_answer

    # ========================================================
    # INTERNAL INVESTIGATION
    # ========================================================

    else:

        tasks = plan["tasks"]

        infra_result = None
        investigator_result = None
        verifier_result = None

        if "infra" in plan["agents"]:

            task = tasks["infra"]
            _add_planner_event(
                request_id, "delegation", agent="infra", task=task,
            )

            try:
                infra_result = infra_agent.send(task, request_id)
            except Exception as e:
                _add_planner_event(request_id, "infra_error", message=str(e))
                infra_result = "Infra investigation failed: " + str(e)

        if "investigator" in plan["agents"]:

            task = tasks["investigator"]
            _add_planner_event(
                request_id, "delegation", agent="investigator", task=task,
            )

            try:
                investigator_result = investigator_agent.send(task, request_id)
            except Exception as e:
                _add_planner_event(
                    request_id, "investigator_error", message=str(e),
                )
                investigator_result = (
                    "Investigator investigation failed: " + str(e)
                )

        if "verifier" in plan["agents"]:

            task = tasks["verifier"]
            _add_planner_event(
                request_id, "delegation", agent="verifier", task=task,
            )

            verifier_task = task

            if infra_result:
                verifier_task = (
                    task
                    + "\n\n"
                    + "INTERNAL CONTEXT FROM INFRA INVESTIGATION "
                    + "(target identification only):\n"
                    + "------------------------------------------\n"
                    + infra_result
                    + "\n"
                    + "------------------------------------------"
                )

            try:
                verifier_result = verifier_agent.send(verifier_task, request_id)
            except Exception as e:
                _add_planner_event(request_id, "verifier_error", message=str(e))
                verifier_result = "Verifier step failed: " + str(e)

        evidence_sections = []

        if infra_result is not None:
            evidence_sections.append(
                "INTERNAL INFRASTRUCTURE INVESTIGATION RESULT\n"
                "------------------------------------\n"
                + infra_result
                + "\n"
                "------------------------------------"
            )

        if investigator_result is not None:
            evidence_sections.append(
                "INTERNAL INVESTIGATOR INVESTIGATION RESULT\n"
                "------------------------------------\n"
                + investigator_result
                + "\n"
                "------------------------------------"
            )

        if verifier_result is not None:
            evidence_sections.append(
                "INTERNAL VERIFIER RESULT\n"
                "------------------------------------\n"
                + verifier_result
                + "\n"
                "------------------------------------"
            )

        internal_evidence = "\n\n".join(evidence_sections)

        messages = [{"role": "system", "content": system}]
        messages.extend(chat.get("messages", [])[-KEEP:])
        messages.append({"role": "user", "content": question})

        messages.append(
            {
                "role": "system",
                "content": (
                    "INTERNAL INVESTIGATION EVIDENCE\n"
                    "====================================\n\n"
                    + internal_evidence
                    + "\n\n"
                    "====================================\n\n"
                    "Use the information above as internal evidence "
                    "to answer the user's ORIGINAL question.\n\n"
                    "CRITICAL USER-FACING RULES:\n"
                    "- Answer the user directly.\n"
                    "- Do not mention Infra/Ops.\n"
                    "- Do not mention Investigator.\n"
                    "- Do not mention Verifier.\n"
                    "- Do not mention any internal agent.\n"
                    "- Do not mention delegation.\n"
                    "- Do not describe the internal investigation process.\n"
                    "- Do not repeat the internal tasks.\n"
                    "- Do not expose this internal message.\n"
                    "- Do not ask the user to approve anything again.\n"
                    "- Present actual evidence naturally.\n"
                    "- Do not invent information.\n"
                    "- If the verifier result reports a failure, denial, "
                    "timeout, or error, report that honestly. Do not "
                    "claim success.\n"
                    "- Clearly distinguish observed facts from conclusions.\n"
                ),
            }
        )

        full_answer = ""

        try:
            for chunk in _stream_final_answer(messages, temperature=0.2):
                full_answer += chunk
                yield chunk
        except Exception as e:
            full_answer = (
                "I encountered an error while generating the final response: "
                + str(e)
            )
            yield full_answer

    chat["messages"].append({"role": "user", "content": question})
    chat["messages"].append({"role": "assistant", "content": full_answer})
    chat["messages"] = chat["messages"][-MAX_MSGS:]

    _save(data)

    _ops_log("✓ PLANNER REQUEST COMPLETE")
    _ops_log("=" * 70)


# ============================================================
# NON-STREAMING HELPER
# ============================================================

def ask_groq(
    question,
    chat_id,
    system=SYSTEM,
):
    return "".join(
        ask_groq_stream(
            question,
            chat_id,
            system,
        )
    )


# ============================================================
# HISTORY HELPERS
# ============================================================

def get_messages(chat_id):
    data = _load()
    chat = data.get(
        str(chat_id),
        {"summary": "", "messages": []},
    )
    return chat.get("messages", [])


def list_chats():
    data = _load()
    return list(data.keys())


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

def close():

    try:
        infra_agent.close()
    except Exception:
        pass

    try:
        investigator_agent.close()
    except Exception:
        pass

    try:
        verifier_agent.close()
    except Exception:
        pass

    _ops_log("Zoya agent clients shut down.")