import asyncio
import json
import os
import shlex
import sys
from typing import Any, AsyncGenerator

from google.protobuf import json_format

from a2a.client import ClientCallContext
from a2a.client.transports.base import ClientTransport

from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    StreamResponse,
    GetTaskRequest,
    ListTasksRequest,
    ListTasksResponse,
    CancelTaskRequest,
    Task,
    TaskPushNotificationConfig,
    GetTaskPushNotificationConfigRequest,
    ListTaskPushNotificationConfigsRequest,
    SubscribeToTaskRequest,
    GetExtendedAgentCardRequest,
)


# ---------------------------------------------------------------------------
# Protobuf helpers
# ---------------------------------------------------------------------------

def proto_to_dict(message) -> dict:
    """Convert an A2A protobuf message into JSON-compatible dict."""
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
    )


def dict_to_proto(data: dict, message_type):
    """Convert a dict into an A2A protobuf message."""
    message = message_type()
    json_format.ParseDict(data, message)
    return message


# ---------------------------------------------------------------------------
# A2A stdio transport
# ---------------------------------------------------------------------------

class StdioTransport(ClientTransport):
    """
    Local A2A transport using a child process.

    Planner
        |
        | stdin/stdout
        v
    Agent process
    """

    def __init__(
        self,
        agent_card: AgentCard,
        command: list[str],
        env: dict[str, str] | None = None,
    ):
        self.agent_card = agent_card
        self.command = command
        self.env = env

        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    async def _ensure_process(self):
        if self.process is not None:
            if self.process.returncode is None:
                return

        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            env=self.env,
        )

    # ------------------------------------------------------------------
    # Low-level request/response
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:

        await self._ensure_process()

        if self.process is None:
            raise RuntimeError("Agent process was not started")

        if self.process.stdin is None:
            raise RuntimeError("Agent stdin unavailable")

        if self.process.stdout is None:
            raise RuntimeError("Agent stdout unavailable")

        async with self._lock:

            self._request_id += 1
            request_id = str(self._request_id)

            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }

            line = json.dumps(
                request,
                separators=(",", ":"),
            ) + "\n"

            self.process.stdin.write(line.encode())
            await self.process.stdin.drain()

            while True:

                raw = await self.process.stdout.readline()

                if not raw:
                    raise RuntimeError(
                        "Agent process closed stdout"
                    )

                try:
                    response = json.loads(raw.decode())
                except json.JSONDecodeError:
                    # Ignore non-JSON lines for now.
                    continue

                # Ignore notifications / unrelated messages.
                if str(response.get("id")) != request_id:
                    continue

                if "error" in response:
                    raise RuntimeError(
                        f"A2A agent error: {response['error']}"
                    )

                return response.get("result", {})

    # ------------------------------------------------------------------
    # A2A message API
    # ------------------------------------------------------------------

    async def send_message(
        self,
        request: SendMessageRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> SendMessageResponse:

        result = await self._request(
            "SendMessage",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            SendMessageResponse,
        )

    async def send_message_streaming(
        self,
        request: SendMessageRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> AsyncGenerator[StreamResponse, None]:

        result = await self._request(
            "SendStreamingMessage",
            proto_to_dict(request),
        )

        yield dict_to_proto(
            result,
            StreamResponse,
        )

    # ------------------------------------------------------------------
    # Task API
    # ------------------------------------------------------------------

    async def get_task(
        self,
        request: GetTaskRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> Task:

        result = await self._request(
            "GetTask",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            Task,
        )

    async def list_tasks(
        self,
        request: ListTasksRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> ListTasksResponse:

        result = await self._request(
            "ListTasks",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            ListTasksResponse,
        )

    async def cancel_task(
        self,
        request: CancelTaskRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> Task:

        result = await self._request(
            "CancelTask",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            Task,
        )

    # ------------------------------------------------------------------
    # Push notification API
    # ------------------------------------------------------------------

    async def create_task_push_notification_config(
        self,
        request: TaskPushNotificationConfig,
        *,
        context: ClientCallContext | None = None,
    ) -> TaskPushNotificationConfig:

        result = await self._request(
            "CreateTaskPushNotificationConfig",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            TaskPushNotificationConfig,
        )

    async def get_task_push_notification_config(
        self,
        request: GetTaskPushNotificationConfigRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> TaskPushNotificationConfig:

        result = await self._request(
            "GetTaskPushNotificationConfig",
            proto_to_dict(request),
        )

        return dict_to_proto(
            result,
            TaskPushNotificationConfig,
        )

    async def list_task_push_notification_configs(
        self,
        request: ListTaskPushNotificationConfigsRequest,
        *,
        context: ClientCallContext | None = None,
    ):

        result = await self._request(
            "ListTaskPushNotificationConfigs",
            proto_to_dict(request),
        )

        return result

    async def delete_task_push_notification_config(
        self,
        request,
        *,
        context: ClientCallContext | None = None,
    ) -> None:

        await self._request(
            "DeleteTaskPushNotificationConfig",
            proto_to_dict(request),
        )

    # ------------------------------------------------------------------
    # Subscription API
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        request: SubscribeToTaskRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> AsyncGenerator[StreamResponse, None]:

        result = await self._request(
            "SubscribeToTask",
            proto_to_dict(request),
        )

        yield dict_to_proto(
            result,
            StreamResponse,
        )

    # ------------------------------------------------------------------
    # Agent card
    # ------------------------------------------------------------------

    async def get_extended_agent_card(
        self,
        request: GetExtendedAgentCardRequest,
        *,
        context: ClientCallContext | None = None,
    ) -> AgentCard:

        # For our local transport the card is already known.
        return self.agent_card

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:

        if self.process is None:
            return

        if self.process.returncode is None:

            self.process.terminate()

            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=2,
                )

            except asyncio.TimeoutError:

                self.process.kill()
                await self.process.wait()

        self.process = None


# ---------------------------------------------------------------------------
# ClientFactory transport producer
# ---------------------------------------------------------------------------

def stdio_transport(
    card: AgentCard,
    url: str,
    config,
) -> StdioTransport:

    """
    Transport producer used by ClientFactory.

    `url` contains the command used to start the agent.

    Example:

        python infra_agent.py
    """

    command = shlex.split(url)

    return StdioTransport(
        agent_card=card,
        command=command,
        env=os.environ.copy(),
    )