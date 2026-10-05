import asyncio

from a2a_stdio import StdioTransport
from a2a.types import AgentCard, Message, Part, SendMessageRequest


async def main():

    transport = StdioTransport(
        agent_card=AgentCard(),
        command=["python", "test_stdio_agent.py"],
    )

    try:
        message = Message()
        message.message_id = "test-001"
        message.role = 1  # user

        part = message.parts.add()
        part.text = "Investigate high CPU usage"

        request = SendMessageRequest()
        request.message.CopyFrom(message)

        response = await transport.send_message(request)

        print("\n=== A2A RESPONSE ===")
        print(response)

    finally:
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())