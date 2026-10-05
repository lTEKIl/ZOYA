import asyncio
from a2a_stdio import StdioTransport
from a2a.types import AgentCard


async def main():
    card = AgentCard()

    transport = StdioTransport(
        agent_card=card,
        command=["python", "test_stdio_agent.py"],
    )

    try:
        result = await transport._request(
            "Ping",
            {"message": "hello from client"}
        )

        print("RESPONSE:")
        print(result)

    finally:
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())