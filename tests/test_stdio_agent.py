import json
import sys


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)

            print(
                f"Agent received: {request['method']}",
                file=sys.stderr,
            )

            params = request["params"]

            # Echo the A2A request back as a response.
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "message": params["message"]
                },
            }

            print(json.dumps(response), flush=True)

        except Exception as e:
            print(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -1,
                        "message": str(e),
                    },
                }),
                flush=True,
            )


if __name__ == "__main__":
    main()