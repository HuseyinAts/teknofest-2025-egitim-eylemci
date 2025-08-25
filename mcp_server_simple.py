import sys
sys.path.insert(0, r"C:\Users\husey\teknofest-2025-egitim-eylemci")

from mcp import Server
import asyncio

server = Server("teknofest-edu")

@server.tool()
async def test_tool(message: str) -> str:
    return f"Test response: {message}"

if __name__ == "__main__":
    asyncio.run(server.run())
