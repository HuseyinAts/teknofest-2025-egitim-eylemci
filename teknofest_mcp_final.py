# -*- coding: utf-8 -*-
import sys
import os
import json
import asyncio
sys.path.insert(0, r"C:\Users\husey\teknofest-2025-egitim-eylemci")

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Veritabani hatalarini atla
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./teknofest.db"
os.environ["ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///./teknofest.db"

async def run():
    server = Server("teknofest-edu")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="generate_learning_path",
                description="Kisisellestirilmis ogrenme yolu olusturur",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "topic": {"type": "string"},
                        "grade_level": {"type": "integer"},
                        "learning_style": {"type": "string"}
                    },
                    "required": ["student_id", "topic", "grade_level", "learning_style"]
                }
            ),
            types.Tool(
                name="generate_quiz",
                description="Adaptif quiz olusturur",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "num_questions": {"type": "integer"},
                        "grade_level": {"type": "integer"}
                    },
                    "required": ["topic", "difficulty", "num_questions", "grade_level"]
                }
            ),
            types.Tool(
                name="answer_question",
                description="Turkce egitim sorularini yanitlar",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "subject": {"type": "string"}
                    },
                    "required": ["question", "subject"]
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "generate_learning_path":
            result = {
                "student_id": arguments["student_id"],
                "topic": arguments["topic"],
                "grade_level": arguments["grade_level"],
                "learning_style": arguments["learning_style"],
                "path": [
                    {"step": 1, "content": f"{arguments['topic']} - Temel Kavramlar"},
                    {"step": 2, "content": f"{arguments['topic']} - Uygulamalar"},
                    {"step": 3, "content": f"{arguments['topic']} - Ileri Seviye"}
                ]
            }
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        
        elif name == "generate_quiz":
            questions = []
            for i in range(arguments["num_questions"]):
                questions.append({
                    "id": i+1,
                    "question": f"{arguments['topic']} ile ilgili {i+1}. soru",
                    "difficulty": arguments["difficulty"]
                })
            return [types.TextContent(type="text", text=json.dumps(questions, ensure_ascii=False))]
        
        elif name == "answer_question":
            answer = f"{arguments['subject']}: {arguments['question']} - Cevap hazirlaniyor..."
            return [types.TextContent(type="text", text=answer)]
        
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="teknofest-edu",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(run())
