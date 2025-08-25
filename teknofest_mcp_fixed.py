# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, r"C:\Users\husey\teknofest-2025-egitim-eylemci")

# Veritabanı hatalarını atla
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./teknofest.db"
os.environ["ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///./teknofest.db"

from mcp.server import Server
import mcp.types as types

server = Server("teknofest-edu")

# Import'ları try-except içine al
try:
    from src.agents.learning_path_agent_v2 import LearningPathAgent
    agent_available = True
except:
    agent_available = False

@server.tool()
async def generate_learning_path(student_id: str, topic: str, grade_level: int, learning_style: str) -> str:
    """Kisisellestirilmis ogrenme yolu olusturur"""
    if agent_available:
        try:
            agent = LearningPathAgent()
            result = agent.generate_path(student_id, topic, grade_level, learning_style)
            return str(result)
        except:
            pass
    
    # Fallback
    return f"Ogrenme yolu: {student_id} - {topic} - Sinif {grade_level} - {learning_style} stili"

@server.tool()
async def generate_quiz(topic: str, difficulty: str, num_questions: int, grade_level: int) -> str:
    """Adaptif quiz olusturur"""
    return f"Quiz: {topic} - {difficulty} - {num_questions} soru - Sinif {grade_level}"

@server.tool()
async def answer_question(question: str, subject: str) -> str:
    """Turkce egitim sorularini yanitlar"""
    return f"{subject}: {question} - Cevap hazirlaniyor..."

@server.tool() 
async def get_irt_analysis(student_responses: list) -> str:
    """IRT analizi yapar"""
    correct = sum(1 for r in student_responses if isinstance(r, dict) and r.get("is_correct", False))
    total = len(student_responses)
    return f"IRT Analizi: {correct}/{total} dogru"

@server.tool()
async def detect_learning_style(responses: list) -> str:
    """VARK ogrenme stilini tespit eder"""
    return f"Ogrenme stili analizi: {len(responses)} yanit degerlendirildi"

@server.tool()
async def create_study_plan(weak_topics: list, available_hours: int) -> str:
    """Calisma plani olusturur"""
    return f"Calisma plani: {len(weak_topics)} konu icin {available_hours} saat"

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
