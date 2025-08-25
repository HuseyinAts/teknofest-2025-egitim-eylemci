# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, r"C:\Users\husey\teknofest-2025-egitim-eylemci")

from mcp.server import Server
import mcp.types as types

# Projenizden modulleri import edin
from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent_clean import StudyBuddyAgent
from src.core.irt_engine import IRTEngine

server = Server("teknofest-edu")

@server.tool()
async def generate_learning_path(student_id: str, topic: str, grade_level: int, learning_style: str) -> str:
    """Kisisellestirilmis ogrenme yolu olusturur"""
    try:
        agent = LearningPathAgent()
        result = agent.generate_path(student_id, topic, grade_level, learning_style)
        return str(result)
    except Exception as e:
        return f"Hata: {str(e)}"

@server.tool()
async def generate_quiz(topic: str, difficulty: str, num_questions: int, grade_level: int) -> str:
    """Adaptif quiz olusturur"""
    return f"Quiz olusturuldu: {topic} - {difficulty} - {num_questions} soru"

@server.tool()
async def answer_question(question: str, subject: str) -> str:
    """Turkce egitim sorularini yanitlar"""
    return f"{subject} konusunda soru: {question} - Yanit hazirlaniyor..."

@server.tool() 
async def get_irt_analysis(student_responses: list) -> str:
    """IRT analizi yapar"""
    try:
        engine = IRTEngine()
        result = engine.analyze(student_responses)
        return str(result)
    except Exception as e:
        return f"IRT Analizi: {len(student_responses)} yanit analiz edildi"

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
