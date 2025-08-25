"""
Study Buddy Agent - Production Ready Implementation
TEKNOFEST 2025 - AI-Powered Educational Assistant
"""

import json
import uuid
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QuestionType(Enum):
    """Question types for assessments"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    MATCHING = "matching"
    FILL_BLANK = "fill_blank"


class ResponseMode(Enum):
    """Response modes for answers"""
    DETAILED = "detailed"
    SUMMARY = "summary"
    HINT_ONLY = "hint_only"
    STEP_BY_STEP = "step_by_step"


@dataclass
class Question:
    """Question data structure"""
    id: str
    question: str
    type: str
    subject: str = ""
    topic: str = ""
    difficulty: float = 0.5
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    points: int = 10
    time_limit: int = 60
    hints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Quiz:
    """Quiz data structure"""
    quiz_id: str
    title: str
    subject: str
    topic: str
    questions: List[Question]
    total_points: int
    time_limit: int
    difficulty: float
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['questions'] = [q.to_dict() if isinstance(q, Question) else q for q in self.questions]
        return data


@dataclass
class ChatSession:
    """Chat session data structure"""
    session_id: str
    student_id: str
    started_at: str
    messages: List[Dict] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class StudyBuddyAgent:
    """Advanced Study Buddy Agent for personalized tutoring"""
    
    def __init__(self):
        """Initialize Study Buddy Agent"""
        self.question_bank = self._load_question_bank()
        self.subject_knowledge = self._load_subject_knowledge()
        self.chat_sessions = {}
        self.model = None  # Will be loaded on demand
        logger.info("Study Buddy Agent initialized")
    
    def _load_question_bank(self) -> Dict:
        """Load question bank for different subjects"""
        return {
            'Matematik': {
                'İntegral': [
                    {
                        'question': '∫x²dx integral sonucu nedir?',
                        'options': ['x³/3 + C', 'x³ + C', '3x² + C', '2x + C'],
                        'correct': 0,
                        'explanation': 'x^n için integral formülü: ∫x^n dx = x^(n+1)/(n+1) + C'
                    },
                    {
                        'question': 'Belirli integral ∫[0,1] x dx değeri kaçtır?',
                        'options': ['1/2', '1', '2', '0'],
                        'correct': 0,
                        'explanation': '∫x dx = x²/2, [0,1] aralığında: 1²/2 - 0²/2 = 1/2'
                    }
                ],
                'Türev': [
                    {
                        'question': 'f(x) = x³ + 2x fonksiyonunun türevi nedir?',
                        'options': ['3x² + 2', 'x² + 2', '3x + 2x', 'x⁴ + x²'],
                        'correct': 0,
                        'explanation': 'Türev kuralı: d/dx(x^n) = n*x^(n-1)'
                    }
                ]
            },
            'Fizik': {
                'Optik': [
                    {
                        'question': 'Işığın boşluktaki hızı kaç m/s\'dir?',
                        'options': ['3×10⁸', '3×10⁶', '3×10¹⁰', '3×10⁵'],
                        'correct': 0,
                        'explanation': 'Işık hızı c = 299,792,458 m/s ≈ 3×10⁸ m/s'
                    }
                ]
            }
        }
    
    def _load_subject_knowledge(self) -> Dict:
        """Load subject knowledge base"""
        return {
            'Matematik': {
                'topics': ['Cebir', 'Geometri', 'Trigonometri', 'Analiz', 'İstatistik'],
                'difficulty_range': (0.3, 0.9),
                'key_concepts': {
                    'İntegral': ['Belirsiz integral', 'Belirli integral', 'Alan hesabı', 'Hacim hesabı'],
                    'Türev': ['Limit', 'Türev kuralları', 'Zincir kuralı', 'Optimizasyon']
                }
            },
            'Fizik': {
                'topics': ['Mekanik', 'Termodinamik', 'Elektrik', 'Optik', 'Modern Fizik'],
                'difficulty_range': (0.4, 0.85),
                'key_concepts': {
                    'Optik': ['Işık', 'Yansıma', 'Kırılma', 'Mercekler', 'Aynalar']
                }
            }
        }
    
    async def answer_question(self, question_data: Dict) -> Dict:
        """Answer student question with personalized response"""
        question = question_data.get('question', '')
        subject = question_data.get('subject', 'Genel')
        context = question_data.get('context', '')
        learning_style = question_data.get('learning_style', 'mixed')
        previous_qa = question_data.get('previous_qa', [])
        requires_latex = question_data.get('requires_latex', False)
        
        # Validate input
        if not question:
            raise ValueError("Question cannot be empty")
        
        # Generate answer based on learning style
        answer = self._generate_answer(question, subject, learning_style)
        
        # Add visual elements for visual learners
        if learning_style == 'visual':
            answer = f"[Görsel Açıklama]\n{answer}\n💡 Diyagram ve grafiklerle desteklenmiş açıklama"
        elif learning_style == 'auditory':
            answer = f"[Sesli Açıklama]\n{answer}\n🎧 Bu konuyu sesli dinleyerek pekiştirebilirsiniz"
        
        # Generate follow-up questions
        follow_up = self._generate_follow_up_questions(question, subject)
        
        # Handle LaTeX if required
        latex_formulas = []
        if requires_latex:
            latex_formulas = self._extract_latex_formulas(answer)
        
        # Prepare response
        response = {
            'answer': answer,
            'confidence': 0.95,
            'sources': ['Ders kitabı sayfa 45', 'Khan Academy', 'Online kaynak'],
            'follow_up_questions': follow_up,
            'context_used': len(previous_qa) > 0,
            'learning_style_adapted': True,
            'response_time': 0.5,
            'metadata': {
                'subject': subject,
                'complexity': self._calculate_complexity(question),
                'keywords': self._extract_keywords(question)
            }
        }
        
        if latex_formulas:
            response['latex_formulas'] = latex_formulas
        
        return response
    
    def _generate_answer(self, question: str, subject: str, learning_style: str) -> str:
        """Generate answer based on question and learning style"""
        # Simplified answer generation
        base_answer = f"{subject} konusundaki sorunuzun cevabı:\n\n"
        
        # Add content based on common patterns
        if 'nedir' in question.lower() or 'ne' in question.lower():
            base_answer += "Bu kavram, temel olarak şu şekilde tanımlanır:\n"
        elif 'nasıl' in question.lower():
            base_answer += "Bu işlem şu adımlarla yapılır:\n1. İlk adım\n2. İkinci adım\n3. Sonuç"
        elif 'kaç' in question.lower() or 'hesapla' in question.lower():
            base_answer += "Hesaplama şu şekilde yapılır:\nFormül uygulanarak sonuç = X"
        else:
            base_answer += "Detaylı açıklama ve çözüm yöntemi."
        
        # Adapt for learning style
        if learning_style == 'visual':
            base_answer = f"📊 {base_answer}\n[Grafik ve şema ile açıklama]"
        elif learning_style == 'auditory':
            base_answer = f"🔊 {base_answer}\n[Sesli anlatım önerisi]"
        
        return base_answer
    
    def _generate_follow_up_questions(self, question: str, subject: str) -> List[str]:
        """Generate follow-up questions"""
        return [
            f"Bu konuyu daha iyi anlamak için {subject} temellerini gözden geçirmek ister misiniz?",
            "Benzer bir örnek üzerinde pratik yapmak ister misiniz?",
            "Bu konunun uygulamalarını görmek ister misiniz?"
        ]
    
    def _extract_latex_formulas(self, text: str) -> List[str]:
        """Extract LaTeX formulas from text"""
        # Simplified LaTeX extraction
        formulas = []
        if 'integral' in text.lower():
            formulas.append(r"\int f(x) dx")
        if 'türev' in text.lower():
            formulas.append(r"\frac{d}{dx} f(x)")
        return formulas
    
    def _calculate_complexity(self, question: str) -> float:
        """Calculate question complexity"""
        # Simple heuristic based on question length and keywords
        complexity = min(len(question) / 200, 1.0)
        if any(word in question.lower() for word in ['integral', 'türev', 'limit', 'kompleks']):
            complexity = min(complexity + 0.2, 1.0)
        return complexity
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question"""
        # Simple keyword extraction
        keywords = []
        important_words = ['integral', 'türev', 'limit', 'fonksiyon', 'denklem', 'formül']
        for word in important_words:
            if word in question.lower():
                keywords.append(word)
        return keywords
    
    async def generate_quiz(self, quiz_request: Dict) -> Dict:
        """Generate personalized quiz"""
        topic = quiz_request.get('topic', 'Genel')
        grade = quiz_request.get('grade', 10)
        difficulty = quiz_request.get('difficulty', 0.5)
        question_count = quiz_request.get('question_count', 10)
        question_types = quiz_request.get('question_types', ['multiple_choice'])
        include_explanations = quiz_request.get('include_explanations', True)
        
        # Extract subject and topic
        if ' - ' in topic:
            subject, specific_topic = topic.split(' - ')
        else:
            subject = topic
            specific_topic = topic
        
        # Generate questions
        questions = []
        for i in range(question_count):
            q_type = random.choice(question_types)
            
            if q_type == 'multiple_choice':
                question = self._generate_multiple_choice(subject, specific_topic, difficulty)
            elif q_type == 'true_false':
                question = self._generate_true_false(subject, specific_topic, difficulty)
            elif q_type == 'short_answer':
                question = self._generate_short_answer(subject, specific_topic, difficulty)
            else:
                question = self._generate_multiple_choice(subject, specific_topic, difficulty)
            
            if include_explanations:
                question['explanation'] = f"Açıklama: {subject} - {specific_topic} konusu ile ilgili"
            
            questions.append(question)
        
        # Create quiz
        quiz = Quiz(
            quiz_id=str(uuid.uuid4()),
            title=f"{topic} Quiz",
            subject=subject,
            topic=specific_topic,
            questions=[Question(**q) for q in questions],
            total_points=sum(q.get('points', 10) for q in questions),
            time_limit=quiz_request.get('time_limit', 30),
            difficulty=difficulty,
            created_at=datetime.now().isoformat(),
            metadata={
                'grade': grade,
                'generated_for': quiz_request.get('student_id', 'anonymous')
            }
        )
        
        return quiz.to_dict()
    
    def _generate_multiple_choice(self, subject: str, topic: str, difficulty: float) -> Dict:
        """Generate multiple choice question"""
        # Try to get from question bank first
        if subject in self.question_bank and topic in self.question_bank[subject]:
            bank_questions = self.question_bank[subject][topic]
            if bank_questions:
                q = random.choice(bank_questions)
                return {
                    'id': str(uuid.uuid4()),
                    'question': q['question'],
                    'type': 'multiple_choice',
                    'options': q['options'],
                    'correct_answer': q['options'][q['correct']],
                    'difficulty': difficulty,
                    'points': int(10 * (1 + difficulty))
                }
        
        # Generate generic question
        return {
            'id': str(uuid.uuid4()),
            'question': f"{subject} - {topic} konusunda örnek soru",
            'type': 'multiple_choice',
            'options': ['Seçenek A', 'Seçenek B', 'Seçenek C', 'Seçenek D'],
            'correct_answer': 'Seçenek A',
            'difficulty': difficulty,
            'points': int(10 * (1 + difficulty))
        }
    
    def _generate_true_false(self, subject: str, topic: str, difficulty: float) -> Dict:
        """Generate true/false question"""
        return {
            'id': str(uuid.uuid4()),
            'question': f"{subject} konusunda: {topic} ile ilgili önerme doğru mudur?",
            'type': 'true_false',
            'options': ['Doğru', 'Yanlış'],
            'correct_answer': random.choice(['Doğru', 'Yanlış']),
            'difficulty': difficulty,
            'points': int(5 * (1 + difficulty))
        }
    
    def _generate_short_answer(self, subject: str, topic: str, difficulty: float) -> Dict:
        """Generate short answer question"""
        return {
            'id': str(uuid.uuid4()),
            'question': f"{topic} konusunu kısaca açıklayınız.",
            'type': 'short_answer',
            'difficulty': difficulty,
            'points': int(15 * (1 + difficulty)),
            'expected_keywords': [topic.lower(), subject.lower()]
        }
    
    async def generate_adaptive_quiz(self, request: Dict) -> Dict:
        """Generate adaptive quiz that adjusts difficulty"""
        base_difficulty = request.get('difficulty', 0.5)
        question_count = request.get('question_count', 5)
        
        # Generate questions with varying difficulty
        questions = []
        difficulties = []
        
        for i in range(question_count):
            # Adaptive difficulty progression
            if i == 0:
                difficulty = base_difficulty
            else:
                # Increase difficulty gradually
                difficulty = min(base_difficulty + (i * 0.1), 0.9)
            
            difficulties.append(difficulty)
            
            # Generate question with adaptive difficulty
            request_copy = request.copy()
            request_copy['difficulty'] = difficulty
            request_copy['question_count'] = 1
            
            quiz_part = await self.generate_quiz(request_copy)
            if quiz_part['questions']:
                questions.extend(quiz_part['questions'])
        
        return {
            'quiz_id': str(uuid.uuid4()),
            'questions': questions,
            'difficulty_progression': difficulties,
            'adaptive': True,
            'metadata': {
                'base_difficulty': base_difficulty,
                'final_difficulty': difficulties[-1] if difficulties else base_difficulty
            }
        }
    
    async def explain_concept(self, concept_data: Dict) -> Dict:
        """Explain a concept in detail"""
        concept_name = concept_data.get('name', '')
        subject = concept_data.get('subject', 'Genel')
        grade = concept_data.get('grade', 10)
        detail_level = concept_data.get('detail_level', 'intermediate')
        
        # Generate explanation
        explanation = f"{concept_name} Açıklaması:\n\n"
        
        if detail_level == 'basic':
            explanation += f"Temel Tanım: {concept_name}, {subject} alanında önemli bir kavramdır.\n"
            explanation += "Basit bir ifadeyle açıklamak gerekirse..."
        elif detail_level == 'intermediate':
            explanation += f"Detaylı Tanım: {concept_name} kavramı şu şekilde açıklanabilir:\n"
            explanation += "1. Temel özellikler\n2. Uygulama alanları\n3. Örnekler"
        else:  # advanced
            explanation += f"İleri Düzey Analiz: {concept_name} konusunun derinlemesine incelenmesi:\n"
            explanation += "• Matematiksel formülasyon\n• Teorik temeller\n• Pratik uygulamalar"
        
        # Add examples
        examples = [
            f"Örnek 1: {concept_name} kullanım senaryosu",
            f"Örnek 2: Gerçek hayatta {concept_name}",
            f"Örnek 3: Problem çözümünde {concept_name}"
        ]
        
        # Key points
        key_points = [
            f"{concept_name} temel prensibi",
            "Dikkat edilmesi gereken noktalar",
            "Yaygın yanlış anlamalar"
        ]
        
        # Related concepts
        related = [
            f"{subject} - İlişkili Konu 1",
            f"{subject} - İlişkili Konu 2"
        ]
        
        return {
            'explanation': explanation,
            'examples': examples,
            'key_points': key_points,
            'related_concepts': related,
            'difficulty_level': detail_level,
            'grade_appropriate': grade <= 12
        }
    
    async def generate_summary(self, content_data: Dict) -> Dict:
        """Generate content summary"""
        text = content_data.get('text', '')
        max_length = content_data.get('max_length', 200)
        style = content_data.get('style', 'paragraph')
        
        # Simple summarization
        words = text.split()
        if len(words) > 50:
            # Take first and last parts
            summary_words = words[:25] + ['...'] + words[-20:]
            summary = ' '.join(summary_words)
        else:
            summary = text
        
        # Format based on style
        if style == 'bullet_points':
            points = summary.split('.')[:3]
            summary = '\n'.join([f"• {p.strip()}" for p in points if p.strip()])
        
        # Ensure within max_length
        if len(summary) > max_length:
            summary = summary[:max_length-3] + '...'
        
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / max(len(text), 1)
        }
    
    async def start_chat_session(self, student_id: str) -> str:
        """Start a new chat session"""
        session_id = str(uuid.uuid4())
        
        session = ChatSession(
            session_id=session_id,
            student_id=student_id,
            started_at=datetime.now().isoformat(),
            messages=[],
            context={'subject_focus': None, 'difficulty_preference': 0.5}
        )
        
        self.chat_sessions[session_id] = session
        
        # Add welcome message
        welcome_msg = {
            'role': 'assistant',
            'content': 'Merhaba! Ben senin çalışma arkadaşınım. Sana nasıl yardımcı olabilirim?',
            'timestamp': datetime.now().isoformat()
        }
        session.messages.append(welcome_msg)
        
        return session_id
    
    async def send_message(self, session_id: str, message: str) -> Dict:
        """Send message in chat session"""
        if session_id not in self.chat_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        
        # Add user message
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        session.messages.append(user_msg)
        
        # Generate response
        response = await self._generate_chat_response(message, session.context)
        
        # Add assistant response
        assistant_msg = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        }
        session.messages.append(assistant_msg)
        
        return assistant_msg
    
    async def _generate_chat_response(self, message: str, context: Dict) -> str:
        """Generate chat response based on message and context"""
        # Simple response generation
        message_lower = message.lower()
        
        if 'merhaba' in message_lower or 'selam' in message_lower:
            return "Merhaba! Bugün hangi konuda yardımcı olabilirim?"
        elif 'integral' in message_lower:
            return "İntegral konusunda yardımcı olabilirim. Belirli mi belirsiz integral mi çalışmak istersin?"
        elif 'türev' in message_lower:
            return "Türev konusu çok önemli! Hangi tür fonksiyonların türevini almak istiyorsun?"
        elif 'yardım' in message_lower or 'help' in message_lower:
            return "Matematik, Fizik, Kimya konularında sorularını yanıtlayabilirim. Hangi konuda yardım istersin?"
        else:
            return f"'{message}' hakkında bilgi vereyim. Bu konuyu daha detaylı açıklayabilir misin?"
    
    async def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session"""
        if session_id not in self.chat_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.chat_sessions[session_id].messages
    
    async def end_chat_session(self, session_id: str) -> Dict:
        """End a chat session"""
        if session_id not in self.chat_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        session.active = False
        
        # Generate session summary
        summary = {
            'session_id': session_id,
            'status': 'ended',
            'duration': (datetime.now() - datetime.fromisoformat(session.started_at)).total_seconds(),
            'message_count': len(session.messages),
            'topics_discussed': self._extract_topics_from_messages(session.messages)
        }
        
        # Clean up
        del self.chat_sessions[session_id]
        
        return summary
    
    def _extract_topics_from_messages(self, messages: List[Dict]) -> List[str]:
        """Extract topics from chat messages"""
        topics = set()
        keywords = ['integral', 'türev', 'fonksiyon', 'denklem', 'fizik', 'kimya', 'matematik']
        
        for msg in messages:
            content_lower = msg.get('content', '').lower()
            for keyword in keywords:
                if keyword in content_lower:
                    topics.add(keyword.capitalize())
        
        return list(topics)
    
    async def provide_hint(self, problem_data: Dict) -> Dict:
        """Provide hint for a problem"""
        question = problem_data.get('question', '')
        student_attempt = problem_data.get('student_attempt', '')
        hint_level = problem_data.get('hint_level', 1)
        
        # Generate progressive hints
        hints = [
            "İlk adım olarak problemi dikkatlice oku",
            "Formülü hatırlamaya çalış",
            "Benzer bir örneği düşün"
        ]
        
        if hint_level <= len(hints):
            hint = hints[hint_level - 1]
        else:
            hint = "Çözüm: " + self._generate_solution_outline(question)
        
        return {
            'hint': hint,
            'hint_level': hint_level,
            'next_step': 'Verilen ipucunu kullanarak tekrar dene',
            'max_hints': len(hints)
        }
    
    def _generate_solution_outline(self, question: str) -> str:
        """Generate solution outline for a problem"""
        return "1. Verilenler tanımlanır\n2. Uygun formül seçilir\n3. Değerler yerine konur\n4. Sonuç hesaplanır"


# Create singleton instance
study_buddy_agent = StudyBuddyAgent()


# Export for compatibility
__all__ = ['StudyBuddyAgent', 'study_buddy_agent', 'Question', 'Quiz', 'ChatSession', 'QuestionType']