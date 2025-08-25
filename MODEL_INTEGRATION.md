# 🚀 Model Integration Guide - TEKNOFEST 2025

## 📦 Fine-tuned Model Information

**Model:** `Huseyin/qwen3-8b-turkish-teknofest2025-private`
- **Base Model:** Qwen3-8B
- **Language:** Turkish
- **Task:** Educational Content Generation
- **Fine-tuned for:** TEKNOFEST 2025 Competition

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements-model.txt
```

### 2. Quick Setup (Minimal)
```bash
pip install torch transformers fastapi uvicorn pandas
```

## 🎯 Quick Start

### Option 1: Interactive Menu
```bash
python quick_start.py
```

### Option 2: Direct Testing
```bash
# Test model loading and inference
python src/model_inference.py

# Test enhanced agents
python src/agents/enhanced_study_buddy.py

# Run full integration test
python test_model_integration.py
```

### Option 3: Start API Server
```bash
python src/mcp_server/enhanced_server.py
```
Then visit: http://localhost:8000/docs

## 📝 Key Features

### 1. Model Inference Module (`src/model_inference.py`)
- **Automatic model loading** from HuggingFace
- **Educational content generation** (explanations, quizzes, examples)
- **Question answering** with subject context
- **Student feedback** generation
- **Study material creation**

### 2. Enhanced Study Buddy Agent (`src/agents/enhanced_study_buddy.py`)
- **AI-powered quiz generation** using fine-tuned model
- **Adaptive difficulty** based on IRT (Item Response Theory)
- **Personalized study plans** with AI recommendations
- **Progressive hint system**
- **Detailed performance feedback**

### 3. Enhanced API Server (`src/mcp_server/enhanced_server.py`)
New endpoints with AI integration:
- `POST /api/v2/quiz` - Generate AI-powered adaptive quiz
- `POST /api/v2/answer-question` - Answer student questions
- `POST /api/v2/feedback` - Provide AI feedback
- `POST /api/v2/study-plan` - Create personalized study plan
- `POST /api/v2/generate-content` - Generate educational content
- `GET /api/v2/stats` - System and model statistics

## 🧪 Testing the Integration

### Run All Tests
```bash
python test_model_integration.py
```

This will test:
1. ✅ Model Loading
2. ✅ Text Generation
3. ✅ Educational Content Creation
4. ✅ Enhanced Agent Functions
5. ✅ API Server Endpoints
6. ✅ AI-Powered Features

## 💡 Usage Examples

### Example 1: Generate Educational Content
```python
from src.model_inference import get_model

model = get_model()

# Generate explanation
content = model.generate_educational_content(
    subject="Matematik",
    topic="Pisagor Teoremi",
    grade_level=9,
    content_type="explanation",
    learning_style="visual"
)
print(content)
```

### Example 2: Create Adaptive Quiz
```python
from src.agents.enhanced_study_buddy import EnhancedStudyBuddyAgent

agent = EnhancedStudyBuddyAgent()

# Generate AI-powered quiz
quiz = agent.generate_adaptive_quiz_with_ai(
    topic="Denklemler",
    subject="Matematik",
    student_ability=0.5,
    num_questions=5,
    grade_level=9
)

for question in quiz:
    print(f"Q{question['number']}: {question['text']}")
    print(f"Options: {question['options']}")
    print(f"AI Generated: {question.get('ai_generated', False)}")
```

### Example 3: API Usage
```python
import requests

# Answer a question
response = requests.post(
    "http://localhost:8000/api/v2/answer-question",
    json={
        "question": "Pisagor teoremi nedir?",
        "subject": "Matematik"
    }
)

answer = response.json()
print(answer['answer'])
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:
```yaml
model:
  base_model: "Huseyin/qwen3-8b-turkish-teknofest2025-private"
  max_length: 2048
  temperature: 0.7
  device: "cuda"  # or "cpu"
  load_in_8bit: false  # Set true for memory optimization
```

## 📊 Performance Metrics

- **Model Loading:** ~5-10 seconds (first time, cached afterwards)
- **Text Generation:** 100-500ms per response
- **Quiz Generation:** 200-300ms per question
- **API Response Time:** <1 second average

## 🔍 Troubleshooting

### Model not loading?
```bash
# Check if you have access to the model
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Huseyin/qwen3-8b-turkish-teknofest2025-private')"
```

### Out of memory?
Enable 8-bit quantization in `configs/config.yaml`:
```yaml
model:
  load_in_8bit: true
```

### API server not starting?
```bash
# Check port 8000 is free
netstat -an | grep 8000

# Use different port
uvicorn src.mcp_server.enhanced_server:app --port 8001
```

---

**Version:** 2.0.0 | **Last Updated:** 2025 | **Competition:** TEKNOFEST 2025
