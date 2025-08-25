"""
Load Testing for TEKNOFEST 2025 API
Usage: locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random
import json

class TeknofestAPIUser(HttpUser):
    """Simulated user for load testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session"""
        self.student_id = f"student_{random.randint(1000, 9999)}"
        self.grade = random.randint(9, 12)
        self.topics = ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe"]
        
    @task(3)
    def health_check(self):
        """Test health endpoint - most frequent"""
        self.client.get("/health")
    
    @task(2)
    def root_endpoint(self):
        """Test root endpoint"""
        self.client.get("/")
    
    @task(5)
    def detect_learning_style(self):
        """Test learning style detection"""
        responses = [
            "Görsel materyalleri tercih ederim",
            "Dinleyerek daha iyi öğrenirim",
            "Okuyarak öğrenmeyi severim",
            "Pratik yaparak öğrenirim"
        ]
        
        payload = {
            "student_responses": random.sample(responses, k=3)
        }
        
        with self.client.post(
            "/api/v1/learning-style",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(8)
    def generate_quiz(self):
        """Test quiz generation - high frequency"""
        payload = {
            "topic": random.choice(self.topics),
            "student_ability": round(random.uniform(0.3, 0.8), 2),
            "num_questions": random.randint(5, 15)
        }
        
        with self.client.post(
            "/api/v1/generate-quiz",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("success"):
                        response.success()
                    else:
                        response.failure("API returned success=false")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(4)
    def generate_text(self):
        """Test text generation"""
        prompts = [
            "Pisagor teoremi nedir?",
            "Newton'un hareket yasaları",
            "Fotosentez nasıl gerçekleşir?",
            "Osmanlı İmparatorluğu'nun kuruluşu",
            "Türkçe'de fiil çekimi"
        ]
        
        payload = {
            "prompt": random.choice(prompts),
            "max_length": random.randint(100, 300)
        }
        
        with self.client.post(
            "/api/v1/generate-text",
            json=payload,
            catch_response=True,
            timeout=10
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def get_curriculum(self):
        """Test curriculum endpoint"""
        grade = random.randint(9, 12)
        
        with self.client.get(
            f"/api/v1/curriculum/{grade}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.failure(f"Curriculum not found for grade {grade}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_data_stats(self):
        """Test data stats endpoint"""
        self.client.get("/api/v1/data/stats")


class AdminUser(HttpUser):
    """Admin user with different behavior patterns"""
    
    wait_time = between(2, 5)
    weight = 1  # Less admin users than regular users
    
    @task
    def check_metrics(self):
        """Admin checking system metrics"""
        endpoints = [
            "/health",
            "/api/v1/data/stats"
        ]
        
        for endpoint in endpoints:
            self.client.get(endpoint)


class MobileUser(HttpUser):
    """Mobile user with slower connection simulation"""
    
    wait_time = between(3, 7)
    weight = 3  # More mobile users
    
    def on_start(self):
        """Simulate mobile headers"""
        self.client.headers.update({
            "User-Agent": "Mobile App v1.0",
            "X-Platform": "iOS"
        })
    
    @task(10)
    def quick_quiz(self):
        """Mobile users mostly do quick quizzes"""
        payload = {
            "topic": random.choice(["Matematik", "Fizik"]),
            "student_ability": 0.5,
            "num_questions": 5  # Fewer questions on mobile
        }
        
        self.client.post("/api/v1/generate-quiz", json=payload)
    
    @task(5)
    def check_health(self):
        """Frequent health checks on mobile"""
        self.client.get("/health")