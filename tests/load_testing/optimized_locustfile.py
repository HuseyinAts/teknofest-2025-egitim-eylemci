"""
Optimized Load Testing Suite for Teknofest 2025 Education Platform
"""

import json
import random
import time
from typing import Dict, List, Any
from locust import HttpUser, task, between, events, LoadTestShape
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import gevent

# Setup logging
setup_logging("INFO", None)


class TeknofestUser(HttpUser):
    """Simulated user behavior for load testing."""
    
    wait_time = between(1, 3)
    host = "http://localhost:8000"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = None
        self.user_id = None
        self.quiz_ids = []
        self.learning_path_id = None
    
    def on_start(self):
        """Initialize user session."""
        # Register or login
        if random.random() < 0.3:
            self.register_user()
        else:
            self.login_user()
    
    def register_user(self):
        """Register a new user."""
        user_data = {
            "username": f"loadtest_user_{random.randint(1000, 99999)}",
            "email": f"loadtest_{random.randint(1000, 99999)}@test.com",
            "password": "LoadTest123!",
            "grade_level": random.randint(9, 12)
        }
        
        with self.client.post(
            "/api/v1/auth/register",
            json=user_data,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                data = response.json()
                self.token = data.get("access_token")
                self.user_id = data.get("user", {}).get("id")
                response.success()
            else:
                response.failure(f"Registration failed: {response.status_code}")
    
    def login_user(self):
        """Login existing user."""
        credentials = {
            "email": "test@example.com",
            "password": "Test123!"
        }
        
        with self.client.post(
            "/api/v1/auth/login",
            json=credentials,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.user_id = data.get("user", {}).get("id")
                response.success()
            else:
                response.failure(f"Login failed: {response.status_code}")
    
    @task(10)
    def view_dashboard(self):
        """View user dashboard."""
        if not self.token:
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        with self.client.get(
            "/api/v1/users/dashboard",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard failed: {response.status_code}")
    
    @task(8)
    def get_quiz_list(self):
        """Get available quizzes."""
        if not self.token:
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        with self.client.get(
            "/api/v1/quiz/list",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.quiz_ids = [q["id"] for q in response.json().get("quizzes", [])]
                response.success()
            else:
                response.failure(f"Quiz list failed: {response.status_code}")
    
    @task(15)
    def take_quiz(self):
        """Take a quiz."""
        if not self.token or not self.quiz_ids:
            return
        
        quiz_id = random.choice(self.quiz_ids)
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Get quiz questions
        with self.client.get(
            f"/api/v1/quiz/{quiz_id}",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Get quiz failed: {response.status_code}")
                return
            
            questions = response.json().get("questions", [])
            response.success()
        
        # Submit quiz answers
        answers = [
            {
                "question_id": q["id"],
                "answer": random.choice(q["options"])["id"]
            }
            for q in questions
        ]
        
        with self.client.post(
            f"/api/v1/quiz/{quiz_id}/submit",
            json={"answers": answers},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Quiz submit failed: {response.status_code}")
    
    @task(5)
    def get_learning_path(self):
        """Get personalized learning path."""
        if not self.token:
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        with self.client.get(
            "/api/v1/learning-paths/my-path",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.learning_path_id = response.json().get("id")
                response.success()
            else:
                response.failure(f"Learning path failed: {response.status_code}")
    
    @task(3)
    def generate_learning_path(self):
        """Generate new learning path."""
        if not self.token:
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {
            "subject": random.choice(["matematik", "fizik", "kimya", "biyoloji"]),
            "grade_level": random.randint(9, 12),
            "learning_style": random.choice(["visual", "auditory", "kinesthetic"])
        }
        
        with self.client.post(
            "/api/v1/learning-paths/generate",
            json=data,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                self.learning_path_id = response.json().get("id")
                response.success()
            else:
                response.failure(f"Generate path failed: {response.status_code}")
    
    @task(7)
    def chat_with_ai(self):
        """Chat with AI study buddy."""
        if not self.token:
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        questions = [
            "Üçgenin iç açıları toplamı nedir?",
            "Mitoz ve mayoz arasındaki fark nedir?",
            "Newton'un hareket yasaları nelerdir?",
            "Periyodik tabloda elementler nasıl sıralanır?"
        ]
        
        data = {
            "message": random.choice(questions),
            "context": {
                "subject": random.choice(["matematik", "fizik", "kimya", "biyoloji"]),
                "grade": random.randint(9, 12)
            }
        }
        
        with self.client.post(
            "/api/v1/study-buddy/chat",
            json=data,
            headers=headers,
            catch_response=True,
            timeout=10
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"AI chat failed: {response.status_code}")
    
    @task(2)
    def view_leaderboard(self):
        """View leaderboard."""
        with self.client.get(
            "/api/v1/leaderboard",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Leaderboard failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Perform health check."""
        with self.client.get(
            "/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    def on_stop(self):
        """Cleanup on user stop."""
        if self.token:
            # Logout
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/v1/auth/logout", headers=headers)


class AdminUser(HttpUser):
    """Admin user behavior for load testing."""
    
    wait_time = between(2, 5)
    weight = 1  # Less admin users than regular users
    
    def on_start(self):
        """Admin login."""
        credentials = {
            "email": "admin@teknofest.com",
            "password": "AdminPass123!"
        }
        
        response = self.client.post("/api/v1/auth/login", json=credentials)
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
    
    @task(5)
    def view_analytics(self):
        """View platform analytics."""
        if not hasattr(self, 'token'):
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.get("/api/v1/admin/analytics", headers=headers)
    
    @task(3)
    def manage_users(self):
        """User management operations."""
        if not hasattr(self, 'token'):
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.get("/api/v1/admin/users", headers=headers)
    
    @task(2)
    def create_quiz(self):
        """Create new quiz."""
        if not hasattr(self, 'token'):
            return
        
        headers = {"Authorization": f"Bearer {self.token}"}
        quiz_data = {
            "title": f"Load Test Quiz {random.randint(1000, 9999)}",
            "subject": random.choice(["matematik", "fizik", "kimya"]),
            "grade_level": random.randint(9, 12),
            "questions": [
                {
                    "text": f"Question {i}",
                    "options": [
                        {"text": f"Option {j}", "is_correct": j == 0}
                        for j in range(4)
                    ]
                }
                for i in range(5)
            ]
        }
        
        self.client.post(
            "/api/v1/admin/quiz/create",
            json=quiz_data,
            headers=headers
        )


class StagesShape(LoadTestShape):
    """
    Custom load test shape with different stages.
    Simulates realistic traffic patterns.
    """
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},   # Warm-up
        {"duration": 180, "users": 50, "spawn_rate": 5},  # Normal load
        {"duration": 300, "users": 100, "spawn_rate": 10}, # Peak load
        {"duration": 120, "users": 200, "spawn_rate": 20}, # Stress test
        {"duration": 180, "users": 50, "spawn_rate": 10}, # Cool down
        {"duration": 60, "users": 10, "spawn_rate": 5},   # Final stage
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data
        
        return None


class SpikeTestShape(LoadTestShape):
    """
    Spike test to simulate sudden traffic increases.
    """
    
    time_limit = 600
    spawn_rate = 20
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < self.time_limit:
            # Normal load for 2 minutes
            if run_time < 120:
                user_count = 50
            # Spike to 500 users for 1 minute
            elif run_time < 180:
                user_count = 500
            # Back to normal for 2 minutes
            elif run_time < 300:
                user_count = 50
            # Another spike for 1 minute
            elif run_time < 360:
                user_count = 300
            # Cool down
            else:
                user_count = 25
            
            return (user_count, self.spawn_rate)
        
        return None


class EnduranceTestShape(LoadTestShape):
    """
    Endurance test for long-running load.
    """
    
    time_limit = 3600  # 1 hour
    target_users = 100
    spawn_rate = 5
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < self.time_limit:
            # Gradual ramp-up for first 5 minutes
            if run_time < 300:
                user_count = int((run_time / 300) * self.target_users)
            else:
                user_count = self.target_users
            
            return (user_count, self.spawn_rate)
        
        return None


# Event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test metrics."""
    print("Load test starting...")
    print(f"Target host: {environment.host}")
    print(f"Number of users: {environment.parsed_options.num_users}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report."""
    print("\nLoad test completed!")
    print("\n=== Test Results ===")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {environment.stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {environment.stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {environment.stats.total.current_rps:.2f}")
    
    # Calculate percentiles
    if environment.stats.total.num_requests > 0:
        print(f"\nResponse time percentiles:")
        for percentile in [50, 75, 90, 95, 99]:
            print(f"  {percentile}%: {environment.stats.total.get_response_time_percentile(percentile):.2f}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log slow requests."""
    if response_time > 5000:  # Log requests slower than 5 seconds
        print(f"Slow request detected: {name} took {response_time}ms")
    
    if exception:
        print(f"Request failed: {name} - {exception}")


# Custom statistics collector
class CustomStats:
    """Collect custom statistics during load test."""
    
    def __init__(self):
        self.ai_response_times = []
        self.quiz_completion_times = []
        self.error_types = {}
    
    def record_ai_response(self, response_time):
        self.ai_response_times.append(response_time)
    
    def record_quiz_completion(self, completion_time):
        self.quiz_completion_times.append(completion_time)
    
    def record_error(self, error_type):
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_summary(self):
        summary = {
            "ai_avg_response": sum(self.ai_response_times) / len(self.ai_response_times) if self.ai_response_times else 0,
            "quiz_avg_completion": sum(self.quiz_completion_times) / len(self.quiz_completion_times) if self.quiz_completion_times else 0,
            "error_distribution": self.error_types
        }
        return summary


# Initialize custom stats
custom_stats = CustomStats()


if __name__ == "__main__":
    # Run with: locust -f optimized_locustfile.py --host=http://localhost:8000
    print("Optimized Locust load testing file ready.")
    print("Run with: locust -f optimized_locustfile.py --host=http://localhost:8000")
    print("\nAvailable test shapes:")
    print("  - Default: Standard load test")
    print("  - StagesShape: Gradual load increase")
    print("  - SpikeTestShape: Spike testing")
    print("  - EnduranceTestShape: Long-running test")