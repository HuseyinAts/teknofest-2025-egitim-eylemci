"""
Production Load Testing Suite
TEKNOFEST 2025 - Performance Validation

This module performs comprehensive load testing for production readiness.
"""

import time
import json
import random
from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import logging

# Setup logging
setup_logging("INFO", None)
logger = logging.getLogger(__name__)

class TeknofestUser(HttpUser):
    """Simulated user behavior for load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Setup before tasks start."""
        # Login and get token
        self.token = self.login()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def login(self):
        """Simulate user login."""
        response = self.client.post("/api/auth/login", json={
            "username": f"test_user_{random.randint(1, 1000)}",
            "password": "Test@Password123"
        }, catch_response=True)
        
        if response.status_code == 200:
            data = response.json()
            response.success()
            return data.get("access_token", "dummy_token")
        else:
            # Use dummy token for testing if login fails
            response.failure(f"Login failed: {response.status_code}")
            return "dummy_token"
    
    @task(3)
    def view_dashboard(self):
        """Most common task - view dashboard."""
        with self.client.get(
            "/api/dashboard",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard failed: {response.status_code}")
    
    @task(2)
    def get_learning_paths(self):
        """Get learning paths."""
        with self.client.get(
            "/api/learning-paths",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Learning paths failed: {response.status_code}")
    
    @task(2)
    def start_quiz(self):
        """Start a quiz session."""
        quiz_data = {
            "subject": random.choice(["Matematik", "Fizik", "Kimya", "Biyoloji"]),
            "grade": random.randint(9, 12),
            "difficulty": random.choice(["easy", "medium", "hard"])
        }
        
        with self.client.post(
            "/api/quiz/start",
            json=quiz_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
                # Submit quiz answers
                self.submit_quiz_answers(response.json().get("quiz_id"))
            else:
                response.failure(f"Quiz start failed: {response.status_code}")
    
    def submit_quiz_answers(self, quiz_id):
        """Submit quiz answers."""
        if not quiz_id:
            return
            
        answers = [
            {"question_id": i, "answer": random.choice(["A", "B", "C", "D"])}
            for i in range(1, 11)
        ]
        
        with self.client.post(
            f"/api/quiz/{quiz_id}/submit",
            json={"answers": answers},
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Quiz submit failed: {response.status_code}")
    
    @task(1)
    def get_progress(self):
        """Get user progress."""
        with self.client.get(
            "/api/progress",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Progress failed: {response.status_code}")
    
    @task(1)
    def chat_with_ai(self):
        """Chat with AI assistant."""
        chat_data = {
            "message": random.choice([
                "Matematik konusunda yardım eder misin?",
                "Fizik formüllerini açıklar mısın?",
                "Kimya dersinde zorlanıyorum",
                "Biyoloji konularını özetler misin?"
            ])
        }
        
        with self.client.post(
            "/api/chat",
            json=chat_data,
            headers=self.headers,
            catch_response=True,
            timeout=10  # AI responses might be slow
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Chat failed: {response.status_code}")
    
    @task(1)
    def search_resources(self):
        """Search for educational resources."""
        search_query = random.choice(["matematik", "fizik", "kimya", "biyoloji"])
        
        with self.client.get(
            f"/api/search?q={search_query}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")

class AdminUser(HttpUser):
    """Simulated admin user behavior."""
    
    wait_time = between(2, 5)
    weight = 1  # Less admin users than regular users
    
    def on_start(self):
        """Setup admin session."""
        self.token = self.admin_login()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def admin_login(self):
        """Admin login."""
        response = self.client.post("/api/auth/login", json={
            "username": "admin",
            "password": "Admin@Password123"
        })
        
        if response.status_code == 200:
            return response.json().get("access_token", "admin_token")
        return "admin_token"
    
    @task(2)
    def view_analytics(self):
        """View analytics dashboard."""
        with self.client.get(
            "/api/admin/analytics",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analytics failed: {response.status_code}")
    
    @task(1)
    def manage_users(self):
        """User management tasks."""
        with self.client.get(
            "/api/admin/users",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"User management failed: {response.status_code}")
    
    @task(1)
    def view_reports(self):
        """View system reports."""
        with self.client.get(
            "/api/admin/reports",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Reports failed: {response.status_code}")

# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom request handler for metrics."""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    elif response_time > 1000:  # Log slow requests (>1s)
        logger.warning(f"Slow request: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    logger.info("=" * 60)
    logger.info("TEKNOFEST 2025 - Production Load Test Starting")
    logger.info(f"Target Host: {environment.host}")
    logger.info(f"Total Users: {environment.parsed_options.num_users if environment.parsed_options else 'N/A'}")
    logger.info("=" * 60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report."""
    logger.info("=" * 60)
    logger.info("Load Test Results Summary")
    logger.info("=" * 60)
    
    # Calculate statistics
    stats = environment.stats
    
    logger.info(f"Total Requests: {stats.total.num_requests}")
    logger.info(f"Total Failures: {stats.total.num_failures}")
    logger.info(f"Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Median Response Time: {stats.total.median_response_time:.2f}ms")
    logger.info(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    logger.info(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    
    # Performance thresholds
    passed = True
    if stats.total.fail_ratio > 0.01:  # >1% failure rate
        logger.error("❌ FAILED: Error rate > 1%")
        passed = False
    
    if stats.total.get_response_time_percentile(0.95) > 2000:  # p95 > 2s
        logger.error("❌ FAILED: 95th percentile response time > 2s")
        passed = False
    
    if stats.total.get_response_time_percentile(0.99) > 5000:  # p99 > 5s
        logger.error("❌ FAILED: 99th percentile response time > 5s")
        passed = False
    
    if passed:
        logger.info("✅ PASSED: All performance criteria met")
    
    # Save detailed report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_requests": stats.total.num_requests,
        "total_failures": stats.total.num_failures,
        "failure_rate": stats.total.fail_ratio,
        "avg_response_time": stats.total.avg_response_time,
        "median_response_time": stats.total.median_response_time,
        "p95_response_time": stats.total.get_response_time_percentile(0.95),
        "p99_response_time": stats.total.get_response_time_percentile(0.99),
        "test_passed": passed
    }
    
    with open("load_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to load_test_report.json")

if __name__ == "__main__":
    # Run standalone test
    import sys
    from locust import run_single_user
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Run single user test
        user = TeknofestUser(environment=Environment())
        user.host = "http://localhost:8000"
        run_single_user(user)
    else:
        print("Use: locust -f production_load_test.py --host=http://localhost:8000")
        print("Or run single user: python production_load_test.py single")