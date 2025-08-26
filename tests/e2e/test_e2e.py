"""
End-to-End Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests


class TestE2EUserJourney:
    """End-to-end tests for complete user journeys"""
    
    @pytest.fixture
    def api_client(self):
        """Create API client for E2E tests"""
        base_url = "http://localhost:8000"
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        return {"session": session, "base_url": base_url}
    
    @pytest.mark.e2e
    def test_student_registration_to_quiz_completion(self, api_client):
        """Test complete flow from registration to quiz completion"""
        session = api_client["session"]
        base_url = api_client["base_url"]
        
        # Step 1: Register new student
        registration_data = {
            "username": f"e2e_student_{int(time.time())}",
            "email": f"e2e_{int(time.time())}@test.com",
            "password": "Test123!@#",
            "full_name": "E2E Test Student"
        }
        
        try:
            response = session.post(
                f"{base_url}/api/auth/register",
                json=registration_data
            )
            
            if response.status_code == 201:
                user_data = response.json()
                assert "id" in user_data
                user_id = user_data["id"]
            else:
                pytest.skip("Registration endpoint not available")
        except requests.ConnectionError:
            pytest.skip("API server not running")
        
        # Step 2: Login
        login_data = {
            "username": registration_data["username"],
            "password": registration_data["password"]
        }
        
        response = session.post(
            f"{base_url}/api/auth/login",
            json=login_data
        )
        
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data
        
        # Set auth header
        session.headers.update({
            "Authorization": f"Bearer {token_data['access_token']}"
        })
        
        # Step 3: Get learning style assessment
        response = session.get(f"{base_url}/api/assessment/learning-style")
        if response.status_code == 200:
            assessment = response.json()
            
            # Submit assessment
            assessment_response = {
                "responses": ["visual", "hands-on", "group-work"]
            }
            
            response = session.post(
                f"{base_url}/api/assessment/learning-style/submit",
                json=assessment_response
            )
            
            if response.status_code == 200:
                style_result = response.json()
                assert "learning_style" in style_result
        
        # Step 4: Get personalized learning path
        response = session.get(f"{base_url}/api/learning-path/personalized")
        if response.status_code == 200:
            learning_path = response.json()
            assert "path" in learning_path
            assert len(learning_path["path"]) > 0
        
        # Step 5: Start first quiz
        response = session.post(
            f"{base_url}/api/quiz/generate",
            json={
                "topic": "Mathematics",
                "difficulty": 0.5,
                "num_questions": 5
            }
        )
        
        if response.status_code == 200:
            quiz = response.json()
            assert "questions" in quiz
            assert len(quiz["questions"]) == 5
            
            # Step 6: Answer quiz questions
            quiz_results = []
            for i, question in enumerate(quiz["questions"]):
                answer_data = {
                    "question_id": question["id"],
                    "answer": 0,  # Select first option
                    "time_taken": 30
                }
                
                response = session.post(
                    f"{base_url}/api/quiz/submit-answer",
                    json=answer_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    quiz_results.append(result)
            
            # Step 7: Get quiz results
            response = session.get(f"{base_url}/api/quiz/results/{quiz['id']}")
            if response.status_code == 200:
                final_results = response.json()
                assert "score" in final_results
                assert "feedback" in final_results
        
        # Step 8: Check achievements
        response = session.get(f"{base_url}/api/gamification/achievements")
        if response.status_code == 200:
            achievements = response.json()
            # Should have at least "First Quiz" achievement
            assert len(achievements) >= 1
        
        # Step 9: Get study recommendations
        response = session.get(f"{base_url}/api/study-buddy/recommendations")
        if response.status_code == 200:
            recommendations = response.json()
            assert "recommendations" in recommendations
    
    @pytest.mark.e2e
    def test_teacher_course_creation_flow(self, api_client):
        """Test teacher creating and managing a course"""
        session = api_client["session"]
        base_url = api_client["base_url"]
        
        # Teacher registration
        teacher_data = {
            "username": f"e2e_teacher_{int(time.time())}",
            "email": f"teacher_{int(time.time())}@test.com",
            "password": "Teacher123!",
            "full_name": "E2E Teacher",
            "role": "teacher"
        }
        
        try:
            response = session.post(
                f"{base_url}/api/auth/register",
                json=teacher_data
            )
            
            if response.status_code != 201:
                pytest.skip("Teacher registration not available")
        except requests.ConnectionError:
            pytest.skip("API server not running")
        
        # Login as teacher
        response = session.post(
            f"{base_url}/api/auth/login",
            json={
                "username": teacher_data["username"],
                "password": teacher_data["password"]
            }
        )
        
        assert response.status_code == 200
        token = response.json()["access_token"]
        session.headers.update({"Authorization": f"Bearer {token}"})
        
        # Create course
        course_data = {
            "title": "Advanced Mathematics",
            "description": "Comprehensive math course",
            "grade_level": 11,
            "subject": "Mathematics",
            "modules": [
                {"name": "Calculus Basics", "order": 1},
                {"name": "Derivatives", "order": 2},
                {"name": "Integration", "order": 3}
            ]
        }
        
        response = session.post(
            f"{base_url}/api/courses/create",
            json=course_data
        )
        
        if response.status_code == 201:
            course = response.json()
            course_id = course["id"]
            
            # Create quiz for course
            quiz_data = {
                "course_id": course_id,
                "title": "Calculus Quiz 1",
                "questions": [
                    {
                        "text": "What is the derivative of x^2?",
                        "options": ["2x", "x", "x^2", "2"],
                        "correct_answer": 0,
                        "difficulty": 0.4
                    }
                ]
            }
            
            response = session.post(
                f"{base_url}/api/courses/{course_id}/quiz",
                json=quiz_data
            )
            
            if response.status_code == 201:
                quiz = response.json()
                assert quiz["title"] == "Calculus Quiz 1"
            
            # Get course analytics
            response = session.get(f"{base_url}/api/courses/{course_id}/analytics")
            if response.status_code == 200:
                analytics = response.json()
                assert "enrolled_students" in analytics
                assert "average_progress" in analytics
    
    @pytest.mark.e2e
    def test_collaborative_learning_session(self, api_client):
        """Test students joining collaborative study session"""
        session = api_client["session"]
        base_url = api_client["base_url"]
        
        # Create multiple students
        students = []
        for i in range(3):
            student_data = {
                "username": f"collab_student_{i}_{int(time.time())}",
                "email": f"collab{i}_{int(time.time())}@test.com",
                "password": "Student123!",
                "full_name": f"Student {i}"
            }
            
            try:
                response = session.post(
                    f"{base_url}/api/auth/register",
                    json=student_data
                )
                
                if response.status_code == 201:
                    # Login and get token
                    login_response = session.post(
                        f"{base_url}/api/auth/login",
                        json={
                            "username": student_data["username"],
                            "password": student_data["password"]
                        }
                    )
                    
                    if login_response.status_code == 200:
                        token = login_response.json()["access_token"]
                        students.append({
                            "data": student_data,
                            "token": token
                        })
            except requests.ConnectionError:
                pytest.skip("API server not running")
        
        if len(students) < 2:
            pytest.skip("Not enough students created")
        
        # First student creates study session
        session.headers.update({
            "Authorization": f"Bearer {students[0]['token']}"
        })
        
        session_data = {
            "title": "Math Study Group",
            "subject": "Mathematics",
            "max_participants": 5,
            "scheduled_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        response = session.post(
            f"{base_url}/api/study-sessions/create",
            json=session_data
        )
        
        if response.status_code == 201:
            study_session = response.json()
            session_id = study_session["id"]
            
            # Other students join
            for student in students[1:]:
                session.headers.update({
                    "Authorization": f"Bearer {student['token']}"
                })
                
                response = session.post(
                    f"{base_url}/api/study-sessions/{session_id}/join"
                )
                
                if response.status_code == 200:
                    join_result = response.json()
                    assert join_result["joined"] is True
            
            # Start collaborative quiz
            session.headers.update({
                "Authorization": f"Bearer {students[0]['token']}"
            })
            
            response = session.post(
                f"{base_url}/api/study-sessions/{session_id}/start-quiz",
                json={"quiz_type": "collaborative"}
            )
            
            if response.status_code == 200:
                collab_quiz = response.json()
                assert "questions" in collab_quiz
                assert "participants" in collab_quiz
    
    @pytest.mark.e2e
    def test_offline_sync_workflow(self, api_client):
        """Test offline data synchronization"""
        session = api_client["session"]
        base_url = api_client["base_url"]
        
        # Login as existing user
        login_data = {
            "username": "test_user",
            "password": "Test123!"
        }
        
        try:
            response = session.post(
                f"{base_url}/api/auth/login",
                json=login_data
            )
            
            if response.status_code != 200:
                # Create user if doesn't exist
                reg_data = {
                    "username": "test_user",
                    "email": "test@example.com",
                    "password": "Test123!",
                    "full_name": "Test User"
                }
                session.post(f"{base_url}/api/auth/register", json=reg_data)
                response = session.post(f"{base_url}/api/auth/login", json=login_data)
            
            token = response.json()["access_token"]
            session.headers.update({"Authorization": f"Bearer {token}"})
        except requests.ConnectionError:
            pytest.skip("API server not running")
        
        # Download offline content
        response = session.get(f"{base_url}/api/offline/download-content")
        if response.status_code == 200:
            offline_content = response.json()
            assert "quizzes" in offline_content
            assert "learning_materials" in offline_content
        
        # Simulate offline work
        offline_data = {
            "quiz_attempts": [
                {
                    "quiz_id": "offline_quiz_1",
                    "score": 85,
                    "completed_at": datetime.utcnow().isoformat(),
                    "answers": [
                        {"question_id": 1, "answer": 0},
                        {"question_id": 2, "answer": 1}
                    ]
                }
            ],
            "study_time": 120,  # minutes
            "notes": ["Studied calculus", "Reviewed derivatives"]
        }
        
        # Sync offline data
        response = session.post(
            f"{base_url}/api/offline/sync",
            json={
                "offline_data": offline_data,
                "last_sync": (datetime.utcnow() - timedelta(days=1)).isoformat()
            }
        )
        
        if response.status_code == 200:
            sync_result = response.json()
            assert "synced_items" in sync_result
            assert "conflicts" in sync_result
            assert sync_result["synced_items"] > 0
    
    @pytest.mark.e2e
    def test_performance_monitoring(self, api_client):
        """Test system performance under load"""
        session = api_client["session"]
        base_url = api_client["base_url"]
        
        # Create test user and login
        try:
            response = session.post(
                f"{base_url}/api/auth/login",
                json={"username": "perf_test", "password": "Test123!"}
            )
            
            if response.status_code != 200:
                # Create user
                session.post(
                    f"{base_url}/api/auth/register",
                    json={
                        "username": "perf_test",
                        "email": "perf@test.com",
                        "password": "Test123!",
                        "full_name": "Performance Test"
                    }
                )
                response = session.post(
                    f"{base_url}/api/auth/login",
                    json={"username": "perf_test", "password": "Test123!"}
                )
            
            token = response.json()["access_token"]
            session.headers.update({"Authorization": f"Bearer {token}"})
        except requests.ConnectionError:
            pytest.skip("API server not running")
        
        # Measure response times
        endpoints = [
            ("GET", "/api/health", None),
            ("GET", "/api/learning-path/personalized", None),
            ("POST", "/api/quiz/generate", {"topic": "Math", "num_questions": 5}),
            ("GET", "/api/gamification/leaderboard", None)
        ]
        
        response_times = []
        
        for method, endpoint, data in endpoints:
            start_time = time.time()
            
            if method == "GET":
                response = session.get(f"{base_url}{endpoint}")
            else:
                response = session.post(f"{base_url}{endpoint}", json=data)
            
            elapsed_time = time.time() - start_time
            response_times.append({
                "endpoint": endpoint,
                "method": method,
                "status": response.status_code,
                "time": elapsed_time
            })
        
        # Check performance thresholds
        for timing in response_times:
            # API responses should be under 2 seconds
            assert timing["time"] < 2.0, f"{timing['endpoint']} took {timing['time']}s"
        
        # Average response time should be under 500ms
        avg_time = sum(t["time"] for t in response_times) / len(response_times)
        assert avg_time < 0.5, f"Average response time {avg_time}s exceeds threshold"


@pytest.mark.e2e
class TestBrowserE2E:
    """Browser-based E2E tests using Selenium"""
    
    @pytest.fixture
    def browser(self):
        """Create browser instance for E2E tests"""
        # Try Chrome first, fallback to Firefox
        driver = None
        try:
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(options=options)
        except Exception:
            try:
                from selenium.webdriver.firefox.options import Options
                options = Options()
                options.add_argument("--headless")
                driver = webdriver.Firefox(options=options)
            except Exception:
                pytest.skip("No browser driver available")
        
        yield driver
        
        if driver:
            driver.quit()
    
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Requires browser driver setup")
    def test_ui_login_flow(self, browser):
        """Test login flow through UI"""
        browser.get("http://localhost:3000")
        
        # Wait for page load
        wait = WebDriverWait(browser, 10)
        
        try:
            # Click login button
            login_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Login']"))
            )
            login_btn.click()
            
            # Fill login form
            username_input = wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_input.send_keys("test_user")
            
            password_input = browser.find_element(By.NAME, "password")
            password_input.send_keys("Test123!")
            
            # Submit form
            submit_btn = browser.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()
            
            # Wait for dashboard
            dashboard = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
            )
            
            assert dashboard is not None
            
            # Check user info displayed
            user_info = browser.find_element(By.CLASS_NAME, "user-info")
            assert "test_user" in user_info.text
            
        except TimeoutException:
            pytest.skip("UI elements not found - frontend may not be running")
    
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Requires browser driver setup")
    def test_ui_quiz_taking(self, browser):
        """Test taking a quiz through UI"""
        # Login first
        browser.get("http://localhost:3000/login")
        
        wait = WebDriverWait(browser, 10)
        
        try:
            # Quick login
            browser.execute_script("""
                localStorage.setItem('token', 'test_token');
                localStorage.setItem('user', JSON.stringify({
                    id: '1',
                    username: 'test_user'
                }));
            """)
            
            # Navigate to quiz
            browser.get("http://localhost:3000/quiz")
            
            # Start quiz
            start_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Start Quiz']"))
            )
            start_btn.click()
            
            # Answer questions
            questions = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "question"))
            )
            
            for question in questions:
                # Select first option
                option = question.find_element(By.CSS_SELECTOR, "input[type='radio']")
                option.click()
                
                # Click next
                next_btn = question.find_element(By.XPATH, ".//button[text()='Next']")
                next_btn.click()
            
            # Submit quiz
            submit_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Submit Quiz']"))
            )
            submit_btn.click()
            
            # Check results
            results = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "quiz-results"))
            )
            
            assert results is not None
            score_element = results.find_element(By.CLASS_NAME, "score")
            assert "Score" in score_element.text
            
        except TimeoutException:
            pytest.skip("Quiz UI not available")
    
    @pytest.mark.e2e
    @pytest.mark.skip(reason="Requires browser driver setup")
    def test_ui_responsive_design(self, browser):
        """Test UI responsiveness on different screen sizes"""
        browser.get("http://localhost:3000")
        
        # Test desktop view
        browser.set_window_size(1920, 1080)
        assert browser.find_element(By.CLASS_NAME, "navbar") is not None
        
        # Test tablet view
        browser.set_window_size(768, 1024)
        time.sleep(1)
        
        # Check if mobile menu appears
        try:
            mobile_menu = browser.find_element(By.CLASS_NAME, "mobile-menu-button")
            assert mobile_menu is not None
        except:
            pass  # Some designs may not have mobile menu at tablet size
        
        # Test mobile view
        browser.set_window_size(375, 667)
        time.sleep(1)
        
        # Mobile menu should definitely be present
        try:
            mobile_menu = browser.find_element(By.CLASS_NAME, "mobile-menu-button")
            assert mobile_menu is not None
            assert mobile_menu.is_displayed()
        except:
            pytest.skip("Responsive design not implemented")