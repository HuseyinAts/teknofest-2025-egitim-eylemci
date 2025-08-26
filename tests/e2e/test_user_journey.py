"""
End-to-end test suite for complete user journey scenarios.
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, Any
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from src.config import Config
from src.database.models import User, Quiz, LearningPath
from src.database.session import DatabaseSession


class TestUserJourney:
    """Complete user journey tests from registration to completion."""
    
    @pytest.fixture(scope="class")
    async def browser(self):
        """Initialize browser for E2E testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            yield browser
            await browser.close()
    
    @pytest.fixture(scope="function")
    async def page(self, browser: Browser):
        """Create a new page for each test."""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.fixture(scope="function")
    def test_user_data(self):
        """Test user data for registration."""
        return {
            'username': f'test_user_{os.urandom(4).hex()}',
            'email': f'test_{os.urandom(4).hex()}@example.com',
            'password': 'Test@Password123',
            'grade_level': 9,
            'learning_style': 'visual'
        }
    
    @pytest.mark.asyncio
    async def test_complete_registration_flow(self, page: Page, test_user_data: Dict[str, Any]):
        """Test complete user registration flow."""
        # Navigate to registration page
        await page.goto('http://localhost:3000/register')
        
        # Fill registration form
        await page.fill('input[name="username"]', test_user_data['username'])
        await page.fill('input[name="email"]', test_user_data['email'])
        await page.fill('input[name="password"]', test_user_data['password'])
        await page.fill('input[name="confirmPassword"]', test_user_data['password'])
        await page.select_option('select[name="gradeLevel"]', str(test_user_data['grade_level']))
        await page.select_option('select[name="learningStyle"]', test_user_data['learning_style'])
        
        # Submit form
        await page.click('button[type="submit"]')
        
        # Wait for navigation to dashboard
        await page.wait_for_url('**/dashboard', timeout=10000)
        
        # Verify dashboard loaded
        assert await page.is_visible('text=Dashboard')
        assert await page.is_visible(f'text={test_user_data["username"]}')
    
    @pytest.mark.asyncio
    async def test_login_and_navigation(self, page: Page, test_user_data: Dict[str, Any]):
        """Test login flow and main navigation."""
        # Navigate to login page
        await page.goto('http://localhost:3000/login')
        
        # Fill login form
        await page.fill('input[name="email"]', test_user_data['email'])
        await page.fill('input[name="password"]', test_user_data['password'])
        
        # Submit login
        await page.click('button[type="submit"]')
        
        # Wait for dashboard
        await page.wait_for_url('**/dashboard', timeout=10000)
        
        # Test navigation to different sections
        await page.click('a[href="/quiz"]')
        await page.wait_for_url('**/quiz')
        assert await page.is_visible('text=Quiz')
        
        await page.click('a[href="/learning-paths"]')
        await page.wait_for_url('**/learning-paths')
        assert await page.is_visible('text=Learning Paths')
        
        await page.click('a[href="/dashboard"]')
        await page.wait_for_url('**/dashboard')
        assert await page.is_visible('text=Dashboard')
    
    @pytest.mark.asyncio
    async def test_quiz_completion_flow(self, page: Page):
        """Test complete quiz taking flow."""
        # Assume user is logged in
        await page.goto('http://localhost:3000/quiz')
        
        # Start a quiz
        await page.click('button:has-text("Start Quiz")')
        
        # Answer questions
        for i in range(5):  # Assume 5 questions
            # Wait for question to load
            await page.wait_for_selector('.question-text', timeout=5000)
            
            # Select an answer
            await page.click('.answer-option:first-child')
            
            # Click next/submit
            if i < 4:
                await page.click('button:has-text("Next")')
            else:
                await page.click('button:has-text("Submit")')
        
        # Wait for results
        await page.wait_for_selector('.quiz-results', timeout=10000)
        
        # Verify results displayed
        assert await page.is_visible('text=Quiz Completed')
        assert await page.is_visible('.score-display')
    
    @pytest.mark.asyncio
    async def test_learning_path_creation(self, page: Page):
        """Test creating and following a learning path."""
        # Navigate to learning paths
        await page.goto('http://localhost:3000/learning-paths')
        
        # Create new learning path
        await page.click('button:has-text("Create Learning Path")')
        
        # Fill learning path details
        await page.fill('input[name="title"]', 'Test Learning Path')
        await page.fill('textarea[name="description"]', 'This is a test learning path')
        await page.select_option('select[name="subject"]', 'mathematics')
        await page.select_option('select[name="difficulty"]', 'intermediate')
        
        # Submit
        await page.click('button:has-text("Create")')
        
        # Wait for creation confirmation
        await page.wait_for_selector('.success-message', timeout=5000)
        
        # Verify path appears in list
        assert await page.is_visible('text=Test Learning Path')
    
    @pytest.mark.asyncio
    async def test_offline_mode_functionality(self, page: Page):
        """Test offline mode capabilities."""
        # Load page normally first
        await page.goto('http://localhost:3000/dashboard')
        
        # Wait for initial load
        await page.wait_for_load_state('networkidle')
        
        # Go offline
        await page.context.set_offline(True)
        
        # Try to navigate
        await page.reload()
        
        # Check offline indicator
        assert await page.is_visible('.offline-indicator')
        
        # Verify cached content still accessible
        assert await page.is_visible('text=Dashboard')
        
        # Go back online
        await page.context.set_offline(False)
        
        # Verify online status
        await page.wait_for_selector('.offline-indicator', state='hidden', timeout=5000)
    
    @pytest.mark.asyncio
    async def test_responsive_design(self, browser: Browser):
        """Test responsive design across different viewports."""
        viewports = [
            {'width': 375, 'height': 667, 'name': 'mobile'},
            {'width': 768, 'height': 1024, 'name': 'tablet'},
            {'width': 1920, 'height': 1080, 'name': 'desktop'}
        ]
        
        for viewport in viewports:
            context = await browser.new_context(
                viewport={'width': viewport['width'], 'height': viewport['height']},
                ignore_https_errors=True
            )
            page = await context.new_page()
            
            await page.goto('http://localhost:3000')
            
            # Check if mobile menu is visible for mobile viewport
            if viewport['name'] == 'mobile':
                assert await page.is_visible('.mobile-menu-toggle')
            else:
                assert await page.is_visible('.desktop-nav')
            
            await context.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, page: Page):
        """Test error handling and recovery."""
        # Test 404 page
        await page.goto('http://localhost:3000/non-existent-page')
        assert await page.is_visible('text=404')
        assert await page.is_visible('button:has-text("Go Home")')
        
        # Test form validation errors
        await page.goto('http://localhost:3000/login')
        await page.click('button[type="submit"]')  # Submit empty form
        assert await page.is_visible('.error-message')
        
        # Test API error handling
        await page.goto('http://localhost:3000/quiz')
        # Simulate network error
        await page.route('**/api/quiz/**', lambda route: route.abort())
        await page.click('button:has-text("Start Quiz")')
        assert await page.is_visible('.error-notification')
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, page: Page):
        """Test performance metrics and loading times."""
        # Enable performance tracking
        await page.goto('http://localhost:3000')
        
        # Measure page load metrics
        metrics = await page.evaluate('''() => {
            const perfData = window.performance.timing;
            return {
                loadTime: perfData.loadEventEnd - perfData.navigationStart,
                domContentLoaded: perfData.domContentLoadedEventEnd - perfData.navigationStart,
                firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0
            };
        }''')
        
        # Assert performance thresholds
        assert metrics['loadTime'] < 3000, "Page load time exceeds 3 seconds"
        assert metrics['domContentLoaded'] < 2000, "DOM content loaded time exceeds 2 seconds"
        assert metrics['firstPaint'] < 1000, "First paint time exceeds 1 second"
    
    @pytest.mark.asyncio
    async def test_accessibility(self, page: Page):
        """Test accessibility features."""
        await page.goto('http://localhost:3000')
        
        # Check for proper ARIA labels
        assert await page.get_attribute('nav', 'aria-label') is not None
        
        # Check for keyboard navigation
        await page.keyboard.press('Tab')
        focused_element = await page.evaluate('() => document.activeElement.tagName')
        assert focused_element in ['A', 'BUTTON', 'INPUT']
        
        # Check for alt text on images
        images = await page.query_selector_all('img')
        for img in images:
            alt_text = await img.get_attribute('alt')
            assert alt_text is not None and alt_text != ''
        
        # Check for proper heading hierarchy
        headings = await page.evaluate('''() => {
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            return Array.from(headings).map(h => h.tagName);
        }''')
        
        # Verify h1 exists and hierarchy is maintained
        assert 'H1' in headings