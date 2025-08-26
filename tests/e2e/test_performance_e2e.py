"""
Performance-focused E2E tests for the Teknofest 2025 Education Platform.
"""

import pytest
import asyncio
import time
import statistics
from playwright.async_api import async_playwright, Page, Browser
import aiohttp
from typing import List, Dict, Any
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))


class TestPerformanceE2E:
    """Performance-focused end-to-end tests."""
    
    @pytest.fixture(scope="class")
    async def performance_browser(self):
        """Browser with performance tracking enabled."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--enable-precise-memory-info',
                    '--disable-dev-shm-usage'
                ]
            )
            yield browser
            await browser.close()
    
    @pytest.fixture(scope="function")
    async def performance_page(self, performance_browser: Browser):
        """Page with performance tracking enabled."""
        context = await performance_browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        
        # Enable CDP for performance metrics
        page = await context.new_page()
        client = await page.context.new_cdp_session(page)
        await client.send('Performance.enable')
        
        yield page, client
        await context.close()
    
    @pytest.mark.asyncio
    async def test_page_load_performance(self, performance_page):
        """Test page load performance metrics."""
        page, cdp_client = performance_page
        
        pages_to_test = [
            ('/', 'Homepage'),
            ('/login', 'Login Page'),
            ('/dashboard', 'Dashboard'),
            ('/quiz', 'Quiz Page'),
            ('/learning-paths', 'Learning Paths')
        ]
        
        results = []
        
        for path, name in pages_to_test:
            # Clear cache and cookies
            await page.context.clear_cookies()
            
            # Start performance measurement
            start_time = time.time()
            
            # Navigate to page
            await page.goto(f'http://localhost:3000{path}')
            await page.wait_for_load_state('networkidle')
            
            # Get performance metrics
            metrics = await cdp_client.send('Performance.getMetrics')
            
            # Get Core Web Vitals
            web_vitals = await page.evaluate('''() => {
                return new Promise((resolve) => {
                    let fcp = 0, lcp = 0, cls = 0, fid = 0;
                    
                    // First Contentful Paint
                    const fcpEntry = performance.getEntriesByName('first-contentful-paint')[0];
                    if (fcpEntry) fcp = fcpEntry.startTime;
                    
                    // Largest Contentful Paint
                    new PerformanceObserver((list) => {
                        const entries = list.getEntries();
                        const lastEntry = entries[entries.length - 1];
                        lcp = lastEntry.startTime;
                    }).observe({ type: 'largest-contentful-paint', buffered: true });
                    
                    // Cumulative Layout Shift
                    let clsValue = 0;
                    new PerformanceObserver((list) => {
                        for (const entry of list.getEntries()) {
                            if (!entry.hadRecentInput) {
                                clsValue += entry.value;
                            }
                        }
                        cls = clsValue;
                    }).observe({ type: 'layout-shift', buffered: true });
                    
                    // First Input Delay (simulated)
                    if (performance.getEntriesByType('first-input').length > 0) {
                        fid = performance.getEntriesByType('first-input')[0].processingStart -
                              performance.getEntriesByType('first-input')[0].startTime;
                    }
                    
                    setTimeout(() => {
                        resolve({
                            FCP: fcp,
                            LCP: lcp,
                            CLS: cls,
                            FID: fid,
                            TTFB: performance.timing.responseStart - performance.timing.fetchStart
                        });
                    }, 2000);
                });
            }''')
            
            load_time = time.time() - start_time
            
            results.append({
                'page': name,
                'load_time': load_time,
                'metrics': metrics['metrics'],
                'web_vitals': web_vitals
            })
            
            # Performance assertions
            assert load_time < 3, f"{name} load time {load_time}s exceeds 3s threshold"
            assert web_vitals['FCP'] < 1800, f"{name} FCP {web_vitals['FCP']}ms exceeds 1800ms"
            assert web_vitals['LCP'] < 2500, f"{name} LCP {web_vitals['LCP']}ms exceeds 2500ms"
            assert web_vitals['CLS'] < 0.1, f"{name} CLS {web_vitals['CLS']} exceeds 0.1"
        
        # Generate performance report
        self._generate_performance_report(results)
    
    @pytest.mark.asyncio
    async def test_api_response_times(self):
        """Test API endpoint response times."""
        async with aiohttp.ClientSession() as session:
            endpoints = [
                ('GET', '/api/v1/health', None),
                ('GET', '/api/v1/quiz/categories', None),
                ('POST', '/api/v1/auth/login', {'email': 'test@test.com', 'password': 'test'}),
                ('GET', '/api/v1/learning-paths', None),
                ('GET', '/api/v1/leaderboard', None)
            ]
            
            response_times = {}
            
            for method, endpoint, data in endpoints:
                times = []
                
                # Make multiple requests to get average
                for _ in range(10):
                    start = time.time()
                    
                    if method == 'GET':
                        async with session.get(f'http://localhost:8000{endpoint}') as response:
                            await response.text()
                    else:
                        async with session.post(f'http://localhost:8000{endpoint}', json=data) as response:
                            await response.text()
                    
                    times.append((time.time() - start) * 1000)  # Convert to ms
                
                avg_time = statistics.mean(times)
                p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
                
                response_times[endpoint] = {
                    'avg': avg_time,
                    'p95': p95_time,
                    'min': min(times),
                    'max': max(times)
                }
                
                # Assert performance requirements
                assert avg_time < 200, f"{endpoint} avg response time {avg_time}ms exceeds 200ms"
                assert p95_time < 500, f"{endpoint} p95 response time {p95_time}ms exceeds 500ms"
            
            return response_times
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test system performance under concurrent requests."""
        async with aiohttp.ClientSession() as session:
            async def make_request(endpoint: str):
                start = time.time()
                try:
                    async with session.get(f'http://localhost:8000{endpoint}') as response:
                        await response.text()
                        return time.time() - start, response.status
                except Exception as e:
                    return time.time() - start, 500
            
            # Test different concurrency levels
            concurrency_levels = [10, 50, 100, 200]
            results = {}
            
            for level in concurrency_levels:
                tasks = [make_request('/api/v1/health') for _ in range(level)]
                responses = await asyncio.gather(*tasks)
                
                response_times = [r[0] for r in responses]
                status_codes = [r[1] for r in responses]
                
                success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)
                
                results[level] = {
                    'avg_response_time': statistics.mean(response_times),
                    'p95_response_time': statistics.quantiles(response_times, n=20)[18],
                    'success_rate': success_rate
                }
                
                # Performance assertions
                assert success_rate > 0.95, f"Success rate {success_rate} at {level} concurrent requests"
                assert results[level]['avg_response_time'] < 1, f"Avg response time too high at {level} concurrent requests"
        
        return results
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, performance_page):
        """Test memory usage patterns."""
        page, cdp_client = performance_page
        
        memory_readings = []
        
        # Navigate through different pages and measure memory
        pages = ['/', '/dashboard', '/quiz', '/learning-paths']
        
        for path in pages:
            await page.goto(f'http://localhost:3000{path}')
            await page.wait_for_load_state('networkidle')
            
            # Get memory info
            memory_info = await page.evaluate('''() => {
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize,
                        totalJSHeapSize: performance.memory.totalJSHeapSize,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                    };
                }
                return null;
            }''')
            
            if memory_info:
                memory_readings.append(memory_info)
                
                # Check for memory leaks (heap should not grow excessively)
                heap_usage_ratio = memory_info['usedJSHeapSize'] / memory_info['jsHeapSizeLimit']
                assert heap_usage_ratio < 0.8, f"High memory usage on {path}: {heap_usage_ratio * 100}%"
        
        # Check for memory leak patterns
        if len(memory_readings) > 1:
            heap_sizes = [m['usedJSHeapSize'] for m in memory_readings]
            # Memory shouldn't consistently increase
            increasing_count = sum(1 for i in range(1, len(heap_sizes)) if heap_sizes[i] > heap_sizes[i-1])
            assert increasing_count < len(heap_sizes) - 1, "Potential memory leak detected"
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """Test database query performance."""
        from src.database.session import DatabaseSession
        
        async with DatabaseSession() as session:
            queries_to_test = [
                ("SELECT COUNT(*) FROM users", "User count"),
                ("SELECT * FROM quizzes LIMIT 100", "Quiz fetch"),
                ("SELECT * FROM questions WHERE quiz_id = 1", "Questions by quiz"),
                ("SELECT * FROM user_progress WHERE user_id = 1", "User progress")
            ]
            
            results = []
            
            for query, name in queries_to_test:
                start = time.time()
                result = await session.execute(query)
                await result.fetchall()
                query_time = (time.time() - start) * 1000
                
                results.append({
                    'query': name,
                    'time_ms': query_time
                })
                
                # Assert query performance
                assert query_time < 100, f"Query '{name}' took {query_time}ms, exceeds 100ms threshold"
            
            return results
    
    @pytest.mark.asyncio
    async def test_static_asset_optimization(self, performance_page):
        """Test static asset optimization and caching."""
        page, _ = performance_page
        
        # Monitor network requests
        network_requests = []
        
        async def log_request(request):
            network_requests.append({
                'url': request.url,
                'method': request.method,
                'resource_type': request.resource_type
            })
        
        page.on('request', log_request)
        
        # First load
        await page.goto('http://localhost:3000')
        await page.wait_for_load_state('networkidle')
        first_load_requests = len(network_requests)
        
        # Clear requests tracker
        network_requests.clear()
        
        # Second load (should use cache)
        await page.reload()
        await page.wait_for_load_state('networkidle')
        second_load_requests = len(network_requests)
        
        # Check if caching is working
        assert second_load_requests < first_load_requests, "Caching not effective"
        
        # Check for compression
        response = await page.goto('http://localhost:3000')
        headers = response.headers
        
        # Check for compression headers
        if 'content-encoding' in headers:
            assert headers['content-encoding'] in ['gzip', 'br', 'deflate'], "Assets not compressed"
    
    @pytest.mark.asyncio
    async def test_websocket_performance(self, performance_page):
        """Test WebSocket connection performance."""
        page, _ = performance_page
        
        await page.goto('http://localhost:3000/dashboard')
        
        # Test WebSocket connection establishment
        ws_performance = await page.evaluate('''async () => {
            return new Promise((resolve) => {
                const startTime = performance.now();
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = () => {
                    const connectionTime = performance.now() - startTime;
                    
                    // Test message round-trip time
                    const messageStart = performance.now();
                    ws.send(JSON.stringify({ type: 'ping' }));
                    
                    ws.onmessage = (event) => {
                        const roundTripTime = performance.now() - messageStart;
                        ws.close();
                        
                        resolve({
                            connectionTime,
                            roundTripTime
                        });
                    };
                };
                
                ws.onerror = () => {
                    resolve({ error: true });
                };
                
                setTimeout(() => {
                    resolve({ timeout: true });
                }, 5000);
            });
        }''')
        
        if not ws_performance.get('error') and not ws_performance.get('timeout'):
            assert ws_performance['connectionTime'] < 1000, "WebSocket connection too slow"
            assert ws_performance['roundTripTime'] < 100, "WebSocket round-trip time too high"
    
    @pytest.mark.asyncio
    async def test_bundle_size_optimization(self):
        """Test JavaScript bundle sizes."""
        async with aiohttp.ClientSession() as session:
            # Get main bundle
            async with session.get('http://localhost:3000/_next/static/chunks/main.js') as response:
                if response.status == 200:
                    content = await response.read()
                    bundle_size = len(content) / 1024  # KB
                    
                    # Check if bundle is minified
                    text = content.decode('utf-8', errors='ignore')
                    is_minified = '\n' not in text[:1000] or len(text.split('\n')) < 10
                    
                    assert is_minified, "JavaScript bundle not minified"
                    assert bundle_size < 500, f"Main bundle size {bundle_size}KB exceeds 500KB"
    
    def _generate_performance_report(self, results: List[Dict[str, Any]]):
        """Generate performance test report."""
        report = {
            'timestamp': time.time(),
            'results': results,
            'summary': {
                'avg_load_time': statistics.mean([r['load_time'] for r in results]),
                'max_load_time': max([r['load_time'] for r in results]),
                'avg_fcp': statistics.mean([r['web_vitals']['FCP'] for r in results]),
                'avg_lcp': statistics.mean([r['web_vitals']['LCP'] for r in results])
            }
        }
        
        # Save report
        with open('performance_e2e_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report