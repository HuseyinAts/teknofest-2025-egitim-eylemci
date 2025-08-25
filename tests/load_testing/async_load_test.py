"""
Async Load Testing Script for TEKNOFEST 2025 API
Standalone script that doesn't require Locust
"""

import asyncio
import aiohttp
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class TestResult:
    """Store individual test result"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: str = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LoadTestStats:
    """Aggregate statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    endpoint_stats: Dict = field(default_factory=lambda: defaultdict(lambda: {
        'count': 0, 
        'success': 0, 
        'fail': 0, 
        'times': []
    }))
    
    def add_result(self, result: TestResult):
        """Add test result to statistics"""
        self.total_requests += 1
        
        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if result.error:
                self.errors.append(result.error)
        
        self.response_times.append(result.response_time)
        
        # Update endpoint-specific stats
        endpoint_key = f"{result.method} {result.endpoint}"
        self.endpoint_stats[endpoint_key]['count'] += 1
        self.endpoint_stats[endpoint_key]['times'].append(result.response_time)
        
        if result.success:
            self.endpoint_stats[endpoint_key]['success'] += 1
        else:
            self.endpoint_stats[endpoint_key]['fail'] += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.response_times:
            return {"error": "No data collected"}
        
        sorted_times = sorted(self.response_times)
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0,
                "p99": sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0,
            },
            "errors_count": len(self.errors),
            "unique_errors": len(set(self.errors))
        }
    
    def get_endpoint_summary(self) -> Dict:
        """Get per-endpoint statistics"""
        summary = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats['times']:
                summary[endpoint] = {
                    "count": stats['count'],
                    "success": stats['success'],
                    "fail": stats['fail'],
                    "success_rate": (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0,
                    "avg_response_time": statistics.mean(stats['times']),
                    "max_response_time": max(stats['times']),
                    "min_response_time": min(stats['times'])
                }
        return summary


class LoadTester:
    """Async load tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.stats = LoadTestStats()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> TestResult:
        """Make HTTP request and measure response time"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.perf_counter() - start_time
                
                # Try to read response body
                try:
                    await response.json()
                except:
                    await response.text()
                
                return TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=(200 <= response.status < 300)
                )
                
        except asyncio.TimeoutError:
            response_time = time.perf_counter() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            response_time = time.perf_counter() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def test_health(self):
        """Test health endpoint"""
        result = await self.make_request("GET", "/health")
        self.stats.add_result(result)
        return result
    
    async def test_root(self):
        """Test root endpoint"""
        result = await self.make_request("GET", "/")
        self.stats.add_result(result)
        return result
    
    async def test_learning_style(self):
        """Test learning style detection"""
        responses = [
            "Görsel materyalleri tercih ederim",
            "Dinleyerek daha iyi öğrenirim",
            "Okuyarak öğrenmeyi severim"
        ]
        
        result = await self.make_request(
            "POST",
            "/api/v1/learning-style",
            json={"student_responses": random.sample(responses, k=2)}
        )
        self.stats.add_result(result)
        return result
    
    async def test_generate_quiz(self):
        """Test quiz generation"""
        topics = ["Matematik", "Fizik", "Kimya", "Biyoloji"]
        
        result = await self.make_request(
            "POST",
            "/api/v1/generate-quiz",
            json={
                "topic": random.choice(topics),
                "student_ability": round(random.uniform(0.3, 0.8), 2),
                "num_questions": random.randint(5, 10)
            }
        )
        self.stats.add_result(result)
        return result
    
    async def test_generate_text(self):
        """Test text generation"""
        prompts = [
            "Pisagor teoremi nedir?",
            "Newton yasaları",
            "Fotosentez"
        ]
        
        result = await self.make_request(
            "POST",
            "/api/v1/generate-text",
            json={
                "prompt": random.choice(prompts),
                "max_length": 200
            }
        )
        self.stats.add_result(result)
        return result
    
    async def test_curriculum(self):
        """Test curriculum endpoint"""
        grade = random.randint(9, 12)
        
        result = await self.make_request(
            "GET",
            f"/api/v1/curriculum/{grade}"
        )
        self.stats.add_result(result)
        return result
    
    async def test_data_stats(self):
        """Test data stats endpoint"""
        result = await self.make_request("GET", "/api/v1/data/stats")
        self.stats.add_result(result)
        return result
    
    async def run_user_scenario(self, user_id: int):
        """Simulate a single user scenario"""
        # Randomize test sequence
        tests = [
            (self.test_health, 0.1),         # 10% chance
            (self.test_root, 0.05),           # 5% chance
            (self.test_learning_style, 0.15), # 15% chance
            (self.test_generate_quiz, 0.35),  # 35% chance
            (self.test_generate_text, 0.2),   # 20% chance
            (self.test_curriculum, 0.1),      # 10% chance
            (self.test_data_stats, 0.05),     # 5% chance
        ]
        
        # Run 10-20 requests per user
        num_requests = random.randint(10, 20)
        
        for _ in range(num_requests):
            # Select test based on probability
            r = random.random()
            cumulative = 0
            
            for test_func, probability in tests:
                cumulative += probability
                if r <= cumulative:
                    await test_func()
                    break
            
            # Random delay between requests (0.5-2 seconds)
            await asyncio.sleep(random.uniform(0.5, 2))
    
    async def run_load_test(self, num_users: int = 10, duration: int = 30):
        """Run load test with concurrent users"""
        print(f"\n[STARTING] Load test with {num_users} concurrent users for {duration} seconds")
        print(f"Target: {self.base_url}")
        print("-" * 60)
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Create user tasks
        user_tasks = []
        user_id = 0
        
        while time.time() < end_time:
            # Add new users gradually
            if len(user_tasks) < num_users:
                user_id += 1
                task = asyncio.create_task(self.run_user_scenario(user_id))
                user_tasks.append(task)
                await asyncio.sleep(0.1)  # Ramp-up delay
            
            # Clean up completed tasks
            user_tasks = [t for t in user_tasks if not t.done()]
            
            # Status update every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                print(f"[PROGRESS] {int(time.time() - start_time)}/{duration}s | "
                      f"Requests: {self.stats.total_requests} | "
                      f"Success Rate: {self.stats.successful_requests/max(1, self.stats.total_requests)*100:.1f}%")
            
            await asyncio.sleep(0.1)
        
        # Wait for remaining tasks to complete
        if user_tasks:
            print("\n[WAITING] Waiting for remaining requests to complete...")
            await asyncio.gather(*user_tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        print(f"\n[COMPLETE] Load test completed in {actual_duration:.1f} seconds")
        
        return self.stats


def print_results(stats: LoadTestStats):
    """Print load test results"""
    summary = stats.get_summary()
    endpoint_summary = stats.get_endpoint_summary()
    
    print("\n" + "=" * 70)
    print("LOAD TEST RESULTS")
    print("=" * 70)
    
    print("\n[STATS] Overall Statistics:")
    print(f"  Total Requests:        {summary['total_requests']}")
    print(f"  Successful:            {summary['successful_requests']} ({summary['success_rate']:.1f}%)")
    print(f"  Failed:                {summary['failed_requests']}")
    print(f"  Errors:                {summary['errors_count']} ({summary['unique_errors']} unique)")
    
    if summary['total_requests'] > 0:
        print(f"\n[TIMING] Response Times (seconds):")
        print(f"  Min:                   {summary['response_times']['min']:.3f}s")
        print(f"  Max:                   {summary['response_times']['max']:.3f}s")
        print(f"  Mean:                  {summary['response_times']['mean']:.3f}s")
        print(f"  Median:                {summary['response_times']['median']:.3f}s")
        print(f"  95th Percentile:       {summary['response_times']['p95']:.3f}s")
        print(f"  99th Percentile:       {summary['response_times']['p99']:.3f}s")
    
    print(f"\n[ENDPOINTS] Per-Endpoint Statistics:")
    for endpoint, endpoint_stats in endpoint_summary.items():
        print(f"\n  {endpoint}:")
        print(f"    Requests:            {endpoint_stats['count']}")
        print(f"    Success Rate:        {endpoint_stats['success_rate']:.1f}%")
        print(f"    Avg Response Time:   {endpoint_stats['avg_response_time']:.3f}s")
        print(f"    Min/Max Time:        {endpoint_stats['min_response_time']:.3f}s / {endpoint_stats['max_response_time']:.3f}s")
    
    # Performance evaluation
    print("\n" + "=" * 70)
    print("PERFORMANCE EVALUATION")
    print("=" * 70)
    
    if summary['total_requests'] > 0:
        # Success rate evaluation
        if summary['success_rate'] >= 99:
            print("[EXCELLENT] Success rate > 99%")
        elif summary['success_rate'] >= 95:
            print("[GOOD] Success rate > 95%")
        elif summary['success_rate'] >= 90:
            print("[WARNING] Success rate 90-95%")
        else:
            print("[CRITICAL] Success rate < 90%")
        
        # Response time evaluation
        mean_time = summary['response_times']['mean']
        if mean_time < 0.2:
            print("[EXCELLENT] Mean response time < 200ms")
        elif mean_time < 0.5:
            print("[GOOD] Mean response time < 500ms")
        elif mean_time < 1.0:
            print("[WARNING] Mean response time 500ms-1s")
        else:
            print("[CRITICAL] Mean response time > 1s")
        
        # P95 evaluation
        p95_time = summary['response_times']['p95']
        if p95_time < 0.5:
            print("[EXCELLENT] P95 response time < 500ms")
        elif p95_time < 1.0:
            print("[GOOD] P95 response time < 1s")
        elif p95_time < 2.0:
            print("[WARNING] P95 response time 1-2s")
        else:
            print("[CRITICAL] P95 response time > 2s")
    
    print("\n" + "=" * 70)


async def main():
    """Main function"""
    # Test configurations
    configs = [
        {"num_users": 5, "duration": 20, "name": "Light Load"},
        {"num_users": 20, "duration": 30, "name": "Medium Load"},
        {"num_users": 50, "duration": 30, "name": "Heavy Load"},
        {"num_users": 100, "duration": 30, "name": "Stress Test"},
    ]
    
    print("\n[TARGET] TEKNOFEST 2025 API Load Testing")
    print("=" * 70)
    
    # Let user choose test configuration
    print("\nAvailable test configurations:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']} - {config['num_users']} users, {config['duration']}s")
    
    # Default to light load for automated testing
    choice = 1
    config = configs[choice - 1]
    
    print(f"\n[SELECTED] {config['name']}")
    
    # Run load test
    async with LoadTester() as tester:
        stats = await tester.run_load_test(
            num_users=config['num_users'],
            duration=config['duration']
        )
        
        # Print results
        print_results(stats)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"load_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "config": config,
                "summary": stats.get_summary(),
                "endpoint_stats": stats.get_endpoint_summary(),
                "timestamp": timestamp
            }, f, indent=2, default=str)
        
        print(f"\n[SAVED] Results saved to: {results_file}")
    
    return stats


if __name__ == "__main__":
    # Run the load test
    asyncio.run(main())