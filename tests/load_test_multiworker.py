"""
Load Testing Script for Multi-Worker Deployment
TEKNOFEST 2025 - Production Performance Verification
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import random
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class TestResult:
    """Store test results for analysis"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    success: bool
    error: str = ""
    worker_id: str = ""
    
@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    base_url: str = "http://localhost:8000"
    num_users: int = 100
    requests_per_user: int = 10
    ramp_up_time: int = 10  # seconds
    test_duration: int = 60  # seconds
    think_time: float = 1.0  # seconds between requests
    
class LoadTester:
    """Load testing orchestrator"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
    async def run_test(self):
        """Run the complete load test"""
        print(f"Starting load test with {self.config.num_users} users")
        print(f"Target: {self.config.base_url}")
        print("-" * 60)
        
        self.start_time = time.time()
        
        # Create user tasks
        tasks = []
        for user_id in range(self.config.num_users):
            # Stagger user start times (ramp-up)
            delay = (user_id / self.config.num_users) * self.config.ramp_up_time
            tasks.append(self.simulate_user(user_id, delay))
        
        # Run all user simulations concurrently
        await asyncio.gather(*tasks)
        
        self.end_time = time.time()
        
        # Generate report
        self.generate_report()
        
    async def simulate_user(self, user_id: int, start_delay: float):
        """Simulate a single user's behavior"""
        # Wait for ramp-up
        await asyncio.sleep(start_delay)
        
        async with aiohttp.ClientSession() as session:
            test_start = time.time()
            request_count = 0
            
            # Run requests for test duration or request limit
            while (time.time() - test_start < self.config.test_duration and 
                   request_count < self.config.requests_per_user):
                
                # Select random endpoint to test
                endpoint_test = random.choice([
                    self.test_health_endpoint,
                    self.test_api_endpoint,
                    self.test_model_inference,
                    self.test_data_processing,
                ])
                
                # Execute test
                await endpoint_test(session, user_id)
                
                # Think time between requests
                await asyncio.sleep(self.config.think_time + random.uniform(-0.5, 0.5))
                
                request_count += 1
    
    async def test_health_endpoint(self, session: aiohttp.ClientSession, user_id: int):
        """Test health check endpoint"""
        await self._make_request(
            session,
            method="GET",
            endpoint="/health",
            user_id=user_id
        )
    
    async def test_api_endpoint(self, session: aiohttp.ClientSession, user_id: int):
        """Test main API endpoint"""
        await self._make_request(
            session,
            method="GET",
            endpoint="/api/v1/status",
            user_id=user_id
        )
    
    async def test_model_inference(self, session: aiohttp.ClientSession, user_id: int):
        """Test model inference endpoint"""
        data = {
            "text": f"Test input from user {user_id}: " + "x" * random.randint(100, 500),
            "model": "default",
            "parameters": {
                "temperature": 0.7,
                "max_length": 100
            }
        }
        
        await self._make_request(
            session,
            method="POST",
            endpoint="/api/v1/inference",
            user_id=user_id,
            json_data=data
        )
    
    async def test_data_processing(self, session: aiohttp.ClientSession, user_id: int):
        """Test data processing endpoint"""
        data = {
            "dataset_id": f"test_dataset_{user_id}",
            "operation": random.choice(["analyze", "transform", "validate"]),
            "parameters": {
                "batch_size": 32,
                "async": True
            }
        }
        
        await self._make_request(
            session,
            method="POST",
            endpoint="/api/v1/process",
            user_id=user_id,
            json_data=data
        )
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        user_id: int,
        json_data: Dict = None
    ):
        """Make HTTP request and record results"""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=url,
                json=json_data,
                headers={
                    "User-Agent": f"LoadTester/User-{user_id}",
                    "X-Request-ID": f"{user_id}-{int(time.time()*1000)}"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                # Try to get worker ID from headers
                worker_id = response.headers.get("X-Worker-ID", "unknown")
                
                result = TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    success=(200 <= response.status < 300),
                    worker_id=worker_id
                )
                
                self.results.append(result)
                
                # Print progress
                if len(self.results) % 100 == 0:
                    print(f"Completed {len(self.results)} requests...")
                    
        except asyncio.TimeoutError:
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=30.0,
                timestamp=datetime.now(),
                success=False,
                error="Timeout"
            )
            self.results.append(result)
            
        except Exception as e:
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
            self.results.append(result)
    
    def generate_report(self):
        """Generate and print load test report"""
        if not self.results:
            print("No results to report")
            return
        
        print("\n" + "=" * 60)
        print("LOAD TEST REPORT")
        print("=" * 60)
        
        # Overall statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100
        
        response_times = [r.response_time for r in self.results if r.success]
        
        print(f"\nTest Duration: {self.end_time - self.start_time:.2f} seconds")
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {successful_requests} ({success_rate:.2f}%)")
        print(f"Failed: {failed_requests}")
        
        if response_times:
            print(f"\nResponse Time Statistics (successful requests):")
            print(f"  Min: {min(response_times):.3f}s")
            print(f"  Max: {max(response_times):.3f}s")
            print(f"  Mean: {statistics.mean(response_times):.3f}s")
            print(f"  Median: {statistics.median(response_times):.3f}s")
            
            # Percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(len(sorted_times) * 0.50)]
            p90 = sorted_times[int(len(sorted_times) * 0.90)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
            
            print(f"\nPercentiles:")
            print(f"  50th (median): {p50:.3f}s")
            print(f"  90th: {p90:.3f}s")
            print(f"  95th: {p95:.3f}s")
            print(f"  99th: {p99:.3f}s")
        
        # Requests per second
        test_duration = self.end_time - self.start_time
        rps = total_requests / test_duration
        print(f"\nThroughput: {rps:.2f} requests/second")
        
        # Per-endpoint statistics
        print("\nPer-Endpoint Statistics:")
        endpoints = {}
        for result in self.results:
            if result.endpoint not in endpoints:
                endpoints[result.endpoint] = {
                    'count': 0,
                    'success': 0,
                    'response_times': []
                }
            
            endpoints[result.endpoint]['count'] += 1
            if result.success:
                endpoints[result.endpoint]['success'] += 1
                endpoints[result.endpoint]['response_times'].append(result.response_time)
        
        for endpoint, stats in endpoints.items():
            success_rate = (stats['success'] / stats['count']) * 100
            avg_time = statistics.mean(stats['response_times']) if stats['response_times'] else 0
            print(f"\n  {endpoint}:")
            print(f"    Requests: {stats['count']}")
            print(f"    Success Rate: {success_rate:.2f}%")
            print(f"    Avg Response Time: {avg_time:.3f}s")
        
        # Worker distribution
        print("\nWorker Distribution:")
        worker_counts = {}
        for result in self.results:
            if result.worker_id:
                worker_counts[result.worker_id] = worker_counts.get(result.worker_id, 0) + 1
        
        for worker_id, count in sorted(worker_counts.items()):
            percentage = (count / total_requests) * 100
            print(f"  {worker_id}: {count} requests ({percentage:.2f}%)")
        
        # Error analysis
        if failed_requests > 0:
            print("\nError Analysis:")
            error_counts = {}
            for result in self.results:
                if not result.success:
                    error_type = result.error or f"HTTP {result.status_code}"
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        # Performance verdict
        print("\n" + "=" * 60)
        print("PERFORMANCE VERDICT:")
        
        if success_rate >= 99 and p95 < 1.0:
            print("✅ EXCELLENT - Production ready!")
        elif success_rate >= 95 and p95 < 2.0:
            print("✅ GOOD - Acceptable for production")
        elif success_rate >= 90 and p95 < 3.0:
            print("⚠️  FAIR - Needs optimization")
        else:
            print("❌ POOR - Not ready for production")
        
        print("=" * 60)
        
        # Save detailed results to file
        self.save_results()
    
    def save_results(self):
        """Save detailed results to JSON file"""
        filename = f"load_test_results_{int(time.time())}.json"
        
        data = {
            "config": {
                "base_url": self.config.base_url,
                "num_users": self.config.num_users,
                "requests_per_user": self.config.requests_per_user,
                "test_duration": self.config.test_duration
            },
            "summary": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.end_time - self.start_time,
                "total_requests": len(self.results),
                "successful_requests": sum(1 for r in self.results if r.success),
                "failed_requests": sum(1 for r in self.results if not r.success)
            },
            "results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "response_time": r.response_time,
                    "timestamp": r.timestamp.isoformat(),
                    "success": r.success,
                    "error": r.error,
                    "worker_id": r.worker_id
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Load test for multi-worker deployment")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rampup", type=int, default=10, help="Ramp-up time in seconds")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        base_url=args.url,
        num_users=args.users,
        requests_per_user=args.requests,
        test_duration=args.duration,
        ramp_up_time=args.rampup
    )
    
    tester = LoadTester(config)
    
    # Run test
    asyncio.run(tester.run_test())

if __name__ == "__main__":
    main()