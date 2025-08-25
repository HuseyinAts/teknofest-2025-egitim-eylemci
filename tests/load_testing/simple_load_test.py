"""
Simple Load Test for Quick Results
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List

async def test_endpoint(session, url: str, num_requests: int = 10) -> List[float]:
    """Test a single endpoint multiple times"""
    response_times = []
    
    for _ in range(num_requests):
        start = time.perf_counter()
        try:
            async with session.get(url, timeout=5) as response:
                await response.text()
                response_time = time.perf_counter() - start
                response_times.append(response_time)
                print(f"  Request completed: {response.status} in {response_time:.3f}s")
        except Exception as e:
            print(f"  Request failed: {e}")
            response_times.append(5.0)  # Timeout value
    
    return response_times

async def quick_load_test():
    """Run quick load test"""
    base_url = "http://localhost:8003"
    
    print("[LOAD TEST] Starting quick load test")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("\n[TEST] Health endpoint (10 requests):")
        health_times = await test_endpoint(session, f"{base_url}/health", 10)
        
        # Test root endpoint
        print("\n[TEST] Root endpoint (10 requests):")
        root_times = await test_endpoint(session, f"{base_url}/", 10)
        
        # Test quiz generation with concurrent requests
        print("\n[TEST] Quiz generation (5 concurrent requests):")
        quiz_url = f"{base_url}/api/v1/generate-quiz"
        quiz_data = {
            "topic": "Matematik",
            "student_ability": 0.5,
            "num_questions": 5
        }
        
        async def post_quiz():
            start = time.perf_counter()
            try:
                async with session.post(quiz_url, json=quiz_data, timeout=10) as response:
                    await response.text()
                    elapsed = time.perf_counter() - start
                    print(f"  Quiz request: {response.status} in {elapsed:.3f}s")
                    return elapsed
            except Exception as e:
                print(f"  Quiz request failed: {e}")
                return 10.0
        
        quiz_tasks = [post_quiz() for _ in range(5)]
        quiz_times = await asyncio.gather(*quiz_tasks)
        
    # Print results
    print("\n" + "=" * 50)
    print("[RESULTS] Load Test Summary")
    print("=" * 50)
    
    all_times = health_times + root_times + list(quiz_times)
    
    print(f"\nTotal requests:     {len(all_times)}")
    print(f"Min response time:  {min(all_times):.3f}s")
    print(f"Max response time:  {max(all_times):.3f}s")
    print(f"Mean response time: {statistics.mean(all_times):.3f}s")
    print(f"Median response:    {statistics.median(all_times):.3f}s")
    
    # Performance evaluation
    mean_time = statistics.mean(all_times)
    if mean_time < 0.2:
        print("\n[EXCELLENT] Mean response time < 200ms")
    elif mean_time < 0.5:
        print("\n[GOOD] Mean response time < 500ms")
    elif mean_time < 1.0:
        print("\n[WARNING] Mean response time 500ms-1s")
    else:
        print("\n[CRITICAL] Mean response time > 1s")
    
    print("\n[COMPLETE] Load test finished")

if __name__ == "__main__":
    asyncio.run(quick_load_test())