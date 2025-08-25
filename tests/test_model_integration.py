# -*- coding: utf-8 -*-
"""
Comprehensive Testing for Model Integration
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import sys
import os
import time
import json
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_inference import TeknofestModel, get_model
from src.agents.learning_path_agent_enhanced import LearningPathAgent, StudentProfile
from src.agents.study_buddy_agent import StudyBuddyAgent


class ModelIntegrationTester:
    """Comprehensive test suite for model integration"""
    
    def __init__(self):
        self.results = []
        self.model = None
        self.learning_agent = None
        self.study_agent = None
        
    def setup(self, use_model: bool = True):
        """Setup test environment"""
        print("\n" + "="*60)
        print("TEKNOFEST 2025 - Model Integration Test Suite")
        print("="*60)
        
        if use_model:
            print("\n[SETUP] Loading fine-tuned model...")
            try:
                self.model = get_model()
                print("[OK] Model loaded successfully")
                model_available = True
            except Exception as e:
                print(f"[WARNING] Model loading failed: {e}")
                model_available = False
        else:
            print("\n[SETUP] Running tests without model...")
            model_available = False
        
        # Initialize agents
        print("[SETUP] Initializing agents...")
        self.learning_agent = LearningPathAgent(use_model=model_available)
        self.study_agent = StudyBuddyAgent(use_model=model_available)
        print(f"[OK] Agents initialized (model_enabled={model_available})")
        
        return model_available
    
    def test_model_inference(self):
        """Test 1: Direct model inference"""
        print("\n[TEST 1] Direct Model Inference")
        print("-" * 40)
        
        if not self.model:
            print("[SKIP] Model not available")
            return {"test": "model_inference", "status": "skipped"}
        
        try:
            # Test basic generation
            start_time = time.time()
            response = self.model.generate_response(
                "Pisagor teoremi nedir?",
                max_new_tokens=100
            )
            elapsed = time.time() - start_time
            
            assert len(response) > 0, "Empty response"
            
            print(f"[OK] Generated response in {elapsed:.2f}s")
            print(f"Response preview: {response[:100]}...")
            
            return {
                "test": "model_inference",
                "status": "passed",
                "time": elapsed,
                "response_length": len(response)
            }
            
        except Exception as e:
            print(f"[FAIL] {e}")
            return {"test": "model_inference", "status": "failed", "error": str(e)}
    
    def test_educational_content_generation(self):
        """Test 2: Educational content generation"""
        print("\n[TEST 2] Educational Content Generation")
        print("-" * 40)
        
        if not self.model:
            print("[SKIP] Model not available")
            return {"test": "educational_content", "status": "skipped"}
        
        test_cases = [
            ("Matematik", "Denklemler", 9, "explanation"),
            ("Fizik", "Newton Yasaları", 10, "quiz"),
            ("Kimya", "Periyodik Tablo", 11, "example")
        ]
        
        results = []
        
        for subject, topic, grade, content_type in test_cases:
            try:
                start_time = time.time()
                content = self.model.generate_educational_content(
                    subject=subject,
                    topic=topic,
                    grade_level=grade,
                    content_type=content_type,
                    learning_style="visual"
                )
                elapsed = time.time() - start_time
                
                assert len(content) > 0, f"Empty content for {topic}"
                
                print(f"[OK] {subject}/{topic} - {content_type}: {elapsed:.2f}s")
                results.append({
                    "subject": subject,
                    "topic": topic,
                    "status": "passed",
                    "time": elapsed
                })
                
            except Exception as e:
                print(f"[FAIL] {subject}/{topic}: {e}")
                results.append({
                    "subject": subject,
                    "topic": topic,
                    "status": "failed",
                    "error": str(e)
                })
        
        passed = sum(1 for r in results if r["status"] == "passed")
        total = len(results)
        
        return {
            "test": "educational_content",
            "status": "passed" if passed == total else "partial",
            "passed": passed,
            "total": total,
            "details": results
        }
    
    def test_learning_path_agent(self):
        """Test 3: Learning Path Agent with model"""
        print("\n[TEST 3] Learning Path Agent Integration")
        print("-" * 40)
        
        try:
            # Create test profile
            profile = StudentProfile(
                student_id="test_123",
                learning_style="visual",
                grade_level=9,
                current_knowledge={'Matematik': 0.4}
            )
            
            # Generate learning path
            start_time = time.time()
            path = self.learning_agent.create_personalized_path(
                profile=profile,
                target_topic="Matematik",
                duration_weeks=3
            )
            elapsed = time.time() - start_time
            
            assert len(path) == 3, f"Expected 3 weeks, got {len(path)}"
            
            # Check if model was used
            model_used = any(
                week.get('içerik', {}).get('model_generated', False) 
                for week in path
            )
            
            print(f"[OK] Generated {len(path)} week path in {elapsed:.2f}s")
            print(f"Model used: {model_used}")
            
            # Test topic suggestions
            suggestions = self.learning_agent.suggest_next_topics(
                profile=profile,
                completed_topics=['Kümeler', 'Sayılar'],
                subject='Matematik'
            )
            
            print(f"[OK] Generated {len(suggestions)} topic suggestions")
            
            return {
                "test": "learning_path_agent",
                "status": "passed",
                "time": elapsed,
                "path_length": len(path),
                "model_used": model_used,
                "suggestions": len(suggestions)
            }
            
        except Exception as e:
            print(f"[FAIL] {e}")
            return {"test": "learning_path_agent", "status": "failed", "error": str(e)}
    
    def test_study_buddy_agent(self):
        """Test 4: Study Buddy Agent with model"""
        print("\n[TEST 4] Study Buddy Agent Integration")
        print("-" * 40)
        
        try:
            # Generate adaptive quiz
            start_time = time.time()
            quiz = self.study_agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=0.5,
                num_questions=5,
                grade_level=9
            )
            elapsed = time.time() - start_time
            
            assert len(quiz) == 5, f"Expected 5 questions, got {len(quiz)}"
            
            # Check if any questions were model-generated
            model_questions = sum(
                1 for q in quiz 
                if isinstance(q.get('text', ''), str) and len(q['text']) > 50
            )
            
            print(f"[OK] Generated {len(quiz)} questions in {elapsed:.2f}s")
            print(f"Model-enhanced questions: {model_questions}/{len(quiz)}")
            
            # Test study plan generation
            study_plan = self.study_agent.generate_study_plan(
                weak_topics=["Denklemler", "Geometri"],
                available_hours=10
            )
            
            assert 'topics' in study_plan, "Study plan missing topics"
            
            print(f"[OK] Generated study plan for {len(study_plan.get('topics', []))} topics")
            
            return {
                "test": "study_buddy_agent",
                "status": "passed",
                "time": elapsed,
                "quiz_length": len(quiz),
                "model_questions": model_questions,
                "study_topics": len(study_plan.get('topics', []))
            }
            
        except Exception as e:
            print(f"[FAIL] {e}")
            return {"test": "study_buddy_agent", "status": "failed", "error": str(e)}
    
    def test_question_answering(self):
        """Test 5: Question answering functionality"""
        print("\n[TEST 5] Question Answering")
        print("-" * 40)
        
        if not self.model:
            print("[SKIP] Model not available")
            return {"test": "question_answering", "status": "skipped"}
        
        test_questions = [
            ("Pisagor teoremi nedir?", "Matematik"),
            ("Fotosentez nasıl gerçekleşir?", "Biyoloji"),
            ("Osmanlı Devleti ne zaman kuruldu?", "Tarih")
        ]
        
        results = []
        
        for question, subject in test_questions:
            try:
                start_time = time.time()
                answer = self.model.answer_question(
                    question=question,
                    subject=subject
                )
                elapsed = time.time() - start_time
                
                assert len(answer) > 10, f"Answer too short for: {question}"
                
                print(f"[OK] Q: {question[:30]}... ({elapsed:.2f}s)")
                results.append({
                    "question": question,
                    "status": "passed",
                    "time": elapsed,
                    "answer_length": len(answer)
                })
                
            except Exception as e:
                print(f"[FAIL] {question}: {e}")
                results.append({
                    "question": question,
                    "status": "failed",
                    "error": str(e)
                })
        
        passed = sum(1 for r in results if r["status"] == "passed")
        
        return {
            "test": "question_answering",
            "status": "passed" if passed == len(results) else "partial",
            "passed": passed,
            "total": len(results),
            "avg_time": sum(r.get("time", 0) for r in results) / len(results) if results else 0
        }
    
    def test_performance_benchmark(self):
        """Test 6: Performance benchmark"""
        print("\n[TEST 6] Performance Benchmark")
        print("-" * 40)
        
        if not self.model:
            print("[SKIP] Model not available")
            return {"test": "performance", "status": "skipped"}
        
        # Measure different prompt lengths
        prompt_lengths = [50, 100, 200, 500]
        results = []
        
        for length in prompt_lengths:
            prompt = "Matematik konusunu açıkla. " * (length // 25)
            prompt = prompt[:length]
            
            try:
                start_time = time.time()
                response = self.model.generate_response(
                    prompt,
                    max_new_tokens=100
                )
                elapsed = time.time() - start_time
                
                print(f"[OK] Prompt length {length}: {elapsed:.2f}s")
                results.append({
                    "prompt_length": length,
                    "time": elapsed,
                    "tokens_per_second": 100 / elapsed
                })
                
            except Exception as e:
                print(f"[FAIL] Length {length}: {e}")
                results.append({
                    "prompt_length": length,
                    "error": str(e)
                })
        
        avg_time = sum(r.get("time", 0) for r in results) / len(results) if results else 0
        
        return {
            "test": "performance",
            "status": "passed" if all("time" in r for r in results) else "partial",
            "avg_response_time": avg_time,
            "benchmarks": results
        }
    
    def run_all_tests(self, use_model: bool = True):
        """Run all tests and generate report"""
        
        # Setup
        model_available = self.setup(use_model)
        
        # Run tests
        test_methods = [
            self.test_model_inference,
            self.test_educational_content_generation,
            self.test_learning_path_agent,
            self.test_study_buddy_agent,
            self.test_question_answering,
            self.test_performance_benchmark
        ]
        
        results = []
        for test_method in test_methods:
            result = test_method()
            results.append(result)
            self.results.append(result)
        
        # Generate report
        self.generate_report(results, model_available)
        
        return results
    
    def generate_report(self, results: List[Dict], model_available: bool):
        """Generate test report"""
        print("\n" + "="*60)
        print("TEST REPORT")
        print("="*60)
        
        total_tests = len(results)
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        partial = sum(1 for r in results if r.get("status") == "partial")
        
        print(f"\nModel Status: {'Available' if model_available else 'Not Available'}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Partial: {partial}")
        print(f"Skipped: {skipped}")
        
        print("\nTest Details:")
        print("-" * 40)
        
        for result in results:
            status_symbol = {
                "passed": "✓",
                "failed": "✗",
                "partial": "⚠",
                "skipped": "-"
            }.get(result.get("status"), "?")
            
            test_name = result.get("test", "Unknown")
            status = result.get("status", "unknown")
            
            print(f"{status_symbol} {test_name}: {status}")
            
            if result.get("time"):
                print(f"  Time: {result['time']:.2f}s")
            
            if result.get("error"):
                print(f"  Error: {result['error']}")
        
        # Save report to file
        report_file = "test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_available": model_available,
                "summary": {
                    "total": total_tests,
                    "passed": passed,
                    "failed": failed,
                    "partial": partial,
                    "skipped": skipped
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Report saved to {report_file}")
        
        # Overall status
        if failed == 0 and passed > 0:
            print("\n🎉 [SUCCESS] All tests passed!")
        elif failed > 0:
            print(f"\n⚠️  [WARNING] {failed} tests failed")
        else:
            print("\n[INFO] Tests completed")


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEKNOFEST Model Integration Tests")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Run tests without loading the model"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = ModelIntegrationTester()
    
    # Run tests
    if args.quick:
        print("[INFO] Running quick tests...")
        model_available = tester.setup(use_model=not args.no_model)
        tester.test_learning_path_agent()
        tester.test_study_buddy_agent()
    else:
        print("[INFO] Running full test suite...")
        tester.run_all_tests(use_model=not args.no_model)
    
    print("\n[COMPLETE] Testing finished")


if __name__ == "__main__":
    main()