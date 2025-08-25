"""
Comprehensive tests for IRT Engine
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta

from src.core.irt_engine import (
    IRTEngine, IRTModel, EstimationMethod,
    ItemParameters, StudentAbility
)


class TestItemParameters:
    """Test ItemParameters dataclass"""
    
    def test_valid_item_creation(self):
        """Test creating valid item parameters"""
        item = ItemParameters(
            item_id="test_item_1",
            difficulty=0.5,
            discrimination=1.2,
            guessing=0.25
        )
        
        assert item.item_id == "test_item_1"
        assert item.difficulty == 0.5
        assert item.discrimination == 1.2
        assert item.guessing == 0.25
        assert item.upper_asymptote == 1.0
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test difficulty range
        with pytest.raises(ValueError, match="Difficulty must be between"):
            ItemParameters(item_id="test", difficulty=5.0)
        
        # Test discrimination range
        with pytest.raises(ValueError, match="Discrimination must be between"):
            ItemParameters(item_id="test", difficulty=0, discrimination=0.05)
        
        # Test guessing range
        with pytest.raises(ValueError, match="Guessing must be between"):
            ItemParameters(item_id="test", difficulty=0, guessing=0.6)
    
    def test_default_values(self):
        """Test default parameter values"""
        item = ItemParameters(item_id="test", difficulty=0)
        
        assert item.discrimination == 1.0
        assert item.guessing == 0.0
        assert item.upper_asymptote == 1.0
        assert item.usage_count == 0
        assert item.exposure_rate == 0.0


class TestIRTEngine:
    """Test IRT Engine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create IRT engine instance"""
        return IRTEngine(
            model=IRTModel.THREE_PL,
            estimation_method=EstimationMethod.EAP
        )
    
    @pytest.fixture
    def sample_items(self):
        """Create sample items for testing"""
        return [
            ItemParameters("item_1", difficulty=-1.0, discrimination=1.5, guessing=0.2),
            ItemParameters("item_2", difficulty=0.0, discrimination=1.2, guessing=0.25),
            ItemParameters("item_3", difficulty=1.0, discrimination=1.0, guessing=0.2),
            ItemParameters("item_4", difficulty=2.0, discrimination=0.8, guessing=0.15),
        ]
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.model == IRTModel.THREE_PL
        assert engine.estimation_method == EstimationMethod.EAP
        assert engine.max_iterations == 50
        assert len(engine.item_bank) == 0
    
    def test_add_items(self, engine, sample_items):
        """Test adding items to item bank"""
        for item in sample_items:
            engine.add_item(item)
        
        assert len(engine.item_bank) == 4
        assert "item_1" in engine.item_bank
        assert engine.item_bank["item_1"].difficulty == -1.0
    
    def test_add_items_batch(self, engine, sample_items):
        """Test batch adding items"""
        engine.add_items_batch(sample_items)
        assert len(engine.item_bank) == 4
    
    def test_probability_3pl(self, engine):
        """Test 3PL probability calculation"""
        # Test average ability, average difficulty
        p = engine.probability_3pl(theta=0, a=1, b=0, c=0.2)
        assert 0.55 < p < 0.65  # Should be around 0.6
        
        # Test high ability, low difficulty
        p = engine.probability_3pl(theta=2, a=1, b=-1, c=0.2)
        assert p > 0.9
        
        # Test low ability, high difficulty
        p = engine.probability_3pl(theta=-2, a=1, b=1, c=0.2)
        assert p < 0.3
        
        # Test guessing parameter effect
        p = engine.probability_3pl(theta=-5, a=1, b=0, c=0.25)
        assert abs(p - 0.25) < 0.01  # Should approach guessing parameter
        
        # Test edge cases
        p = engine.probability_3pl(theta=10, a=1, b=0, c=0.2)
        assert abs(p - 1.0) < 0.01
        
        p = engine.probability_3pl(theta=-10, a=1, b=0, c=0.2)
        assert abs(p - 0.2) < 0.01
    
    def test_information_function(self, engine, sample_items):
        """Test Fisher information calculation"""
        item = sample_items[0]
        
        # Information should be maximum near item difficulty
        info_at_difficulty = engine.information_function(-1.0, item)
        info_away = engine.information_function(2.0, item)
        
        assert info_at_difficulty > info_away
        
        # Test with different discrimination values
        high_disc = ItemParameters("test", difficulty=0, discrimination=2.5)
        low_disc = ItemParameters("test", difficulty=0, discrimination=0.5)
        
        info_high = engine.information_function(0, high_disc)
        info_low = engine.information_function(0, low_disc)
        
        assert info_high > info_low
    
    def test_test_information(self, engine, sample_items):
        """Test total test information"""
        engine.add_items_batch(sample_items)
        
        total_info = engine.test_information(0, sample_items)
        assert total_info > 0
        
        # Test information should be sum of item informations
        item_infos = [engine.information_function(0, item) for item in sample_items]
        assert abs(total_info - sum(item_infos)) < 0.001
    
    def test_standard_error(self, engine, sample_items):
        """Test standard error calculation"""
        se = engine.standard_error(0, sample_items)
        assert se > 0
        
        # More items should reduce standard error
        se_one = engine.standard_error(0, sample_items[:1])
        se_all = engine.standard_error(0, sample_items)
        assert se_all < se_one
    
    def test_estimate_ability_mle(self, engine, sample_items):
        """Test Maximum Likelihood Estimation"""
        engine.add_items_batch(sample_items)
        
        # Test with mixed responses
        responses = [1, 1, 0, 0]  # Correct on easy, incorrect on hard
        theta, se, iterations = engine.estimate_ability_mle(responses, sample_items)
        
        assert -2 < theta < 2  # Should be around average
        assert se > 0
        assert iterations > 0
        
        # Test extreme patterns
        theta_all_correct, _, _ = engine.estimate_ability_mle([1, 1, 1, 1], sample_items)
        assert theta_all_correct == 3.0
        
        theta_all_wrong, _, _ = engine.estimate_ability_mle([0, 0, 0, 0], sample_items)
        assert theta_all_wrong == -3.0
    
    def test_estimate_ability_eap(self, engine, sample_items):
        """Test Expected A Posteriori estimation"""
        engine.add_items_batch(sample_items)
        
        responses = [1, 1, 0, 0]
        theta, se, iterations = engine.estimate_ability_eap(
            responses, sample_items,
            prior_mean=0, prior_sd=1,
            quadrature_points=40
        )
        
        assert -2 < theta < 2
        assert se > 0
        assert iterations == 1  # EAP doesn't iterate
        
        # Test with different prior
        theta_high_prior, _, _ = engine.estimate_ability_eap(
            responses, sample_items,
            prior_mean=1.0, prior_sd=0.5
        )
        
        # Should be pulled toward prior
        assert theta_high_prior > theta
    
    def test_estimate_ability_full(self, engine, sample_items):
        """Test full ability estimation"""
        engine.add_items_batch(sample_items)
        
        student_id = "test_student"
        responses = [1, 0, 1, 0]
        item_ids = [item.item_id for item in sample_items]
        
        ability = engine.estimate_ability(
            student_id, responses, item_ids,
            method=EstimationMethod.EAP
        )
        
        assert ability.student_id == student_id
        assert -4 <= ability.theta <= 4
        assert ability.standard_error > 0
        assert len(ability.confidence_interval) == 2
        assert ability.confidence_interval[0] < ability.theta < ability.confidence_interval[1]
        assert 0 <= ability.reliability <= 1
        assert ability.response_pattern == responses
        assert ability.items_administered == item_ids
    
    def test_select_next_item(self, engine, sample_items):
        """Test adaptive item selection"""
        engine.add_items_batch(sample_items)
        
        # Test maximum information selection
        next_item = engine.select_next_item(
            current_theta=0,
            administered_items=[],
            selection_criteria="max_info"
        )
        
        assert next_item is not None
        assert next_item.item_id in [item.item_id for item in sample_items]
        
        # Test with administered items
        next_item = engine.select_next_item(
            current_theta=0,
            administered_items=["item_1", "item_2"],
            selection_criteria="max_info"
        )
        
        assert next_item.item_id not in ["item_1", "item_2"]
        
        # Test random selection
        next_item = engine.select_next_item(
            current_theta=0,
            administered_items=[],
            selection_criteria="random"
        )
        
        assert next_item is not None
        
        # Test when all items administered
        next_item = engine.select_next_item(
            current_theta=0,
            administered_items=[item.item_id for item in sample_items]
        )
        
        assert next_item is None
    
    def test_simulate_adaptive_test(self, engine, sample_items):
        """Test adaptive test simulation"""
        engine.add_items_batch(sample_items)
        
        # Add more items for better simulation
        for i in range(5, 20):
            engine.add_item(
                ItemParameters(
                    f"item_{i}",
                    difficulty=np.random.uniform(-2, 2),
                    discrimination=np.random.uniform(0.8, 1.5),
                    guessing=0.2
                )
            )
        
        true_theta = 0.5
        result = engine.simulate_adaptive_test(
            true_theta=true_theta,
            max_items=10,
            min_items=3,
            stopping_se=0.4
        )
        
        assert result["true_theta"] == true_theta
        assert -4 <= result["final_estimate"] <= 4
        assert result["final_se"] > 0
        assert result["items_administered"] >= 3
        assert result["items_administered"] <= 10
        assert len(result["responses"]) == result["items_administered"]
        assert len(result["theta_trajectory"]) == result["items_administered"]
        assert 0 <= result["reliability"] <= 1
        
        # Bias should be reasonable
        assert abs(result["bias"]) < 2
    
    @pytest.mark.asyncio
    async def test_estimate_abilities_batch_async(self, engine, sample_items):
        """Test batch ability estimation"""
        engine.add_items_batch(sample_items)
        
        student_data = [
            {
                "student_id": "student_1",
                "responses": [1, 1, 0, 0],
                "item_ids": [item.item_id for item in sample_items]
            },
            {
                "student_id": "student_2",
                "responses": [0, 1, 1, 0],
                "item_ids": [item.item_id for item in sample_items]
            },
            {
                "student_id": "student_3",
                "responses": [1, 1, 1, 1],
                "item_ids": [item.item_id for item in sample_items]
            }
        ]
        
        abilities = await engine.estimate_abilities_batch_async(student_data)
        
        assert len(abilities) == 3
        assert all(isinstance(a, StudentAbility) for a in abilities)
        assert abilities[0].student_id == "student_1"
        assert abilities[2].theta > abilities[0].theta  # All correct should be higher
    
    def test_calibrate_items(self, engine):
        """Test item calibration from response matrix"""
        # Create synthetic response data
        np.random.seed(42)
        n_students = 100
        n_items = 10
        
        # Generate response matrix based on true parameters
        response_matrix = np.zeros((n_students, n_items))
        true_abilities = np.random.normal(0, 1, n_students)
        true_difficulties = np.linspace(-2, 2, n_items)
        
        for i in range(n_students):
            for j in range(n_items):
                p = 1 / (1 + np.exp(-(true_abilities[i] - true_difficulties[j])))
                response_matrix[i, j] = 1 if np.random.random() < p else 0
        
        calibrated_items = engine.calibrate_items(response_matrix)
        
        assert len(calibrated_items) == n_items
        assert all(isinstance(item, ItemParameters) for item in calibrated_items)
        assert all(-4 <= item.difficulty <= 4 for item in calibrated_items)
        assert all(0.5 <= item.discrimination <= 2.5 for item in calibrated_items)
    
    def test_export_import_item_bank(self, engine, sample_items, tmp_path):
        """Test item bank export and import"""
        engine.add_items_batch(sample_items)
        
        # Export
        export_path = tmp_path / "item_bank.json"
        engine.export_item_bank(str(export_path))
        
        assert export_path.exists()
        
        # Load and verify JSON
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 4
        assert "item_1" in exported_data
        assert exported_data["item_1"]["difficulty"] == -1.0
        
        # Import into new engine
        new_engine = IRTEngine()
        new_engine.import_item_bank(str(export_path))
        
        assert len(new_engine.item_bank) == 4
        assert "item_1" in new_engine.item_bank
        assert new_engine.item_bank["item_1"].difficulty == -1.0
    
    def test_caching(self, engine):
        """Test LRU cache functionality"""
        # Call probability function multiple times with same parameters
        for _ in range(100):
            p = engine.probability_3pl(0, 1, 0, 0.2)
        
        # Cache info should show hits
        cache_info = engine.probability_3pl.cache_info()
        assert cache_info.hits > 0
    
    def test_metrics_tracking(self, engine, sample_items):
        """Test performance metrics tracking"""
        engine.add_items_batch(sample_items)
        
        # Perform some operations
        responses = [1, 0, 1, 0]
        item_ids = [item.item_id for item in sample_items]
        
        engine.estimate_ability("student_1", responses, item_ids)
        engine.calibrate_items(np.random.randint(0, 2, (50, 5)))
        
        metrics = engine.get_metrics()
        
        assert metrics["estimations_performed"] == 1
        assert metrics["items_calibrated"] == 5
    
    def test_edge_cases(self, engine):
        """Test edge cases and error handling"""
        # Empty responses
        with pytest.raises(Exception):
            engine.estimate_ability_mle([], [])
        
        # Mismatched responses and items
        item = ItemParameters("test", difficulty=0)
        engine.add_item(item)
        
        # This should handle gracefully
        ability = engine.estimate_ability(
            "student",
            [1],
            ["test"]
        )
        assert ability is not None
    
    def test_thread_safety(self, engine, sample_items):
        """Test thread safety of engine operations"""
        import threading
        import concurrent.futures
        
        engine.add_items_batch(sample_items)
        results = []
        
        def estimate_ability_thread():
            ability = engine.estimate_ability(
                "student",
                [1, 0, 1, 0],
                [item.item_id for item in sample_items]
            )
            results.append(ability.theta)
        
        # Run multiple estimations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(estimate_ability_thread) for _ in range(10)]
            concurrent.futures.wait(futures)
        
        assert len(results) == 10
        # All estimates should be similar (same data)
        assert all(abs(r - results[0]) < 0.1 for r in results)