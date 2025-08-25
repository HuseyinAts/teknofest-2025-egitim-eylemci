"""
Production-Ready Item Response Theory (IRT) Engine
TEKNOFEST 2025 - Eğitim Teknolojileri

Implements 3-Parameter Logistic (3PL) Model for adaptive testing
and student ability estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache
import json
from datetime import datetime
from scipy import optimize
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


class IRTModel(Enum):
    """Supported IRT models"""
    RASCH = "1PL"  # 1-Parameter Logistic (difficulty only)
    TWO_PL = "2PL"  # 2-Parameter Logistic (difficulty + discrimination)
    THREE_PL = "3PL"  # 3-Parameter Logistic (difficulty + discrimination + guessing)
    GRADED_RESPONSE = "GRM"  # Graded Response Model for polytomous items


class EstimationMethod(Enum):
    """Ability estimation methods"""
    MLE = "maximum_likelihood"  # Maximum Likelihood Estimation
    MAP = "maximum_aposteriori"  # Maximum A Posteriori
    EAP = "expected_aposteriori"  # Expected A Posteriori
    WLE = "weighted_likelihood"  # Weighted Likelihood Estimation


@dataclass
class ItemParameters:
    """IRT item parameters for 3PL model"""
    item_id: str
    difficulty: float  # b parameter (-3 to 3)
    discrimination: float = 1.0  # a parameter (0.5 to 2.5)
    guessing: float = 0.0  # c parameter (0 to 0.3)
    upper_asymptote: float = 1.0  # d parameter (usually 1)
    
    # Metadata
    subject: Optional[str] = None
    topic: Optional[str] = None
    grade_level: Optional[int] = None
    usage_count: int = 0
    exposure_rate: float = 0.0
    
    # Statistical properties
    standard_error: Optional[Dict[str, float]] = field(default_factory=dict)
    fit_statistics: Optional[Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameters"""
        if not -4 <= self.difficulty <= 4:
            raise ValueError(f"Difficulty must be between -4 and 4, got {self.difficulty}")
        if not 0.1 <= self.discrimination <= 3:
            raise ValueError(f"Discrimination must be between 0.1 and 3, got {self.discrimination}")
        if not 0 <= self.guessing <= 0.5:
            raise ValueError(f"Guessing must be between 0 and 0.5, got {self.guessing}")
        if not 0.5 <= self.upper_asymptote <= 1:
            raise ValueError(f"Upper asymptote must be between 0.5 and 1, got {self.upper_asymptote}")


@dataclass
class StudentAbility:
    """Student ability estimation with metadata"""
    student_id: str
    theta: float  # Ability estimate (-4 to 4)
    standard_error: float  # Standard error of measurement
    confidence_interval: Tuple[float, float]  # 95% CI
    estimation_method: EstimationMethod
    response_pattern: List[int]  # 0/1 responses
    items_administered: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metrics
    test_information: float = 0.0
    reliability: float = 0.0
    convergence_iterations: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "student_id": self.student_id,
            "theta": self.theta,
            "standard_error": self.standard_error,
            "confidence_interval": self.confidence_interval,
            "estimation_method": self.estimation_method.value,
            "response_count": len(self.response_pattern),
            "timestamp": self.timestamp.isoformat(),
            "test_information": self.test_information,
            "reliability": self.reliability
        }


class IRTEngine:
    """
    Production-ready IRT engine for adaptive testing and ability estimation.
    
    Features:
    - 3PL model implementation
    - Multiple estimation methods (MLE, MAP, EAP)
    - Adaptive item selection
    - Parallel processing for batch operations
    - Caching for performance
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        model: IRTModel = IRTModel.THREE_PL,
        estimation_method: EstimationMethod = EstimationMethod.EAP,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001,
        cache_size: int = 1000
    ):
        self.model = model
        self.estimation_method = estimation_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.cache_size = cache_size
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Item bank
        self.item_bank: Dict[str, ItemParameters] = {}
        
        # Performance metrics
        self.metrics = {
            "estimations_performed": 0,
            "items_calibrated": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"IRT Engine initialized with {model.value} model")
    
    def add_item(self, item: ItemParameters) -> None:
        """Add item to the item bank"""
        self.item_bank[item.item_id] = item
        logger.debug(f"Added item {item.item_id} to bank")
    
    def add_items_batch(self, items: List[ItemParameters]) -> None:
        """Add multiple items to the bank"""
        for item in items:
            self.add_item(item)
        logger.info(f"Added {len(items)} items to bank")
    
    @lru_cache(maxsize=1000)
    def probability_3pl(
        self,
        theta: float,
        a: float,
        b: float,
        c: float,
        d: float = 1.0
    ) -> float:
        """
        Calculate probability of correct response using 3PL model.
        
        P(θ) = c + (d - c) / (1 + exp(-a * (θ - b)))
        
        Args:
            theta: Student ability
            a: Discrimination parameter
            b: Difficulty parameter
            c: Guessing parameter
            d: Upper asymptote parameter
        
        Returns:
            Probability of correct response
        """
        try:
            exponent = -a * (theta - b)
            # Prevent overflow
            if exponent > 700:
                return c
            elif exponent < -700:
                return d
            
            probability = c + (d - c) / (1 + np.exp(exponent))
            return float(np.clip(probability, 0, 1))
        except Exception as e:
            logger.error(f"Error calculating probability: {e}")
            return 0.5
    
    def information_function(
        self,
        theta: float,
        item: ItemParameters
    ) -> float:
        """
        Calculate Fisher information for an item at given ability level.
        
        I(θ) = a² * (P - c)² / ((1 - c)² * P * Q)
        
        Higher information means more precise measurement.
        """
        p = self.probability_3pl(
            theta,
            item.discrimination,
            item.difficulty,
            item.guessing,
            item.upper_asymptote
        )
        q = 1 - p
        
        if p == 0 or q == 0 or p == item.guessing:
            return 0.0
        
        numerator = item.discrimination ** 2 * (p - item.guessing) ** 2
        denominator = (1 - item.guessing) ** 2 * p * q
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def test_information(
        self,
        theta: float,
        items: List[ItemParameters]
    ) -> float:
        """Calculate total test information at ability level"""
        return sum(self.information_function(theta, item) for item in items)
    
    def standard_error(
        self,
        theta: float,
        items: List[ItemParameters]
    ) -> float:
        """Calculate standard error of measurement"""
        info = self.test_information(theta, items)
        return 1 / np.sqrt(info) if info > 0 else float('inf')
    
    def estimate_ability_mle(
        self,
        responses: List[int],
        items: List[ItemParameters],
        initial_theta: float = 0.0
    ) -> Tuple[float, float, int]:
        """
        Maximum Likelihood Estimation of ability.
        
        Returns:
            Tuple of (theta estimate, standard error, iterations)
        """
        def neg_log_likelihood(theta):
            ll = 0
            for response, item in zip(responses, items):
                p = self.probability_3pl(
                    theta[0],
                    item.discrimination,
                    item.difficulty,
                    item.guessing
                )
                if response == 1:
                    ll += np.log(p) if p > 0 else -100
                else:
                    ll += np.log(1 - p) if p < 1 else -100
            return -ll
        
        # Check for extreme response patterns
        if all(r == 1 for r in responses):
            return 3.0, 1.0, 1  # Maximum ability
        elif all(r == 0 for r in responses):
            return -3.0, 1.0, 1  # Minimum ability
        
        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            [initial_theta],
            method='L-BFGS-B',
            bounds=[(-4, 4)],
            options={'maxiter': self.max_iterations}
        )
        
        theta_est = result.x[0]
        se = self.standard_error(theta_est, items)
        
        return theta_est, se, result.nit
    
    def estimate_ability_eap(
        self,
        responses: List[int],
        items: List[ItemParameters],
        prior_mean: float = 0.0,
        prior_sd: float = 1.0,
        quadrature_points: int = 40
    ) -> Tuple[float, float, int]:
        """
        Expected A Posteriori estimation with normal prior.
        
        More robust than MLE, especially for extreme scores.
        """
        # Quadrature points and weights
        theta_points = np.linspace(-4, 4, quadrature_points)
        prior_weights = norm.pdf(theta_points, prior_mean, prior_sd)
        
        # Calculate likelihood for each quadrature point
        likelihoods = np.zeros(quadrature_points)
        
        for i, theta in enumerate(theta_points):
            likelihood = 1.0
            for response, item in zip(responses, items):
                p = self.probability_3pl(
                    theta,
                    item.discrimination,
                    item.difficulty,
                    item.guessing
                )
                likelihood *= p if response == 1 else (1 - p)
            likelihoods[i] = likelihood
        
        # Posterior distribution
        posterior = likelihoods * prior_weights
        posterior_sum = np.sum(posterior)
        
        if posterior_sum == 0:
            return prior_mean, prior_sd, 1
        
        posterior /= posterior_sum
        
        # EAP estimate
        theta_eap = np.sum(theta_points * posterior)
        
        # Posterior standard deviation
        variance = np.sum((theta_points - theta_eap) ** 2 * posterior)
        se = np.sqrt(variance)
        
        return theta_eap, se, 1
    
    def estimate_ability(
        self,
        student_id: str,
        responses: List[int],
        item_ids: List[str],
        method: Optional[EstimationMethod] = None
    ) -> StudentAbility:
        """
        Estimate student ability from responses.
        
        Args:
            student_id: Student identifier
            responses: List of 0/1 responses
            item_ids: List of item identifiers
            method: Estimation method (uses default if None)
        
        Returns:
            StudentAbility object with estimates
        """
        method = method or self.estimation_method
        
        # Get item parameters
        items = [self.item_bank[item_id] for item_id in item_ids]
        
        # Estimate based on method
        if method == EstimationMethod.MLE:
            theta, se, iterations = self.estimate_ability_mle(responses, items)
        elif method == EstimationMethod.EAP:
            theta, se, iterations = self.estimate_ability_eap(responses, items)
        else:
            # Default to EAP
            theta, se, iterations = self.estimate_ability_eap(responses, items)
        
        # Calculate confidence interval
        ci_lower = theta - 1.96 * se
        ci_upper = theta + 1.96 * se
        
        # Calculate test information and reliability
        test_info = self.test_information(theta, items)
        reliability = 1 - (1 / test_info) if test_info > 1 else 0
        
        # Update metrics
        self.metrics["estimations_performed"] += 1
        
        return StudentAbility(
            student_id=student_id,
            theta=theta,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            estimation_method=method,
            response_pattern=responses,
            items_administered=item_ids,
            test_information=test_info,
            reliability=reliability,
            convergence_iterations=iterations
        )
    
    def select_next_item(
        self,
        current_theta: float,
        administered_items: List[str],
        selection_criteria: str = "max_info",
        exposure_control: bool = True
    ) -> Optional[ItemParameters]:
        """
        Select next item for adaptive testing.
        
        Args:
            current_theta: Current ability estimate
            administered_items: Already administered items
            selection_criteria: "max_info", "random", or "balanced"
            exposure_control: Apply exposure control
        
        Returns:
            Selected item or None if no items available
        """
        # Get available items
        available_items = [
            item for item_id, item in self.item_bank.items()
            if item_id not in administered_items
        ]
        
        if not available_items:
            return None
        
        if selection_criteria == "max_info":
            # Select item with maximum information at current theta
            item_infos = [
                (item, self.information_function(current_theta, item))
                for item in available_items
            ]
            
            # Apply exposure control
            if exposure_control:
                # Randomize among top 5 most informative items
                item_infos.sort(key=lambda x: x[1], reverse=True)
                top_items = item_infos[:min(5, len(item_infos))]
                if top_items:
                    selected = np.random.choice([item for item, _ in top_items])
                else:
                    selected = item_infos[0][0]
            else:
                selected = max(item_infos, key=lambda x: x[1])[0]
            
            # Update exposure rate
            selected.exposure_rate = selected.usage_count / max(1, len(self.item_bank))
            selected.usage_count += 1
            
            return selected
        
        elif selection_criteria == "random":
            return np.random.choice(available_items)
        
        else:  # balanced
            # Balance information and content coverage
            target_info = 0.5  # Target medium information
            distances = [
                abs(self.information_function(current_theta, item) - target_info)
                for item in available_items
            ]
            min_distance_idx = np.argmin(distances)
            return available_items[min_distance_idx]
    
    def simulate_adaptive_test(
        self,
        true_theta: float,
        max_items: int = 20,
        min_items: int = 5,
        stopping_se: float = 0.3
    ) -> Dict:
        """
        Simulate an adaptive test for validation.
        
        Args:
            true_theta: True ability level
            max_items: Maximum number of items
            min_items: Minimum number of items
            stopping_se: Stop when SE below this threshold
        
        Returns:
            Test results and metrics
        """
        administered_items = []
        responses = []
        theta_estimates = []
        se_estimates = []
        
        current_theta = 0.0  # Start with average ability
        
        for i in range(max_items):
            # Select next item
            next_item = self.select_next_item(
                current_theta,
                [item.item_id for item in administered_items]
            )
            
            if not next_item:
                break
            
            administered_items.append(next_item)
            
            # Simulate response based on true theta
            p_correct = self.probability_3pl(
                true_theta,
                next_item.discrimination,
                next_item.difficulty,
                next_item.guessing
            )
            response = 1 if np.random.random() < p_correct else 0
            responses.append(response)
            
            # Update ability estimate
            ability = self.estimate_ability(
                "simulation",
                responses,
                [item.item_id for item in administered_items]
            )
            
            current_theta = ability.theta
            theta_estimates.append(current_theta)
            se_estimates.append(ability.standard_error)
            
            # Check stopping criteria
            if i >= min_items - 1 and ability.standard_error < stopping_se:
                break
        
        # Final estimate
        final_ability = self.estimate_ability(
            "simulation",
            responses,
            [item.item_id for item in administered_items]
        )
        
        return {
            "true_theta": true_theta,
            "final_estimate": final_ability.theta,
            "final_se": final_ability.standard_error,
            "bias": final_ability.theta - true_theta,
            "items_administered": len(administered_items),
            "responses": responses,
            "theta_trajectory": theta_estimates,
            "se_trajectory": se_estimates,
            "reliability": final_ability.reliability
        }
    
    async def estimate_abilities_batch_async(
        self,
        student_responses: List[Dict]
    ) -> List[StudentAbility]:
        """
        Asynchronously estimate abilities for multiple students.
        
        Args:
            student_responses: List of dicts with student_id, responses, item_ids
        
        Returns:
            List of StudentAbility objects
        """
        loop = asyncio.get_event_loop()
        
        tasks = []
        for data in student_responses:
            task = loop.run_in_executor(
                self.executor,
                self.estimate_ability,
                data["student_id"],
                data["responses"],
                data["item_ids"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def calibrate_items(
        self,
        response_matrix: np.ndarray,
        method: str = "marginal_maximum_likelihood"
    ) -> List[ItemParameters]:
        """
        Calibrate item parameters from response data.
        
        Args:
            response_matrix: N x M matrix (N students, M items)
            method: Calibration method
        
        Returns:
            List of calibrated ItemParameters
        
        Note: This is a simplified implementation. Production systems
        should use specialized IRT packages like mirt or ltm.
        """
        n_students, n_items = response_matrix.shape
        calibrated_items = []
        
        for item_idx in range(n_items):
            item_responses = response_matrix[:, item_idx]
            
            # Simple difficulty estimation (proportion correct)
            p_correct = np.mean(item_responses)
            
            # Convert to logit scale
            if p_correct == 0:
                difficulty = 3.0
            elif p_correct == 1:
                difficulty = -3.0
            else:
                difficulty = -np.log(p_correct / (1 - p_correct))
            
            # Simple discrimination estimation (point-biserial correlation)
            total_scores = np.sum(response_matrix, axis=1)
            if np.std(total_scores) > 0 and np.std(item_responses) > 0:
                discrimination = np.corrcoef(item_responses, total_scores)[0, 1]
                discrimination = max(0.5, min(2.5, discrimination * 2))
            else:
                discrimination = 1.0
            
            # Guessing parameter (for multiple choice, typically 1/n_options)
            guessing = 0.2  # Assume 5-option multiple choice
            
            calibrated_items.append(
                ItemParameters(
                    item_id=f"item_{item_idx}",
                    difficulty=difficulty,
                    discrimination=discrimination,
                    guessing=guessing
                )
            )
        
        self.metrics["items_calibrated"] += len(calibrated_items)
        return calibrated_items
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def export_item_bank(self, filepath: str) -> None:
        """Export item bank to JSON file"""
        items_dict = {
            item_id: {
                "difficulty": item.difficulty,
                "discrimination": item.discrimination,
                "guessing": item.guessing,
                "subject": item.subject,
                "topic": item.topic,
                "grade_level": item.grade_level,
                "usage_count": item.usage_count,
                "exposure_rate": item.exposure_rate
            }
            for item_id, item in self.item_bank.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(items_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(items_dict)} items to {filepath}")
    
    def import_item_bank(self, filepath: str) -> None:
        """Import item bank from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            items_dict = json.load(f)
        
        for item_id, params in items_dict.items():
            item = ItemParameters(
                item_id=item_id,
                difficulty=params["difficulty"],
                discrimination=params.get("discrimination", 1.0),
                guessing=params.get("guessing", 0.0),
                subject=params.get("subject"),
                topic=params.get("topic"),
                grade_level=params.get("grade_level"),
                usage_count=params.get("usage_count", 0),
                exposure_rate=params.get("exposure_rate", 0.0)
            )
            self.add_item(item)
        
        logger.info(f"Imported {len(items_dict)} items from {filepath}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)