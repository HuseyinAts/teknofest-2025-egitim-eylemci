"""
IRT Service Layer for Production Integration
TEKNOFEST 2025 - Eğitim Teknolojileri

Provides high-level IRT operations with caching, validation, and database integration.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import asdict
import logging

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from redis import Redis
from pydantic import BaseModel, Field, validator

from src.core.irt_engine import (
    IRTEngine, IRTModel, EstimationMethod,
    ItemParameters, StudentAbility
)
from src.database.models import (
    Question, QuizAttempt, StudentProfile,
    IRTItemBank, IRTStudentAbility, IRTTestSession
)
from src.core.cache_manager import CacheManager
from src.monitoring import metrics_collector

logger = logging.getLogger(__name__)


class IRTItemRequest(BaseModel):
    """Request model for IRT item creation"""
    question_id: str
    difficulty: float = Field(ge=-4, le=4)
    discrimination: float = Field(ge=0.1, le=3, default=1.0)
    guessing: float = Field(ge=0, le=0.5, default=0.2)
    subject: str
    topic: Optional[str] = None
    grade_level: int = Field(ge=1, le=12)
    
    @validator('guessing')
    def validate_guessing(cls, v, values):
        """Ensure guessing parameter is reasonable for item type"""
        if v > 0.33:  # More than 1/3 is suspicious
            logger.warning(f"High guessing parameter: {v}")
        return v


class IRTEstimationRequest(BaseModel):
    """Request model for ability estimation"""
    student_id: str
    responses: List[int] = Field(..., min_items=3)  # Need at least 3 responses
    item_ids: List[str]
    test_id: Optional[str] = None
    estimation_method: EstimationMethod = EstimationMethod.EAP
    
    @validator('responses')
    def validate_responses(cls, v):
        """Ensure responses are binary"""
        if not all(r in [0, 1] for r in v):
            raise ValueError("Responses must be 0 or 1")
        return v
    
    @validator('item_ids')
    def validate_item_match(cls, v, values):
        """Ensure item count matches response count"""
        if 'responses' in values and len(v) != len(values['responses']):
            raise ValueError("Number of items must match number of responses")
        return v


class AdaptiveTestRequest(BaseModel):
    """Request model for adaptive test configuration"""
    student_id: str
    subject: str
    topic: Optional[str] = None
    max_items: int = Field(default=20, ge=5, le=50)
    min_items: int = Field(default=5, ge=3, le=20)
    target_se: float = Field(default=0.3, ge=0.1, le=1.0)
    time_limit_minutes: Optional[int] = Field(default=60, ge=10, le=180)


class IRTService:
    """
    Production-ready IRT service with database integration and caching.
    
    Features:
    - Async database operations
    - Redis caching for performance
    - Comprehensive error handling
    - Metrics collection
    - Batch processing support
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        cache_manager: Optional[CacheManager] = None,
        redis_client: Optional[Redis] = None
    ):
        self.db = db_session
        self.cache = cache_manager or CacheManager()
        self.redis = redis_client
        
        # Initialize IRT engine
        self.engine = IRTEngine(
            model=IRTModel.THREE_PL,
            estimation_method=EstimationMethod.EAP,
            max_iterations=50,
            cache_size=2000
        )
        
        # Performance tracking
        self.performance_stats = {
            "estimations": 0,
            "cache_hits": 0,
            "avg_estimation_time": 0,
            "active_sessions": 0
        }
        
        logger.info("IRT Service initialized")
    
    async def initialize_item_bank(self) -> None:
        """Load item bank from database into engine"""
        try:
            # Query all active items with IRT parameters
            result = await self.db.execute(
                select(IRTItemBank).where(IRTItemBank.is_active == True)
            )
            items = result.scalars().all()
            
            # Convert to ItemParameters and add to engine
            for item in items:
                item_params = ItemParameters(
                    item_id=item.item_id,
                    difficulty=item.difficulty,
                    discrimination=item.discrimination,
                    guessing=item.guessing,
                    subject=item.subject,
                    topic=item.topic,
                    grade_level=item.grade_level,
                    usage_count=item.usage_count,
                    exposure_rate=item.exposure_rate
                )
                self.engine.add_item(item_params)
            
            logger.info(f"Loaded {len(items)} items into IRT engine")
            
            # Cache the item bank
            if self.redis:
                self.redis.setex(
                    "irt:item_bank:loaded",
                    timedelta(hours=1),
                    json.dumps({"count": len(items), "timestamp": datetime.now().isoformat()})
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize item bank: {e}")
            raise
    
    async def add_or_update_item(
        self,
        item_request: IRTItemRequest
    ) -> ItemParameters:
        """Add or update IRT parameters for an item"""
        try:
            # Check if item exists
            result = await self.db.execute(
                select(IRTItemBank).where(IRTItemBank.item_id == item_request.question_id)
            )
            existing_item = result.scalar_one_or_none()
            
            if existing_item:
                # Update existing item
                existing_item.difficulty = item_request.difficulty
                existing_item.discrimination = item_request.discrimination
                existing_item.guessing = item_request.guessing
                existing_item.updated_at = datetime.now()
                
                await self.db.commit()
                logger.info(f"Updated IRT parameters for item {item_request.question_id}")
            else:
                # Create new item
                new_item = IRTItemBank(
                    item_id=item_request.question_id,
                    difficulty=item_request.difficulty,
                    discrimination=item_request.discrimination,
                    guessing=item_request.guessing,
                    subject=item_request.subject,
                    topic=item_request.topic,
                    grade_level=item_request.grade_level
                )
                self.db.add(new_item)
                await self.db.commit()
                logger.info(f"Added new IRT item {item_request.question_id}")
            
            # Update engine
            item_params = ItemParameters(
                item_id=item_request.question_id,
                difficulty=item_request.difficulty,
                discrimination=item_request.discrimination,
                guessing=item_request.guessing,
                subject=item_request.subject,
                topic=item_request.topic,
                grade_level=item_request.grade_level
            )
            self.engine.add_item(item_params)
            
            # Invalidate cache
            if self.redis:
                self.redis.delete(f"irt:item:{item_request.question_id}")
            
            return item_params
            
        except Exception as e:
            logger.error(f"Failed to add/update item: {e}")
            await self.db.rollback()
            raise
    
    async def estimate_ability(
        self,
        request: IRTEstimationRequest,
        save_to_db: bool = True
    ) -> StudentAbility:
        """
        Estimate student ability from responses.
        
        Args:
            request: Estimation request with responses
            save_to_db: Whether to save result to database
        
        Returns:
            StudentAbility object
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if self.redis:
                cached_result = self.redis.get(cache_key)
                if cached_result:
                    self.performance_stats["cache_hits"] += 1
                    metrics_collector.increment("irt.cache_hits")
                    return StudentAbility(**json.loads(cached_result))
            
            # Ensure items are in the engine
            missing_items = [
                item_id for item_id in request.item_ids
                if item_id not in self.engine.item_bank
            ]
            
            if missing_items:
                # Load missing items from database
                await self._load_items_from_db(missing_items)
            
            # Perform estimation
            ability = self.engine.estimate_ability(
                student_id=request.student_id,
                responses=request.responses,
                item_ids=request.item_ids,
                method=request.estimation_method
            )
            
            # Save to database if requested
            if save_to_db:
                await self._save_ability_to_db(ability, request.test_id)
            
            # Cache the result
            if self.redis:
                self.redis.setex(
                    cache_key,
                    timedelta(minutes=30),
                    json.dumps(ability.to_dict())
                )
            
            # Update metrics
            estimation_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["estimations"] += 1
            self.performance_stats["avg_estimation_time"] = (
                (self.performance_stats["avg_estimation_time"] * 
                 (self.performance_stats["estimations"] - 1) + estimation_time) /
                self.performance_stats["estimations"]
            )
            
            metrics_collector.histogram(
                "irt.estimation_time",
                estimation_time,
                tags={"method": request.estimation_method.value}
            )
            metrics_collector.increment("irt.estimations_total")
            
            logger.info(
                f"Ability estimated for student {request.student_id}: "
                f"θ={ability.theta:.3f}, SE={ability.standard_error:.3f}"
            )
            
            return ability
            
        except Exception as e:
            logger.error(f"Ability estimation failed: {e}")
            metrics_collector.increment("irt.estimation_errors")
            raise
    
    async def start_adaptive_test(
        self,
        request: AdaptiveTestRequest
    ) -> Dict:
        """
        Start an adaptive test session.
        
        Args:
            request: Adaptive test configuration
        
        Returns:
            Test session information with first item
        """
        try:
            # Create test session
            session = IRTTestSession(
                session_id=self._generate_session_id(),
                student_id=request.student_id,
                subject=request.subject,
                topic=request.topic,
                max_items=request.max_items,
                min_items=request.min_items,
                target_se=request.target_se,
                current_theta=0.0,  # Start with average ability
                current_se=1.0,
                items_administered=[],
                responses=[],
                start_time=datetime.now(),
                time_limit=timedelta(minutes=request.time_limit_minutes) 
                          if request.time_limit_minutes else None
            )
            
            self.db.add(session)
            await self.db.commit()
            
            # Select first item (medium difficulty)
            first_item = await self._select_adaptive_item(
                current_theta=0.0,
                administered_items=[],
                subject=request.subject,
                topic=request.topic
            )
            
            if not first_item:
                raise ValueError(f"No items available for {request.subject}")
            
            # Update session
            session.items_administered = [first_item.item_id]
            session.current_item = first_item.item_id
            await self.db.commit()
            
            # Track active session
            self.performance_stats["active_sessions"] += 1
            
            # Cache session
            if self.redis:
                self.redis.setex(
                    f"irt:session:{session.session_id}",
                    timedelta(hours=2),
                    json.dumps({
                        "student_id": request.student_id,
                        "subject": request.subject,
                        "start_time": session.start_time.isoformat(),
                        "current_theta": session.current_theta,
                        "items_count": len(session.items_administered)
                    })
                )
            
            return {
                "session_id": session.session_id,
                "student_id": request.student_id,
                "subject": request.subject,
                "first_item": {
                    "item_id": first_item.item_id,
                    "difficulty": first_item.difficulty,
                    "estimated_probability": self.engine.probability_3pl(
                        0.0,
                        first_item.discrimination,
                        first_item.difficulty,
                        first_item.guessing
                    )
                },
                "max_items": request.max_items,
                "time_limit_minutes": request.time_limit_minutes
            }
            
        except Exception as e:
            logger.error(f"Failed to start adaptive test: {e}")
            await self.db.rollback()
            raise
    
    async def submit_adaptive_response(
        self,
        session_id: str,
        response: int
    ) -> Dict:
        """
        Submit response for adaptive test and get next item.
        
        Args:
            session_id: Test session ID
            response: Student response (0 or 1)
        
        Returns:
            Updated test state with next item or final results
        """
        try:
            # Get session
            result = await self.db.execute(
                select(IRTTestSession).where(IRTTestSession.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Check time limit
            if session.time_limit:
                elapsed = datetime.now() - session.start_time
                if elapsed > session.time_limit:
                    return await self._finalize_test(session, timeout=True)
            
            # Add response
            session.responses.append(response)
            
            # Update ability estimate
            ability = self.engine.estimate_ability(
                student_id=session.student_id,
                responses=session.responses,
                item_ids=session.items_administered
            )
            
            session.current_theta = ability.theta
            session.current_se = ability.standard_error
            
            # Check stopping criteria
            items_count = len(session.items_administered)
            
            if (items_count >= session.min_items and 
                ability.standard_error <= session.target_se) or \
               items_count >= session.max_items:
                # Test complete
                return await self._finalize_test(session)
            
            # Select next item
            next_item = await self._select_adaptive_item(
                current_theta=ability.theta,
                administered_items=session.items_administered,
                subject=session.subject,
                topic=session.topic
            )
            
            if not next_item:
                # No more items available
                return await self._finalize_test(session)
            
            # Update session
            session.items_administered.append(next_item.item_id)
            session.current_item = next_item.item_id
            await self.db.commit()
            
            # Calculate progress
            progress = min(100, (items_count / session.max_items) * 100)
            
            return {
                "session_id": session_id,
                "status": "in_progress",
                "current_theta": ability.theta,
                "current_se": ability.standard_error,
                "confidence_interval": ability.confidence_interval,
                "items_administered": items_count,
                "progress": progress,
                "next_item": {
                    "item_id": next_item.item_id,
                    "difficulty": next_item.difficulty,
                    "estimated_probability": self.engine.probability_3pl(
                        ability.theta,
                        next_item.discrimination,
                        next_item.difficulty,
                        next_item.guessing
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process adaptive response: {e}")
            await self.db.rollback()
            raise
    
    async def get_student_ability_history(
        self,
        student_id: str,
        subject: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get historical ability estimates for a student"""
        try:
            query = select(IRTStudentAbility).where(
                IRTStudentAbility.student_id == student_id
            )
            
            if subject:
                query = query.where(IRTStudentAbility.subject == subject)
            
            query = query.order_by(IRTStudentAbility.timestamp.desc()).limit(limit)
            
            result = await self.db.execute(query)
            abilities = result.scalars().all()
            
            return [
                {
                    "timestamp": ability.timestamp.isoformat(),
                    "theta": ability.theta,
                    "standard_error": ability.standard_error,
                    "subject": ability.subject,
                    "test_id": ability.test_id,
                    "reliability": ability.reliability,
                    "items_count": ability.items_count
                }
                for ability in abilities
            ]
            
        except Exception as e:
            logger.error(f"Failed to get ability history: {e}")
            raise
    
    async def calibrate_items_from_responses(
        self,
        subject: str,
        min_responses: int = 30
    ) -> List[ItemParameters]:
        """
        Calibrate item parameters from historical response data.
        
        Args:
            subject: Subject to calibrate
            min_responses: Minimum responses needed per item
        
        Returns:
            List of calibrated items
        """
        try:
            # Get response data from database
            query = """
                SELECT 
                    qa.quiz_id as item_id,
                    qa.student_id,
                    qa.answers->>'correct' as response
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE q.subject = :subject
                AND qa.status = 'completed'
                ORDER BY qa.student_id, qa.quiz_id
            """
            
            result = await self.db.execute(query, {"subject": subject})
            responses = result.fetchall()
            
            if not responses:
                logger.warning(f"No response data found for {subject}")
                return []
            
            # Organize into response matrix
            student_ids = list(set(r.student_id for r in responses))
            item_ids = list(set(r.item_id for r in responses))
            
            # Filter items with sufficient responses
            item_response_counts = {}
            for item_id in item_ids:
                count = sum(1 for r in responses if r.item_id == item_id)
                if count >= min_responses:
                    item_response_counts[item_id] = count
            
            if not item_response_counts:
                logger.warning(f"No items with sufficient responses for {subject}")
                return []
            
            # Create response matrix
            filtered_items = list(item_response_counts.keys())
            student_idx_map = {sid: i for i, sid in enumerate(student_ids)}
            item_idx_map = {iid: i for i, iid in enumerate(filtered_items)}
            
            response_matrix = np.full(
                (len(student_ids), len(filtered_items)), 
                np.nan
            )
            
            for r in responses:
                if r.item_id in item_idx_map:
                    student_idx = student_idx_map[r.student_id]
                    item_idx = item_idx_map[r.item_id]
                    response_matrix[student_idx, item_idx] = int(r.response == 'true')
            
            # Calibrate items
            calibrated_items = self.engine.calibrate_items(response_matrix)
            
            # Update database
            for item, item_id in zip(calibrated_items, filtered_items):
                item.item_id = item_id
                item.subject = subject
                
                # Update or create in database
                await self.add_or_update_item(
                    IRTItemRequest(
                        question_id=item_id,
                        difficulty=item.difficulty,
                        discrimination=item.discrimination,
                        guessing=item.guessing,
                        subject=subject
                    )
                )
            
            logger.info(f"Calibrated {len(calibrated_items)} items for {subject}")
            
            metrics_collector.gauge(
                "irt.calibrated_items",
                len(calibrated_items),
                tags={"subject": subject}
            )
            
            return calibrated_items
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise
    
    async def get_test_information_curve(
        self,
        item_ids: List[str],
        theta_range: Tuple[float, float] = (-4, 4),
        points: int = 100
    ) -> Dict:
        """
        Calculate test information curve for a set of items.
        
        Returns:
            Dictionary with theta values and corresponding information
        """
        try:
            # Ensure items are loaded
            missing_items = [
                item_id for item_id in item_ids
                if item_id not in self.engine.item_bank
            ]
            if missing_items:
                await self._load_items_from_db(missing_items)
            
            # Get items
            items = [self.engine.item_bank[item_id] for item_id in item_ids]
            
            # Calculate information across theta range
            theta_values = np.linspace(theta_range[0], theta_range[1], points)
            information_values = []
            se_values = []
            
            for theta in theta_values:
                info = self.engine.test_information(theta, items)
                se = self.engine.standard_error(theta, items)
                information_values.append(info)
                se_values.append(se)
            
            return {
                "theta": theta_values.tolist(),
                "information": information_values,
                "standard_error": se_values,
                "max_information": max(information_values),
                "max_info_theta": theta_values[np.argmax(information_values)],
                "reliability": 1 - (1 / max(information_values)) if max(information_values) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate test information: {e}")
            raise
    
    # Helper methods
    
    async def _load_items_from_db(self, item_ids: List[str]) -> None:
        """Load items from database into engine"""
        result = await self.db.execute(
            select(IRTItemBank).where(IRTItemBank.item_id.in_(item_ids))
        )
        items = result.scalars().all()
        
        for item in items:
            item_params = ItemParameters(
                item_id=item.item_id,
                difficulty=item.difficulty,
                discrimination=item.discrimination,
                guessing=item.guessing,
                subject=item.subject,
                topic=item.topic,
                grade_level=item.grade_level
            )
            self.engine.add_item(item_params)
    
    async def _save_ability_to_db(
        self,
        ability: StudentAbility,
        test_id: Optional[str] = None
    ) -> None:
        """Save ability estimate to database"""
        db_ability = IRTStudentAbility(
            student_id=ability.student_id,
            theta=ability.theta,
            standard_error=ability.standard_error,
            confidence_lower=ability.confidence_interval[0],
            confidence_upper=ability.confidence_interval[1],
            test_id=test_id,
            items_count=len(ability.items_administered),
            reliability=ability.reliability,
            timestamp=ability.timestamp
        )
        
        self.db.add(db_ability)
        await self.db.commit()
    
    async def _select_adaptive_item(
        self,
        current_theta: float,
        administered_items: List[str],
        subject: str,
        topic: Optional[str] = None
    ) -> Optional[ItemParameters]:
        """Select next item for adaptive testing"""
        # Filter items by subject and topic
        available_items = []
        for item_id, item in self.engine.item_bank.items():
            if item_id not in administered_items and item.subject == subject:
                if topic is None or item.topic == topic:
                    available_items.append(item)
        
        if not available_items:
            return None
        
        # Calculate information for each item
        item_infos = [
            (item, self.engine.information_function(current_theta, item))
            for item in available_items
        ]
        
        # Sort by information
        item_infos.sort(key=lambda x: x[1], reverse=True)
        
        # Apply exposure control - randomize among top 5
        top_items = item_infos[:min(5, len(item_infos))]
        
        if top_items:
            selected = np.random.choice([item for item, _ in top_items])
            
            # Update usage statistics
            selected.usage_count += 1
            selected.exposure_rate = selected.usage_count / max(1, len(available_items))
            
            # Update in database
            await self.db.execute(
                update(IRTItemBank)
                .where(IRTItemBank.item_id == selected.item_id)
                .values(
                    usage_count=selected.usage_count,
                    exposure_rate=selected.exposure_rate
                )
            )
            await self.db.commit()
            
            return selected
        
        return None
    
    async def _finalize_test(
        self,
        session: IRTTestSession,
        timeout: bool = False
    ) -> Dict:
        """Finalize adaptive test session"""
        # Final ability estimate
        if session.responses:
            final_ability = self.engine.estimate_ability(
                student_id=session.student_id,
                responses=session.responses,
                item_ids=session.items_administered
            )
            
            # Save to database
            await self._save_ability_to_db(final_ability, session.session_id)
            
            # Update session
            session.final_theta = final_ability.theta
            session.final_se = final_ability.standard_error
            session.status = "timeout" if timeout else "completed"
            session.end_time = datetime.now()
            await self.db.commit()
            
            # Update stats
            self.performance_stats["active_sessions"] -= 1
            
            # Clear cache
            if self.redis:
                self.redis.delete(f"irt:session:{session.session_id}")
            
            return {
                "session_id": session.session_id,
                "status": session.status,
                "final_theta": final_ability.theta,
                "final_se": final_ability.standard_error,
                "confidence_interval": final_ability.confidence_interval,
                "reliability": final_ability.reliability,
                "items_administered": len(session.items_administered),
                "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60,
                "responses": session.responses
            }
        else:
            session.status = "aborted"
            session.end_time = datetime.now()
            await self.db.commit()
            
            return {
                "session_id": session.session_id,
                "status": "aborted",
                "message": "No responses recorded"
            }
    
    def _generate_cache_key(self, request: IRTEstimationRequest) -> str:
        """Generate cache key for estimation request"""
        key_data = f"{request.student_id}:{request.item_ids}:{request.responses}"
        return f"irt:estimation:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"irt_session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def get_performance_stats(self) -> Dict:
        """Get service performance statistics"""
        stats = self.performance_stats.copy()
        stats.update(self.engine.get_metrics())
        return stats