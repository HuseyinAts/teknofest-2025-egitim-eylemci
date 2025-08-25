"""
Enhanced Model Integration with Versioning System
Production-ready integration using the ML model versioning infrastructure
"""

import os
import sys
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import logging
from datetime import datetime, timezone
import yaml
from pathlib import Path

from src.config import get_settings
from src.ml.model_registry import (
    ModelRegistry, ModelMetadata, ModelStatus,
    ModelFramework, ModelType, get_model_registry
)
from src.ml.model_versioning_service import (
    ModelVersioningService, DeploymentConfig, DeploymentStrategy,
    get_versioning_service
)

# Fix Windows encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedModelIntegration:
    """Enhanced model integration with versioning, deployment, and monitoring"""
    
    def __init__(self):
        """Initialize enhanced model integration"""
        self.settings = settings
        self.registry = get_model_registry()
        self.versioning_service = get_versioning_service()
        
        # Load configuration
        self.config = self._load_config()
        
        # Model cache
        self.model_cache = {}
        
        # Initialize default models
        asyncio.create_task(self._initialize_default_models())
        
        logger.info("Enhanced Model Integration initialized with versioning system")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_path = Path(__file__).parent / "model_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    async def _initialize_default_models(self):
        """Initialize and deploy default models"""
        try:
            # Check for content generation model
            content_gen_config = self.config.get('models', {}).get('types', {}).get('content_generation', {})
            
            if content_gen_config:
                model_name = content_gen_config.get('default_model', settings.model_name)
                
                # Check if model is already deployed
                try:
                    await self.versioning_service.predict(
                        model_id="content-generation",
                        input_data="test"
                    )
                    logger.info("Content generation model already deployed")
                except:
                    # Deploy default model
                    await self._deploy_default_model(
                        model_id="content-generation",
                        model_name=model_name,
                        model_type=ModelType.CONTENT_GENERATION
                    )
            
        except Exception as e:
            logger.error(f"Failed to initialize default models: {e}")
    
    async def _deploy_default_model(self,
                                   model_id: str,
                                   model_name: str,
                                   model_type: ModelType):
        """Deploy a default model"""
        try:
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=model_name,
                version="1.0.0",
                framework=ModelFramework.HUGGINGFACE,
                model_type=model_type,
                description=f"Default {model_type.value} model",
                status=ModelStatus.PRODUCTION,
                tags=["default", "auto-deployed"],
                labels={"environment": settings.app_env.value}
            )
            
            # For HuggingFace models, we don't need to download the actual model
            # The versioning service will handle loading on-demand
            model_stub = {"model_name": model_name, "framework": "huggingface"}
            
            # Register in the registry
            self.registry.register_model(
                model=model_stub,
                metadata=metadata
            )
            
            # Deploy with immediate strategy
            config = DeploymentConfig(
                strategy=DeploymentStrategy.IMMEDIATE,
                auto_rollback=True,
                health_check_interval=60
            )
            
            await self.versioning_service.deploy_model(
                model_id=model_id,
                version="1.0.0",
                config=config
            )
            
            logger.info(f"Default model {model_id} deployed successfully")
            
        except Exception as e:
            logger.error(f"Failed to deploy default model {model_id}: {e}")
    
    async def generate(self,
                      prompt: str,
                      model_type: str = "content_generation",
                      max_length: int = 200,
                      version: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate text using versioned model with monitoring"""
        try:
            # Map model type to model ID
            model_id = self._get_model_id(model_type)
            
            # Prepare input
            input_data = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": self.config.get('models', {}).get('types', {}).get(model_type, {}).get('temperature', 0.7)
            }
            
            # Get prediction from versioning service
            result, version_used = await self.versioning_service.predict(
                model_id=model_id,
                input_data=input_data,
                version=version
            )
            
            # Extract generated text
            if isinstance(result, dict):
                generated_text = result.get('generated_text', str(result))
            else:
                generated_text = str(result)
            
            # Return with metadata
            metadata = {
                "model_id": model_id,
                "version": version_used,
                "model_type": model_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return generated_text, metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback to rule-based generation
            return self._rule_based_generation(prompt), {"fallback": True, "error": str(e)}
    
    async def generate_educational_content(self,
                                         subject: str,
                                         topic: str,
                                         grade_level: int,
                                         content_type: str = "explanation",
                                         language: str = "tr") -> Tuple[str, Dict[str, Any]]:
        """Generate educational content with versioned models"""
        try:
            # Create structured prompt
            if content_type == "quiz":
                prompt = f"""
                Konu: {subject} - {topic}
                Sınıf Seviyesi: {grade_level}
                
                Lütfen bu konuyla ilgili 4 seçenekli bir çoktan seçmeli soru oluştur.
                Format:
                Soru: [soru metni]
                A) [seçenek]
                B) [seçenek] 
                C) [seçenek]
                D) [seçenek]
                Doğru Cevap: [harf]
                Açıklama: [kısa açıklama]
                """
            elif content_type == "explanation":
                prompt = f"""
                Konu: {subject} - {topic}
                Sınıf Seviyesi: {grade_level}
                
                Bu konuyu {grade_level}. sınıf öğrencisine uygun şekilde açıkla.
                Açıklama net, anlaşılır ve örneklerle desteklenmiş olmalı.
                """
            elif content_type == "exercise":
                prompt = f"""
                Konu: {subject} - {topic}
                Sınıf Seviyesi: {grade_level}
                
                Bu konuyla ilgili bir alıştırma veya problem oluştur.
                Zorluk seviyesi {grade_level}. sınıfa uygun olmalı.
                Çözümü de ekle.
                """
            else:
                prompt = f"{grade_level}. sınıf {subject} dersi {topic} konusu {content_type}"
            
            # Generate content
            content, metadata = await self.generate(
                prompt=prompt,
                model_type="content_generation",
                max_length=500
            )
            
            # Add educational metadata
            metadata.update({
                "subject": subject,
                "topic": topic,
                "grade_level": grade_level,
                "content_type": content_type,
                "language": language
            })
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Educational content generation failed: {e}")
            # Fallback content
            if content_type == "quiz":
                return self._generate_fallback_quiz(subject, topic, grade_level), {"fallback": True}
            else:
                return self._generate_fallback_explanation(subject, topic, grade_level), {"fallback": True}
    
    async def assess_answer(self,
                           question: str,
                           student_answer: str,
                           correct_answer: str,
                           rubric: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assess student answer using ML model"""
        try:
            # Prepare assessment prompt
            prompt = f"""
            Soru: {question}
            Öğrenci Cevabı: {student_answer}
            Doğru Cevap: {correct_answer}
            
            Öğrencinin cevabını değerlendir:
            1. Doğruluk (0-100 puan)
            2. Eksik veya yanlış kısımlar
            3. Öneriler
            """
            
            # Get assessment from model
            assessment_text, metadata = await self.generate(
                prompt=prompt,
                model_type="assessment",
                max_length=300
            )
            
            # Parse assessment (in production, use structured output)
            assessment = {
                "score": self._extract_score(assessment_text),
                "feedback": assessment_text,
                "correct": student_answer.lower() == correct_answer.lower(),
                "metadata": metadata
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            # Simple fallback assessment
            return {
                "score": 100 if student_answer.lower() == correct_answer.lower() else 0,
                "feedback": "Cevabınız değerlendirildi.",
                "correct": student_answer.lower() == correct_answer.lower(),
                "error": str(e)
            }
    
    async def get_model_info(self, model_type: str = "content_generation") -> Dict[str, Any]:
        """Get information about deployed model"""
        try:
            model_id = self._get_model_id(model_type)
            
            # Get metrics from versioning service
            metrics = await self.versioning_service.get_model_metrics(model_id)
            
            # Get lineage from registry
            lineage = self.registry.get_model_lineage(model_id)
            
            # Find current production version
            production_version = None
            for version_info in lineage:
                if version_info['status'] == ModelStatus.PRODUCTION.value:
                    production_version = version_info['version']
                    break
            
            return {
                "model_id": model_id,
                "model_type": model_type,
                "production_version": production_version,
                "metrics": metrics,
                "total_versions": len(lineage),
                "deployment_strategy": self.config.get('deployment', {}).get('default_strategy', 'immediate')
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def compare_model_outputs(self,
                                   prompt: str,
                                   versions: List[str],
                                   model_type: str = "content_generation") -> Dict[str, Any]:
        """Compare outputs from different model versions"""
        try:
            model_id = self._get_model_id(model_type)
            results = {}
            
            for version in versions:
                try:
                    output, metadata = await self.generate(
                        prompt=prompt,
                        model_type=model_type,
                        version=version
                    )
                    results[version] = {
                        "output": output,
                        "metadata": metadata
                    }
                except Exception as e:
                    results[version] = {
                        "error": str(e)
                    }
            
            # Compare models
            if len(versions) == 2:
                comparison = await self.versioning_service.compare_versions(
                    model_id=model_id,
                    version1=versions[0],
                    version2=versions[1],
                    test_data={"prompt": prompt}
                )
            else:
                comparison = None
            
            return {
                "prompt": prompt,
                "results": results,
                "comparison": comparison
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": str(e)}
    
    def _get_model_id(self, model_type: str) -> str:
        """Map model type to model ID"""
        mapping = {
            "content_generation": "content-generation",
            "question_answering": "qa-model",
            "assessment": "assessment-model",
            "recommendation": "recommendation-model"
        }
        return mapping.get(model_type, model_type)
    
    def _extract_score(self, text: str) -> float:
        """Extract numerical score from assessment text"""
        import re
        # Look for patterns like "85 puan", "90/100", etc.
        patterns = [
            r'(\d+)\s*puan',
            r'(\d+)/100',
            r'%(\d+)',
            r'(\d+)\s*point'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _rule_based_generation(self, prompt: str) -> str:
        """Fallback rule-based generation"""
        prompt_lower = prompt.lower()
        
        # Educational responses
        knowledge_base = {
            "matematik": "Matematik, sayılar ve şekiller üzerine kurulu bir bilim dalıdır.",
            "fizik": "Fizik, doğadaki olayları inceleyen temel bilim dalıdır.",
            "kimya": "Kimya, maddelerin yapısını ve değişimlerini inceler.",
            "biyoloji": "Biyoloji, canlı organizmaları inceleyen bilim dalıdır.",
            "tarih": "Tarih, geçmişte yaşanan olayları ve toplumları inceler.",
            "coğrafya": "Coğrafya, Dünya'nın fiziki özelliklerini ve insan faaliyetlerini inceler."
        }
        
        for key, value in knowledge_base.items():
            if key in prompt_lower:
                return value
        
        return "Bu konuda size yardımcı olmak için daha fazla bilgiye ihtiyacım var."
    
    def _generate_fallback_quiz(self, subject: str, topic: str, grade_level: int) -> str:
        """Generate fallback quiz question"""
        return f"""
        Soru: {topic} konusuyla ilgili temel bir soru
        A) Seçenek 1
        B) Seçenek 2
        C) Seçenek 3
        D) Seçenek 4
        
        Doğru Cevap: A
        Açıklama: {grade_level}. sınıf seviyesine uygun {subject} sorusu.
        """
    
    def _generate_fallback_explanation(self, subject: str, topic: str, grade_level: int) -> str:
        """Generate fallback explanation"""
        return f"""
        {topic} Konusu ({grade_level}. Sınıf {subject})
        
        Bu konu, {subject} dersinin önemli konularından biridir.
        Öğrenciler bu konuyu öğrenerek temel kavramları anlayabilirler.
        
        Detaylı bilgi için ders kitabınıza ve öğretmeninize başvurabilirsiniz.
        """


# Singleton instance
_integration_instance = None

def get_model_integration() -> EnhancedModelIntegration:
    """Get or create enhanced model integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = EnhancedModelIntegration()
    return _integration_instance


if __name__ == "__main__":
    import asyncio
    
    async def test_integration():
        """Test enhanced model integration"""
        integration = get_model_integration()
        
        print("=" * 60)
        print("Enhanced Model Integration Test")
        print("=" * 60)
        
        # Test 1: Generate content
        print("\n[TEST 1] Content Generation")
        print("-" * 40)
        content, metadata = await integration.generate(
            prompt="Pisagor teoremi nedir?",
            model_type="content_generation"
        )
        print(f"Generated: {content[:200]}...")
        print(f"Version used: {metadata.get('version')}")
        
        # Test 2: Educational content
        print("\n[TEST 2] Educational Content")
        print("-" * 40)
        quiz, metadata = await integration.generate_educational_content(
            subject="Matematik",
            topic="Denklemler",
            grade_level=10,
            content_type="quiz"
        )
        print(f"Quiz: {quiz[:300]}...")
        
        # Test 3: Model info
        print("\n[TEST 3] Model Information")
        print("-" * 40)
        info = await integration.get_model_info("content_generation")
        print(f"Model Info: {info}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    
    # Run tests
    asyncio.run(test_integration())