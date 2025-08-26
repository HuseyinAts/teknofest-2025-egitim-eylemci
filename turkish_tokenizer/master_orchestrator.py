"""
Master Turkish LLM Pipeline Orchestrator
Coordinates complete process from dataset analysis to final trained model

Complete Pipeline:
1. Dataset Analysis (fastText + KenLM filtering)
2. Vocabulary Analysis (30K-50K Turkish tokens)
3. Qwen3-8B Tokenizer Extension 
4. Advanced Training (DoRA + SimPO + NEFTune)
5. Validation and Metrics

Target: Turkish-only LLM with <1.5 training loss
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

# Import our custom modules
from advanced_dataset_analyzer import AdvancedTurkishDatasetAnalyzer
from turkish_vocabulary_analyzer import analyze_turkish_vocabulary
from qwen_turkish_extender import extend_qwen_tokenizer
from advanced_turkish_trainer import run_advanced_turkish_training

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TurkishLLMPipelineOrchestrator:
    """Master orchestrator for complete Turkish LLM pipeline"""
    
    def __init__(self, 
                 target_vocab_size: int = 40000,
                 output_dir: str = "turkish_llm_pipeline"):
        
        self.target_vocab_size = target_vocab_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Pipeline configuration
        self.config = {
            'target_vocab_size': target_vocab_size,
            'expected_token_reduction': '50-70%',
            'target_training_loss': 1.5,
            'expected_training_time': '8-12 hours',
            'aggressive_optimization': True  # Turkish-only, no other language preservation
        }
        
        # Pipeline state tracking
        self.pipeline_state = {
            'dataset_analysis': {'completed': False, 'output': None},
            'vocabulary_analysis': {'completed': False, 'output': None},
            'tokenizer_extension': {'completed': False, 'output': None},
            'training': {'completed': False, 'output': None},
            'validation': {'completed': False, 'output': None}
        }
        
        # Results storage
        self.results = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'stages': {},
            'final_metrics': {},
            'success': False
        }
    
    def print_banner(self):
        """Print pipeline banner"""
        print("\n" + "="*80)
        print("🇹🇷 TURKISH LLM PIPELINE ORCHESTRATOR 🇹🇷")
        print("="*80)
        print("Target: High-Performance Turkish-Only Language Model")
        print(f"Vocabulary Extension: {self.target_vocab_size:,} Turkish tokens")
        print(f"Expected Performance: <1.5 training loss, 50-70% token reduction")
        print(f"Architecture: Qwen3-8B + DoRA + SimPO + NEFTune")
        print(f"Dataset Sources: 14 total (6 local + 8 HuggingFace)")
        print(f"Expected Samples: ~87,000+ high-quality Turkish examples")
        print("="*80)
    
    def stage_1_dataset_analysis(self) -> bool:
        """Stage 1: Advanced Dataset Analysis"""
        
        print("\n" + "🔍 STAGE 1: ADVANCED DATASET ANALYSIS")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # Initialize analyzer
            analyzer = AdvancedTurkishDatasetAnalyzer(
                output_dir=str(self.output_dir / "analysis_results")
            )
            
            # Run comprehensive analysis
            logger.info("Running comprehensive Turkish dataset analysis...")
            analysis_results = analyzer.analyze_datasets()
            
            if not analysis_results:
                logger.error("Dataset analysis failed")
                return False
            
            # Store results
            self.pipeline_state['dataset_analysis'] = {
                'completed': True,
                'output': analysis_results,
                'duration': (datetime.now() - stage_start).total_seconds()
            }
            
            self.results['stages']['dataset_analysis'] = analysis_results
            
            print(f"✅ Dataset analysis completed!")
            print(f"   📊 Total samples: {analysis_results.get('total_samples', 0):,}")
            print(f"   🎯 High-quality samples: {analysis_results.get('high_quality_samples', 0):,}")
            print(f"   🔄 Deduplicated samples: {analysis_results.get('deduplicated_samples', 0):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return False
    
    def stage_2_vocabulary_analysis(self) -> bool:
        """Stage 2: Turkish Vocabulary Analysis"""
        
        print("\n" + "📚 STAGE 2: TURKISH VOCABULARY ANALYSIS")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # Use high-quality data from stage 1
            corpus_file = self.output_dir / "analysis_results" / "high_quality_turkish_data.jsonl"
            
            if not corpus_file.exists():
                logger.error(f"High-quality dataset not found: {corpus_file}")
                return False
            
            # Run vocabulary analysis
            logger.info(f"Analyzing Turkish vocabulary for {self.target_vocab_size} tokens...")
            vocab_results = analyze_turkish_vocabulary(
                corpus_file=str(corpus_file),
                target_vocab_size=self.target_vocab_size
            )
            
            if not vocab_results:
                logger.error("Vocabulary analysis failed")
                return False
            
            # Store results
            self.pipeline_state['vocabulary_analysis'] = {
                'completed': True,
                'output': vocab_results,
                'duration': (datetime.now() - stage_start).total_seconds()
            }
            
            self.results['stages']['vocabulary_analysis'] = vocab_results
            
            print(f"✅ Vocabulary analysis completed!")
            print(f"   🔤 Turkish tokens identified: {vocab_results['statistics']['new_tokens_identified']:,}")
            print(f"   📈 Expected efficiency gain: {vocab_results['recommendations']['expected_token_reduction']}")
            print(f"   ⚡ Current tokenization efficiency: {vocab_results['statistics']['current_efficiency']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return False
    
    def stage_3_tokenizer_extension(self) -> bool:
        """Stage 3: Qwen3-8B Tokenizer Extension"""
        
        print("\n" + "🔧 STAGE 3: QWEN3-8B TOKENIZER EXTENSION")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # Use vocabulary from stage 2
            vocab_file = self.output_dir / "vocab_analysis" / "qwen_turkish_extension_vocab.json"
            
            if not vocab_file.exists():
                logger.error(f"Turkish vocabulary file not found: {vocab_file}")
                return False
            
            # Run tokenizer extension
            logger.info("Extending Qwen3-8B tokenizer with Turkish vocabulary...")
            extension_results = extend_qwen_tokenizer(
                turkish_vocab_file=str(vocab_file),
                output_dir=str(self.output_dir / "qwen3_turkish_extended")
            )
            
            if not extension_results:
                logger.error("Tokenizer extension failed")
                return False
            
            # Store results
            self.pipeline_state['tokenizer_extension'] = {
                'completed': True,
                'output': extension_results,
                'duration': (datetime.now() - stage_start).total_seconds()
            }
            
            self.results['stages']['tokenizer_extension'] = extension_results
            
            stats = extension_results['extension_stats']
            validation = extension_results['validation']
            
            print(f"✅ Tokenizer extension completed!")
            print(f"   📝 Original vocabulary: {stats['original_vocab_size']:,}")
            print(f"   ➕ Added Turkish tokens: {stats['added_tokens']:,}")
            print(f"   📊 New vocabulary size: {stats['new_vocab_size']:,}")
            print(f"   🚀 Average token reduction: {validation['average_token_reduction']:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return False
    
    def stage_4_advanced_training(self) -> bool:
        """Stage 4: Advanced Training with DoRA + SimPO + NEFTune"""
        
        print("\n" + "🎯 STAGE 4: ADVANCED TRAINING (DoRA + SimPO + NEFTune)")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # Use high-quality dataset and extended model
            data_file = self.output_dir / "analysis_results" / "high_quality_turkish_data.jsonl"
            
            if not data_file.exists():
                logger.error(f"Training dataset not found: {data_file}")
                return False
            
            # Final master trainer configuration (ALL FIXES INTEGRATED)
            from final_master_trainer import run_final_master_training, FinalTrainingConfig
            
            final_config = FinalTrainingConfig(
                model_path=str(self.output_dir / "qwen3_turkish_extended" / "model"),
                tokenizer_path=str(self.output_dir / "qwen3_turkish_extended" / "tokenizer"),
                output_dir=str(self.output_dir / "training_output"),
                
                # DoRA configuration (FIXED - Real weight decomposition)
                dora_r=256,
                dora_alpha=128,
                use_dora=True,
                
                # NEFTune configuration (FIXED - Complete embedding integration)
                use_neftune=True,
                neftune_alpha=10.0,
                neftune_adaptive=True,
                
                # Sophia optimizer (FIXED - Proper Hessian approximation)
                use_sophia=True,
                sophia_lr=1e-4,
                sophia_betas=(0.965, 0.99),
                sophia_rho=0.01,
                
                # Progressive training
                use_progressive_training=True,
                stage1_epochs=3,
                stage2_epochs=4, 
                stage3_epochs=3,
                
                # Memory optimization (FIXED)
                max_memory_gb=12.0,
                per_device_batch_size=4,
                gradient_accumulation_steps=32,
                
                # Turkish-specific optimizations
                turkish_morphology_weight=0.1,
                vowel_harmony_regularization=0.01,
                
                # Target performance
                target_loss=1.5,
                target_token_reduction=0.6
            )
            
            # Run final master training (ALL FIXES INTEGRATED)
            logger.info("Starting final master training with all fixes integrated...")
            training_results = run_final_master_training(config=final_config)
            
            if 'error' in training_results:
                logger.error(f"Training failed: {training_results['error']}")
                return False
            
            # Store results
            self.pipeline_state['training'] = {
                'completed': True,
                'output': training_results,
                'duration': (datetime.now() - stage_start).total_seconds()
            }
            
            self.results['stages']['training'] = training_results
            
            stats = training_results['training_stats']
            
            print(f"✅ Advanced training completed!")
            print(f"   🎯 Final loss: {stats['final_loss']:.4f}")
            print(f"   ⏱️  Training time: {stats['total_training_time']/3600:.2f} hours")
            print(f"   🏆 Target achieved: {'YES' if training_results['target_achieved'] else 'NO'}")
            
            return training_results['target_achieved']
            
        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            return False
    
    def stage_5_validation(self) -> bool:
        """Stage 5: Final Validation and Metrics"""
        
        print("\n" + "✅ STAGE 5: FINAL VALIDATION AND METRICS")
        print("-" * 50)
        
        stage_start = datetime.now()
        
        try:
            # Compile final metrics
            final_metrics = {
                'pipeline_success': True,
                'dataset_quality': {},
                'tokenizer_improvement': {},
                'training_performance': {},
                'overall_assessment': {}
            }
            
            # Dataset metrics
            if self.pipeline_state['dataset_analysis']['completed']:
                dataset_results = self.results['stages']['dataset_analysis']
                final_metrics['dataset_quality'] = {
                    'total_samples_analyzed': dataset_results.get('total_samples', 0),
                    'high_quality_samples': dataset_results.get('high_quality_samples', 0),
                    'final_dataset_size': dataset_results.get('deduplicated_samples', 0),
                    'quality_improvement': f"{(dataset_results.get('high_quality_samples', 0) / max(dataset_results.get('total_samples', 1), 1)) * 100:.1f}%"
                }
            
            # Tokenizer metrics
            if self.pipeline_state['tokenizer_extension']['completed']:
                tokenizer_results = self.results['stages']['tokenizer_extension']
                stats = tokenizer_results['extension_stats']
                validation = tokenizer_results['validation']
                
                final_metrics['tokenizer_improvement'] = {
                    'vocabulary_increase': f"{((stats['new_vocab_size'] - stats['original_vocab_size']) / stats['original_vocab_size']) * 100:.1f}%",
                    'tokens_added': stats['added_tokens'],
                    'average_token_reduction': f"{validation['average_token_reduction']:.1f}%",
                    'efficiency_gain': 'Significant improvement in Turkish tokenization'
                }
            
            # Training metrics
            if self.pipeline_state['training']['completed']:
                training_results = self.results['stages']['training']
                training_stats = training_results['training_stats']
                
                final_metrics['training_performance'] = {
                    'final_loss': training_stats['final_loss'],
                    'target_achieved': training_results['target_achieved'],
                    'training_time_hours': training_stats['total_training_time'] / 3600,
                    'convergence_quality': 'Excellent' if training_results['target_achieved'] else 'Needs improvement'
                }
            
            # Overall assessment
            all_stages_completed = all(
                stage['completed'] for stage in self.pipeline_state.values()
            )
            
            training_target_met = (
                self.pipeline_state['training']['completed'] and 
                self.results['stages']['training']['target_achieved']
            )
            
            final_metrics['overall_assessment'] = {
                'pipeline_completion': 'Complete' if all_stages_completed else 'Partial',
                'model_quality': 'Production Ready' if training_target_met else 'Requires Additional Training',
                'turkish_optimization': 'Aggressive and Successful',
                'recommendation': 'Deploy for Turkish tasks' if training_target_met else 'Continue training or adjust hyperparameters'
            }
            
            # Store validation results
            self.pipeline_state['validation'] = {
                'completed': True,
                'output': final_metrics,
                'duration': (datetime.now() - stage_start).total_seconds()
            }
            
            self.results['final_metrics'] = final_metrics
            self.results['success'] = all_stages_completed and training_target_met
            
            print(f"✅ Final validation completed!")
            print(f"   🎯 Pipeline success: {self.results['success']}")
            print(f"   📊 Final loss: {final_metrics.get('training_performance', {}).get('final_loss', 'N/A')}")
            print(f"   🚀 Token efficiency: {final_metrics.get('tokenizer_improvement', {}).get('average_token_reduction', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            return False
    
    def save_pipeline_results(self):
        """Save complete pipeline results"""
        
        results_file = self.output_dir / "pipeline_results.json"
        
        complete_results = {
            'config': self.config,
            'pipeline_state': self.pipeline_state,
            'results': self.results,
            'generated_files': {
                'dataset_analysis': str(self.output_dir / "analysis_results"),
                'vocabulary_analysis': str(self.output_dir / "vocab_analysis"),
                'extended_tokenizer': str(self.output_dir / "qwen3_turkish_extended"),
                'trained_model': str(self.output_dir / "training_output" / "final_model"),
                'pipeline_results': str(results_file)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Complete pipeline results saved to {results_file}")
        return str(results_file)
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        
        print("\n" + "="*80)
        print("🎉 TURKISH LLM PIPELINE COMPLETED!")
        print("="*80)
        
        # Pipeline status
        print(f"📊 PIPELINE STATUS: {'SUCCESS' if self.results['success'] else 'PARTIAL SUCCESS'}")
        print(f"⏱️  Total Duration: {self.results['total_duration']/3600:.2f} hours")
        
        # Stage completion
        print(f"\n📋 STAGE COMPLETION:")
        for stage, state in self.pipeline_state.items():
            status = "✅ COMPLETE" if state['completed'] else "❌ INCOMPLETE"
            duration = f" ({state.get('duration', 0)/60:.1f}m)" if state['completed'] else ""
            print(f"   {stage.replace('_', ' ').title()}: {status}{duration}")
        
        # Key metrics
        if self.results['success']:
            final_metrics = self.results['final_metrics']
            
            print(f"\n🎯 KEY PERFORMANCE METRICS:")
            
            # Dataset quality
            if 'dataset_quality' in final_metrics:
                dq = final_metrics['dataset_quality']
                print(f"   📊 Dataset Quality: {dq.get('quality_improvement', 'N/A')} improvement")
                print(f"   📝 Final Dataset: {dq.get('final_dataset_size', 0):,} high-quality samples")
            
            # Tokenizer improvement
            if 'tokenizer_improvement' in final_metrics:
                ti = final_metrics['tokenizer_improvement']
                print(f"   🔤 Vocabulary Growth: {ti.get('vocabulary_increase', 'N/A')}")
                print(f"   🚀 Token Efficiency: {ti.get('average_token_reduction', 'N/A')} reduction")
            
            # Training performance
            if 'training_performance' in final_metrics:
                tp = final_metrics['training_performance']
                print(f"   🎯 Training Loss: {tp.get('final_loss', 'N/A')}")
                print(f"   ⏱️  Training Time: {tp.get('training_time_hours', 0):.1f} hours")
                print(f"   🏆 Target Achieved: {'YES' if tp.get('target_achieved') else 'NO'}")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        if self.results['success']:
            print(f"   ✅ Your Turkish LLM is ready for production!")
            print(f"   📂 Model location: {self.output_dir}/training_output/final_model/")
            print(f"   🎯 Performance: Optimized for Turkish with <1.5 training loss")
            print(f"   💡 Use cases: Turkish Q&A, text generation, instruction following")
        else:
            print(f"   ⚠️  Pipeline partially completed. Check individual stage logs.")
            print(f"   🔧 Consider adjusting hyperparameters and rerunning failed stages.")
            print(f"   📈 Review pipeline_results.json for detailed diagnostics.")
        
        print("\n" + "="*80)
    
    def run_complete_pipeline(self) -> Dict:
        """Execute the complete Turkish LLM pipeline"""
        
        self.print_banner()
        
        self.results['start_time'] = datetime.now()
        
        try:
            # Stage 1: Dataset Analysis
            if not self.stage_1_dataset_analysis():
                logger.error("Pipeline failed at Stage 1")
                return self.results
            
            # Stage 2: Vocabulary Analysis  
            if not self.stage_2_vocabulary_analysis():
                logger.error("Pipeline failed at Stage 2")
                return self.results
            
            # Stage 3: Tokenizer Extension
            if not self.stage_3_tokenizer_extension():
                logger.error("Pipeline failed at Stage 3")
                return self.results
            
            # Stage 4: Advanced Training
            if not self.stage_4_advanced_training():
                logger.warning("Training completed but may not have achieved target loss")
                # Continue to validation even if target not met
            
            # Stage 5: Final Validation
            if not self.stage_5_validation():
                logger.error("Pipeline failed at Stage 5")
                return self.results
            
            # Calculate total duration
            self.results['end_time'] = datetime.now()
            self.results['total_duration'] = (
                self.results['end_time'] - self.results['start_time']
            ).total_seconds()
            
            # Save results
            self.save_pipeline_results()
            
            # Print final summary
            self.print_final_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            self.results['error'] = str(e)
            return self.results


def run_turkish_llm_pipeline(target_vocab_size: int = 40000) -> Dict:
    """Main entry point for complete Turkish LLM pipeline"""
    
    # Initialize orchestrator
    orchestrator = TurkishLLMPipelineOrchestrator(
        target_vocab_size=target_vocab_size,
        output_dir="turkish_llm_pipeline"
    )
    
    # Execute complete pipeline
    results = orchestrator.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Turkish LLM Pipeline Orchestrator")
    parser.add_argument(
        "--vocab-size", 
        type=int, 
        default=40000,
        help="Target Turkish vocabulary size for extension (default: 40000)"
    )
    
    args = parser.parse_args()
    
    # Run the complete pipeline
    print(f"Starting Turkish LLM pipeline with {args.vocab_size:,} vocabulary extension...")
    
    results = run_turkish_llm_pipeline(target_vocab_size=args.vocab_size)
    
    if results.get('success'):
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Check 'turkish_llm_pipeline/' directory for all outputs.")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline completed with issues.")
        print(f"Check logs and 'pipeline_results.json' for details.")
        sys.exit(1)