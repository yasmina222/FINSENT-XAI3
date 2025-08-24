"""
FINSENT-XAI Pipeline Orchestrator
Manages the complete workflow from data collection to evaluation results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from pathlib import Path
import logging
import subprocess
from typing import Dict, List, Optional
import pandas as pd


class FINSENTPipeline:
    """
    Orchestrates the complete FINSENT-XAI pipeline:
    1. Data Collection (scraping)
    2. Data Validation
    3. Model Training
    4. Evaluation
    5. Results Generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize pipeline with configuration"""
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        self.results = {}
        
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            'min_articles': 100,
            'train_epochs': 3,
            'batch_size': 16,
            'test_size': 50,
            'output_dir': 'data/pipeline_output',
            'enable_training': True,  # Set to False to skip training
            'enable_api': False,  # Set to True to auto-start API
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Configure pipeline logger"""
        logger = logging.getLogger('FINSENTPipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
        
    def run_complete_pipeline(self):
        """Run the complete FINSENT-XAI pipeline"""
        self.logger.info("="*60)
        self.logger.info("STARTING FINSENT-XAI PIPELINE")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Step 1: Data Collection
            self.logger.info("\n[Step 1/5] Data Collection")
            scraped_file = self._run_data_collection()
            
            # Step 2: Data Validation
            self.logger.info("\n[Step 2/5] Data Validation")
            validated_file = self._run_data_validation(scraped_file)
            
            # Step 3: Model Training (optional)
            self.logger.info("\n[Step 3/5] Model Training")
            if self.config['enable_training']:
                model_path = self._run_model_training(validated_file)
            else:
                self.logger.info("Skipping training - using base FinBERT model")
                model_path = "ProsusAI/finbert"
                
            # Step 4: Evaluation
            self.logger.info("\n[Step 4/5] Model Evaluation")
            evaluation_results = self._run_evaluation(validated_file, model_path)
            
            # Step 5: Generate Reports
            self.logger.info("\n[Step 5/5] Generating Reports")
            self._generate_reports(evaluation_results)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Print summary
            self._print_summary(total_time)
            
            # Optionally start API
            if self.config['enable_api']:
                self._start_api_server()
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
            
    def _run_data_collection(self) -> str:
        """Run UK news scraper"""
        self.logger.info("Running UK news scraper...")
        
        try:
            # Import and run scraper
            from scrapers.uk_news_scraper import UKFinancialNewsScraper
            
            scraper = UKFinancialNewsScraper()
            articles = scraper.run(min_articles=self.config['min_articles'])
            
            if len(articles) < self.config['min_articles']:
                self.logger.warning(
                    f"Only collected {len(articles)} articles, "
                    f"minimum is {self.config['min_articles']}"
                )
                
            # Get output file
            output_file = "data/raw/uk_news/latest_articles.json"
            
            self.results['data_collection'] = {
                'articles_collected': len(articles),
                'output_file': output_file,
                'sources': list(set(a['source'] for a in articles))
            }
            
            self.logger.info(f"Collected {len(articles)} articles")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
            
    def _run_data_validation(self, input_file: str) -> str:
        """Run data validator"""
        self.logger.info("Validating scraped data...")
        
        try:
            from scrapers.data_validator import UKFinanceDataValidator
            
            # Load data
            with open(input_file, 'r') as f:
                data = json.load(f)
                articles = data.get('articles', [])
                
            # Validate
            validator = UKFinanceDataValidator()
            validation_results = validator.validate_dataset(articles)
            
            os.makedirs(self.config['output_dir'], exist_ok=True)
            
            # Generate report
            validator.generate_quality_report(
                validation_results,
                save_path=os.path.join(self.config['output_dir'], 'validation_report.txt')
            )
            
            # Save validated data
            output_file = validator.clean_and_save_valid_data(validation_results)
            
            self.results['data_validation'] = {
                'total_articles': len(articles),
                'valid_articles': len(validation_results['valid_articles']),
                'validity_rate': validation_results['statistics']['validity_rate'],
                'output_file': output_file
            }
            
            self.logger.info(
                f"Validated {len(validation_results['valid_articles'])}"
                f"/{len(articles)} articles "
                f"({validation_results['statistics']['validity_rate']:.1%} valid)"
            )
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
            
    def _run_model_training(self, training_data_file: str) -> str:
        """Run FinBERT training"""
        self.logger.info("Training FinBERT on UK financial data...")
        
        try:
            from ml.finbert_trainer import FinBERTTrainer
            
            trainer = FinBERTTrainer()
            
            # Set data path
            trainer.data_path = training_data_file
            
            # Run training
            trained_model, eval_results = trainer.run_full_pipeline()
            
            self.results['model_training'] = {
                'model_path': 'data/models/finbert-uk-final',
                'test_accuracy': eval_results['test_metrics']['test_accuracy'],
                'test_f1': eval_results['test_metrics']['test_f1'],
                'training_completed': True
            }
            
            self.logger.info(
                f"Training complete - "
                f"Accuracy: {eval_results['test_metrics']['test_accuracy']:.2%}, "
                f"F1: {eval_results['test_metrics']['test_f1']:.3f}"
            )
            
            return 'data/models/finbert-uk-final'
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.logger.info("Falling back to base FinBERT model")
            return "ProsusAI/finbert"
            
    def _run_evaluation(self, test_data_file: str, model_path: str) -> Dict:
        """Run comprehensive XAI evaluation"""
        self.logger.info("Running XAI method comparison...")
        
        try:
            from ml.xai_comparator import XAIComparator
            
            # Load test texts
            with open(test_data_file, 'r') as f:
                data = json.load(f)
                articles = data.get('articles', [])
                
            # Select test texts
            test_texts = [a['title'] for a in articles[:self.config['test_size']]]
            
            # Initialize comparator
            comparator = XAIComparator(model_path)
            
            # Run evaluation
            evaluation_results = comparator.run_comprehensive_evaluation(test_texts)
            
            # Save results
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparator.save_evaluation_results(
                evaluation_results,
                output_dir / 'xai_evaluation_results.json'
            )
            
            # Generate visualizations
            comparator.generate_comparison_visualizations(
                evaluation_results,
                output_dir=str(output_dir)
            )
            
            self.results['evaluation'] = {
                'num_texts_evaluated': len(test_texts),
                'average_processing_time': evaluation_results['average_processing_time'],
                'faithfulness_scores': evaluation_results['faithfulness_scores'],
                'stability_scores': evaluation_results['stability_scores'],
                'output_dir': str(output_dir)
            }
            
            self.logger.info("Evaluation complete - visualizations saved")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
            
    def _generate_reports(self, evaluation_results: Dict):
        """Generate final reports and summaries"""
        self.logger.info("Generating final reports...")
        
        output_dir = Path(self.config['output_dir'])
        
        # 1. Pipeline Summary Report
        summary_report = []
        summary_report.append("FINSENT-XAI PIPELINE EXECUTION SUMMARY")
        summary_report.append("=" * 60)
        summary_report.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_report.append("")
        
        # Data Collection Summary
        if 'data_collection' in self.results:
            dc = self.results['data_collection']
            summary_report.append("DATA COLLECTION:")
            summary_report.append(f"  Articles collected: {dc['articles_collected']}")
            summary_report.append(f"  Sources: {', '.join(dc['sources'])}")
            summary_report.append("")
            
        # Validation Summary
        if 'data_validation' in self.results:
            dv = self.results['data_validation']
            summary_report.append("DATA VALIDATION:")
            summary_report.append(f"  Valid articles: {dv['valid_articles']}/{dv['total_articles']}")
            summary_report.append(f"  Validity rate: {dv['validity_rate']:.1%}")
            summary_report.append("")
            
        # Training Summary
        if 'model_training' in self.results:
            mt = self.results['model_training']
            summary_report.append("MODEL TRAINING:")
            summary_report.append(f"  Test Accuracy: {mt['test_accuracy']:.2%}")
            summary_report.append(f"  Test F1-Score: {mt['test_f1']:.3f}")
            summary_report.append("")
            
        # Evaluation Summary
        if 'evaluation' in self.results:
            ev = self.results['evaluation']
            summary_report.append("XAI EVALUATION:")
            summary_report.append(f"  Texts evaluated: {ev['num_texts_evaluated']}")
            summary_report.append(f"  Average times: {ev['average_processing_time']}")
            summary_report.append(f"  Faithfulness: {ev['faithfulness_scores']}")
            summary_report.append(f"  Stability: {ev['stability_scores']}")
            
        # Save summary
        summary_path = output_dir / 'pipeline_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_report))
            
        self.logger.info(f"Reports saved to {output_dir}")
        
        # 2. Results for dissertation
        dissertation_data = {
            'experiment_date': datetime.now().isoformat(),
            'pipeline_results': self.results,
            'evaluation_summary': {
                'best_method_speed': 'attention' if evaluation_results['average_processing_time']['attention'] < evaluation_results['average_processing_time']['lime'] else 'lime',
                'best_method_faithfulness': 'attention' if evaluation_results['faithfulness_scores']['attention'] > evaluation_results['faithfulness_scores']['lime'] else 'lime',
                'best_method_stability': 'attention' if evaluation_results['stability_scores']['attention'] > evaluation_results['stability_scores']['lime'] else 'lime'
            }
        }
        
        dissertation_path = output_dir / 'dissertation_results.json'
        with open(dissertation_path, 'w') as f:
            json.dump(dissertation_data, f, indent=2)
            
        # 3. Create results table for dissertation
        self._create_results_table(evaluation_results, output_dir)
        
    def _create_results_table(self, evaluation_results: Dict, output_dir: Path):
        """Create formatted results table for dissertation"""
        # Create DataFrame for nice formatting - ONLY TWO METHODS NOW
        methods = ['attention', 'lime']
        
        data = {
            'Method': methods,
            'Avg Time (s)': [
                evaluation_results['average_processing_time'][m] 
                for m in methods
            ],
            'Faithfulness': [
                evaluation_results['faithfulness_scores'][m] 
                for m in methods
            ],
            'Stability': [
                evaluation_results['stability_scores'][m] 
                for m in methods
            ]
        }
        
        df = pd.DataFrame(data)
        df = df.round(3)
        
        # Save as CSV
        df.to_csv(output_dir / 'results_table.csv', index=False)
        
        # Save as formatted text
        with open(output_dir / 'results_table.txt', 'w') as f:
            f.write("XAI METHOD COMPARISON RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))
            
    def _print_summary(self, total_time: float):
        """Print pipeline execution summary"""
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        
        print("\nKey Results:")
        if 'data_collection' in self.results:
            print(f"  - Articles collected: {self.results['data_collection']['articles_collected']}")
            
        if 'data_validation' in self.results:
            print(f"  - Valid articles: {self.results['data_validation']['valid_articles']}")
            
        if 'model_training' in self.results:
            print(f"  - Model accuracy: {self.results['model_training']['test_accuracy']:.2%}")
            
        if 'evaluation' in self.results:
            ev = self.results['evaluation']
            print(f"  - XAI methods compared on {ev['num_texts_evaluated']} texts")
            
        print(f"\nResults saved to: {self.config['output_dir']}/")
        
        print("\nVISUAL OUTPUTS GENERATED:")
        print(f"  - {self.config['output_dir']}/attention_lime_speed_comparison.png")
        print(f"  - {self.config['output_dir']}/attention_lime_agreement.png")
        print(f"  - {self.config['output_dir']}/attention_lime_faithfulness_stability.png")
        
        print("\nDATA FILES FOR DISSERTATION:")
        print(f"  - {self.config['output_dir']}/dissertation_results.json")
        print(f"  - {self.config['output_dir']}/results_table.csv")
        print(f"  - {self.config['output_dir']}/pipeline_summary.txt")
        
    def _start_api_server(self):
        """Start the API server"""
        self.logger.info("\nStarting API server...")
        
        try:
            # Start API in subprocess
            api_process = subprocess.Popen(
                [sys.executable, "src/api/main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a bit for startup
            time.sleep(5)
            
            if api_process.poll() is None:
                print("\nAPI server started at http://localhost:8000")
                print("  - API docs: http://localhost:8000/docs")
                print("  - To stop: Ctrl+C")
            else:
                print("\nAPI server failed to start")
                
        except Exception as e:
            self.logger.error(f"Failed to start API: {e}")


def main():
    """Run the complete pipeline"""
    print("\nFINSENT-XAI PIPELINE RUNNER")
    print("This will run the complete workflow from data to results\n")
    
    # Configuration options
    config = {
        'min_articles': 100,      # Minimum articles to collect
        'train_epochs': 2,        # Reduced for faster testing
        'batch_size': 16,         
        'test_size': 30,          # Number of texts for evaluation
        'output_dir': f'data/pipeline_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'enable_training': False,  # Set True to train model (takes ~30 min)
        'enable_api': False       # Set True to auto-start API
    }
    
    # Ask user about training
    response = input("Enable model training? (y/n, default=n): ").strip().lower()
    if response == 'y':
        config['enable_training'] = True
        print("Training enabled - this will take 20-30 minutes")
    else:
        print("Using base FinBERT model (no training)")
        
    # Create pipeline
    pipeline = FINSENTPipeline(config)
    
    try:
        # Run pipeline
        pipeline.run_complete_pipeline()
        
        print("\nPipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check visual outputs in:", config['output_dir'])
        print("2. Start API: python src/api/main.py")
        print("3. Launch dashboard: streamlit run src/dashboard/app.py")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        print("Check logs/pipeline_*.log for details")
        

if __name__ == "__main__":
    main()