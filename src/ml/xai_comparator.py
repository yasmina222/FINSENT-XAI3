"""
XAI Method Comparison Framework for FINSENT-XAI
Compares attention mechanisms and LIME for financial sentiment explainability
Enhanced with better visualizations for dissertation
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau

from .attention_extractor import AttentionExtractor
from .lime_explainer import FinancialLimeExplainer


class XAIComparator:
    """
    Comparison framework for attention mechanisms and LIME explainability methods
    in financial sentiment analysis.
    """
    
    def __init__(self, model_path: str = "data/models/finbert-uk-final"):
        """
        Initialize attention and LIME explainability methods.
        
        Args:
            model_path: Path to fine-tuned FinBERT model
        """
        self.logger = self._setup_logger()
        self.model_path = model_path
        
        # Initialize explainers
        self.logger.info("Initializing explainability methods...")
        self.attention_extractor = AttentionExtractor(model_path)
        self.lime_explainer = FinancialLimeExplainer(model_path)
        
        # Results storage
        self.comparison_results = []
        
        # Set up academic style for visualizations
        self._setup_visualization_style()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for comparator"""
        logger = logging.getLogger('XAIComparator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _setup_visualization_style(self):
        """Set up academic style for visualizations"""
        # Use classic academic style
        plt.style.use('classic')
        
        # Set font sizes for readability in Word documents
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        
        # Set figure DPI for screen viewing (not print)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        
        # Academic color palette
        self.colors = {
            'attention': '#2E86AB',  # Academic blue
            'lime': '#A23B72',       # Academic burgundy
            'positive': '#2E7D32',   # Academic green
            'negative': '#C62828',   # Academic red
            'neutral': '#616161'     # Academic gray
        }
        
    def compare_single_instance(self, text: str) -> Dict:
        """
        Compare attention and LIME methods on a single text instance.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Comprehensive comparison results
        """
        self.logger.info(f"Comparing methods for: {text[:50]}...")
        
        results = {
            'text': text,
            'attention': {},
            'lime': {},
            'comparisons': {},
            'timings': {}
        }
        
        # 1. Attention-based explanation
        start_time = time.time()
        attention_data = self.attention_extractor.extract_attention(text)
        attention_importance = self.attention_extractor.get_token_importance(attention_data)
        results['timings']['attention'] = time.time() - start_time
        results['attention']['importance'] = attention_importance
        
        # 2. LIME explanation
        start_time = time.time()
        lime_result = self.lime_explainer.explain_instance(text)
        
        # Handle both dict and object access patterns
        if isinstance(lime_result, dict):
            lime_importance = {
                feature.lower(): abs(score) 
                for feature, score in lime_result['feature_importance']
            }
            results['lime']['prediction'] = lime_result['prediction']
        else:
            lime_importance = {
                feature.lower(): abs(score) 
                for feature, score in lime_result.feature_importance
            }
            results['lime']['prediction'] = lime_result.prediction
            
        results['timings']['lime'] = time.time() - start_time
        results['lime']['importance'] = lime_importance
        
        # 3. Calculate agreement metrics
        results['comparisons'] = self._calculate_agreement_metrics(
            attention_importance, 
            lime_importance
        )
        
        # 4. Identify consensus features
        results['consensus_features'] = self._find_consensus_features(
            attention_importance,
            lime_importance
        )
        
        return results
        
    def _calculate_agreement_metrics(self,
                                   attention_imp: Dict[str, float],
                                   lime_imp: Dict[str, float]) -> Dict:
        """Calculate agreement between attention and LIME methods"""
        
        # Get common tokens across both methods
        all_tokens = set()
        all_tokens.update(attention_imp.keys())
        all_tokens.update(lime_imp.keys())
        
        # Convert to normalized vectors for comparison
        tokens_list = sorted(list(all_tokens))
        
        attention_vec = np.array([attention_imp.get(t.lower(), 0) for t in tokens_list])
        lime_vec = np.array([lime_imp.get(t.lower(), 0) for t in tokens_list])
        
        # Normalize vectors
        if np.sum(attention_vec) > 0:
            attention_vec = attention_vec / np.sum(attention_vec)
        if np.sum(lime_vec) > 0:
            lime_vec = lime_vec / np.sum(lime_vec)
            
        # Calculate correlations (with safety checks)
        def safe_correlation(vec1, vec2):
            if len(vec1) > 1 and np.std(vec1) > 0 and np.std(vec2) > 0:
                return float(np.corrcoef(vec1, vec2)[0, 1])
            return 0.0
            
        metrics = {
            'attention_lime_correlation': safe_correlation(attention_vec, lime_vec)
        }
        
        # Calculate rank correlations (Spearman) with safety
        try:
            metrics['attention_lime_spearman'] = float(spearmanr(attention_vec, lime_vec)[0])
        except:
            metrics['attention_lime_spearman'] = 0.0
        
        # Top-k agreement (k=5)
        k = 5
        top_attention = set(sorted(attention_imp.items(), key=lambda x: x[1], reverse=True)[:k])
        top_lime = set(sorted(lime_imp.items(), key=lambda x: x[1], reverse=True)[:k])
        
        top_attention_tokens = set(t[0] for t in top_attention)
        top_lime_tokens = set(t[0] for t in top_lime)
        
        metrics['top5_overlap'] = len(top_attention_tokens & top_lime_tokens) / k
        
        return metrics
        
    def _find_consensus_features(self,
                               attention_imp: Dict[str, float],
                               lime_imp: Dict[str, float],
                               top_k: int = 5) -> List[str]:
        """Find features that both methods agree are important"""
        
        # Get top-k features from each method
        top_attention = set(sorted(attention_imp.items(), key=lambda x: x[1], reverse=True)[:top_k])
        top_lime = set(sorted(lime_imp.items(), key=lambda x: x[1], reverse=True)[:top_k])
        
        top_attention_tokens = set(t[0] for t in top_attention)
        top_lime_tokens = set(t[0] for t in top_lime)
        
        # Find consensus (tokens in both methods' top-k)
        consensus = list(top_attention_tokens & top_lime_tokens)
        
        return consensus
        
    def evaluate_faithfulness(self, texts: List[str], num_perturbations: int = 10) -> Dict:
        """
        Evaluate faithfulness of each explanation method.
        Tests if removing important features changes predictions.
        """
        self.logger.info(f"Evaluating faithfulness on {len(texts)} texts...")
        
        faithfulness_scores = {
            'attention': [],
            'lime': []
        }
        
        for text in texts:
            # Get explanations from both methods
            comparison = self.compare_single_instance(text)
            
            # Get original prediction
            original_proba = self.lime_explainer._predict_proba([text])[0]
            original_pred = np.argmax(original_proba)
            
            # Test each method's faithfulness
            for method in ['attention', 'lime']:
                importance = comparison[method]['importance']
                
                # Get top important tokens
                top_tokens = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                # Create perturbations by removing important tokens
                perturbed_texts = []
                for _ in range(num_perturbations):
                    perturbed = text
                    for token, _ in top_tokens[:3]:  # Remove top 3
                        perturbed = perturbed.replace(token, '')
                    perturbed_texts.append(perturbed)
                    
                # Get predictions for perturbed texts
                perturbed_probas = self.lime_explainer._predict_proba(perturbed_texts)
                perturbed_preds = np.argmax(perturbed_probas, axis=1)
                
                # Calculate faithfulness (how often prediction changes)
                prediction_changes = np.mean(perturbed_preds != original_pred)
                faithfulness_scores[method].append(prediction_changes)
                
        # Average faithfulness scores
        avg_faithfulness = {
            method: float(np.mean(scores))
            for method, scores in faithfulness_scores.items()
        }
        
        return avg_faithfulness
        
    def evaluate_stability(self, texts: List[str], num_runs: int = 5) -> Dict:
        """
        Evaluate stability/consistency of explanations across multiple runs.
        """
        self.logger.info(f"Evaluating stability with {num_runs} runs per text...")
        
        stability_scores = {
            'attention': [],
            'lime': []
        }
        
        for text in texts:
            # Collect multiple explanations for each method
            attention_runs = []
            lime_runs = []
            
            for _ in range(num_runs):
                # Attention (deterministic, but test anyway)
                att_data = self.attention_extractor.extract_attention(text)
                att_imp = self.attention_extractor.get_token_importance(att_data)
                attention_runs.append(att_imp)
                
                # LIME (stochastic)
                lime_result = self.lime_explainer.explain_instance(text, num_samples=1000)
                if isinstance(lime_result, dict):
                    lime_imp = {
                        feature.lower(): abs(score)
                        for feature, score in lime_result['feature_importance']
                    }
                else:
                    lime_imp = {
                        feature.lower(): abs(score)
                        for feature, score in lime_result.feature_importance
                    }
                lime_runs.append(lime_imp)
                
            # Calculate stability for each method
            for method, runs in [
                ('attention', attention_runs),
                ('lime', lime_runs)
            ]:
                # Get all tokens across runs
                all_tokens = set()
                for run in runs:
                    all_tokens.update(run.keys())
                    
                # Calculate variance in importance scores
                token_variances = []
                for token in all_tokens:
                    scores = [run.get(token, 0) for run in runs]
                    if np.mean(scores) > 0:  # Only consider tokens that appear
                        variance = np.var(scores) / (np.mean(scores) + 1e-8)
                        token_variances.append(variance)
                        
                # Average variance (lower is more stable)
                avg_variance = np.mean(token_variances) if token_variances else 0
                stability_scores[method].append(1 - min(avg_variance, 1))  # Convert to stability
                
        # Average stability scores
        avg_stability = {
            method: float(np.mean(scores))
            for method, scores in stability_scores.items()
        }
        
        return avg_stability
        
    def run_comprehensive_evaluation(self, test_texts: List[str]) -> Dict:
        """
        Run complete evaluation comparing attention and LIME methods.
        
        Args:
            test_texts: List of financial texts to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"Running comprehensive evaluation on {len(test_texts)} texts...")
        
        # 1. Compare on each instance
        instance_comparisons = []
        for i, text in enumerate(test_texts):
            self.logger.info(f"Processing text {i+1}/{len(test_texts)}")
            comparison = self.compare_single_instance(text)
            instance_comparisons.append(comparison)
            
        # 2. Aggregate timing results
        avg_timings = {
            'attention': np.mean([c['timings']['attention'] for c in instance_comparisons]),
            'lime': np.mean([c['timings']['lime'] for c in instance_comparisons])
        }
        
        # 3. Aggregate agreement metrics
        all_correlations = [c['comparisons']['attention_lime_correlation'] 
                           for c in instance_comparisons]
        
        avg_correlation = float(np.mean([v for v in all_correlations if not np.isnan(v)]))
        
        # 4. Evaluate faithfulness
        faithfulness = self.evaluate_faithfulness(test_texts[:10])  # Subset for speed
        
        # 5. Evaluate stability
        stability = self.evaluate_stability(test_texts[:10])  # Subset for speed
        
        # 6. Compile results
        evaluation_results = {
            'num_texts': len(test_texts),
            'average_processing_time': avg_timings,
            'average_correlation': avg_correlation,
            'faithfulness_scores': faithfulness,
            'stability_scores': stability,
            'instance_comparisons': instance_comparisons
        }
        
        return evaluation_results
        
    def generate_comparison_visualizations(self, 
                                         evaluation_results: Dict,
                                         output_dir: str = "data/processed"):
        """Generate enhanced visualizations comparing attention and LIME methods"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Enhanced Processing Time Comparison
        self._create_enhanced_speed_comparison(evaluation_results, output_path)
        
        # 2. Enhanced Faithfulness and Stability Comparison
        self._create_enhanced_performance_metrics(evaluation_results, output_path)
        
        # 3. Enhanced Agreement Visualization
        self._create_enhanced_agreement_plot(evaluation_results, output_path)
        
        # 4. Create detailed attention heatmap for first example
        if evaluation_results['instance_comparisons']:
            self._create_enhanced_attention_heatmap(
                evaluation_results['instance_comparisons'][0],
                output_path
            )
        
        self.logger.info(f"Enhanced visualizations saved to {output_path}")
    
    def _create_enhanced_speed_comparison(self, results: Dict, output_path: Path):
        """Create enhanced speed comparison visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['Attention', 'LIME']
        times = [
            results['average_processing_time']['attention'],
            results['average_processing_time']['lime']
        ]
        
        # Create bars with academic styling
        bars = ax.bar(methods, times, 
                      color=[self.colors['attention'], self.colors['lime']],
                      width=0.6, edgecolor='black', linewidth=1.5)
        
        # Add value labels with better formatting
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{time:.3f}s', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Styling
        ax.set_ylabel('Average Processing Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Computational Efficiency: Attention vs LIME', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(times) * 1.2)
        
        # Add grid for better readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'attention_lime_speed_comparison.png', 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_enhanced_performance_metrics(self, results: Dict, output_path: Path):
        """Create enhanced performance metrics visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = ['Attention', 'LIME']
        
        # Faithfulness scores
        faithfulness = [
            results['faithfulness_scores']['attention'],
            results['faithfulness_scores']['lime']
        ]
        
        bars1 = ax1.bar(methods, faithfulness,
                        color=[self.colors['attention'], self.colors['lime']],
                        width=0.6, edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars1, faithfulness):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.set_ylabel('Faithfulness Score', fontsize=14, fontweight='bold')
        ax1.set_title('Faithfulness Evaluation\n(Higher = More Faithful)', fontsize=16, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Stability scores
        stability = [
            results['stability_scores']['attention'],
            results['stability_scores']['lime']
        ]
        
        bars2 = ax2.bar(methods, stability,
                       color=[self.colors['attention'], self.colors['lime']],
                       width=0.6, edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars2, stability):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax2.set_ylabel('Stability Score', fontsize=14, fontweight='bold')
        ax2.set_title('Stability Evaluation\n(Higher = More Stable)', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.suptitle('XAI Method Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'attention_lime_faithfulness_stability.png',
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_enhanced_agreement_plot(self, results: Dict, output_path: Path):
        """Create enhanced agreement visualization"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create data for multiple agreement metrics
        metrics = ['Pearson\nCorrelation', 'Top-5 Token\nOverlap', 'Overall\nAgreement']
        
        # Calculate average top-5 overlap from instance comparisons
        avg_top5_overlap = np.mean([
            comp['comparisons'].get('top5_overlap', 0) 
            for comp in results['instance_comparisons']
        ])
        
        values = [
            results['average_correlation'],
            avg_top5_overlap,
            (results['average_correlation'] + avg_top5_overlap) / 2  # Combined metric
        ]
        
        # Create horizontal bar chart for better label readability
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, color=['#4A90E2', '#7B68EE', '#50C878'],
                      height=0.6, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=14, fontweight='bold')
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=14)
        ax.set_xlabel('Agreement Score', fontsize=14, fontweight='bold')
        ax.set_title('Method Agreement Analysis: Attention vs LIME', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.1)
        
        # Add reference line
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate Agreement')
        
        # Grid and styling
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / 'attention_lime_agreement.png',
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_enhanced_attention_heatmap(self, comparison: Dict, output_path: Path):
        """Create enhanced attention heatmap with smaller squares"""
        if 'data' not in comparison['attention']:
            return
            
        attention_data = comparison['attention']['data']
        attention_matrix = attention_data.aggregated_attention
        tokens = attention_data.tokens
        
        # Determine figure size based on number of tokens
        n_tokens = len(tokens)
        fig_size = max(12, n_tokens * 0.3)  # Scale with tokens but cap at reasonable size
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Create heatmap with smaller squares
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_tokens))
        ax.set_yticks(np.arange(n_tokens))
        ax.set_xticklabels(tokens, rotation=90, ha='center', fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
        
        # Add grid for better visibility
        ax.set_xticks(np.arange(n_tokens + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_tokens + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Remove tick lines
        ax.tick_params(which='both', length=0)
        
        # Labels and title
        ax.set_xlabel('Target Tokens', fontsize=14, fontweight='bold')
        ax.set_ylabel('Source Tokens', fontsize=14, fontweight='bold')
        ax.set_title('Attention Mechanism Visualization\n' + comparison['text'][:80] + '...', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path / 'enhanced_attention_heatmap.png',
                   bbox_inches='tight', facecolor='white', edgecolor='none', dpi=200)
        plt.close()
        
    def save_evaluation_results(self, 
                              evaluation_results: Dict,
                              output_path: str = "data/processed/xai_comparison.json"):
        """Save comprehensive evaluation results"""
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
            
        serializable_results = convert_to_serializable(evaluation_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Evaluation results saved to {output_path}")
        
    def generate_summary_report(self, evaluation_results: Dict) -> str:
        """Generate human-readable summary of comparison results"""
        report = []
        report.append("ATTENTION vs LIME COMPARISON SUMMARY")
        report.append("=" * 50)
        
        # Speed comparison
        report.append("\n1. PROCESSING SPEED:")
        times = evaluation_results['average_processing_time']
        faster = min(times, key=times.get)
        speed_ratio = times['lime'] / times['attention']
        report.append(f"   Attention: {times['attention']:.3f}s")
        report.append(f"   LIME: {times['lime']:.3f}s")
        report.append(f"   Attention is {speed_ratio:.1f}x faster than LIME")
            
        # Agreement analysis
        report.append("\n2. METHOD AGREEMENT:")
        report.append(f"   Correlation: {evaluation_results['average_correlation']:.3f}")
        
        # Faithfulness
        report.append("\n3. FAITHFULNESS SCORES:")
        faith = evaluation_results['faithfulness_scores']
        report.append(f"   Attention: {faith['attention']:.3f}")
        report.append(f"   LIME: {faith['lime']:.3f}")
        
        # Stability
        report.append("\n4. STABILITY SCORES:")
        stab = evaluation_results['stability_scores']
        report.append(f"   Attention: {stab['attention']:.3f}")
        report.append(f"   LIME: {stab['lime']:.3f}")
        
        # Overall recommendation
        report.append("\n5. KEY FINDINGS:")
        if times['attention'] < times['lime'] and stab['attention'] > stab['lime']:
            report.append("   ✓ Attention provides better speed and stability")
        elif faith['lime'] > faith['attention']:
            report.append("   ✓ LIME shows higher faithfulness despite slower speed")
        else:
            report.append("   ✓ Both methods show complementary strengths")
            
        return "\n".join(report)


def main():
    """Test the XAI comparison framework"""
    # Initialize comparator
    comparator = XAIComparator()
    
    # Test texts
    test_texts = [
        "Barclays shares plunged 15% after the bank reported significant losses.",
        "FTSE 100 reaches all time high as banking stocks rally strongly.",
        "Sterling falls against dollar amid Bank of England policy concerns.",
        "BP announces record profits driven by high oil prices.",
        "Tesco sales beat expectations during crucial Christmas period."
    ]
    
    print("Running Attention vs LIME comparison...")
    
    # Run evaluation
    evaluation_results = comparator.run_comprehensive_evaluation(test_texts)
    
    # Generate visualizations
    comparator.generate_comparison_visualizations(evaluation_results)
    
    # Save results
    comparator.save_evaluation_results(evaluation_results)
    
    # Print summary
    summary = comparator.generate_summary_report(evaluation_results)
    print("\n" + summary)
    
    print("\nComparison complete! Check data/processed/ for results and visualizations.")
    

if __name__ == "__main__":
    main()
    