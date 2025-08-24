"""
Evaluation Metrics Module for FINSENT-XAI
Implements comprehensive metrics for sentiment analysis and explainability evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    cohen_kappa_score, classification_report
)
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns


class FinancialMetrics:
    """
    Comprehensive metrics for evaluating financial sentiment analysis
    and explainability methods.
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.labels = ['negative', 'neutral', 'positive']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
    def calculate_classification_metrics(self,
                                       y_true: List[int],
                                       y_pred: List[int],
                                       sample_weights: Optional[List[float]] = None) -> Dict:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weights: Optional weights for samples
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, sample_weight=sample_weights
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', sample_weight=sample_weights
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', sample_weight=sample_weights
        )
        
        # Cohen's Kappa (agreement measure)
        kappa = cohen_kappa_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i, label in enumerate(self.labels):
            class_correct = cm[i, i]
            class_total = cm[i].sum()
            per_class_accuracy[label] = class_correct / class_total if class_total > 0 else 0
            
        metrics = {
            'accuracy': float(accuracy),
            'precision': {
                'per_class': {label: float(p) for label, p in zip(self.labels, precision)},
                'weighted': float(precision_weighted),
                'macro': float(precision_macro)
            },
            'recall': {
                'per_class': {label: float(r) for label, r in zip(self.labels, recall)},
                'weighted': float(recall_weighted),
                'macro': float(recall_macro)
            },
            'f1_score': {
                'per_class': {label: float(f) for label, f in zip(self.labels, f1)},
                'weighted': float(f1_weighted),
                'macro': float(f1_macro)
            },
            'support': {label: int(s) for label, s in zip(self.labels, support)},
            'per_class_accuracy': per_class_accuracy,
            'cohen_kappa': float(kappa),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
        
    def calculate_company_specific_metrics(self,
                                         texts: List[str],
                                         y_true: List[int],
                                         y_pred: List[int],
                                         companies: List[str]) -> Dict:
        """
        Calculate metrics for specific companies.
        
        Args:
            texts: List of input texts
            y_true: True labels
            y_pred: Predicted labels
            companies: List of company names to analyze
            
        Returns:
            Company-specific metrics
        """
        company_metrics = {}
        
        for company in companies:
            # Find texts mentioning this company
            company_indices = [
                i for i, text in enumerate(texts) 
                if company.lower() in text.lower()
            ]
            
            if company_indices:
                company_true = [y_true[i] for i in company_indices]
                company_pred = [y_pred[i] for i in company_indices]
                
                # Calculate metrics
                accuracy = accuracy_score(company_true, company_pred)
                
                # Distribution of sentiments
                true_dist = Counter(company_true)
                pred_dist = Counter(company_pred)
                
                company_metrics[company] = {
                    'num_articles': len(company_indices),
                    'accuracy': float(accuracy),
                    'true_sentiment_distribution': {
                        self.labels[k]: v for k, v in true_dist.items()
                    },
                    'predicted_sentiment_distribution': {
                        self.labels[k]: v for k, v in pred_dist.items()
                    }
                }
            else:
                company_metrics[company] = {
                    'num_articles': 0,
                    'accuracy': None,
                    'true_sentiment_distribution': {},
                    'predicted_sentiment_distribution': {}
                }
                
        return company_metrics
        
    def calculate_xai_faithfulness(self,
                                 original_predictions: List[float],
                                 perturbed_predictions: List[float],
                                 importance_scores: List[float]) -> Dict:
        """
        Calculate faithfulness metrics for XAI methods.
        
        Args:
            original_predictions: Original model predictions
            perturbed_predictions: Predictions after removing important features
            importance_scores: Feature importance from XAI method
            
        Returns:
            Faithfulness metrics
        """
        # Prediction difference
        pred_diff = np.abs(np.array(original_predictions) - np.array(perturbed_predictions))
        
        # Correlation between importance and prediction change
        if len(importance_scores) > 1:
            correlation = np.corrcoef(importance_scores, pred_diff)[0, 1]
        else:
            correlation = 0.0
            
        # Average prediction change
        avg_change = np.mean(pred_diff)
        
        # Faithfulness score (higher change = more faithful)
        faithfulness_score = avg_change
        
        return {
            'faithfulness_score': float(faithfulness_score),
            'avg_prediction_change': float(avg_change),
            'importance_change_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'num_samples': len(original_predictions)
        }
        
    def calculate_xai_stability(self,
                              explanations_multiple_runs: List[Dict[str, float]]) -> Dict:
        """
        Calculate stability metrics for XAI explanations.
        
        Args:
            explanations_multiple_runs: List of explanation dicts from multiple runs
            
        Returns:
            Stability metrics
        """
        if len(explanations_multiple_runs) < 2:
            return {
                'stability_score': 1.0,
                'avg_variance': 0.0,
                'consistency_ratio': 1.0
            }
            
        # Get all features across runs
        all_features = set()
        for exp in explanations_multiple_runs:
            all_features.update(exp.keys())
            
        # Calculate variance for each feature
        feature_variances = []
        consistent_features = 0
        
        for feature in all_features:
            scores = [exp.get(feature, 0) for exp in explanations_multiple_runs]
            
            if np.mean(scores) > 0:  # Feature appears in some runs
                variance = np.var(scores)
                feature_variances.append(variance)
                
                # Check if feature appears consistently
                appearances = sum(1 for s in scores if s > 0)
                if appearances == len(explanations_multiple_runs):
                    consistent_features += 1
                    
        # Average variance (normalized)
        avg_variance = np.mean(feature_variances) if feature_variances else 0
        
        # Consistency ratio
        consistency_ratio = consistent_features / len(all_features) if all_features else 0
        
        # Stability score (inverse of variance)
        stability_score = 1 / (1 + avg_variance)
        
        return {
            'stability_score': float(stability_score),
            'avg_variance': float(avg_variance),
            'consistency_ratio': float(consistency_ratio),
            'num_features': len(all_features),
            'consistent_features': consistent_features
        }
        
    def calculate_xai_agreement(self,
                              method1_scores: Dict[str, float],
                              method2_scores: Dict[str, float]) -> Dict:
        """
        Calculate agreement between two XAI methods.
        
        Args:
            method1_scores: Feature importance from method 1
            method2_scores: Feature importance from method 2
            
        Returns:
            Agreement metrics
        """
        # Get common features
        common_features = set(method1_scores.keys()) & set(method2_scores.keys())
        
        if not common_features:
            return {
                'correlation': 0.0,
                'rank_correlation': 0.0,
                'top_k_overlap': {},
                'num_common_features': 0
            }
            
        # Extract scores for common features
        scores1 = [method1_scores[f] for f in common_features]
        scores2 = [method2_scores[f] for f in common_features]
        
        # Pearson correlation
        if len(scores1) > 1:
            correlation = np.corrcoef(scores1, scores2)[0, 1]
        else:
            correlation = 0.0
            
        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(scores1, scores2)
        
        # Top-k overlap
        top_k_overlap = {}
        for k in [3, 5, 10]:
            top_k_1 = set(sorted(method1_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:k])
            top_k_2 = set(sorted(method2_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:k])
            
            overlap = len(set(f[0] for f in top_k_1) & set(f[0] for f in top_k_2)) / k
            top_k_overlap[f'top_{k}'] = float(overlap)
            
        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'rank_correlation': float(rank_corr) if not np.isnan(rank_corr) else 0.0,
            'top_k_overlap': top_k_overlap,
            'num_common_features': len(common_features)
        }
        
    def create_confusion_matrix_plot(self,
                                   y_true: List[int],
                                   y_pred: List[int],
                                   save_path: str = "confusion_matrix.png") -> None:
        """
        Create and save confusion matrix visualization.
        
        ⚠️ VISUAL OUTPUT: This creates a confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Sentiment Classification Confusion Matrix', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentage annotations
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(len(self.labels)):
            for j in range(len(self.labels)):
                percentage = cm_normalized[i, j] * 100
                text = plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%',
                              ha='center', va='center', fontsize=9, color='gray')
                              
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 VISUAL OUTPUT: Confusion matrix saved to {save_path}")
        
    def create_metrics_comparison_chart(self,
                                      metrics_dict: Dict[str, Dict],
                                      save_path: str = "metrics_comparison.png") -> None:
        """
        Create comparison chart of metrics across methods.
        
        ⚠️ VISUAL OUTPUT: This creates a bar chart comparing precision/recall/F1
        
        Args:
            metrics_dict: Dictionary with method names and their metrics
            save_path: Path to save the plot
        """
        methods = list(metrics_dict.keys())
        metrics = ['Precision', 'Recall', 'F1-Score']
        
        # Extract values
        values = []
        for method in methods:
            method_values = [
                metrics_dict[method]['precision']['weighted'],
                metrics_dict[method]['recall']['weighted'],
                metrics_dict[method]['f1_score']['weighted']
            ]
            values.append(method_values)
            
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (method, vals) in enumerate(zip(methods, values)):
            offset = (i - len(methods)/2) * width + width/2
            bars = ax.bar(x + offset, vals, width, label=method)
            
            # Add value labels
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                       
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 VISUAL OUTPUT: Metrics comparison chart saved to {save_path}")
        
    def save_metrics_report(self,
                          all_metrics: Dict,
                          save_path: str = "metrics_report.json") -> None:
        """
        Save comprehensive metrics report.
        
        Args:
            all_metrics: All calculated metrics
            save_path: Path to save JSON report
        """
        with open(save_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
        print(f"📄 Metrics report saved to {save_path}")
        
    def generate_summary_statistics(self, metrics: Dict) -> str:
        """
        Generate human-readable summary of metrics.
        
        Args:
            metrics: Calculated metrics dictionary
            
        Returns:
            Summary string
        """
        summary = []
        summary.append("PERFORMANCE METRICS SUMMARY")
        summary.append("=" * 50)
        
        # Overall accuracy
        summary.append(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
        summary.append(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
        
        # Per-class performance
        summary.append("\nPer-Class Performance:")
        for label in self.labels:
            summary.append(f"\n{label.upper()}:")
            summary.append(f"  Precision: {metrics['precision']['per_class'][label]:.2%}")
            summary.append(f"  Recall: {metrics['recall']['per_class'][label]:.2%}")
            summary.append(f"  F1-Score: {metrics['f1_score']['per_class'][label]:.3f}")
            summary.append(f"  Accuracy: {metrics['per_class_accuracy'][label]:.2%}")
            
        # Weighted averages
        summary.append("\nWeighted Averages:")
        summary.append(f"  Precision: {metrics['precision']['weighted']:.2%}")
        summary.append(f"  Recall: {metrics['recall']['weighted']:.2%}")
        summary.append(f"  F1-Score: {metrics['f1_score']['weighted']:.3f}")
        
        return "\n".join(summary)


def main():
    """Test metrics calculation"""
    metrics_calc = FinancialMetrics()
    
    # Simulated predictions for testing
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic true labels (slightly imbalanced)
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Generate predictions with 85% accuracy
    y_pred = y_true.copy()
    n_errors = int(0.15 * n_samples)
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    for idx in error_indices:
        y_pred[idx] = np.random.choice([i for i in range(3) if i != y_true[idx]])
        
    # Calculate metrics
    metrics = metrics_calc.calculate_classification_metrics(y_true, y_pred)
    
    # Print summary
    print(metrics_calc.generate_summary_statistics(metrics))
    
    # Create visualizations
    metrics_calc.create_confusion_matrix_plot(y_true, y_pred)
    
    print("\nMetrics testing complete!")
    

if __name__ == "__main__":
    main()