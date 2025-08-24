"""
Visualization Module for FINSENT-XAI
Generates interactive and static visualizations for explainability analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import base64
from io import BytesIO


class XAIVisualizer:
    """
    Creates visualizations for attention mechanisms and XAI method comparisons.
    Focuses on interpretability for financial domain experts.
    """
    
    def __init__(self, style: str = 'professional'):
        """
        Initialize visualizer with consistent styling.
        
        Args:
            style: Visual style preset ('professional', 'academic', 'presentation')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Configure matplotlib and seaborn styling"""
        if self.style == 'professional':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            self.cmap_attention = 'Blues'
            self.figsize_default = (10, 6)
        elif self.style == 'academic':
            plt.style.use('seaborn-v0_8-paper')
            self.color_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']
            self.cmap_attention = 'viridis'
            self.figsize_default = (8, 6)
        else:  # presentation
            plt.style.use('seaborn-v0_8-darkgrid')
            self.color_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            self.cmap_attention = 'plasma'
            self.figsize_default = (12, 8)
            
        # Set general parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
    def create_attention_heatmap(self,
                               attention_weights: np.ndarray,
                               tokens: List[str],
                               title: str = "Attention Weights Visualization",
                               save_path: Optional[str] = None,
                               interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create attention weight heatmap.
        
        Args:
            attention_weights: 2D array of attention weights
            tokens: List of tokens
            title: Plot title
            save_path: Path to save figure
            interactive: If True, create interactive Plotly figure
            
        Returns:
            Matplotlib or Plotly figure object
        """
        if interactive:
            return self._create_interactive_attention_heatmap(
                attention_weights, tokens, title, save_path
            )
            
        # Static matplotlib version
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=self.cmap_attention,
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Target Tokens', fontsize=12)
        ax.set_ylabel('Source Tokens', fontsize=12)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
        
    def _create_interactive_attention_heatmap(self,
                                            attention_weights: np.ndarray,
                                            tokens: List[str],
                                            title: str,
                                            save_path: Optional[str] = None) -> go.Figure:
        """Create interactive Plotly heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='Source: %{y}<br>Target: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Target Tokens",
            yaxis_title="Source Tokens",
            width=800,
            height=800,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def create_token_importance_comparison(self,
                                         attention_scores: Dict[str, float],
                                         lime_scores: Dict[str, float],
                                         shap_scores: Dict[str, float],
                                         top_k: int = 10,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison of token importance across methods.
        
        Args:
            attention_scores: Token importance from attention
            lime_scores: Token importance from LIME
            shap_scores: Token importance from SHAP
            top_k: Number of top tokens to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get union of top-k tokens from each method
        top_attention = set(sorted(attention_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:top_k])
        top_lime = set(sorted(lime_scores.items(), 
                            key=lambda x: x[1], reverse=True)[:top_k])
        top_shap = set(sorted(shap_scores.items(), 
                            key=lambda x: x[1], reverse=True)[:top_k])
        
        # Get unique tokens
        all_top_tokens = set()
        for (token, _) in (top_attention | top_lime | top_shap):
            all_top_tokens.add(token)
            
        tokens = sorted(list(all_top_tokens))
        
        # Prepare data
        attention_vals = [attention_scores.get(t, 0) for t in tokens]
        lime_vals = [lime_scores.get(t, 0) for t in tokens]
        shap_vals = [shap_scores.get(t, 0) for t in tokens]
        
        # Create grouped bar chart
        x = np.arange(len(tokens))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, attention_vals, width, label='Attention', 
                       color=self.color_palette[0])
        bars2 = ax.bar(x, lime_vals, width, label='LIME', 
                       color=self.color_palette[1])
        bars3 = ax.bar(x + width, shap_vals, width, label='SHAP', 
                       color=self.color_palette[2])
        
        ax.set_xlabel('Tokens', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title('Token Importance Comparison Across XAI Methods', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
        
    def create_method_agreement_matrix(self,
                                     agreement_scores: Dict[str, float],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix showing agreement between methods.
        
        Args:
            agreement_scores: Dictionary with correlation values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract correlation values
        methods = ['Attention', 'LIME', 'SHAP']
        
        # Build correlation matrix
        corr_matrix = np.array([
            [1.0, 
             agreement_scores.get('attention_lime_correlation', 0),
             agreement_scores.get('attention_shap_correlation', 0)],
            [agreement_scores.get('attention_lime_correlation', 0),
             1.0,
             agreement_scores.get('lime_shap_correlation', 0)],
            [agreement_scores.get('attention_shap_correlation', 0),
             agreement_scores.get('lime_shap_correlation', 0),
             1.0]
        ])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=methods,
            yticklabels=methods,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title('XAI Method Agreement Matrix', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
        
    def create_performance_radar_chart(self,
                                     performance_metrics: Dict[str, Dict[str, float]],
                                     save_path: Optional[str] = None) -> go.Figure:
        """
        Create radar chart comparing methods across multiple metrics.
        
        Args:
            performance_metrics: Nested dict with methods and their metrics
            save_path: Path to save figure
            
        Returns:
            Plotly figure
        """
        # Prepare data
        categories = ['Speed', 'Faithfulness', 'Stability', 'Agreement', 'Interpretability']
        
        fig = go.Figure()
        
        for i, (method, metrics) in enumerate(performance_metrics.items()):
            values = [
                1 - metrics.get('avg_time', 1),  # Invert time (faster = higher)
                metrics.get('faithfulness', 0),
                metrics.get('stability', 0),
                metrics.get('avg_agreement', 0),
                metrics.get('interpretability', 0.8)  # Subjective baseline
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=method.capitalize(),
                line_color=self.color_palette[i % len(self.color_palette)]
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="XAI Method Performance Comparison",
            width=700,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def create_explanation_summary_plot(self,
                                      text: str,
                                      tokens: List[str],
                                      explanations: Dict[str, np.ndarray],
                                      prediction: str,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary plot showing all explanations.
        
        Args:
            text: Original text
            tokens: List of tokens
            explanations: Dict with method names and importance arrays
            prediction: Sentiment prediction
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[1, 1])
        
        # Text and prediction
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.text(0.5, 0.7, f"Text: {text}", ha='center', va='center', 
                    fontsize=12, wrap=True)
        ax_text.text(0.5, 0.3, f"Prediction: {prediction.upper()}", 
                    ha='center', va='center', fontsize=14, 
                    fontweight='bold', color=self._get_sentiment_color(prediction))
        ax_text.axis('off')
        
        # Individual method visualizations
        method_positions = {
            'attention': (1, 0),
            'lime': (1, 1),
            'shap': (2, 0)
        }
        
        for method, (row, col) in method_positions.items():
            if method in explanations:
                ax = fig.add_subplot(gs[row, col])
                
                importance = explanations[method]
                
                # Create bar plot
                y_pos = np.arange(len(tokens))
                colors = [self._get_importance_color(val) for val in importance]
                
                bars = ax.barh(y_pos, importance, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(tokens, fontsize=8)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'{method.upper()} Explanation', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, importance):
                    width = bar.get_width()
                    if width != 0:
                        ax.text(width + 0.01 if width > 0 else width - 0.01,
                               bar.get_y() + bar.get_height()/2,
                               f'{val:.3f}', ha='left' if width > 0 else 'right',
                               va='center', fontsize=8)
                               
        # Consensus view
        ax_consensus = fig.add_subplot(gs[2, 1])
        
        # Calculate average importance
        all_importance = np.zeros(len(tokens))
        count = 0
        for method_imp in explanations.values():
            all_importance += np.abs(method_imp)
            count += 1
            
        if count > 0:
            avg_importance = all_importance / count
            
            # Create consensus bar plot
            y_pos = np.arange(len(tokens))
            sorted_idx = np.argsort(avg_importance)[::-1]
            
            bars = ax_consensus.barh(y_pos, avg_importance[sorted_idx], 
                                    color=self.color_palette[3], alpha=0.7)
            ax_consensus.set_yticks(y_pos)
            ax_consensus.set_yticklabels([tokens[i] for i in sorted_idx], fontsize=8)
            ax_consensus.set_xlabel('Average Importance Score')
            ax_consensus.set_title('Consensus Importance', fontsize=12)
            ax_consensus.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
        
    def _get_sentiment_color(self, sentiment: str) -> str:
        """Get color for sentiment label"""
        colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
        return colors.get(sentiment.lower(), '#34495e')
        
    def _get_importance_color(self, value: float) -> str:
        """Get color based on importance value"""
        if value > 0:
            return '#2ecc71'  # Green for positive
        elif value < 0:
            return '#e74c3c'  # Red for negative
        else:
            return '#95a5a6'  # Gray for zero
            
    def save_all_visualizations(self,
                              results_path: str,
                              output_dir: str = "visualizations"):
        """
        Generate all visualizations from saved results.
        
        Args:
            results_path: Path to JSON results file
            output_dir: Directory to save visualizations
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        # Generate various plots
        if 'instance_comparisons' in results:
            for i, comparison in enumerate(results['instance_comparisons'][:5]):
                # Token importance comparison
                self.create_token_importance_comparison(
                    comparison['attention']['importance'],
                    comparison['lime']['importance'],
                    comparison['shap']['importance'],
                    save_path=output_path / f'token_importance_{i}.png'
                )
                
        # Method agreement matrix
        if 'average_correlations' in results:
            self.create_method_agreement_matrix(
                results['average_correlations'],
                save_path=output_path / 'method_agreement.png'
            )
            
        # Performance radar chart
        if all(k in results for k in ['average_processing_time', 
                                      'faithfulness_scores', 
                                      'stability_scores']):
            performance_data = {}
            
            for method in ['attention', 'lime', 'shap']:
                performance_data[method] = {
                    'avg_time': results['average_processing_time'][method],
                    'faithfulness': results['faithfulness_scores'][method],
                    'stability': results['stability_scores'][method],
                    'avg_agreement': 0.7  # Placeholder
                }
                
            self.create_performance_radar_chart(
                performance_data,
                save_path=output_path / 'performance_radar.html'
            )


def main():
    """Test visualization creation"""
    visualizer = XAIVisualizer(style='professional')
    
    # Test data
    tokens = ['Barclays', 'profits', 'plunged', '15%', 'amid', 'rising', 'costs']
    attention_weights = np.random.rand(len(tokens), len(tokens))
    
    # Make it symmetric and normalized
    attention_weights = (attention_weights + attention_weights.T) / 2
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    # Create attention heatmap
    fig = visualizer.create_attention_heatmap(
        attention_weights,
        tokens,
        title="Attention Weights for Financial Sentiment",
        save_path="test_attention.png"
    )
    
    print("Visualization tests complete! Check generated images.")
    

if __name__ == "__main__":
    main()