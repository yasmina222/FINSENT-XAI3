"""
Attention Weight Extraction Module for FINSENT-XAI
Implements attention mechanism analysis for financial sentiment explainability
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class AttentionData:
    """Container for attention analysis results"""
    tokens: List[str]
    attention_weights: np.ndarray  # Shape: (num_layers, num_heads, seq_len, seq_len)
    layer_weights: np.ndarray      # Shape: (num_layers, seq_len, seq_len)
    head_weights: Dict[int, np.ndarray]  # Layer -> (num_heads, seq_len, seq_len)
    aggregated_attention: np.ndarray  # Shape: (seq_len, seq_len)
    

class AttentionExtractor:
    """
    Extracts and analyzes attention patterns from FinBERT for interpretable
    sentiment analysis of financial text.
    """
    
    def __init__(self, model_path: str = "data/models/finbert-uk-final"):
        """
        Initialize attention extractor with trained model.
        
        Args:
            model_path: Path to fine-tuned FinBERT model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
        # Load model and tokenizer
        self.logger.info(f"Loading model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Model configuration
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        
        # Attention hooks storage
        self.attention_weights = {}
        self._register_hooks()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for attention extraction"""
        logger = logging.getLogger('AttentionExtractor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # BERT attention output format: (attention_weights, attention_output)
                if isinstance(output, tuple) and len(output) > 1:
                    attention_probs = output[1]  # Shape: (batch, num_heads, seq_len, seq_len)
                    self.attention_weights[f'layer_{layer_idx}'] = attention_probs.detach().cpu()
            return hook
            
        # Register hooks on all transformer layers
        for idx, layer in enumerate(self.model.encoder.layer):
            layer.attention.self.register_forward_hook(get_attention_hook(idx))
            
    def extract_attention(self, text: str) -> AttentionData:
        """
        Extract attention weights for input text.
        
        Args:
            text: Financial news text to analyze
            
        Returns:
            AttentionData object containing tokens and attention patterns
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model outputs with attention
        self.attention_weights.clear()
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Extract tokens (excluding special tokens for visualization)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Process attention weights
        attention_data = self._process_attention_weights(tokens)
        
        return attention_data
        
    def _process_attention_weights(self, tokens: List[str]) -> AttentionData:
        """Process raw attention weights into analyzable format"""
        num_tokens = len(tokens)
        
        # Initialize storage arrays
        all_layer_weights = np.zeros((self.num_layers, self.num_heads, num_tokens, num_tokens))
        layer_aggregated = np.zeros((self.num_layers, num_tokens, num_tokens))
        head_weights_dict = {}
        
        # Process each layer's attention
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in self.attention_weights:
                # Get attention for this layer (batch_size=1)
                layer_attention = self.attention_weights[layer_key][0].numpy()
                
                # Store full attention
                all_layer_weights[layer_idx] = layer_attention
                
                # Average across heads for layer-level view
                layer_aggregated[layer_idx] = layer_attention.mean(axis=0)
                
                # Store per-head weights
                head_weights_dict[layer_idx] = layer_attention
                
        # Compute final aggregated attention using attention rollout
        aggregated = self._compute_attention_rollout(all_layer_weights)
        
        return AttentionData(
            tokens=tokens,
            attention_weights=all_layer_weights,
            layer_weights=layer_aggregated,
            head_weights=head_weights_dict,
            aggregated_attention=aggregated
        )
        
    def _compute_attention_rollout(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Compute attention rollout for better interpretability.
        Following Abnar & Zuidema (2020) quantifying attention flow.
        """
        # Average attention weights across all heads
        attention_averaged = attention_weights.mean(axis=1)  # (layers, seq_len, seq_len)
        
        # Initialize with first layer
        rollout = attention_averaged[0]
        
        # Propagate through layers
        for layer_idx in range(1, self.num_layers):
            # Add residual connection
            rollout = rollout + np.eye(rollout.shape[0])
            rollout = rollout / rollout.sum(axis=-1, keepdims=True)
            
            # Multiply with next layer's attention
            rollout = np.matmul(attention_averaged[layer_idx], rollout)
            
        return rollout
        
    def get_token_importance(self, attention_data: AttentionData, 
                           target_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate importance scores for each token.
        
        Args:
            attention_data: Extracted attention data
            target_idx: Target token index (e.g., [CLS] token for classification)
            
        Returns:
            Dictionary mapping tokens to importance scores
        """
        if target_idx is None:
            target_idx = 0  # Use [CLS] token by default
            
        # Get attention from all tokens to target token
        attention_to_target = attention_data.aggregated_attention[:, target_idx]
        
        # Create token importance mapping
        importance_scores = {}
        for idx, (token, score) in enumerate(zip(attention_data.tokens, attention_to_target)):
            # Skip special tokens and punctuation
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                # Handle subword tokens
                clean_token = token.replace('##', '')
                if clean_token in importance_scores:
                    importance_scores[clean_token] += score
                else:
                    importance_scores[clean_token] = score
                    
        # Normalize scores
        total_score = sum(importance_scores.values())
        if total_score > 0:
            importance_scores = {k: v/total_score for k, v in importance_scores.items()}
            
        return importance_scores
        
    def analyze_attention_patterns(self, attention_data: AttentionData) -> Dict:
        """
        Analyze attention patterns for financial insights.
        
        Returns:
            Dictionary containing various attention analytics
        """
        analytics = {
            'head_specialization': self._analyze_head_specialization(attention_data),
            'layer_behavior': self._analyze_layer_behavior(attention_data),
            'token_interactions': self._analyze_token_interactions(attention_data),
            'attention_entropy': self._calculate_attention_entropy(attention_data)
        }
        
        return analytics
        
    def _analyze_head_specialization(self, attention_data: AttentionData) -> Dict:
        """Analyze what different attention heads focus on"""
        specialization = {}
        
        for layer_idx in range(self.num_layers):
            layer_special = []
            
            for head_idx in range(self.num_heads):
                head_attention = attention_data.attention_weights[layer_idx, head_idx]
                
                # Calculate metrics for head behavior
                # Diagonal attention (self-attention strength)
                diagonal_score = np.mean(np.diagonal(head_attention))
                
                # Adjacent attention (local context)
                adjacent_score = np.mean([head_attention[i, i+1] for i in range(len(head_attention)-1)])
                
                # Global attention (long-range dependencies)
                global_score = np.mean(head_attention) - diagonal_score
                
                layer_special.append({
                    'head_idx': head_idx,
                    'diagonal_attention': float(diagonal_score),
                    'adjacent_attention': float(adjacent_score),
                    'global_attention': float(global_score)
                })
                
            specialization[f'layer_{layer_idx}'] = layer_special
            
        return specialization
        
    def _analyze_layer_behavior(self, attention_data: AttentionData) -> List[Dict]:
        """Analyze how attention patterns evolve across layers"""
        layer_analysis = []
        
        for layer_idx in range(self.num_layers):
            layer_attention = attention_data.layer_weights[layer_idx]
            
            # Calculate attention statistics
            attention_mean = np.mean(layer_attention)
            attention_std = np.std(layer_attention)
            attention_max = np.max(layer_attention)
            
            # Measure attention diffusion
            entropy = -np.sum(layer_attention * np.log(layer_attention + 1e-9))
            
            layer_analysis.append({
                'layer': layer_idx,
                'mean_attention': float(attention_mean),
                'std_attention': float(attention_std),
                'max_attention': float(attention_max),
                'entropy': float(entropy)
            })
            
        return layer_analysis
        
    def _analyze_token_interactions(self, attention_data: AttentionData) -> Dict:
        """Analyze interactions between financial entities and sentiment words"""
        interactions = {
            'entity_sentiment_pairs': [],
            'strong_connections': []
        }
        
        # Identify financial entities and sentiment words in tokens
        entities = []
        sentiment_words = []
        
        positive_indicators = {'gain', 'rise', 'surge', 'profit', 'growth', 'strong'}
        negative_indicators = {'loss', 'fall', 'drop', 'decline', 'weak', 'risk'}
        
        for idx, token in enumerate(attention_data.tokens):
            token_lower = token.lower()
            
            # Simple entity detection (could be enhanced)
            if token[0].isupper() and token not in ['[CLS]', '[SEP]']:
                entities.append((idx, token))
                
            # Sentiment word detection
            if token_lower in positive_indicators:
                sentiment_words.append((idx, token, 'positive'))
            elif token_lower in negative_indicators:
                sentiment_words.append((idx, token, 'negative'))
                
        # Analyze connections between entities and sentiment words
        aggregated_attention = attention_data.aggregated_attention
        
        for entity_idx, entity_token in entities:
            for sent_idx, sent_token, sentiment in sentiment_words:
                # Bidirectional attention strength
                attention_strength = (
                    aggregated_attention[entity_idx, sent_idx] + 
                    aggregated_attention[sent_idx, entity_idx]
                ) / 2
                
                if attention_strength > 0.1:  # Threshold for significant connection
                    interactions['entity_sentiment_pairs'].append({
                        'entity': entity_token,
                        'sentiment_word': sent_token,
                        'sentiment': sentiment,
                        'strength': float(attention_strength)
                    })
                    
        return interactions
        
    def _calculate_attention_entropy(self, attention_data: AttentionData) -> List[float]:
        """Calculate attention entropy for each layer (measure of focus vs. diffusion)"""
        entropies = []
        
        for layer_idx in range(self.num_layers):
            layer_attention = attention_data.layer_weights[layer_idx]
            
            # Flatten and normalize
            flat_attention = layer_attention.flatten()
            flat_attention = flat_attention / flat_attention.sum()
            
            # Calculate entropy
            entropy = -np.sum(flat_attention * np.log(flat_attention + 1e-9))
            entropies.append(float(entropy))
            
        return entropies

    def generate_attention_visualization(self, attention_data: AttentionData, 
                                       save_path: str = "attention_heatmap.png"):
        """Generate attention heatmap visualization"""
        # Use aggregated attention for visualization
        attention_matrix = attention_data.aggregated_attention
        
        # Create figure with smaller cells
        plt.figure(figsize=(20, 16))  # Bigger figure for more detail
        
        # Plot heatmap with more detail
        ax = sns.heatmap(
            attention_matrix,
            xticklabels=attention_data.tokens,
            yticklabels=attention_data.tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.5,  # Add gridlines between cells
            linecolor='lightgray',
            cbar=True,
            vmin=0,
            vmax=attention_matrix.max()
        )
        
        # Make it more detailed
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.title('Detailed Attention Weights Matrix', fontsize=16)
        plt.xlabel('Target Tokens', fontsize=12)
        plt.ylabel('Source Tokens', fontsize=12)
        
        # Rotate labels for readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save with high DPI for detail
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Detailed attention visualization saved to {save_path}")
    
    def save_attention_data(self, attention_data: AttentionData, 
                          analytics: Dict, output_path: str):
        """Save attention analysis results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'tokens': attention_data.tokens,
            'token_importance': self.get_token_importance(attention_data),
            'analytics': analytics,
            'aggregated_attention_shape': attention_data.aggregated_attention.shape,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        self.logger.info(f"Attention analysis saved to {output_path}")


def main():
    """Test attention extraction on sample text"""
    # Initialize extractor
    extractor = AttentionExtractor()
    
    # Sample financial text
    sample_text = "Barclays shares plunged 15% after the bank reported significant losses."
    
    print("Extracting attention weights...")
    attention_data = extractor.extract_attention(sample_text)
    
    # Get token importance
    importance = extractor.get_token_importance(attention_data)
    print("\nToken Importance Scores:")
    for token, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {token}: {score:.4f}")
        
    # Analyze patterns
    analytics = extractor.analyze_attention_patterns(attention_data)
    
    # Generate visualization
    extractor.generate_attention_visualization(attention_data)
    
    # Save results
    extractor.save_attention_data(
        attention_data, 
        analytics, 
        "data/processed/attention_analysis.json"
    )
    
    print("\nAttention analysis complete!")
    

if __name__ == "__main__":
    main()