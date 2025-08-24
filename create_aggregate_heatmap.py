"""
Create aggregate attention heatmap from saved results
This runs AFTER your main analysis - completely safe!
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.ml.attention_extractor import AttentionExtractor

def create_aggregate_heatmap(headlines, output_path="aggregate_attention_heatmap.png"):
    """
    Create an aggregate heatmap showing average attention patterns across all headlines
    """
    print("Creating aggregate attention heatmap...")
    
    # Initialize extractor (read-only, won't affect your system)
    extractor = AttentionExtractor("ProsusAI/finbert")
    
    # We'll aggregate attention patterns by position (first word, second word, etc.)
    # Most headlines are 5-10 words, so we'll use a 10x10 grid
    max_len = 10
    aggregate_matrix = np.zeros((max_len, max_len))
    count_matrix = np.zeros((max_len, max_len))  # Track how many samples contribute to each cell
    
    # Process each headline
    for i, headline in enumerate(headlines):
        if i % 20 == 0:  # Progress update every 20
            print(f"Processing headline {i+1}/{len(headlines)}...")
            
        try:
            # Extract attention
            attention_data = extractor.extract_attention(headline)
            
            # Use middle layer (layer 6) for clearest patterns
            attention = attention_data.layer_weights[6]
            
            # Get actual tokens (excluding [CLS] and [SEP])
            tokens = [t for t in attention_data.tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
            
            # Only use first max_len tokens
            n_tokens = min(len(tokens), max_len)
            
            # Add to aggregate (position-based)
            for i in range(n_tokens):
                for j in range(n_tokens):
                    # Map token positions to our fixed grid
                    # Skip [CLS] token (position 0) by using i+1, j+1
                    aggregate_matrix[i, j] += attention[i+1, j+1]
                    count_matrix[i, j] += 1
                    
        except Exception as e:
            print(f"Skipping headline due to error: {e}")
            continue
    
    # Average the attention scores
    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    average_attention = aggregate_matrix / count_matrix
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Create position labels
    position_labels = ['Word 1', 'Word 2', 'Word 3', 'Word 4', 'Word 5', 
                      'Word 6', 'Word 7', 'Word 8', 'Word 9', 'Word 10']
    
    # Plot heatmap
    sns.heatmap(
        average_attention,
        xticklabels=position_labels,
        yticklabels=position_labels,
        cmap='Blues',
        cbar_kws={'label': 'Average Attention Weight'},
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        annot=True,  # Show values
        fmt='.3f',   # 3 decimal places
        annot_kws={'size': 8}
    )
    
    plt.title('Aggregate Attention Patterns Across 210 Financial Headlines\n(Position-Based Average)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Target Position in Sentence', fontsize=14, fontweight='bold')
    plt.ylabel('Source Position in Sentence', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregate heatmap saved to {output_path}")
    
    # Return statistics for dissertation
    return {
        'total_headlines_processed': len(headlines),
        'average_attention_mean': float(np.mean(average_attention)),
        'average_attention_std': float(np.std(average_attention)),
        'strongest_position_connection': np.unravel_index(np.argmax(average_attention), average_attention.shape)
    }

# Main execution
if __name__ == "__main__":
    # Load your RESULTS file to get the headlines
    with open('/Users/Lyons/Desktop/FINSENT.XAI/test_results_20250810_234225/results.json', 'r') as f:
        results = json.load(f)
    
    # Extract headlines from the results
    headlines = [comp['text'] for comp in results['instance_comparisons']]
    print(f"Loaded {len(headlines)} headlines from results")
    
    # Create aggregate heatmap
    stats = create_aggregate_heatmap(headlines)
    
    print("\nStatistics for dissertation:")
    print(f"- Headlines analyzed: {stats['total_headlines_processed']}")
    print(f"- Mean attention weight: {stats['average_attention_mean']:.4f}")
    print(f"- Std deviation: {stats['average_attention_std']:.4f}")
    strong_pos = stats['strongest_position_connection']
    print(f"- Strongest connection: Word {strong_pos[0]+1} → Word {strong_pos[1]+1}")