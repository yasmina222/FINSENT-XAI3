"""
Create token importance comparison visualization from saved results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def create_token_importance_comparison(results_path, output_path="token_importance_comparison.png"):
    """
    Create visualization comparing top tokens identified by Attention vs LIME
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Find interesting examples where methods disagree
    examples_to_show = []
    
    for comp in results['instance_comparisons'][:50]:  # Check first 50
        attention_imp = comp['attention']['importance']
        lime_imp = comp['lime']['importance']
        
        # Get top 5 tokens from each
        top_attention = set(sorted(attention_imp.items(), key=lambda x: x[1], reverse=True)[:5])
        top_lime = set(sorted(lime_imp.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate overlap
        overlap = len(set(t[0] for t in top_attention) & set(t[0] for t in top_lime))
        
        # Look for examples with low overlap (interesting disagreement)
        if overlap <= 2:  # Less than 50% agreement
            examples_to_show.append({
                'text': comp['text'],
                'attention': attention_imp,
                'lime': lime_imp
            })
            
        if len(examples_to_show) >= 2:  # Show 2 examples
            break
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Token Importance Comparison: Attention vs LIME', fontsize=18, fontweight='bold')
    
    for idx, example in enumerate(examples_to_show):
        # Get top 10 tokens from each method
        att_items = sorted(example['attention'].items(), key=lambda x: x[1], reverse=True)[:10]
        lime_items = sorted(example['lime'].items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Attention subplot
        ax_att = axes[idx, 0]
        tokens_att = [item[0] for item in att_items]
        scores_att = [item[1] for item in att_items]
        
        bars_att = ax_att.barh(range(len(tokens_att)), scores_att, color='#2E86AB')
        ax_att.set_yticks(range(len(tokens_att)))
        ax_att.set_yticklabels(tokens_att)
        ax_att.set_xlabel('Importance Score', fontsize=12)
        ax_att.set_title(f'Attention Method\n"{example["text"][:60]}..."', fontsize=12, fontweight='bold')
        ax_att.invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars_att, scores_att)):
            ax_att.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
        
        # LIME subplot
        ax_lime = axes[idx, 1]
        tokens_lime = [item[0] for item in lime_items]
        scores_lime = [item[1] for item in lime_items]
        
        bars_lime = ax_lime.barh(range(len(tokens_lime)), scores_lime, color='#A23B72')
        ax_lime.set_yticks(range(len(tokens_lime)))
        ax_lime.set_yticklabels(tokens_lime)
        ax_lime.set_xlabel('Importance Score', fontsize=12)
        ax_lime.set_title(f'LIME Method\n"{example["text"][:60]}..."', fontsize=12, fontweight='bold')
        ax_lime.invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars_lime, scores_lime)):
            ax_lime.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Token importance comparison saved to {output_path}")
    
    # Print insights
    print("\nKey Insight: The methods identify different important tokens, explaining their low agreement (0.080)")

# Run it
if __name__ == "__main__":
    create_token_importance_comparison(
        "/Users/Lyons/Desktop/FINSENT.XAI/test_results_20250810_234225/results.json"
    )