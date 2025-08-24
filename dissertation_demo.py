"""
FINSENT-XAI Research Demonstration
Streamlit app for dissertation viva - Comparing Attention vs LIME explanations
Author: [Your Name]
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from src.ml.attention_extractor import AttentionExtractor
from src.ml.lime_explainer import FinancialLimeExplainer

# Page configuration
st.set_page_config(
    page_title="FINSENT-XAI: Attention vs LIME",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for academic presentation
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stSelectbox > div > div {
        font-size: 16px;
    }
    .headline-display {
        font-size: 24px;
        font-weight: 500;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        border: 2px solid #e0e0e0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0;
    }
    .processing-time {
        font-size: 24px;
        font-weight: bold;
    }
    .method-title {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    h1 {
        text-align: center;
        color: #1f2937;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Pre-selected headlines
HEADLINES = [
    "BT Group fibre rollout ahead of schedule saves millions",
    "Aviva insurance profits beat expectations by 20%",
    "Sainsbury's market share gains accelerate amid rival struggles",
    "National Grid green energy investments yield strong returns",
    "British retail footfall matches seasonal average",
    "UK online sales growth continues steady trend",
    "London hedge fund assets stable quarter-on-quarter",
    "British private equity activity at expected levels",
    "Vodafone UK revenues decline sharply amid fierce competition",
    "Shell cuts dividend for first time since 1945",
    "Rolls-Royce announces 3,000 job cuts globally",
    "Tesco hit by £900m accounting scandal fallout"
]

# Cache model loading
@st.cache_resource
def load_models():
    """Load models once at startup"""
    with st.spinner("Loading AI models..."):
        attention_extractor = AttentionExtractor("ProsusAI/finbert")
        lime_explainer = FinancialLimeExplainer("ProsusAI/finbert")
    return attention_extractor, lime_explainer

def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment"""
    colors = {
        'positive': '#10b981',
        'negative': '#ef4444', 
        'neutral': '#6b7280'
    }
    return colors.get(sentiment.lower(), '#6b7280')

def create_attention_visualization(attention_data, save_path="attention_plot.png"):
    """Create attention heatmap"""
    tokens = attention_data.tokens
    attention_matrix = attention_data.aggregated_attention
    
    # Limit tokens for visibility
    max_tokens = 15
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        attention_matrix = attention_matrix[:max_tokens, :max_tokens]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=attention_matrix.max()
    )
    
    plt.title('Attention Mechanism: Which Tokens the Model "Looks At"', fontsize=14, pad=20)
    plt.xlabel('Target Tokens', fontsize=12)
    plt.ylabel('Source Tokens', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_lime_visualization(lime_explanation, save_path="lime_plot.png"):
    """Create LIME bar chart"""
    # Extract top features
    feature_importance = lime_explanation['feature_importance'][:10]
    
    # Separate features and scores
    features = [f[0] for f in feature_importance]
    scores = [f[1] for f in feature_importance]
    
    # Create color map (positive green, negative red)
    colors = ['#10b981' if s > 0 else '#ef4444' for s in scores]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, scores, color=colors, alpha=0.8)
    plt.yticks(y_pos, features)
    plt.xlabel('Impact on Prediction', fontsize=12)
    plt.title('LIME: Which Words Drive the Prediction', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_insights(attention_importance: Dict, lime_importance: List) -> str:
    """Generate insights about the differences between methods"""
    # Get top tokens from attention
    attention_top = sorted(attention_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    attention_tokens = [t[0] for t in attention_top]
    
    # Get top features from LIME
    lime_tokens = [f[0] for f in lime_importance[:3]]
    
    # Find what's unique to each
    attention_unique = [t for t in attention_tokens if t not in lime_tokens]
    lime_unique = [t for t in lime_tokens if t not in attention_tokens]
    
    insights = []
    
    if attention_unique:
        insights.append(f"**Attention focuses on**: {', '.join(attention_unique)} (structural/positional patterns)")
    
    if lime_unique:
        insights.append(f"**LIME identifies**: {', '.join(lime_unique)} (predictive features)")
    
    # Add interpretation
    if len(set(attention_tokens) & set(lime_tokens)) == 0:
        insights.append("⚠️ **No overlap in top features** - Methods explain different aspects")
    else:
        overlap = set(attention_tokens) & set(lime_tokens)
        insights.append(f"✓ **Both methods agree on**: {', '.join(overlap)}")
    
    return "\n\n".join(insights)

def main():
    # Header
    st.markdown("<h1>🏦 FINSENT-XAI: Attention vs LIME</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comparing Explainable AI Methods for Financial Sentiment Analysis</p>', unsafe_allow_html=True)
    
    # Load models
    attention_extractor, lime_explainer = load_models()
    
    # Headline selector
    st.markdown("### Select a Financial Headline for Analysis")
    selected_headline = st.selectbox(
        "Choose from pre-selected UK financial news headlines:",
        HEADLINES,
        index=10  # Default to Rolls-Royce
    )
    
    # Display selected headline prominently
    st.markdown(f'<div class="headline-display">{selected_headline}</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🔍 Analyze with Both Methods", type="primary", use_container_width=True):
        
        # Create columns for results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Sentiment Prediction**")
            with st.spinner("Analyzing..."):
                # Get prediction (using LIME's model)
                pred_proba = lime_explainer._predict_proba([selected_headline])[0]
                pred_idx = np.argmax(pred_proba)
                sentiment = lime_explainer.labels[pred_idx]
                confidence = float(pred_proba[pred_idx])
                
            sentiment_color = get_sentiment_color(sentiment)
            st.markdown(f'<h2 style="color: {sentiment_color}; margin: 0;">{sentiment.upper()}</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 18px; margin: 0;">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Attention Analysis
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**Attention Analysis**")
            
            start_time = time.time()
            with st.spinner("Extracting attention patterns..."):
                attention_data = attention_extractor.extract_attention(selected_headline)
                attention_importance = attention_extractor.get_token_importance(attention_data)
            attention_time = time.time() - start_time
            
            st.markdown(f'<p class="processing-time" style="color: #10b981;">⚡ {attention_time:.3f}s</p>', unsafe_allow_html=True)
            st.markdown('<p style="margin: 0;">~832x faster</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # LIME Analysis
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("**LIME Analysis**")
            
            # Show processing message
            lime_container = st.empty()
            
            start_time = time.time()
            with lime_container.container():
                with st.spinner("Generating perturbations and testing impact... (this takes ~30s)"):
                    lime_result = lime_explainer.explain_instance(selected_headline, num_samples=1000)
            lime_time = time.time() - start_time
            
            lime_container.markdown(f'<p class="processing-time" style="color: #ef4444;">🔬 {lime_time:.1f}s</p>', unsafe_allow_html=True)
            lime_container.markdown('<p style="margin: 0;">More faithful explanations</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("### Explanation Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown('<p class="method-title">Attention Mechanism</p>', unsafe_allow_html=True)
            attention_img = create_attention_visualization(attention_data)
            st.image(attention_img, use_column_width=True)
            
            # Top attention tokens
            st.markdown("**Top Attention Focus:**")
            for token, score in sorted(attention_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {token}: {score:.3f}")
        
        with viz_col2:
            st.markdown('<p class="method-title">LIME Explanation</p>', unsafe_allow_html=True)
            lime_img = create_lime_visualization(lime_result)
            st.image(lime_img, use_column_width=True)
            
            # Top LIME features
            st.markdown("**Top LIME Features:**")
            for feature, score in lime_result['feature_importance'][:5]:
                direction = "↑" if score > 0 else "↓"
                st.write(f"• {feature}: {score:.3f} {direction}")
        
        # Insights
        st.markdown("---")
        st.markdown("### 🔍 Key Insights")
        
        insights = generate_insights(attention_importance, lime_result['feature_importance'])
        st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)
        
        # Research findings summary
        st.markdown("---")
        st.info(
            "**Research Finding**: Attention mechanisms (832x faster) and LIME provide complementary explanations "
            "with only 0.080 correlation. Attention reveals what the model 'looks at' (structural patterns), "
            "while LIME identifies which words actually drive the prediction (causal features)."
        )

if __name__ == "__main__":
    main()