"""
FINSENT-XAI Interactive Dashboard
Streamlit-based interface for financial sentiment analysis with explainability
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import time
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="FINSENT-XAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        margin: 5px;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_BASE = "http://localhost:8000"

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def create_attention_heatmap(attention_data):
    """Create interactive attention heatmap"""
    tokens = list(attention_data.keys())
    scores = list(attention_data.values())
    
    # Create matrix for heatmap (tokens x 1 for simple view)
    z = [[score] for score in scores]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=tokens,
        x=['Importance'],
        colorscale='Blues',
        showscale=True,
        hovertemplate='Token: %{y}<br>Importance: %{z[0]:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Attention Weights",
        height=400,
        xaxis_title="",
        yaxis_title="Tokens"
    )
    
    return fig


def create_feature_importance_chart(importance_data, title="Feature Importance"):
    """Create horizontal bar chart for feature importance"""
    if isinstance(importance_data, dict):
        features = list(importance_data.keys())
        scores = list(importance_data.values())
    else:
        # Handle list of tuples (LIME format)
        features = [item[0] for item in importance_data]
        scores = [item[1] for item in importance_data]
    
    # Sort by absolute importance
    sorted_indices = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)[:10]
    features = [features[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Color based on positive/negative
    colors = ['green' if s > 0 else 'red' for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color=colors,
            hovertemplate='%{y}: %{x:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        showlegend=False
    )
    
    return fig


def create_method_comparison_radar(comparison_data):
    """Create radar chart comparing XAI methods"""
    methods = ['Attention', 'LIME', 'SHAP']
    
    # Extract metrics
    metrics = {
        'Speed': [
            1 / (1 + comparison_data['attention_explanation']['processing_time']),
            1 / (1 + comparison_data['lime_explanation']['processing_time']),
            1 / (1 + comparison_data['shap_explanation']['processing_time'])
        ],
        'Top-5 Agreement': [
            comparison_data['agreement_metrics'].get('top5_attention_lime_overlap', 0),
            comparison_data['agreement_metrics'].get('top5_lime_shap_overlap', 0),
            comparison_data['agreement_metrics'].get('top5_attention_shap_overlap', 0)
        ],
        'Correlation': [
            comparison_data['agreement_metrics'].get('attention_lime_correlation', 0),
            comparison_data['agreement_metrics'].get('lime_shap_correlation', 0),
            comparison_data['agreement_metrics'].get('attention_shap_correlation', 0)
        ]
    }
    
    fig = go.Figure()
    
    for i, method in enumerate(methods):
        values = [metrics['Speed'][i], metrics['Top-5 Agreement'][i], metrics['Correlation'][i]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Speed', 'Agreement', 'Correlation'],
            fill='toself',
            name=method
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="XAI Method Comparison",
        height=400
    )
    
    return fig


def main():
    # Header
    st.title("🏦 FINSENT-XAI: Financial Sentiment Explainable AI")
    st.markdown("### UK Financial News Sentiment Analysis with Transparent Explanations")
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Check API health
        health = check_api_health()
        if health and health['status'] == 'healthy':
            st.success("✅ API Connected")
            
            # Show model status
            st.subheader("Models Loaded")
            for model, loaded in health['models_loaded'].items():
                if loaded:
                    st.write(f"✅ {model.capitalize()}")
                else:
                    st.write(f"❌ {model.capitalize()}")
        else:
            st.error("❌ API Not Connected")
            st.info("Please start the API: `python src/api/main.py`")
            
        st.divider()
        
        # Analysis options
        st.subheader("Analysis Options")
        include_explanations = st.checkbox("Include Explanations", value=True)
        
        if include_explanations:
            st.write("Select Methods:")
            use_attention = st.checkbox("Attention", value=True)
            use_lime = st.checkbox("LIME", value=True)
            use_shap = st.checkbox("SHAP", value=True)
            
            methods = []
            if use_attention: methods.append("attention")
            if use_lime: methods.append("lime")
            if use_shap: methods.append("shap")
        else:
            methods = []
    
    # Main content area
    tabs = st.tabs(["📝 Single Analysis", "🔍 Method Comparison", "📊 Batch Analysis", "📈 Results Gallery"])
    
    # Tab 1: Single Analysis
    with tabs[0]:
        st.header("Analyze Single Text")
        
        # Predefined examples
        examples = {
            "Positive": "FTSE 100 surges to record high as banking stocks rally strongly",
            "Negative": "Barclays shares plunged 15% after reporting significant losses",
            "Neutral": "Bank of England announces quarterly meeting schedule for 2025"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_example = st.selectbox("Choose an example or enter custom text:", 
                                          ["Custom"] + list(examples.keys()))
        
        if selected_example == "Custom":
            text_input = st.text_area("Enter UK financial news text:", 
                                    height=100,
                                    placeholder="e.g., Sterling falls against dollar amid economic concerns...")
        else:
            text_input = examples[selected_example]
            st.info(f"Using {selected_example} example: {text_input}")
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("Analyzing..."):
                    try:
                        # Call API
                        response = requests.post(
                            f"{API_BASE}/analyze",
                            json={
                                "text": text_input,
                                "include_explanations": include_explanations,
                                "explanation_methods": methods
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_color = {
                                    'positive': '🟢',
                                    'negative': '🔴',
                                    'neutral': '🟡'
                                }
                                st.metric(
                                    "Sentiment",
                                    f"{sentiment_color.get(result['sentiment'], '⚪')} {result['sentiment'].upper()}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Confidence",
                                    f"{result['confidence']:.1%}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Processing Time",
                                    f"{result['processing_time']:.3f}s"
                                )
                            
                            # Display explanations if included
                            if include_explanations and result.get('explanations'):
                                st.divider()
                                st.subheader("📊 Explanations")
                                
                                # Create columns for each method
                                exp_cols = st.columns(len(methods))
                                
                                for idx, method in enumerate(methods):
                                    if method in result['explanations']:
                                        with exp_cols[idx]:
                                            exp_data = result['explanations'][method]
                                            
                                            if method == "attention":
                                                fig = create_attention_heatmap(exp_data['token_importance'])
                                            elif method == "lime":
                                                fig = create_feature_importance_chart(
                                                    exp_data['feature_importance'],
                                                    title="LIME Feature Importance"
                                                )
                                            else:  # SHAP
                                                fig = create_feature_importance_chart(
                                                    exp_data['token_importance'],
                                                    title="SHAP Token Importance"
                                                )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Show top features
                                            st.markdown("**Top Features:**")
                                            top_features = exp_data.get('top_features') or exp_data.get('top_tokens', [])
                                            for feature in top_features[:3]:
                                                if isinstance(feature, tuple):
                                                    st.write(f"• {feature[0]}: {feature[1]:.3f}")
                                                else:
                                                    st.write(f"• {feature}")
                            
                            # Save to history
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'text': text_input,
                                'result': result
                            })
                            
                        else:
                            st.error(f"API Error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure the API is running at http://localhost:8000")
    
    # Tab 2: Method Comparison
    with tabs[1]:
        st.header("Compare XAI Methods")
        st.markdown("Compare how different explainability methods analyze the same text")
        
        comparison_text = st.text_area(
            "Enter text for comparison:",
            value="Barclays profits plunged 15% amid rising regulatory costs",
            height=100
        )
        
        if st.button("🔄 Compare All Methods", type="primary", use_container_width=True):
            with st.spinner("Running comparison analysis..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/compare_methods",
                        json={"text": comparison_text}
                    )
                    
                    if response.status_code == 200:
                        comparison = response.json()
                        st.session_state.comparison_results = comparison
                        
                        # Display sentiment
                        st.success(f"Sentiment: **{comparison['sentiment'].upper()}**")
                        
                        # Method agreement metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Agreement Metrics")
                            metrics = comparison['agreement_metrics']
                            
                            # Create agreement matrix
                            agreement_df = pd.DataFrame({
                                'Attention-LIME': [metrics.get('attention_lime_correlation', 0)],
                                'Attention-SHAP': [metrics.get('attention_shap_correlation', 0)],
                                'LIME-SHAP': [metrics.get('lime_shap_correlation', 0)]
                            })
                            
                            st.dataframe(agreement_df.style.format("{:.3f}"))
                            
                            # Consensus features
                            if comparison['consensus_features']:
                                st.markdown("**Consensus Features:**")
                                for feature in comparison['consensus_features']:
                                    st.write(f"• {feature}")
                        
                        with col2:
                            # Radar chart comparison
                            fig = create_method_comparison_radar(comparison)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed explanations
                        st.divider()
                        st.markdown("### 🔍 Detailed Explanations")
                        
                        exp_cols = st.columns(3)
                        
                        # Attention
                        with exp_cols[0]:
                            st.markdown("**Attention**")
                            att_fig = create_attention_heatmap(
                                comparison['attention_explanation']['importance']
                            )
                            st.plotly_chart(att_fig, use_container_width=True)
                            st.caption(f"Time: {comparison['attention_explanation']['processing_time']:.3f}s")
                        
                        # LIME
                        with exp_cols[1]:
                            st.markdown("**LIME**")
                            lime_fig = create_feature_importance_chart(
                                comparison['lime_explanation']['importance'],
                                "LIME Features"
                            )
                            st.plotly_chart(lime_fig, use_container_width=True)
                            st.caption(f"Time: {comparison['lime_explanation']['processing_time']:.3f}s")
                        
                        # SHAP
                        with exp_cols[2]:
                            st.markdown("**SHAP**")
                            shap_fig = create_feature_importance_chart(
                                comparison['shap_explanation']['importance'],
                                "SHAP Features"
                            )
                            st.plotly_chart(shap_fig, use_container_width=True)
                            st.caption(f"Time: {comparison['shap_explanation']['processing_time']:.3f}s")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 3: Batch Analysis
    with tabs[2]:
        st.header("Batch Analysis")
        st.markdown("Analyze multiple texts at once")
        
        # Text input methods
        input_method = st.radio("Input Method:", ["Text Box", "Upload File"])
        
        if input_method == "Text Box":
            batch_text = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="FTSE 100 rises on strong earnings\nBarclays faces regulatory challenges\nSterling strengthens against euro"
            )
            texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        else:
            uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
            texts = []
            if uploaded_file:
                data = json.load(uploaded_file)
                texts = data.get('texts', [])
                st.info(f"Loaded {len(texts)} texts from file")
        
        if st.button("📊 Analyze Batch", type="primary", use_container_width=True):
            if texts:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process batch
                    results = []
                    for i, text in enumerate(texts):
                        status_text.text(f"Processing {i+1}/{len(texts)}...")
                        progress_bar.progress((i + 1) / len(texts))
                        
                        response = requests.post(
                            f"{API_BASE}/analyze",
                            json={"text": text, "include_explanations": False}
                        )
                        
                        if response.status_code == 200:
                            results.append(response.json())
                        else:
                            results.append({"text": text, "error": "Failed to analyze"})
                    
                    # Display results
                    st.success(f"Analyzed {len(results)} texts")
                    
                    # Create results dataframe
                    df_data = []
                    for r in results:
                        df_data.append({
                            'Text': r['text'][:50] + '...' if len(r['text']) > 50 else r['text'],
                            'Sentiment': r.get('sentiment', 'Error'),
                            'Confidence': f"{r.get('confidence', 0):.1%}",
                            'Time (s)': f"{r.get('processing_time', 0):.3f}"
                        })
                    
                    results_df = pd.DataFrame(df_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Sentiment distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment_counts = results_df['Sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={
                                'positive': '#2ecc71',
                                'negative': '#e74c3c',
                                'neutral': '#95a5a6'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Processing time stats
                        times = [float(t.replace('s', '')) for t in results_df['Time (s)']]
                        st.metric("Average Processing Time", f"{np.mean(times):.3f}s")
                        st.metric("Total Processing Time", f"{sum(times):.3f}s")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    # Tab 4: Results Gallery
    with tabs[3]:
        st.header("Results Gallery")
        st.markdown("View analysis history and saved results")
        
        if st.session_state.analysis_history:
            st.subheader(f"📝 Recent Analyses ({len(st.session_state.analysis_history)})")
            
            for idx, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"{item['text'][:50]}... - {item['result']['sentiment']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Full Text:** {item['text']}")
                        st.write(f"**Sentiment:** {item['result']['sentiment']}")
                        st.write(f"**Confidence:** {item['result']['confidence']:.1%}")
                    
                    with col2:
                        st.write(f"**Timestamp:** {item['timestamp']}")
                        st.write(f"**Processing Time:** {item['result']['processing_time']:.3f}s")
        else:
            st.info("No analysis history yet. Start analyzing texts to see results here.")
        
        # Export functionality
        st.divider()
        st.subheader("📥 Export Results")
        
        if st.button("Export History as JSON"):
            if st.session_state.analysis_history:
                # Create download
                history_json = json.dumps(st.session_state.analysis_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"finsent_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No history to export")


if __name__ == "__main__":
    main()