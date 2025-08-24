"""
FINSENT-XAI REST API
Provides endpoints for financial sentiment analysis with explainability
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import torch
import numpy as np
from pathlib import Path

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.attention_extractor import AttentionExtractor
from ml.lime_explainer import FinancialLimeExplainer
from ml.shap_explainer import FinancialShapExplainer
from ml.xai_comparator import XAIComparator
from utils.uk_finance_utils import UKFinanceUtils
from utils.metrics import FinancialMetrics


# Initialize FastAPI app
app = FastAPI(
    title="FINSENT-XAI API",
    description="Financial Sentiment Analysis with Explainable AI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SentimentRequest(BaseModel):
    text: str
    include_explanations: bool = True
    explanation_methods: List[str] = ["attention", "lime", "shap"]


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    explanations: Optional[Dict] = None
    processing_time: float


class BatchSentimentRequest(BaseModel):
    texts: List[str]
    include_explanations: bool = False


class ExplanationComparison(BaseModel):
    text: str
    sentiment: str
    attention_explanation: Dict
    lime_explanation: Dict
    shap_explanation: Dict
    agreement_metrics: Dict
    consensus_features: List[str]


# Global model instances (initialized on startup)
attention_extractor = None
lime_explainer = None
shap_explainer = None
xai_comparator = None
uk_utils = None
metrics_calc = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global attention_extractor, lime_explainer, shap_explainer, xai_comparator, uk_utils, metrics_calc
    
    print("Initializing FINSENT-XAI models...")
    
    model_path = "data/models/finbert-uk-final"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Warning: Trained model not found. Using base FinBERT.")
        model_path = "ProsusAI/finbert"
    
    # Initialize components
    try:
        attention_extractor = AttentionExtractor(model_path)
        lime_explainer = FinancialLimeExplainer(model_path)
        shap_explainer = FinancialShapExplainer(model_path)
        xai_comparator = XAIComparator(model_path)
        uk_utils = UKFinanceUtils()
        metrics_calc = FinancialMetrics()
        
        print("✓ All models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("API will start but some features may not work")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "FINSENT-XAI API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze - Analyze single text",
            "/analyze_batch - Analyze multiple texts",
            "/compare_methods - Compare all XAI methods",
            "/health - Check API health",
            "/docs - Interactive API documentation"
        ]
    }


@app.get("/health")
def health_check():
    """Check API and model health"""
    health_status = {
        "status": "healthy",
        "models_loaded": {
            "attention": attention_extractor is not None,
            "lime": lime_explainer is not None,
            "shap": shap_explainer is not None,
            "comparator": xai_comparator is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Check if all models loaded
    all_loaded = all(health_status["models_loaded"].values())
    if not all_loaded:
        health_status["status"] = "partial"
        
    return health_status


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of financial text with optional explanations.
    
    - **text**: UK financial news text to analyze
    - **include_explanations**: Whether to include XAI explanations
    - **explanation_methods**: Which methods to use ["attention", "lime", "shap"]
    """
    start_time = datetime.now()
    
    # Validate and preprocess text
    processed = uk_utils.preprocess_for_analysis(request.text)
    
    if not processed['contains_uk_content']:
        raise HTTPException(
            status_code=400, 
            detail="Text does not appear to be UK financial content"
        )
    
    # Get base prediction (using LIME explainer's model)
    prediction_proba = lime_explainer._predict_proba([processed['processed_text']])[0]
    predicted_idx = np.argmax(prediction_proba)
    sentiment = lime_explainer.labels[predicted_idx]
    confidence = float(prediction_proba[predicted_idx])
    
    explanations = {}
    
    if request.include_explanations:
        # Get explanations from requested methods
        if "attention" in request.explanation_methods:
            att_data = attention_extractor.extract_attention(processed['processed_text'])
            att_importance = attention_extractor.get_token_importance(att_data)
            explanations["attention"] = {
                "token_importance": att_importance,
                "top_tokens": sorted(att_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            }
            
        if "lime" in request.explanation_methods:
            lime_exp = lime_explainer.explain_instance(
                processed['processed_text'], 
                num_features=10
            )
            explanations["lime"] = {
                "feature_importance": lime_exp.feature_importance,
                "top_features": lime_exp.feature_importance[:5]
            }
            
        if "shap" in request.explanation_methods:
            shap_exp = shap_explainer.explain_instance(processed['processed_text'])
            shap_importance = shap_explainer.get_token_importance(shap_exp)
            explanations["shap"] = {
                "token_importance": shap_importance,
                "top_tokens": sorted(shap_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            }
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        explanations=explanations if request.include_explanations else None,
        processing_time=processing_time
    )


@app.post("/analyze_batch")
async def analyze_batch(request: BatchSentimentRequest):
    """
    Analyze multiple texts efficiently.
    
    - **texts**: List of texts to analyze
    - **include_explanations**: Whether to include explanations (slower)
    """
    results = []
    
    for text in request.texts:
        try:
            # Create single request
            single_request = SentimentRequest(
                text=text,
                include_explanations=request.include_explanations
            )
            
            # Analyze
            result = await analyze_sentiment(single_request)
            results.append(result.dict())
            
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e),
                "sentiment": None,
                "confidence": 0.0
            })
            
    return {
        "results": results,
        "total_processed": len(results),
        "successful": sum(1 for r in results if "error" not in r)
    }


@app.post("/compare_methods", response_model=ExplanationComparison)
async def compare_explanation_methods(request: SentimentRequest):
    """
    Compare all three explanation methods on the same text.
    This is the key endpoint for dissertation evaluation.
    """
    # Get comprehensive comparison
    comparison = xai_comparator.compare_single_instance(request.text)
    
    # Format response
    return ExplanationComparison(
        text=request.text,
        sentiment=comparison['lime']['prediction'],
        attention_explanation={
            "importance": comparison['attention']['importance'],
            "processing_time": comparison['timings']['attention']
        },
        lime_explanation={
            "importance": dict(comparison['lime']['importance']),
            "processing_time": comparison['timings']['lime']
        },
        shap_explanation={
            "importance": comparison['shap']['importance'],
            "processing_time": comparison['timings']['shap']
        },
        agreement_metrics=comparison['comparisons'],
        consensus_features=comparison['consensus_features']
    )


@app.post("/evaluate_dataset")
async def evaluate_on_dataset(file: UploadFile = File(...)):
    """
    Run evaluation on uploaded dataset.
    Expects JSON file with list of texts.
    """
    # Read uploaded file
    contents = await file.read()
    data = json.loads(contents)
    
    if "texts" not in data:
        raise HTTPException(status_code=400, detail="File must contain 'texts' field")
    
    texts = data["texts"][:50]  # Limit to 50 for API demo
    
    # Run comprehensive evaluation
    evaluation_results = xai_comparator.run_comprehensive_evaluation(texts)
    
    # Generate visualizations
    output_dir = "data/api_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    xai_comparator.generate_comparison_visualizations(
        evaluation_results,
        output_dir=output_dir
    )
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    xai_comparator.save_evaluation_results(evaluation_results, results_path)
    
    # Generate summary report
    summary = xai_comparator.generate_summary_report(evaluation_results)
    
    return {
        "summary": summary,
        "detailed_results_path": results_path,
        "visualizations_path": output_dir,
        "num_texts_evaluated": len(texts)
    }


@app.get("/download_results/{filename}")
async def download_results(filename: str):
    """Download generated results or visualizations"""
    file_path = os.path.join("data/api_outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)


@app.get("/model_info")
def get_model_info():
    """Get information about loaded models"""
    return {
        "models": {
            "sentiment_model": "FinBERT (fine-tuned on UK financial news)" 
                             if os.path.exists("data/models/finbert-uk-final") 
                             else "FinBERT (base model)",
            "attention_mechanism": "Multi-head attention from BERT",
            "lime_version": "0.2.0.1",
            "shap_version": "0.44.0"
        },
        "uk_finance_terms": len(uk_utils.uk_financial_terms),
        "ftse_companies": len(uk_utils.ftse_companies),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/test_uk_relevance")
async def test_uk_relevance(request: SentimentRequest):
    """Test if text is relevant to UK financial markets"""
    processed = uk_utils.preprocess_for_analysis(request.text)
    is_relevant, score = uk_utils.validate_uk_relevance(request.text)
    
    return {
        "text": request.text,
        "is_uk_financial": is_relevant,
        "relevance_score": score,
        "uk_entities_found": processed['entities'],
        "uk_terms_found": processed['uk_specific_terms']
    }


def custom_openapi():
    """Customize OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
        
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title="FINSENT-XAI API",
        version="1.0.0",
        description="""
        ## Financial Sentiment Explainable AI API
        
        This API provides sentiment analysis for UK financial news with 
        three explainability methods:
        
        1. **Attention Mechanisms** - Neural attention weights
        2. **LIME** - Local Interpretable Model-agnostic Explanations
        3. **SHAP** - SHapley Additive exPlanations
        
        ### Key Features:
        - UK financial content validation
        - Real-time sentiment analysis
        - Comparative explainability analysis
        - Batch processing support
        
        ### For dissertation evaluation:
        Use `/compare_methods` endpoint to get all three explanations
        for the same text.
        """,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FINSENT-XAI API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)