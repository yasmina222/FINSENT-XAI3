"""
LIME Explainer for UK Financial Sentiment Analysis
Fixed to handle FinBERT's subword tokenization properly
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re


class FinancialLimeExplainer:
    """LIME explainer adapted for financial sentiment with FinBERT"""
    
    def __init__(self, model_path: str = "ProsusAI/finbert"):
        """Initialize LIME explainer with FinBERT model"""
        self.logger = logging.getLogger("LimeExplainer")
        self.logger.info(f"Loading model from {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Custom word tokenizer for LIME (not subword)
        def word_tokenizer(text):
            # Simple word-level tokenization
            return re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Initialize LIME with custom tokenizer
        self.explainer = LimeTextExplainer(
            class_names=['positive', 'negative', 'neutral'],
            split_expression=word_tokenizer,  # Use custom tokenizer
            mask_string='[UNK]',  # Use FinBERT's unknown token
            bow=False
        )
        
        self.labels = ['positive', 'negative', 'neutral']
        
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for texts
        This is what LIME calls internally with perturbed texts
        """
        results = []
        
        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Handle empty or invalid texts
            processed_texts = []
            for text in batch_texts:
                if not text or not text.strip() or text.strip() == '[UNK]':
                    # For empty text, use a neutral placeholder
                    processed_texts.append("The market remains stable.")
                else:
                    processed_texts.append(text)
            
            # Tokenize batch with FinBERT tokenizer
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                results.extend(probabilities.cpu().numpy())
                
        return np.array(results)
    
    def explain_instance(self, text: str, num_features: int = 10, num_samples: int = 5000) -> Dict:
        """
        Generate LIME explanation for a single text
        
        Args:
            text: Input text to explain
            num_features: Number of important features to return
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Dictionary containing explanation data
        """
        try:
            # Get the model's prediction first
            prediction_probs = self._predict_proba([text])[0]
            predicted_class = np.argmax(prediction_probs)
            
            # Generate LIME explanation
            explanation = self.explainer.explain_instance(
                text,
                self._predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=[predicted_class]  # Only explain the predicted class
            )
            
            # Extract feature importance for the predicted class
            feature_importance = explanation.as_list(label=predicted_class)
            
            # Also get feature importance as a dictionary
            feature_dict = dict(feature_importance)
            
            return {
                'text': text,
                'prediction': self.labels[predicted_class],
                'confidence': float(prediction_probs[predicted_class]),
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.labels, prediction_probs)
                },
                'feature_importance': feature_importance,
                'feature_dict': feature_dict,
                'num_features': len(feature_importance),
                'explanation_object': explanation
            }
            
        except Exception as e:
            self.logger.error(f"Error explaining instance: {e}")
            # Return a basic result even if explanation fails
            return {
                'text': text,
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {label: 0.0 for label in self.labels},
                'feature_importance': [],
                'feature_dict': {},
                'error': str(e)
            }
    
    def explain_batch(self, texts: List[str], num_features: int = 10) -> List[Dict]:
        """Explain multiple texts"""
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Explaining text {i+1}/{len(texts)}: {text[:50]}...")
            result = self.explain_instance(text, num_features)
            results.append(result)
        return results


# Test function
if __name__ == "__main__":
    explainer = FinancialLimeExplainer()
    
    # Test texts
    test_texts = [
        "Barclays profits increased significantly",
        "FTSE drops amid market concerns",
        "Bank of England maintains interest rates"
    ]
    
    for text in test_texts:
        print(f"\nAnalyzing: {text}")
        result = explainer.explain_instance(text, num_features=5)
        
        if 'error' not in result:
            print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
            print("Important features:")
            for feature, score in result['feature_importance'][:5]:
                print(f"  '{feature}': {score:.4f}")
        else:
            print(f"Error: {result['error']}")