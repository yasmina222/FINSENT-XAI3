"""
FinBERT Fine-tuning for UK Financial Sentiment Analysis
Core ML component of FINSENT-XAI
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from typing import Dict, List, Tuple
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class UKFinancialDataset(Dataset):
    """Dataset class for UK financial news"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FinBERTTrainer:
    """Fine-tunes FinBERT for UK financial sentiment analysis"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_directories()
        
        # Label mappings
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # UK financial terms for validation
        self.uk_terms = [
            'FTSE', 'sterling', 'gilt', 'London Stock Exchange', 'Bank of England',
            'pound', '£', 'UK economy', 'British', 'Brexit', 'Chancellor'
        ]
        
        self.logger.info(f"Using device: {self.device}")
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def load_and_prepare_data(self, data_path: str = None) -> Tuple:
        """Load and prepare UK financial news data"""
        self.logger.info("Loading UK financial news data...")
        
        # Load scraped articles
        if data_path is None:
            data_path = "data/raw/uk_news/latest_articles.json"
            
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                articles = data['articles']
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            self.logger.info("Creating sample data for demonstration...")
            articles = self.create_sample_data()
            
        # Extract texts
        texts = [article['title'] for article in articles]
        
        # For initial training, we'll use rule-based labels
        # In production, these would come from human annotations
        labels = [self.assign_sentiment_label(text) for text in texts]
        
        # Convert to numeric labels
        numeric_labels = [self.label2id[label] for label in labels]
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, numeric_labels, test_size=0.3, random_state=42, stratify=numeric_labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        self.logger.info(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
        
    def assign_sentiment_label(self, text: str) -> str:
        """Assign sentiment label based on financial keywords (for demo)"""
        text_lower = text.lower()
        
        # Negative indicators
        negative_words = [
            'plunge', 'crash', 'fall', 'drop', 'decline', 'loss', 'cut', 'weak',
            'concern', 'fear', 'risk', 'warning', 'threat', 'crisis', 'recession',
            'slump', 'tumble', 'sink', 'struggle', 'pressure', 'negative'
        ]
        
        # Positive indicators
        positive_words = [
            'surge', 'rise', 'gain', 'grow', 'increase', 'profit', 'strong',
            'boost', 'improve', 'recover', 'rally', 'advance', 'success', 'positive',
            'outperform', 'upgrade', 'optimis', 'record', 'high', 'beat'
        ]
        
        # Count occurrences
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if neg_count > pos_count:
            return 'negative'
        elif pos_count > neg_count:
            return 'positive'
        else:
            return 'neutral'
            
    def create_sample_data(self) -> List[Dict]:
        """Create sample UK financial data for testing"""
        samples = [
            {"title": "FTSE 100 surges to record high as banking stocks rally", "source": "FT"},
            {"title": "Barclays profits plunge 15% amid rising regulatory costs", "source": "BBC"},
            {"title": "Bank of England holds interest rates steady at 5.25%", "source": "Reuters"},
            {"title": "UK economy shows signs of recovery with 0.3% growth", "source": "Guardian"},
            {"title": "Sterling falls to six-month low against dollar on Brexit fears", "source": "FT"},
            {"title": "Tesco reports strong Christmas sales beating analyst expectations", "source": "BBC"},
            {"title": "HSBC warns of challenging year ahead as profits decline", "source": "Reuters"},
            {"title": "London Stock Exchange announces new sustainability measures", "source": "Guardian"},
        ]
        
        return samples
        
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'negative_f1': class_f1[0],
            'neutral_f1': class_f1[1],
            'positive_f1': class_f1[2]
        }
        
        return metrics
        
    def train(self, train_data, val_data, num_epochs: int = 3, batch_size: int = 16):
        """Fine-tune FinBERT on UK financial data"""
        self.logger.info("Starting FinBERT fine-tuning...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)
        
        # Prepare datasets
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data
        
        train_dataset = UKFinancialDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = UKFinancialDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./data/models/finbert-uk',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=50,
            save_strategy='steps',
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train model
        self.logger.info("Training started...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model('./data/models/finbert-uk-final')
        self.tokenizer.save_pretrained('./data/models/finbert-uk-final')
        
        # Save training metrics
        metrics = train_result.metrics
        with open('data/models/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info("Training completed!")
        
        return trainer
        
    def evaluate(self, trainer, test_data):
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")
        
        test_texts, test_labels = test_data
        test_dataset = UKFinancialDataset(test_texts, test_labels, self.tokenizer)
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        
        # Calculate detailed metrics
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, pred_labels)
        
        # Per-class accuracy
        class_accuracies = {}
        for i, label in enumerate(['negative', 'neutral', 'positive']):
            class_correct = cm[i, i]
            class_total = cm[i].sum()
            class_accuracies[label] = class_correct / class_total if class_total > 0 else 0
            
        # Company-specific accuracy (for key FTSE companies)
        company_accuracies = self.evaluate_company_specific(test_texts, test_labels, pred_labels)
        
        # Save evaluation results
        eval_results = {
            'test_metrics': predictions.metrics,
            'confusion_matrix': cm.tolist(),
            'class_accuracies': class_accuracies,
            'company_accuracies': company_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('data/processed/evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
            
        self.logger.info(f"Test Accuracy: {predictions.metrics['test_accuracy']:.4f}")
        self.logger.info(f"Test F1: {predictions.metrics['test_f1']:.4f}")
        
        return eval_results
        
    def evaluate_company_specific(self, texts, true_labels, pred_labels):
        """Evaluate accuracy for specific companies"""
        companies = ['Barclays', 'HSBC', 'BP', 'Tesco', 'Vodafone']
        company_results = {}
        
        for company in companies:
            # Find texts mentioning this company
            company_indices = [i for i, text in enumerate(texts) if company.lower() in text.lower()]
            
            if company_indices:
                company_true = [true_labels[i] for i in company_indices]
                company_pred = [pred_labels[i] for i in company_indices]
                accuracy = accuracy_score(company_true, company_pred)
                company_results[company] = {
                    'accuracy': accuracy,
                    'num_articles': len(company_indices)
                }
                
        return company_results
        
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        self.logger.info("Starting FINSENT-XAI training pipeline...")
        
        # Load data
        train_data, val_data, test_data = self.load_and_prepare_data()
        
        # Train model
        trainer = self.train(train_data, val_data)
        
        # Evaluate
        eval_results = self.evaluate(trainer, test_data)
        
        # Save pipeline summary
        summary = {
            'model_name': self.model_name,
            'device': str(self.device),
            'train_size': len(train_data[0]),
            'val_size': len(val_data[0]),
            'test_size': len(test_data[0]),
            'final_test_accuracy': eval_results['test_metrics']['test_accuracy'],
            'final_test_f1': eval_results['test_metrics']['test_f1']
        }
        
        with open('data/models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n Training Pipeline Complete!")
        print(f" Test Accuracy: {eval_results['test_metrics']['test_accuracy']:.2%}")
        print(f" Test F1-Score: {eval_results['test_metrics']['test_f1']:.4f}")
        print("\n Model saved to: data/models/finbert-uk-final/")
        
        return trainer, eval_results


def main():
    """Run the training pipeline"""
    print(" FINSENT-XAI FinBERT Training")
    print("=" * 50)
    
    trainer = FinBERTTrainer()
    trainer.run_full_pipeline()
    

if __name__ == "__main__":
    main()