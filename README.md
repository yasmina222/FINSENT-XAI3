# FINSENT-XAI: Financial Sentiment Explainable AI

## Project Overview
FINSENT-XAI analyzes UK financial news (synthetic data) to determine market sentiment (positive/negative/neutral) while providing transparent explanations using attention mechanisms and LIME.
Previously we tried to use a news scraper but could not scrape enough data so we had to change out strategy to syntheic data based of off new headlines that mimic real world financial headlines. 

We tried to use SHAP as a compariative method also, but the problems with tokenization and finbert made it not possibly, therefore we are just comparing Lime and attention Mechanisms.

Due to these other attemps that failed, tehre may be evidence in the github repo of these methods.

## Quick Start Guide

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sms