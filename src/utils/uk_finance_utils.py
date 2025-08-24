"""
UK Financial Utilities for FINSENT-XAI
Handles UK-specific terminology, currency formatting, and entity standardization
"""

import re
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class UKFinanceUtils:
    """
    Utility class for processing UK financial text.
    Handles currency conversion, entity normalization, and terminology mapping.
    """
    
    def __init__(self):
        """Initialize UK finance utilities with domain-specific mappings"""
        
        # FTSE 100 companies and common variations
        self.ftse_companies = self._load_ftse_companies()
        
        # UK financial institutions
        self.uk_institutions = {
            'boe': 'Bank of England',
            'bank of england': 'Bank of England',
            'fca': 'Financial Conduct Authority',
            'pra': 'Prudential Regulation Authority',
            'hmrc': 'HM Revenue & Customs',
            'hm treasury': 'HM Treasury',
            'lse': 'London Stock Exchange',
            'london stock exchange': 'London Stock Exchange'
        }
        
        # Currency patterns
        self.currency_patterns = [
            (r'£(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|bn)', lambda m: float(m.group(1).replace(',', '')) * 1e9),
            (r'£(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|mn|m)', lambda m: float(m.group(1).replace(',', '')) * 1e6),
            (r'£(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|k)', lambda m: float(m.group(1).replace(',', '')) * 1e3),
            (r'£(\d+(?:,\d{3})*(?:\.\d+)?)', lambda m: float(m.group(1).replace(',', ''))),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|bn)\s*(?:pounds|sterling|GBP)', lambda m: float(m.group(1).replace(',', '')) * 1e9),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|mn|m)\s*(?:pounds|sterling|GBP)', lambda m: float(m.group(1).replace(',', '')) * 1e6),
            (r'(\d+(?:\.\d+)?)\s*p(?:ence)?', lambda m: float(m.group(1)) / 100)  # Pence to pounds
        ]
        
        # UK market indices
        self.uk_indices = {
            'ftse 100': 'FTSE 100',
            'ftse100': 'FTSE 100',
            'footsie': 'FTSE 100',
            'ftse 250': 'FTSE 250',
            'ftse250': 'FTSE 250',
            'ftse all-share': 'FTSE All-Share',
            'ftse aim': 'FTSE AIM',
            'ftse small cap': 'FTSE SmallCap'
        }
        
        # UK-specific financial terms
        self.uk_financial_terms = {
            # Regulatory terms
            'mifid': 'Markets in Financial Instruments Directive',
            'ring-fencing': 'Ring-fencing',
            'senior managers regime': 'Senior Managers Regime',
            'smr': 'Senior Managers Regime',
            
            # Market terms
            'gilts': 'UK Government Bonds',
            'gilt-edged': 'UK Government Bonds',
            'isa': 'Individual Savings Account',
            'sipp': 'Self-Invested Personal Pension',
            'ns&i': 'National Savings & Investments',
            
            # Corporate terms
            'plc': 'Public Limited Company',
            'ltd': 'Limited Company',
            'rights issue': 'Rights Issue',
            'scrip dividend': 'Scrip Dividend',
            'interim dividend': 'Interim Dividend',
            'final dividend': 'Final Dividend',
            
            # Economic terms
            'base rate': 'Bank of England Base Rate',
            'libor': 'London Interbank Offered Rate',
            'sonia': 'Sterling Overnight Index Average',
            'rpi': 'Retail Prices Index',
            'cpi': 'Consumer Prices Index'
        }
        
    def _load_ftse_companies(self) -> Dict[str, str]:
        """Load FTSE 100 companies with variations"""
        # Major FTSE 100 companies and their variations
        companies = {
            # Banks
            'barclays': 'Barclays',
            'barclays bank': 'Barclays',
            'barclays plc': 'Barclays',
            'hsbc': 'HSBC',
            'hsbc holdings': 'HSBC',
            'lloyds': 'Lloyds Banking Group',
            'lloyds bank': 'Lloyds Banking Group',
            'lloyds banking group': 'Lloyds Banking Group',
            'natwest': 'NatWest Group',
            'natwest group': 'NatWest Group',
            'rbs': 'NatWest Group',  # Former name
            'standard chartered': 'Standard Chartered',
            'stanchart': 'Standard Chartered',
            
            # Energy
            'bp': 'BP',
            'british petroleum': 'BP',
            'shell': 'Shell',
            'royal dutch shell': 'Shell',
            
            # Pharma
            'astrazeneca': 'AstraZeneca',
            'az': 'AstraZeneca',
            'gsk': 'GlaxoSmithKline',
            'glaxosmithkline': 'GlaxoSmithKline',
            'glaxo': 'GlaxoSmithKline',
            
            # Consumer
            'tesco': 'Tesco',
            'tesco plc': 'Tesco',
            'sainsbury': 'Sainsbury\'s',
            'sainsburys': 'Sainsbury\'s',
            'j sainsbury': 'Sainsbury\'s',
            'unilever': 'Unilever',
            'diageo': 'Diageo',
            
            # Telecom
            'vodafone': 'Vodafone',
            'vodafone group': 'Vodafone',
            'bt': 'BT Group',
            'bt group': 'BT Group',
            'british telecom': 'BT Group',
            
            # Other major companies
            'rolls-royce': 'Rolls-Royce',
            'rolls royce': 'Rolls-Royce',
            'rr': 'Rolls-Royce',
            'bae': 'BAE Systems',
            'bae systems': 'BAE Systems',
            'rio': 'Rio Tinto',
            'rio tinto': 'Rio Tinto',
            'prudential': 'Prudential',
            'pru': 'Prudential',
            'aviva': 'Aviva',
            'legal & general': 'Legal & General',
            'l&g': 'Legal & General'
        }
        
        return companies
        
    def standardize_company_name(self, text: str) -> str:
        """
        Standardize company names in text to canonical forms.
        
        Args:
            text: Input text containing company names
            
        Returns:
            Text with standardized company names
        """
        text_lower = text.lower()
        
        # Replace company variations with standard names
        for variant, standard in self.ftse_companies.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(variant) + r'\b'
            
            # Find all matches to preserve original case where possible
            matches = list(re.finditer(pattern, text_lower))
            
            # Replace from end to start to maintain positions
            for match in reversed(matches):
                start, end = match.span()
                # Try to preserve original case pattern
                original = text[start:end]
                if original.isupper():
                    replacement = standard.upper()
                elif original[0].isupper():
                    replacement = standard
                else:
                    replacement = standard.lower()
                    
                text = text[:start] + replacement + text[end:]
                
        return text
        
    def standardize_currency(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Standardize currency expressions to numerical values.
        
        Args:
            text: Input text containing currency expressions
            
        Returns:
            Tuple of (standardized text, list of currency extractions)
        """
        standardized = text
        extractions = []
        
        for pattern, converter in self.currency_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in reversed(matches):  # Process from end to maintain positions
                try:
                    value = converter(match)
                    
                    extraction = {
                        'original': match.group(0),
                        'value_gbp': value,
                        'position': match.span()
                    }
                    extractions.append(extraction)
                    
                    # Format large numbers appropriately
                    if value >= 1e9:
                        formatted = f"£{value/1e9:.1f}bn"
                    elif value >= 1e6:
                        formatted = f"£{value/1e6:.1f}m"
                    elif value >= 1e3:
                        formatted = f"£{value/1e3:.0f}k"
                    else:
                        formatted = f"£{value:.2f}"
                        
                    standardized = (
                        standardized[:match.start()] + 
                        formatted + 
                        standardized[match.end():]
                    )
                    
                except ValueError:
                    continue
                    
        return standardized, extractions
        
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract UK financial entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their occurrences
        """
        entities = {
            'companies': [],
            'institutions': [],
            'indices': [],
            'currencies': []
        }
        
        text_lower = text.lower()
        
        # Extract companies
        for variant, standard in self.ftse_companies.items():
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                if standard not in entities['companies']:
                    entities['companies'].append(standard)
                    
        # Extract institutions
        for variant, standard in self.uk_institutions.items():
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                if standard not in entities['institutions']:
                    entities['institutions'].append(standard)
                    
        # Extract indices
        for variant, standard in self.uk_indices.items():
            if variant in text_lower:
                if standard not in entities['indices']:
                    entities['indices'].append(standard)
                    
        # Extract currency amounts
        _, currency_extractions = self.standardize_currency(text)
        for extraction in currency_extractions:
            entities['currencies'].append({
                'amount': extraction['value_gbp'],
                'formatted': extraction['original']
            })
            
        return entities
        
    def detect_uk_specific_terms(self, text: str) -> List[str]:
        """
        Detect UK-specific financial terms in text.
        
        Args:
            text: Input text
            
        Returns:
            List of UK-specific terms found
        """
        found_terms = []
        text_lower = text.lower()
        
        for term in self.uk_financial_terms:
            if term in text_lower:
                found_terms.append(term)
                
        return found_terms
        
    def preprocess_for_analysis(self, text: str) -> Dict:
        """
        Comprehensive preprocessing for UK financial text.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary with processed text and metadata
        """
        # Standardize companies
        processed_text = self.standardize_company_name(text)
        
        # Standardize currency
        processed_text, currency_data = self.standardize_currency(processed_text)
        
        # Extract entities
        entities = self.extract_financial_entities(text)
        
        # Detect UK terms
        uk_terms = self.detect_uk_specific_terms(text)
        
        # Clean up extra whitespace
        processed_text = ' '.join(processed_text.split())
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'entities': entities,
            'uk_specific_terms': uk_terms,
            'currency_data': currency_data,
            'contains_uk_content': bool(
                entities['companies'] or 
                entities['institutions'] or 
                entities['indices'] or 
                uk_terms
            )
        }
        
    def validate_uk_relevance(self, text: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Validate if text is relevant to UK financial markets.
        
        Args:
            text: Input text
            threshold: Minimum relevance score (0-1)
            
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        # Process text
        processed = self.preprocess_for_analysis(text)
        
        # Calculate relevance score
        score_components = {
            'has_uk_company': len(processed['entities']['companies']) > 0,
            'has_uk_institution': len(processed['entities']['institutions']) > 0,
            'has_uk_index': len(processed['entities']['indices']) > 0,
            'has_uk_terms': len(processed['uk_specific_terms']) > 0,
            'has_gbp_currency': any('£' in c['formatted'] for c in processed['currency_data'])
        }
        
        # Weight components
        weights = {
            'has_uk_company': 0.4,
            'has_uk_institution': 0.2,
            'has_uk_index': 0.2,
            'has_uk_terms': 0.1,
            'has_gbp_currency': 0.1
        }
        
        relevance_score = sum(
            weights[component] for component, present in score_components.items() 
            if present
        )
        
        is_relevant = relevance_score >= threshold
        
        return is_relevant, relevance_score


def main():
    """Test UK finance utilities"""
    utils = UKFinanceUtils()
    
    # Test texts
    test_texts = [
        "Barclays plunged 15% to £1.2bn market cap after profit warning",
        "HSBC announces £2.5 billion share buyback program",
        "The FTSE 100 rose 120 points as BP gained 3.5%",
        "Bank of England holds base rate at 5.25% amid inflation concerns",
        "Tesco sales reach £28.3m during Christmas period"
    ]
    
    print("Testing UK Finance Utilities\n")
    
    for text in test_texts:
        print(f"Original: {text}")
        
        # Process text
        result = utils.preprocess_for_analysis(text)
        
        print(f"Processed: {result['processed_text']}")
        print(f"Companies: {result['entities']['companies']}")
        print(f"Institutions: {result['entities']['institutions']}")
        print(f"UK Terms: {result['uk_specific_terms']}")
        
        # Check relevance
        is_relevant, score = utils.validate_uk_relevance(text)
        print(f"UK Relevance: {is_relevant} (score: {score:.2f})")
        print("-" * 50)
        

if __name__ == "__main__":
    main()