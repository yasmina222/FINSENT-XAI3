"""
Data Validation Module for FINSENT-XAI
Ensures quality and relevance of scraped UK financial news data
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import Counter
import pandas as pd


class UKFinanceDataValidator:
    """
    Validates scraped financial news data for quality, relevance, and completeness.
    Ensures only high-quality UK financial content enters the ML pipeline.
    """
    
    def __init__(self):
        """Initialize validator with UK-specific criteria"""
        self.logger = self._setup_logger()
        
        # Validation thresholds
        self.min_title_length = 10
        self.max_title_length = 200
        self.min_article_length = 20
        self.max_days_old = 365  # Articles older than 1 year are considered stale
        
        # UK finance indicators
        self.uk_indicators = [
            'uk', 'british', 'britain', 'london', 'sterling', '£', 'gbp',
            'ftse', 'lse', 'bank of england', 'boe', 'fca', 'brexit'
        ]
        
        # Financial indicators
        self.finance_indicators = [
            'bank', 'profit', 'loss', 'earnings', 'revenue', 'share', 'stock',
            'market', 'investor', 'trading', 'financial', 'economy', 'growth',
            'decline', 'rise', 'fall', 'dividend', 'acquisition', 'merger'
        ]
        
        # Blacklist patterns (non-financial content)
        self.blacklist_patterns = [
            r'cookie\s*policy', r'privacy\s*policy', r'terms\s*of\s*service',
            r'subscribe\s*now', r'sign\s*up', r'advertisement', r'sponsored'
        ]
        
        # Quality metrics storage
        self.validation_stats = {
            'total_articles': 0,
            'valid_articles': 0,
            'invalid_reasons': Counter(),
            'sources': Counter(),
            'uk_relevance_scores': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for validator"""
        logger = logging.getLogger('DataValidator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def validate_article(self, article: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single article for quality and relevance.
        
        Args:
            article: Article dictionary with title, url, source, etc.
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. Check required fields
        required_fields = ['title', 'url', 'source']
        for field in required_fields:
            if field not in article or not article[field]:
                issues.append(f"missing_{field}")
                
        if issues:
            return False, issues
            
        # 2. Validate title
        title = article['title']
        if len(title) < self.min_title_length:
            issues.append("title_too_short")
        elif len(title) > self.max_title_length:
            issues.append("title_too_long")
            
        # 3. Check for blacklisted content
        title_lower = title.lower()
        for pattern in self.blacklist_patterns:
            if re.search(pattern, title_lower):
                issues.append(f"blacklisted_pattern_{pattern}")
                
        # 4. Validate URL
        url = article['url']
        if not url.startswith(('http://', 'https://')):
            issues.append("invalid_url_format")
            
        # 5. Check timestamp if available
        if 'scraped_at' in article:
            try:
                scraped_time = datetime.fromisoformat(article['scraped_at'].replace('Z', '+00:00'))
                age_days = (datetime.now() - scraped_time.replace(tzinfo=None)).days
                if age_days > self.max_days_old:
                    issues.append("article_too_old")
            except:
                issues.append("invalid_timestamp")
                
        # 6. Check UK relevance
        uk_score = self._calculate_uk_relevance_score(title)
        if uk_score < 0.1:  # No UK indicators
            issues.append("not_uk_relevant")
            
        # 7. Check financial relevance
        finance_score = self._calculate_finance_relevance_score(title)
        if finance_score < 0.1:  # No financial indicators
            issues.append("not_finance_relevant")
            
        # 8. Check for duplicates (simplified - just exact title match)
        # In production, would use more sophisticated deduplication
        
        return len(issues) == 0, issues
        
    def _calculate_uk_relevance_score(self, text: str) -> float:
        """
        Calculate UK relevance score (0-1).
        
        Args:
            text: Text to analyze
            
        Returns:
            Relevance score
        """
        text_lower = text.lower()
        matches = sum(1 for indicator in self.uk_indicators if indicator in text_lower)
        score = min(matches / 3.0, 1.0)  # Normalize to 0-1
        return score
        
    def _calculate_finance_relevance_score(self, text: str) -> float:
        """
        Calculate financial content relevance score (0-1).
        
        Args:
            text: Text to analyze
            
        Returns:
            Relevance score
        """
        text_lower = text.lower()
        matches = sum(1 for indicator in self.finance_indicators if indicator in text_lower)
        score = min(matches / 5.0, 1.0)  # Normalize to 0-1
        return score
        
    def validate_dataset(self, articles: List[Dict]) -> Dict:
        """
        Validate entire dataset of articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating {len(articles)} articles...")
        
        valid_articles = []
        invalid_articles = []
        
        for article in articles:
            is_valid, issues = self.validate_article(article)
            
            if is_valid:
                valid_articles.append(article)
                # Add relevance scores
                article['uk_relevance_score'] = self._calculate_uk_relevance_score(article['title'])
                article['finance_relevance_score'] = self._calculate_finance_relevance_score(article['title'])
            else:
                invalid_articles.append({
                    'article': article,
                    'issues': issues
                })
                # Update stats
                for issue in issues:
                    self.validation_stats['invalid_reasons'][issue] += 1
                    
            # Track source
            if 'source' in article:
                self.validation_stats['sources'][article['source']] += 1
                
        # Update overall stats
        self.validation_stats['total_articles'] = len(articles)
        self.validation_stats['valid_articles'] = len(valid_articles)
        
        # Calculate quality metrics
        validity_rate = len(valid_articles) / len(articles) if articles else 0
        
        # UK relevance distribution
        uk_scores = [a['uk_relevance_score'] for a in valid_articles if 'uk_relevance_score' in a]
        avg_uk_relevance = sum(uk_scores) / len(uk_scores) if uk_scores else 0
        
        # Financial relevance distribution
        fin_scores = [a['finance_relevance_score'] for a in valid_articles if 'finance_relevance_score' in a]
        avg_finance_relevance = sum(fin_scores) / len(fin_scores) if fin_scores else 0
        
        validation_results = {
            'valid_articles': valid_articles,
            'invalid_articles': invalid_articles,
            'statistics': {
                'total_articles': len(articles),
                'valid_count': len(valid_articles),
                'invalid_count': len(invalid_articles),
                'validity_rate': validity_rate,
                'avg_uk_relevance': avg_uk_relevance,
                'avg_finance_relevance': avg_finance_relevance,
                'invalid_reasons': dict(self.validation_stats['invalid_reasons']),
                'source_distribution': dict(self.validation_stats['sources'])
            }
        }
        
        return validation_results
        
    def generate_quality_report(self, validation_results: Dict, save_path: str = "data_quality_report.txt"):
        """
        Generate human-readable quality report.
        
        Args:
            validation_results: Results from validate_dataset
            save_path: Path to save report
        """
        stats = validation_results['statistics']
        
        report = []
        report.append("DATA QUALITY VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall statistics
        report.append("\n\nOVERALL STATISTICS:")
        report.append(f"Total articles processed: {stats['total_articles']}")
        report.append(f"Valid articles: {stats['valid_count']} ({stats['validity_rate']:.1%})")
        report.append(f"Invalid articles: {stats['invalid_count']}")
        
        # Relevance scores
        report.append("\n\nRELEVANCE METRICS:")
        report.append(f"Average UK relevance: {stats['avg_uk_relevance']:.2f}")
        report.append(f"Average finance relevance: {stats['avg_finance_relevance']:.2f}")
        
        # Source distribution
        report.append("\n\nSOURCE DISTRIBUTION:")
        for source, count in sorted(stats['source_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
            report.append(f"  {source}: {count} articles")
            
        # Invalid reasons
        if stats['invalid_reasons']:
            report.append("\n\nINVALIDATION REASONS:")
            for reason, count in sorted(stats['invalid_reasons'].items(), 
                                      key=lambda x: x[1], reverse=True):
                report.append(f"  {reason}: {count} occurrences")
                
        # Recommendations
        report.append("\n\nRECOMMENDATIONS:")
        if stats['validity_rate'] < 0.8:
            report.append("⚠️  Low validity rate - review scraping logic")
        if stats['avg_uk_relevance'] < 0.5:
            report.append("⚠️  Low UK relevance - adjust source selection")
        if stats['avg_finance_relevance'] < 0.5:
            report.append("⚠️  Low finance relevance - refine content filters")
            
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
            
        print(f"\n📊 Quality report saved to: {save_path}")
        print("\nReport Summary:")
        print(report_text)
        
        return report_text
        
    def clean_and_save_valid_data(self, 
                                 validation_results: Dict,
                                 output_path: str = "data/raw/uk_news/validated_articles.json"):
        """
        Save only valid, cleaned articles.
        
        Args:
            validation_results: Results from validate_dataset
            output_path: Path to save cleaned data
        """
        valid_articles = validation_results['valid_articles']
        
        # Additional cleaning
        for article in valid_articles:
            # Clean title
            article['title'] = article['title'].strip()
            article['title'] = re.sub(r'\s+', ' ', article['title'])  # Normalize whitespace
            
            # Ensure consistent timestamp format
            if 'scraped_at' not in article:
                article['scraped_at'] = datetime.now().isoformat()
                
        # Save with metadata
        output_data = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'total_articles': len(valid_articles),
                'avg_uk_relevance': validation_results['statistics']['avg_uk_relevance'],
                'avg_finance_relevance': validation_results['statistics']['avg_finance_relevance'],
                'sources': list(validation_results['statistics']['source_distribution'].keys())
            },
            'articles': valid_articles
        }
        
        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved {len(valid_articles)} validated articles to {output_path}")
        
        return output_path
        
    def check_minimum_requirements(self, valid_articles: List[Dict]) -> Tuple[bool, str]:
        """
        Check if we have minimum data for training.
        
        Args:
            valid_articles: List of validated articles
            
        Returns:
            Tuple of (meets_requirements, message)
        """
        min_articles = 100  # Minimum for decent training
        min_sources = 2     # Need diversity
        
        num_articles = len(valid_articles)
        num_sources = len(set(a['source'] for a in valid_articles if 'source' in a))
        
        if num_articles < min_articles:
            return False, f"Insufficient articles: {num_articles} < {min_articles} required"
            
        if num_sources < min_sources:
            return False, f"Insufficient source diversity: {num_sources} < {min_sources} required"
            
        # Check sentiment diversity (need examples of each)
        # This is a simple heuristic - in reality would need labeled data
        has_positive = any('rise' in a['title'].lower() or 'gain' in a['title'].lower() 
                          for a in valid_articles[:50])
        has_negative = any('fall' in a['title'].lower() or 'loss' in a['title'].lower() 
                          for a in valid_articles[:50])
                          
        if not (has_positive and has_negative):
            return False, "Insufficient sentiment diversity in dataset"
            
        return True, "Dataset meets minimum requirements for training"


def main():
    """Test data validation"""
    validator = UKFinanceDataValidator()
    
    # Load scraped data
    data_file = "data/raw/uk_news/latest_articles.json"
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            articles = data.get('articles', [])
    except FileNotFoundError:
        print(f"Creating sample data for testing (no scraped data found)")
        # Create sample data
        articles = [
            {
                "title": "Barclays profits surge 20% as UK economy recovers",
                "url": "https://example.com/article1",
                "source": "Financial Times",
                "scraped_at": datetime.now().isoformat()
            },
            {
                "title": "Cookie Policy",  # Should be filtered
                "url": "https://example.com/cookies",
                "source": "BBC",
                "scraped_at": datetime.now().isoformat()
            },
            {
                "title": "FTSE 100 closes at record high",
                "url": "https://example.com/article2",
                "source": "Reuters",
                "scraped_at": datetime.now().isoformat()
            }
        ]
        
    print(f"Validating {len(articles)} articles...\n")
    
    # Validate dataset
    validation_results = validator.validate_dataset(articles)
    
    # Generate report
    validator.generate_quality_report(validation_results)
    
    # Check requirements
    meets_req, message = validator.check_minimum_requirements(validation_results['valid_articles'])
    print(f"\n✓ Training Requirements: {message}")
    
    # Save cleaned data
    if validation_results['valid_articles']:
        output_path = validator.clean_and_save_valid_data(validation_results)
        print(f"\n✓ Cleaned data saved to: {output_path}")
        

if __name__ == "__main__":
    main()