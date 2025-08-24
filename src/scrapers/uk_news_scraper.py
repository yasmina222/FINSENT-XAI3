"""
UK Financial News Scraper for FINSENT-XAI
Collects articles from UK financial news sources
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import os
from typing import List, Dict
import logging
from urllib.robotparser import RobotFileParser

class UKFinancialNewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.setup_logging()
        self.setup_directories()
        
        # FTSE 100 companies for filtering
        self.ftse_companies = [
            'Barclays', 'HSBC', 'Lloyds', 'NatWest', 'Standard Chartered',
            'BP', 'Shell', 'AstraZeneca', 'GlaxoSmithKline', 'Unilever',
            'Tesco', 'Sainsbury', 'Vodafone', 'BT Group', 'Rolls-Royce',
            'BAE Systems', 'Diageo', 'British American Tobacco', 'Reckitt',
            'London Stock Exchange', 'Aviva', 'Prudential', 'Legal & General'
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraping.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs('data/raw/uk_news', exist_ok=True)
        os.makedirs('data/raw/cache', exist_ok=True)
        
    def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        try:
            rp = RobotFileParser()
            rp.set_url(url + "/robots.txt")
            rp.read()
            return rp.can_fetch(self.headers['User-Agent'], url)
        except:
            return True  # If can't check, assume allowed
            
    def is_relevant_article(self, title: str, content: str = "") -> bool:
        """Check if article is relevant to UK finance"""
        text = (title + " " + content).lower()
        
        # Check for company mentions
        for company in self.ftse_companies:
            if company.lower() in text:
                return True
                
        # Check for UK financial terms
        uk_terms = ['ftse', 'london stock', 'sterling', 'gilt', 'uk economy', 
                    'bank of england', 'chancellor', 'brexit', 'city of london']
        
        return any(term in text for term in uk_terms)
        
    def scrape_bbc_business(self) -> List[Dict]:
        """Scrape BBC Business section"""
        articles = []
        url = "https://www.bbc.com/news/business"
        
        try:
            self.logger.info("Scraping BBC Business...")
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                for article in soup.find_all('a', {'class': 'gs-c-promo-heading'}):
                    title = article.get_text(strip=True)
                    link = article.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.bbc.com' + link
                        
                    if title and self.is_relevant_article(title):
                        articles.append({
                            'title': title,
                            'url': link,
                            'source': 'BBC Business',
                            'scraped_at': datetime.now().isoformat(),
                            'category': 'uk_finance'
                        })
                        
                self.logger.info(f"Found {len(articles)} relevant BBC articles")
                
        except Exception as e:
            self.logger.error(f"Error scraping BBC: {e}")
            
        return articles
        
    def scrape_guardian_business(self) -> List[Dict]:
        """Scrape Guardian Business section"""
        articles = []
        url = "https://www.theguardian.com/uk/business"
        
        try:
            self.logger.info("Scraping Guardian Business...")
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article headlines
                for article in soup.find_all('h3', {'class': 'dcr-lv2v9o'}):
                    link_elem = article.find('a')
                    if link_elem:
                        title = link_elem.get_text(strip=True)
                        link = link_elem.get('href', '')
                        
                        if link and not link.startswith('http'):
                            link = 'https://www.theguardian.com' + link
                            
                        if title and self.is_relevant_article(title):
                            articles.append({
                                'title': title,
                                'url': link,
                                'source': 'Guardian Business',
                                'scraped_at': datetime.now().isoformat(),
                                'category': 'uk_finance'
                            })
                            
                self.logger.info(f"Found {len(articles)} relevant Guardian articles")
                
        except Exception as e:
            self.logger.error(f"Error scraping Guardian: {e}")
            
        return articles
        
    def scrape_ft_headlines(self) -> List[Dict]:
        """Scrape Financial Times headlines (limited without subscription)"""
        articles = []
        # Using Google News RSS for FT headlines
        url = "https://news.google.com/rss/search?q=site:ft.com+UK+finance+FTSE&hl=en-GB&gl=GB&ceid=GB:en"
        
        try:
            self.logger.info("Scraping FT headlines via RSS...")
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                
                for item in soup.find_all('item')[:20]:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    
                    if title_elem and link_elem:
                        title = title_elem.text.split(' - ')[0]  # Remove source
                        
                        if self.is_relevant_article(title):
                            articles.append({
                                'title': title,
                                'url': link_elem.text,
                                'source': 'Financial Times',
                                'scraped_at': datetime.now().isoformat(),
                                'category': 'uk_finance'
                            })
                            
                self.logger.info(f"Found {len(articles)} relevant FT articles")
                
        except Exception as e:
            self.logger.error(f"Error scraping FT: {e}")
            
        return articles
        
    def scrape_reuters_uk(self) -> List[Dict]:
        """Scrape Reuters UK Business"""
        articles = []
        url = "https://www.reuters.com/world/uk/"
        
        try:
            self.logger.info("Scraping Reuters UK...")
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                for article in soup.find_all('a', {'data-testid': 'Heading'}):
                    title = article.get_text(strip=True)
                    link = article.get('href', '')
                    
                    if link and not link.startswith('http'):
                        link = 'https://www.reuters.com' + link
                        
                    if title and self.is_relevant_article(title):
                        articles.append({
                            'title': title,
                            'url': link,
                            'source': 'Reuters UK',
                            'scraped_at': datetime.now().isoformat(),
                            'category': 'uk_finance'
                        })
                        
                self.logger.info(f"Found {len(articles)} relevant Reuters articles")
                
        except Exception as e:
            self.logger.error(f"Error scraping Reuters: {e}")
            
        return articles
        
    def scrape_all_sources(self) -> List[Dict]:
        """Scrape all configured news sources"""
        all_articles = []
        
        # List of scraper methods
        scrapers = [
            self.scrape_bbc_business,
            self.scrape_guardian_business,
            self.scrape_ft_headlines,
            self.scrape_reuters_uk
        ]
        
        for scraper in scrapers:
            try:
                articles = scraper()
                all_articles.extend(articles)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in {scraper.__name__}: {e}")
                
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
                
        return unique_articles
        
    def save_articles(self, articles: List[Dict]) -> str:
        """Save articles to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/raw/uk_news/articles_{timestamp}.json"
        
        # Add metadata
        data = {
            'metadata': {
                'scraped_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'sources': list(set(a['source'] for a in articles)),
                'scraper_version': '1.0'
            },
            'articles': articles
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved {len(articles)} articles to {filename}")
        
        # Also save as latest
        latest_file = "data/raw/uk_news/latest_articles.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return filename
        
    def run(self, min_articles: int = 100):
        """Main scraping process"""
        self.logger.info("Starting UK financial news scraping...")
        
        articles = self.scrape_all_sources()
        
        if len(articles) < min_articles:
            self.logger.warning(f"Only found {len(articles)} articles, minimum is {min_articles}")
            self.logger.info("Consider running again later or adjusting filters")
        
        if articles:
            filename = self.save_articles(articles)
            print(f"\n✅ Scraping complete! Found {len(articles)} articles")
            print(f"📁 Saved to: {filename}")
        else:
            self.logger.error("No articles found!")
            
        return articles


def main():
    """Run the scraper"""
    print("🔍 FINSENT-XAI UK News Scraper")
    print("=" * 50)
    
    scraper = UKFinancialNewsScraper()
    articles = scraper.run()
    
    # Print summary
    if articles:
        print("\n📊 Summary by Source:")
        source_counts = {}
        for article in articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
            
        for source, count in sorted(source_counts.items()):
            print(f"  - {source}: {count} articles")
            
        print("\n📰 Sample Articles:")
        for article in articles[:5]:
            print(f"  - {article['title'][:80]}...")
            

if __name__ == "__main__":
    main()