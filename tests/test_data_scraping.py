import pytest
from ..scripts.data_scraping import scrape_nyt_articles, scrape_guardian_articles

# Replace these placeholders with your actual API keys for testing
NYT_API_KEY = "YOUR_NYT_API_KEY"
GUARDIAN_API_KEY = "YOUR_GUARDIAN_API_KEY"
QUERY = "climate"
NUM_PAGES = 5

def test_scrape_nyt_articles():
    # Test if scrape_nyt_articles returns a non-empty list
    headlines_urls = scrape_nyt_articles(NYT_API_KEY, QUERY)
    assert isinstance(headlines_urls, list)
    assert len(headlines_urls) > 0

def test_scrape_guardian_articles():
    # Test if scrape_guardian_articles returns a non-empty list
    articles_sentiment_headlines_urls = scrape_guardian_articles(GUARDIAN_API_KEY, QUERY, NUM_PAGES)
    assert isinstance(articles_sentiment_headlines_urls, list)
    assert len(articles_sentiment_headlines_urls) > 0

if __name__ == "__main__":
    pytest.main()
