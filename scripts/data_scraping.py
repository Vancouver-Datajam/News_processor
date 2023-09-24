# cretaed by: Sirpreet (Siri) Dhillon 
# original work by: Siri, Alice, Elgin, Sama 
# Project: News Processor | Vancouver DataJams 2023
# Date created: Sept 23, 2023 
# Python version: Python 3.9.13

import requests
import csv

NYT_API_KEY = "YOUR_API_HERE"  # Replace with your actual API key
GUARDIAN_API_KEY = "YOUR_API_HERE"  # Replace with your API key
NYT_API_ENDPOINT = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
GUARDIAN_API_ENDPOINT = "https://content.guardianapis.com/search"
QUERY = "climate"  # Replace with your desired query
NUM_PAGES = 5  # Number of pages to scrape

def scrape_nyt_articles(api_key, query):
    params = {
        'q': query,
        'api-key': api_key
    }
    response = requests.get(NYT_API_ENDPOINT, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data['response']['docs']

        # Extract headlines and URLs
        headlines_urls = [{'headline': article['headline']['main'], 'url': article['web_url']} for article in articles]
        return headlines_urls
    else:
        print("Failed to fetch data")
        return []
    
def scrape_guardian_articles(api_key, query, num_pages):
    articles = []
    articles_sentiment_headlines_urls = []
    for page in range(1, num_pages + 1):
        params = {
            'q': query,
            'page': page,
            'api-key': api_key
        }
        response = requests.get(GUARDIAN_API_ENDPOINT, params=params)

        if response.status_code == 200:
            data = response.json()
            articles += data['response']['results']
            for article in articles:
              # Fetch the content of each article
              article_url = article['webUrl']
              articles_sentiment_headlines_urls.append({
                      'headline': article['webTitle'],
                      'url':  article['webUrl']
                  })
        else:
            print(f"Failed to fetch data for page {page}")

    return articles_sentiment_headlines_urls

if __name__ == "__main__":
    headlines_urls = scrape_nyt_articles(NYT_API_KEY, QUERY)
    articles_sentiment_headlines_urls = scrape_guardian_articles(GUARDIAN_API_KEY, QUERY, NUM_PAGES)

    # combining the data sets
    headlines_urls = headlines_urls + articles_sentiment_headlines_urls
   
    # creating output csv 
    with open("./../data/news_headlines.csv","w",newline="") as f:  
        title = "headline,url".split(",") # quick hack
        cw = csv.DictWriter(f,title,delimiter=',', quoting=csv.QUOTE_MINIMAL)
        cw.writeheader()
        cw.writerows(headlines_urls)
