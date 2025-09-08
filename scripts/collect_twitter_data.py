#!/usr/bin/env python3
"""
Twitter Data Collection Script for PhishGuard Framework

This script helps collect real Twitter data using the Twitter API v2
and formats it for the PhishGuard phishing detection system.

Requirements:
- Twitter API Bearer Token
- tweepy library: pip install tweepy
"""

import tweepy
import pandas as pd
from typing import List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataCollector:
    def __init__(self, bearer_token: str):
        """Initialize Twitter API client."""
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        self.phishing_keywords = [
            # Common phishing terms
            "free iphone", "urgent account", "verify now", "click here",
            "suspended account", "security alert", "confirm identity",
            "limited time offer", "act now", "claim reward",
            # Suspicious URL patterns
            "bit.ly", "tinyurl", "shorturl", "t.co"
        ]
        self.legitimate_keywords = [
            # News and general content
            "breaking news", "weather update", "sports", "technology",
            "education", "health", "science", "business news"
        ]
    
    def collect_phishing_tweets(self, count: int = 1000) -> List[Dict]:
        """Collect potential phishing tweets based on keywords."""
        phishing_tweets = []
        
        logger.info(f"Collecting {count} potential phishing tweets...")
        
        for keyword in self.phishing_keywords:
            try:
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=f'"{keyword}" -is:retweet lang:en',
                    tweet_fields=['created_at', 'author_id', 'context_annotations', 
                                'entities', 'public_metrics', 'referenced_tweets'],
                    user_fields=['public_metrics', 'verified'],
                    expansions=['author_id', 'referenced_tweets.id'],
                    max_results=min(100, count // len(self.phishing_keywords))
                ).flatten(limit=count // len(self.phishing_keywords))
                
                for tweet in tweets:
                    # Extract URLs from tweet
                    urls = []
                    if tweet.entities and 'urls' in tweet.entities:
                        urls = [url['expanded_url'] for url in tweet.entities['urls']]
                    
                    phishing_tweets.append({
                        'text': tweet.text,
                        'label': 1,  # Potential phishing (requires manual verification)
                        'user_id': str(tweet.author_id),
                        'timestamp': tweet.created_at.isoformat() + 'Z',
                        'parent_user_id': self._get_parent_user_id(tweet),
                        'url': urls[0] if urls else '',
                        'engagement_count': self._get_engagement_count(tweet),
                        'tweet_id': str(tweet.id)
                    })
                
            except Exception as e:
                logger.warning(f"Error collecting tweets for '{keyword}': {e}")
                continue
        
        logger.info(f"Collected {len(phishing_tweets)} potential phishing tweets")
        return phishing_tweets
    
    def collect_legitimate_tweets(self, count: int = 1000) -> List[Dict]:
        """Collect legitimate tweets for balanced dataset."""
        legitimate_tweets = []
        
        logger.info(f"Collecting {count} legitimate tweets...")
        
        for keyword in self.legitimate_keywords:
            try:
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=f'"{keyword}" -is:retweet lang:en',
                    tweet_fields=['created_at', 'author_id', 'context_annotations',
                                'entities', 'public_metrics', 'referenced_tweets'],
                    user_fields=['public_metrics', 'verified'],
                    expansions=['author_id', 'referenced_tweets.id'],
                    max_results=min(100, count // len(self.legitimate_keywords))
                ).flatten(limit=count // len(self.legitimate_keywords))
                
                for tweet in tweets:
                    # Filter out tweets that might be phishing
                    if self._is_likely_phishing(tweet.text):
                        continue
                    
                    urls = []
                    if tweet.entities and 'urls' in tweet.entities:
                        urls = [url['expanded_url'] for url in tweet.entities['urls']]
                    
                    legitimate_tweets.append({
                        'text': tweet.text,
                        'label': 0,  # Legitimate
                        'user_id': str(tweet.author_id),
                        'timestamp': tweet.created_at.isoformat() + 'Z',
                        'parent_user_id': self._get_parent_user_id(tweet),
                        'url': urls[0] if urls else '',
                        'engagement_count': self._get_engagement_count(tweet),
                        'tweet_id': str(tweet.id)
                    })
                
            except Exception as e:
                logger.warning(f"Error collecting tweets for '{keyword}': {e}")
                continue
        
        logger.info(f"Collected {len(legitimate_tweets)} legitimate tweets")
        return legitimate_tweets
    
    def collect_user_interactions(self, user_ids: List[str], max_users: int = 100) -> List[Dict]:
        """Collect user interaction data for social graph construction."""
        edges = []
        
        logger.info(f"Collecting interactions for {min(len(user_ids), max_users)} users...")
        
        for user_id in user_ids[:max_users]:
            try:
                # Get user's recent tweets to find interactions
                tweets = self.client.get_users_tweets(
                    id=user_id,
                    tweet_fields=['referenced_tweets', 'created_at'],
                    max_results=50
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        if tweet.referenced_tweets:
                            for ref in tweet.referenced_tweets:
                                if ref.type in ['retweeted', 'replied_to']:
                                    # Get the referenced tweet's author
                                    try:
                                        ref_tweet = self.client.get_tweet(ref.id, expansions=['author_id'])
                                        if ref_tweet.includes and 'users' in ref_tweet.includes:
                                            parent_user_id = str(ref_tweet.includes['users'][0].id)
                                            
                                            # Calculate interaction weight based on relationship
                                            weight = 0.1 if ref.type == 'retweeted' else 0.05
                                            
                                            edges.append({
                                                'src': parent_user_id,
                                                'dst': user_id,
                                                'weight': weight,
                                                'timestamp': tweet.created_at.isoformat() + 'Z',
                                                'interaction_type': ref.type
                                            })
                                    except Exception:
                                        continue
                
            except Exception as e:
                logger.warning(f"Error collecting interactions for user {user_id}: {e}")
                continue
        
        logger.info(f"Collected {len(edges)} user interactions")
        return edges
    
    def _get_parent_user_id(self, tweet) -> str:
        """Extract parent user ID for retweets/replies."""
        if tweet.referenced_tweets:
            # This is a simplified approach - in practice, you'd need to fetch the referenced tweet
            return ""  # Would need additional API call to get parent user
        return ""
    
    def _get_engagement_count(self, tweet) -> int:
        """Calculate total engagement count."""
        if tweet.public_metrics:
            return (tweet.public_metrics.get('retweet_count', 0) +
                   tweet.public_metrics.get('like_count', 0) +
                   tweet.public_metrics.get('reply_count', 0))
        return 0
    
    def _is_likely_phishing(self, text: str) -> bool:
        """Simple heuristic to filter out potential phishing from legitimate set."""
        phishing_indicators = [
            'click here', 'verify now', 'urgent', 'suspended',
            'free iphone', 'limited time', 'act now', 'claim'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in phishing_indicators)

def main():
    """Main data collection workflow."""
    
    # Twitter API credentials
    BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    if not BEARER_TOKEN:
        logger.error("Please set TWITTER_BEARER_TOKEN environment variable")
        return
    
    # Initialize collector
    collector = TwitterDataCollector(BEARER_TOKEN)
    
    # Collect tweets
    logger.info("Starting Twitter data collection...")
    phishing_tweets = collector.collect_phishing_tweets(count=2000)
    legitimate_tweets = collector.collect_legitimate_tweets(count=2000)
    
    # Combine datasets
    all_tweets = phishing_tweets + legitimate_tweets
    
    # Create DataFrame and save
    df_tweets = pd.DataFrame(all_tweets)
    
    # Basic preprocessing
    df_tweets = df_tweets.drop_duplicates(subset=['text'])
    df_tweets = df_tweets[df_tweets['text'].str.len() >= 10]  # Minimum length
    
    # Save tweets data
    output_path = 'data/tweets.csv'
    df_tweets.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df_tweets)} tweets to {output_path}")
    
    # Collect user interactions for social graph
    unique_users = df_tweets['user_id'].unique()
    edges = collector.collect_user_interactions(unique_users, max_users=200)
    
    if edges:
        df_edges = pd.DataFrame(edges)
        edges_path = 'data/edges.csv'
        df_edges.to_csv(edges_path, index=False)
        logger.info(f"Saved {len(df_edges)} edges to {edges_path}")
    
    logger.info("Data collection complete!")
    logger.info("Dataset summary:")
    logger.info(f"  - Total tweets: {len(df_tweets)}")
    logger.info(f"  - Phishing: {len(df_tweets[df_tweets['label'] == 1])}")
    logger.info(f"  - Legitimate: {len(df_tweets[df_tweets['label'] == 0])}")
    logger.info(f"  - Unique users: {len(unique_users)}")
    logger.info(f"  - User interactions: {len(edges)}")

if __name__ == "__main__":
    main()
