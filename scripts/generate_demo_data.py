#!/usr/bin/env python3
"""
Demo Data Generator for PhishGuard Framework

This script generates realistic synthetic Twitter data for demonstration
and testing purposes when real Twitter data is not available.

The generated data mimics real phishing patterns and social interactions.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re
import logging
from typing import List, Tuple
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Phishing templates and patterns
        self.phishing_templates = [
            "🚨 URGENT: Your {service} account has been {action}! Verify immediately: {url}",
            "Congratulations! You've won a {prize}! Claim now: {url} Limited time offer!",
            "Security Alert: Suspicious activity on your {service}. Confirm identity: {url}",
            "Your {service} payment failed. Update info to avoid suspension: {url}",
            "Final Notice: {service} account expires today. Renew now: {url}",
            "BREAKING: {company} is giving away {prize}! Get yours: {url}",
            "⚠️ Account Locked: Your {service} needs immediate verification: {url}",
            "Limited Time: Get {prize} absolutely FREE! Click here: {url}",
            "Your package could not be delivered. Reschedule: {url}",
            "Tax refund of ${amount} pending. Claim here: {url}"
        ]
        
        self.legitimate_templates = [
            "Just finished reading an interesting article about {topic}. Thoughts?",
            "Beautiful sunset today in {location}! 🌅 #photography",
            "Working on a new {project} project. Exciting times ahead!",
            "Reminder: {event} is coming up next week. Looking forward to it!",
            "Great meeting with the {team} team today. Productive discussions!",
            "Weather update: It's {weather} in {location} today",
            "Celebrating {milestone} with the family today! 🎉",
            "Reading about {topic} - fascinating developments in the field",
            "Coffee break thoughts: {opinion} What do you think?",
            "Weekend plans include {activity}. Should be fun!"
        ]
        
        # Data for template filling
        self.services = ["PayPal", "Amazon", "Netflix", "Bank of America", "Apple", "Google", "Microsoft", "Facebook"]
        self.actions = ["suspended", "locked", "compromised", "flagged", "restricted"]
        self.prizes = ["iPhone 15", "$1000 gift card", "vacation package", "laptop", "smart TV"]
        self.companies = ["Apple", "Google", "Amazon", "Tesla", "Microsoft"]
        self.amounts = ["850", "1,200", "500", "750", "2,100"]
        
        self.topics = ["AI", "climate change", "space exploration", "renewable energy", "technology trends"]
        self.locations = ["New York", "California", "London", "Tokyo", "Sydney"]
        self.projects = ["machine learning", "mobile app", "website", "data analysis", "research"]
        self.events = ["conference", "workshop", "meetup", "webinar", "presentation"]
        self.teams = ["marketing", "engineering", "design", "sales", "product"]
        self.weather = ["sunny", "rainy", "cloudy", "windy", "snowy"]
        self.milestones = ["anniversary", "birthday", "graduation", "promotion", "achievement"]
        self.activities = ["hiking", "reading", "coding", "cooking", "traveling"]
        self.opinions = ["innovation drives progress", "collaboration is key", "learning never stops"]
        
        # Suspicious URL patterns
        self.suspicious_domains = [
            "bit.ly", "tinyurl.com", "t.co", "short.link", "click.me",
            "secure-verify.net", "account-update.org", "urgent-action.com",
            "verify-now.net", "claim-reward.org"
        ]
        
        self.legitimate_domains = [
            "twitter.com", "medium.com", "github.com", "linkedin.com", "youtube.com",
            "wikipedia.org", "news.bbc.co.uk", "cnn.com", "techcrunch.com"
        ]
    
    def generate_url(self, is_phishing: bool) -> str:
        """Generate a realistic URL."""
        if is_phishing:
            domain = random.choice(self.suspicious_domains)
            path = random.choice(["verify", "secure", "update", "claim", "confirm"])
            return f"https://{domain}/{path}{random.randint(100, 999)}"
        else:
            domain = random.choice(self.legitimate_domains)
            if random.random() < 0.3:  # 30% chance of having a URL
                return f"https://{domain}/article/{random.randint(1000, 9999)}"
            return ""
    
    def generate_tweet_text(self, is_phishing: bool) -> Tuple[str, str]:
        """Generate tweet text and associated URL."""
        if is_phishing:
            template = random.choice(self.phishing_templates)
            url = self.generate_url(True)
            
            text = template.format(
                service=random.choice(self.services),
                action=random.choice(self.actions),
                prize=random.choice(self.prizes),
                company=random.choice(self.companies),
                amount=random.choice(self.amounts),
                url=url
            )
        else:
            template = random.choice(self.legitimate_templates)
            url = self.generate_url(False)
            
            text = template.format(
                topic=random.choice(self.topics),
                location=random.choice(self.locations),
                project=random.choice(self.projects),
                event=random.choice(self.events),
                team=random.choice(self.teams),
                weather=random.choice(self.weather),
                milestone=random.choice(self.milestones),
                activity=random.choice(self.activities),
                opinion=random.choice(self.opinions)
            )
        
        return text, url
    
    def generate_users(self, num_users: int) -> List[dict]:
        """Generate user profiles with varying risk levels."""
        users = []
        
        for i in range(num_users):
            # Some users are more likely to post phishing content
            risk_level = random.choices(
                ["low", "medium", "high"], 
                weights=[0.7, 0.2, 0.1]
            )[0]
            
            user = {
                "user_id": f"user_{i+1:06d}",
                "risk_level": risk_level,
                "activity_level": random.choices(
                    ["low", "medium", "high"], 
                    weights=[0.3, 0.5, 0.2]
                )[0],
                "follower_count": random.randint(10, 10000),
                "following_count": random.randint(50, 5000)
            }
            
            users.append(user)
        
        return users
    
    def generate_tweets(self, users: List[dict], num_tweets: int) -> pd.DataFrame:
        """Generate tweet dataset."""
        tweets = []
        
        # Time span for tweets (last 6 months)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=180)
        
        for i in range(num_tweets):
            # Select user based on activity level
            high_activity_users = [u for u in users if u["activity_level"] == "high"]
            medium_activity_users = [u for u in users if u["activity_level"] == "medium"]
            low_activity_users = [u for u in users if u["activity_level"] == "low"]
            
            user = random.choices(
                high_activity_users + medium_activity_users + low_activity_users,
                weights=[3] * len(high_activity_users) + 
                        [2] * len(medium_activity_users) + 
                        [1] * len(low_activity_users)
            )[0]
            
            # Determine if tweet is phishing based on user risk level
            phishing_prob = {
                "low": 0.05,    # 5% chance
                "medium": 0.15, # 15% chance  
                "high": 0.40    # 40% chance
            }[user["risk_level"]]
            
            is_phishing = random.random() < phishing_prob
            
            # Generate content
            text, url = self.generate_tweet_text(is_phishing)
            
            # Random timestamp
            timestamp = start_time + timedelta(
                seconds=random.randint(0, int((end_time - start_time).total_seconds()))
            )
            
            # Check if this is a retweet (10% chance)
            parent_user_id = ""
            if random.random() < 0.1 and len(tweets) > 100:
                # Find a random previous tweet to retweet
                parent_tweet = random.choice(tweets[-100:])  # From recent tweets
                parent_user_id = parent_tweet["user_id"]
                text = f"RT @{parent_user_id}: {parent_tweet['text']}"
                is_phishing = parent_tweet["label"]  # Inherit label
                url = parent_tweet["url"]
            
            tweet = {
                "text": text,
                "label": int(is_phishing),
                "user_id": user["user_id"],
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "parent_user_id": parent_user_id,
                "url": url,
                "follower_count": user["follower_count"],
                "engagement_count": random.randint(0, user["follower_count"] // 10)
            }
            
            tweets.append(tweet)
        
        return pd.DataFrame(tweets)
    
    def generate_social_graph(self, users: List[dict], tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Generate social interaction graph."""
        edges = []
        
        # Create follow relationships
        for user in users:
            # Each user follows 20-200 other users
            num_follows = random.randint(20, min(200, len(users) // 2))
            followed_users = random.sample(
                [u for u in users if u["user_id"] != user["user_id"]], 
                num_follows
            )
            
            for followed in followed_users:
                # Follow probability based on similar risk levels and activity
                base_weight = 0.02
                
                # Users with similar risk levels are more likely to interact
                if user["risk_level"] == followed["risk_level"]:
                    base_weight *= 2
                
                # High activity users have stronger connections
                if user["activity_level"] == "high":
                    base_weight *= 1.5
                
                edges.append({
                    "src": user["user_id"],
                    "dst": followed["user_id"],
                    "weight": min(0.8, base_weight * random.uniform(0.5, 2.0)),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "interaction_type": "follow"
                })
        
        # Add retweet/reply edges from tweet data
        for _, tweet in tweets_df.iterrows():
            if tweet["parent_user_id"]:
                # Retweet relationship
                edges.append({
                    "src": tweet["parent_user_id"],
                    "dst": tweet["user_id"],
                    "weight": 0.15,  # Higher weight for actual interactions
                    "timestamp": tweet["timestamp"],
                    "interaction_type": "retweet"
                })
        
        return pd.DataFrame(edges)

def main():
    parser = argparse.ArgumentParser(description="Generate demo Twitter data for PhishGuard")
    parser.add_argument("--tweets", "-t", type=int, default=5000, help="Number of tweets to generate")
    parser.add_argument("--users", "-u", type=int, default=1000, help="Number of users to generate") 
    parser.add_argument("--output-dir", "-o", default="data", help="Output directory")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logger.info(f"Generating demo data with {args.tweets} tweets and {args.users} users")
    
    # Initialize generator
    generator = PhishingDataGenerator(seed=args.seed)
    
    # Generate users
    users = generator.generate_users(args.users)
    logger.info(f"Generated {len(users)} user profiles")
    
    # Generate tweets
    tweets_df = generator.generate_tweets(users, args.tweets)
    logger.info(f"Generated {len(tweets_df)} tweets")
    
    # Class distribution
    class_dist = tweets_df["label"].value_counts()
    logger.info(f"Class distribution: Legitimate: {class_dist[0]}, Phishing: {class_dist[1]}")
    
    # Generate social graph
    edges_df = generator.generate_social_graph(users, tweets_df)
    logger.info(f"Generated {len(edges_df)} social interactions")
    
    # Save datasets
    tweets_path = f"{args.output_dir}/tweets.csv"
    edges_path = f"{args.output_dir}/edges.csv"
    
    # Select required columns for tweets
    tweets_output = tweets_df[["text", "label", "user_id", "timestamp", "parent_user_id", "url"]]
    tweets_output.to_csv(tweets_path, index=False)
    
    # Select required columns for edges
    edges_output = edges_df[["src", "dst", "weight", "timestamp"]]
    edges_output.to_csv(edges_path, index=False)
    
    logger.info(f"✅ Demo data generated successfully!")
    logger.info(f"📁 Tweets saved to: {tweets_path}")
    logger.info(f"📁 Edges saved to: {edges_path}")
    logger.info(f"🔬 Dataset ready for PhishGuard training!")
    
    # Show sample data
    print("\n📋 Sample tweets:")
    print(tweets_output.head(3).to_string())
    
    print("\n📋 Sample edges:")
    print(edges_output.head(3).to_string())
    
    print(f"\n🚀 Ready to train! Run:")
    print(f"python -m training.train --config configs/config.yaml")

if __name__ == "__main__":
    main()
