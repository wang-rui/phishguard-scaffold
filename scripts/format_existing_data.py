#!/usr/bin/env python3
"""
Data Formatting Script for Existing Twitter Datasets

This script helps convert existing Twitter phishing datasets 
to the format expected by the PhishGuard framework.

Supports common dataset formats and provides flexible column mapping.
"""

import pandas as pd
import numpy as np
import re
import argparse
from datetime import datetime
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self):
        # Common column name mappings from various datasets
        self.column_mappings = {
            'text_columns': ['text', 'tweet', 'content', 'message', 'tweet_text', 'full_text'],
            'label_columns': ['label', 'class', 'is_phishing', 'phishing', 'target', 'classification'],
            'user_columns': ['user_id', 'author_id', 'userid', 'user', 'screen_name', 'username'],
            'time_columns': ['timestamp', 'created_at', 'date', 'time', 'datetime', 'tweet_created_at'],
            'url_columns': ['url', 'urls', 'expanded_url', 'link', 'links']
        }
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Automatically detect column names from common patterns."""
        detected = {}
        
        for column_type, possible_names in self.column_mappings.items():
            detected_column = None
            for col in df.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    detected_column = col
                    break
            detected[column_type.replace('_columns', '')] = detected_column
        
        logger.info("Detected columns:")
        for k, v in detected.items():
            logger.info(f"  {k}: {v}")
        
        return detected
    
    def standardize_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Standardize labels to 0 (legitimate) and 1 (phishing)."""
        df = df.copy()
        
        # Get unique values
        unique_labels = df[label_col].unique()
        logger.info(f"Original labels: {unique_labels}")
        
        # Common label mappings
        label_map = {}
        
        for label in unique_labels:
            if pd.isna(label):
                continue
                
            label_str = str(label).lower().strip()
            
            # Phishing indicators
            if label_str in ['1', 'true', 'phishing', 'spam', 'malicious', 'positive', 'yes']:
                label_map[label] = 1
            # Legitimate indicators  
            elif label_str in ['0', 'false', 'legitimate', 'ham', 'benign', 'negative', 'no']:
                label_map[label] = 0
            else:
                # Ask user or make best guess
                logger.warning(f"Unknown label '{label}' - assuming legitimate (0)")
                label_map[label] = 0
        
        df[label_col] = df[label_col].map(label_map)
        logger.info(f"Label mapping: {label_map}")
        
        return df
    
    def standardize_timestamp(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Convert timestamp to ISO format."""
        df = df.copy()
        
        try:
            # Try to parse as datetime
            df[time_col] = pd.to_datetime(df[time_col])
            df[time_col] = df[time_col].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.info("Successfully standardized timestamps")
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")
            # Generate fake timestamps if needed
            start_date = datetime(2024, 1, 1)
            df[time_col] = pd.date_range(start=start_date, periods=len(df), freq='H').strftime('%Y-%m-%dT%H:%M:%SZ')
            logger.info("Generated synthetic timestamps")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters (keep emojis)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def extract_urls(self, text: str) -> str:
        """Extract URLs from text."""
        if pd.isna(text):
            return ""
        
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, str(text))
        
        return urls[0] if urls else ""
    
    def generate_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate user IDs if not present."""
        df = df.copy()
        
        # Create synthetic user IDs
        num_users = max(100, len(df) // 10)  # Assume 10 tweets per user on average
        user_ids = [f"user_{i:06d}" for i in range(1, num_users + 1)]
        
        # Randomly assign users (some users will have multiple tweets)
        np.random.seed(42)  # For reproducibility
        df['user_id'] = np.random.choice(user_ids, size=len(df))
        
        logger.info(f"Generated {len(user_ids)} synthetic user IDs")
        return df
    
    def format_dataset(self, input_path: str, output_path: str, 
                      column_overrides: Optional[Dict] = None) -> None:
        """Main formatting function."""
        
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Detect columns
        detected = self.detect_columns(df)
        
        # Apply manual overrides if provided
        if column_overrides:
            detected.update(column_overrides)
            logger.info(f"Applied column overrides: {column_overrides}")
        
        # Check required columns
        if not detected['text']:
            raise ValueError("Could not detect text column. Please specify manually.")
        if not detected['label']:
            raise ValueError("Could not detect label column. Please specify manually.")
        
        # Start building output DataFrame
        output_df = pd.DataFrame()
        
        # Text column
        output_df['text'] = df[detected['text']].apply(self.clean_text)
        
        # Label column
        temp_df = self.standardize_labels(df, detected['label'])
        output_df['label'] = temp_df[detected['label']]
        
        # User ID column
        if detected['user']:
            output_df['user_id'] = df[detected['user']].astype(str)
        else:
            logger.info("No user column detected, generating synthetic user IDs")
            temp_df = self.generate_user_ids(df)
            output_df['user_id'] = temp_df['user_id']
        
        # Timestamp column
        if detected['time']:
            temp_df = self.standardize_timestamp(df, detected['time'])
            output_df['timestamp'] = temp_df[detected['time']]
        else:
            logger.info("No timestamp column detected, generating synthetic timestamps")
            start_date = datetime(2024, 1, 1)
            output_df['timestamp'] = pd.date_range(start=start_date, periods=len(output_df), freq='H').strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # URL column
        if detected['url']:
            output_df['url'] = df[detected['url']].fillna('')
        else:
            logger.info("No URL column detected, extracting from text")
            output_df['url'] = output_df['text'].apply(self.extract_urls)
        
        # Parent user ID (for retweets/replies) - usually not in datasets
        output_df['parent_user_id'] = ''
        
        # Remove rows with empty text or invalid labels
        initial_count = len(output_df)
        output_df = output_df[
            (output_df['text'].str.len() >= 5) &
            (output_df['label'].isin([0, 1]))
        ]
        final_count = len(output_df)
        
        if initial_count != final_count:
            logger.info(f"Filtered out {initial_count - final_count} invalid rows")
        
        # Balance check
        label_counts = output_df['label'].value_counts()
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Save formatted dataset
        output_df.to_csv(output_path, index=False)
        logger.info(f"Saved formatted dataset to {output_path}")
        logger.info(f"Final dataset: {len(output_df)} rows")
        
        return output_df

def main():
    parser = argparse.ArgumentParser(description='Format existing Twitter dataset for PhishGuard')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='data/tweets.csv', help='Output CSV file path')
    parser.add_argument('--text-col', help='Text column name (if not auto-detected)')
    parser.add_argument('--label-col', help='Label column name (if not auto-detected)')
    parser.add_argument('--user-col', help='User ID column name (if not auto-detected)')
    parser.add_argument('--time-col', help='Timestamp column name (if not auto-detected)')
    parser.add_argument('--url-col', help='URL column name (if not auto-detected)')
    
    args = parser.parse_args()
    
    # Build column overrides
    overrides = {}
    if args.text_col:
        overrides['text'] = args.text_col
    if args.label_col:
        overrides['label'] = args.label_col
    if args.user_col:
        overrides['user'] = args.user_col
    if args.time_col:
        overrides['time'] = args.time_col
    if args.url_col:
        overrides['url'] = args.url_col
    
    # Format dataset
    formatter = DataFormatter()
    try:
        formatter.format_dataset(args.input, args.output, overrides)
        logger.info("✅ Dataset formatting complete!")
        logger.info("You can now run: python -m training.train --config configs/config.yaml")
    except Exception as e:
        logger.error(f"❌ Error formatting dataset: {e}")

if __name__ == "__main__":
    main()
