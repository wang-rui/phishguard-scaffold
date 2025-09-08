import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import logging
try:
    from langdetect import detect, DetectorFactory
    LANGDETECT_AVAILABLE = True
    DetectorFactory.seed = 42  # For reproducible results
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Language filtering will be skipped.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def preprocess_text(text: str) -> str:
    """Clean and standardize text data for LLaMA model input.
    
    Args:
        text: Raw text from tweet
    
    Returns:
        Cleaned and standardized text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and basic cleaning
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize encoding (remove non-printable characters but keep emojis)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text.strip()

def is_english_text(text: str) -> bool:
    """Detect if text is in English.
    
    Args:
        text: Text to analyze
    
    Returns:
        True if text is English, False otherwise
    """
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 3:
        return True  # Default to True if detection unavailable or text too short
    
    try:
        detected_lang = detect(text)
        return detected_lang == 'en'
    except Exception:
        return True  # Default to True on detection failure

def remove_duplicates(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Remove duplicate tweets based on text content.
    
    Args:
        df: DataFrame with tweet data
        text_col: Column name containing text
    
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=[text_col])
    
    # Remove near-duplicates (texts with very high similarity)
    # For efficiency, we'll use a simple approach based on cleaned text
    df['_cleaned_text'] = df[text_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x).lower().strip()))
    df = df.drop_duplicates(subset=['_cleaned_text'])
    df = df.drop(columns=['_cleaned_text'])
    
    final_count = len(df)
    logger.info(f"Removed {initial_count - final_count} duplicates ({initial_count} -> {final_count})")
    
    return df

def filter_by_length(df: pd.DataFrame, text_col: str, min_length: int = 10, max_length: int = 512) -> pd.DataFrame:
    """Filter tweets by text length.
    
    Args:
        df: DataFrame with tweet data
        text_col: Column name containing text
        min_length: Minimum text length
        max_length: Maximum text length
    
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    
    # Filter by length
    df = df[
        (df[text_col].str.len() >= min_length) & 
        (df[text_col].str.len() <= max_length)
    ]
    
    final_count = len(df)
    logger.info(f"Filtered by length ({min_length}-{max_length} chars): {initial_count} -> {final_count}")
    
    return df

def enhanced_preprocessing(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Apply comprehensive preprocessing as described in the research.
    
    This includes:
    - Removing duplicates
    - Filtering non-English tweets
    - Text cleaning and tokenization preparation
    - Length filtering
    - Standardizing encoding for LLaMA model input
    
    Args:
        df: Raw DataFrame
        cfg: Configuration dictionary
    
    Returns:
        Preprocessed DataFrame ready for model training
    """
    text_col = cfg["data"]["text_col"]
    label_col = cfg["data"]["label_col"]
    
    logger.info(f"Starting preprocessing with {len(df)} samples")
    
    # Step 1: Basic cleaning and null removal
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(int).clip(0, 1)
    logger.info(f"After null removal: {len(df)} samples")
    
    # Step 2: Text preprocessing
    df[text_col] = df[text_col].apply(preprocess_text)
    
    # Step 3: Remove duplicates (as mentioned in research)
    if cfg["data"].get("remove_duplicates", True):
        df = remove_duplicates(df, text_col)
    
    # Step 4: Filter non-English tweets (as mentioned in research)
    if cfg["data"].get("filter_non_english", True):
        initial_count = len(df)
        df = df[df[text_col].apply(is_english_text)]
        final_count = len(df)
        logger.info(f"Filtered non-English tweets: {initial_count} -> {final_count}")
    
    # Step 5: Length filtering
    min_length = cfg["data"].get("min_text_length", 10)
    max_length = cfg["data"].get("max_text_length", 512)
    df = filter_by_length(df, text_col, min_length, max_length)
    
    # Step 6: Remove any remaining empty texts
    df = df[df[text_col].str.strip() != ""]
    
    logger.info(f"Final preprocessing result: {len(df)} samples")
    
    # Ensure balanced class distribution logging
    class_dist = df[label_col].value_counts()
    logger.info(f"Class distribution: {dict(class_dist)}")
    
    return df.reset_index(drop=True)

def load_and_split(tweets_csv: str, cfg: Dict) -> SplitData:
    """Load and split Twitter phishing dataset with enhanced preprocessing.
    
    This implementation follows the research methodology:
    - Load ~100k real tweets labeled as phishing or legitimate
    - Apply comprehensive preprocessing (deduplication, language filtering, etc.)
    - Split into train/val/test (8:1:1) with balanced class distribution
    - Retain key metadata for social network graph construction
    
    Args:
        tweets_csv: Path to tweets CSV file
        cfg: Configuration dictionary
    
    Returns:
        SplitData object with train, validation, and test DataFrames
    """
    # Load raw data
    df = pd.read_csv(tweets_csv)
    logger.info(f"Loaded {len(df)} raw samples from {tweets_csv}")
    
    # Apply enhanced preprocessing
    df = enhanced_preprocessing(df, cfg)
    
    # Ensure we have enough data for splitting
    if len(df) < 100:
        logger.warning(f"Very few samples after preprocessing ({len(df)}). Consider adjusting filters.")
    
    text_col = cfg["data"]["text_col"]
    label_col = cfg["data"]["label_col"]
    
    # Split configuration
    train_size = cfg["data"]["split"]["train"]
    val_size = cfg["data"]["split"]["val"]
    test_size = cfg["data"]["split"]["test"]
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    
    # Stratified split to maintain balanced class distribution
    try:
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1-train_size), 
            stratify=df[label_col], 
            random_state=42
        )
        
        val_rel = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1-val_rel), 
            stratify=temp_df[label_col], 
            random_state=42
        )
        
        logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return SplitData(train=train_df, val=val_df, test=test_df)
        
    except ValueError as e:
        logger.error(f"Stratification failed: {e}. Falling back to random split.")
        # Fallback to random split if stratification fails
        train_df, temp_df = train_test_split(df, test_size=(1-train_size), random_state=42)
        val_rel = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, test_size=(1-val_rel), random_state=42)
        
        return SplitData(train=train_df, val=val_df, test=test_df)
