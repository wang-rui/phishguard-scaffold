# Real Twitter Data Integration Guide

This guide helps you integrate real Twitter datasets with the PhishGuard framework for authentic phishing detection research.

## Available Twitter Phishing Datasets

### 1. **PhishStorm Dataset** (Recommended)
- **Size**: ~100k tweets with phishing/legitimate labels
- **Source**: Academic research, available on request from universities
- **Features**: Text, URLs, user metadata, engagement metrics
- **Labels**: Derived from blacklists/whitelists + manual annotation

### 2. **TweepFake Dataset with Phishing Extension**
- **Size**: ~25k+ tweets including phishing samples
- **Source**: Available on Kaggle/IEEE research
- **Features**: Tweet content, user features, temporal data

### 3. **Phishing Tweets Dataset (Kaggle)**
- **URL**: https://www.kaggle.com/datasets/danielwillgeorge/phishing-email-detection
- **Size**: Variable, multiple datasets available
- **Note**: Some focus on emails but include social media adaptations

### 4. **Custom Twitter Collection via API**
- **Method**: Use Twitter API v2 to collect real-time data
- **Labeling**: Manual labeling or URL-based classification
- **Advantage**: Most current and relevant data

## Data Format Requirements

Your data must match this schema for the PhishGuard framework:

### tweets.csv
```csv
text,label,user_id,timestamp,parent_user_id,url
"Check out this amazing deal: bit.ly/scam123",1,user_12345,2024-01-01T00:00:00Z,,https://bit.ly/scam123
"Just had coffee with @friend",0,user_67890,2024-01-01T01:00:00Z,,
"URGENT: Your account will be suspended! Click here: suspicious.link",1,user_11111,2024-01-01T02:00:00Z,,https://suspicious.link
"RT @user_12345: Check out this amazing deal",1,user_22222,2024-01-01T03:00:00Z,user_12345,
```

**Required Columns:**
- `text`: Tweet content (string)
- `label`: 0 = legitimate, 1 = phishing (integer)
- `user_id`: Unique user identifier (string/integer)
- `timestamp`: ISO format timestamp (string)

**Optional Columns (enhance propagation modeling):**
- `parent_user_id`: For retweets/replies (string)
- `url`: Extracted URLs from tweet (string)
- `engagement_count`: Likes + retweets + replies (integer)
- `follower_count`: User's follower count (integer)
- `verified`: User verification status (boolean)

### edges.csv
```csv
src,dst,weight,timestamp
user_12345,user_67890,0.15,2024-01-01T00:30:00Z
user_67890,user_11111,0.08,2024-01-01T01:15:00Z
user_11111,user_22222,0.22,2024-01-01T02:45:00Z
```

**Required Columns:**
- `src`: Source user ID (string/integer)  
- `dst`: Destination user ID (string/integer)
- `weight`: Influence probability [0,1] (float)

**Optional Columns:**
- `timestamp`: When interaction occurred (string)
- `interaction_type`: follow, retweet, reply, mention (string)

## Integration Methods

### Method 1: Direct Dataset Download

If you have access to academic datasets:

```bash
# 1. Download your dataset (example for a common format)
# Place files in the data/ directory

# 2. Rename columns to match expected format
python scripts/format_twitter_data.py --input your_dataset.csv --output data/tweets.csv

# 3. Run the framework
python -m training.train --config configs/config.yaml
```

### Method 2: Twitter API Collection

Use the provided `scripts/collect_twitter_data.py` script:

```bash
# 1. Get Twitter API Bearer Token from developer.twitter.com
export TWITTER_BEARER_TOKEN="your_bearer_token_here"

# 2. Install additional dependencies
pip install tweepy

# 3. Run collection script
python scripts/collect_twitter_data.py

# 4. This will create data/tweets.csv and data/edges.csv with real data
```

### Method 3: Format Existing Dataset

If you already have a Twitter dataset, use the formatting script:

```bash
# Format existing dataset to PhishGuard format
python scripts/format_existing_data.py \
    --input your_dataset.csv \
    --output data/tweets.csv \
    --text-col "tweet_content" \
    --label-col "is_phishing"
```

## Popular Real Datasets to Use

### 1. Academic Research Datasets

**PhishStorm Dataset**
- Request from: University research groups studying social engineering
- Contains: ~100k labeled tweets with rich metadata
- Format: Usually requires formatting with our script

**TweepFake + Phishing Extension**
- Available: Some versions on IEEE/ACM digital libraries  
- Size: 25k+ tweets with user features
- Format: CSV, usually needs column mapping

### 2. Public Datasets

**Kaggle Phishing Detection Datasets**
```bash
# Example: Download from Kaggle
kaggle datasets download -d danielwillgeorge/phishing-email-detection
unzip phishing-email-detection.zip
python scripts/format_existing_data.py --input dataset.csv --output data/tweets.csv
```

**HatEval/OffensEval Social Media Datasets**
- Some contain phishing examples mixed with other content
- Available through competition organizers
- Usually require filtering for phishing-specific content

### 3. Real-time Collection

**Current Method (Recommended for Research)**
```bash
# Set up Twitter Developer Account
# Get Bearer Token from developer.twitter.com
export TWITTER_BEARER_TOKEN="your_token"

# Collect current data
python scripts/collect_twitter_data.py

# This gives you the most current and relevant data
```

## Data Quality Considerations

### For Academic Research:
- **Minimum Size**: 10k+ tweets (5k phishing, 5k legitimate)
- **Temporal Spread**: At least 3-6 months of data  
- **User Diversity**: 1k+ unique users
- **URL Coverage**: Mix of shortened and direct URLs

### For Production Systems:
- **Minimum Size**: 50k+ tweets  
- **Real-time Updates**: Continuous collection
- **Quality Control**: Manual verification of samples
- **Balanced Classes**: Equal phishing/legitimate distribution

## Validation Steps

After integrating your data, validate the setup:

```bash
# 1. Check data format
head -n 5 data/tweets.csv
head -n 5 data/edges.csv

# 2. Run a quick test
python -m training.train --config configs/config.yaml --eval_only

# 3. Check preprocessing results
python -c "
from data.dataset import load_and_split
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
split = load_and_split('data/tweets.csv', cfg)
print(f'Train: {len(split.train)}, Val: {len(split.val)}, Test: {len(split.test)}')
print(f'Class distribution: {split.train.label.value_counts().to_dict()}')
"
```

## Integration Troubleshooting

### Common Issues and Solutions:

**1. Column Name Mismatch**
```bash
# Check your data structure
head -n 1 your_dataset.csv

# Use explicit column mapping
python scripts/format_existing_data.py \
    --input your_dataset.csv \
    --text-col "actual_text_column" \
    --label-col "actual_label_column"
```

**2. Label Format Issues**
- Ensure labels are 0 (legitimate) and 1 (phishing)
- The formatter handles common formats automatically

**3. Missing Social Graph**
- If you don't have edges.csv, the system will work but with limited propagation modeling
- Use the Twitter API collector to build interaction graphs
- Or create synthetic edges based on user activity patterns

**4. Large Dataset Memory Issues**
```yaml
# In configs/config.yaml, reduce batch size:
train:
  batch_size: 4  # Reduce from 8
  gradient_checkpointing: true
  fp16: true
```

## Performance Expectations

With real Twitter data (10k+ tweets):

**Training Time:**
- CPU: 2-6 hours (depending on model size)
- GPU: 30-90 minutes with LLaMA + LoRA
- Data preprocessing: 5-15 minutes

**Memory Requirements:**
- LLaMA-2-7B: 16GB+ GPU memory (with LoRA + FP16)
- DistilBERT fallback: 4GB GPU memory
- Data loading: 2-8GB RAM (depending on dataset size)

**Expected Results:**
- Accuracy: 85-95% (depending on data quality)
- F1-Score: 80-92% (with balanced dataset)
- Propagation Control: 15-40% spread reduction

Ready to get started? Choose your data source and follow the corresponding method above!
