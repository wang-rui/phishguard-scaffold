# Quick Start: Using Real Twitter Data

This guide gets you up and running with real Twitter data in under 30 minutes.

## Option 1: Use Existing Dataset (Fastest)

If you have a phishing dataset already:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Format your dataset
python scripts/format_existing_data.py \
    --input your_dataset.csv \
    --output data/tweets.csv

# 3. Train the model
python -m src.training.train --config configs/config.yaml
```

## Option 2: Collect Live Twitter Data

```bash
# 1. Get Twitter API access
# - Go to developer.twitter.com
# - Create a developer account
# - Create a new app and get your Bearer Token

# 2. Set up environment
export TWITTER_BEARER_TOKEN="your_bearer_token_here"
pip install tweepy

# 3. Collect data (this takes 10-15 minutes)
python scripts/collect_twitter_data.py

# 4. Train the model
python -m src.training.train --config configs/config.yaml
```

## Option 3: Download Public Dataset

```bash
# Example with Kaggle dataset
pip install kaggle

# Download a phishing detection dataset
kaggle datasets download -d username/phishing-dataset
unzip phishing-dataset.zip

# Format for PhishGuard
python scripts/format_existing_data.py \
    --input dataset.csv \
    --output data/tweets.csv \
    --text-col "text_column_name" \
    --label-col "label_column_name"

# Train
python -m src.training.train --config configs/config.yaml
```

## Validate Your Data

After setting up data, verify it's working:

```bash
# Check data format
head -n 5 data/tweets.csv

# Should show:
# text,label,user_id,timestamp,parent_user_id,url
# "Some tweet text",0,user_001,2024-01-01T00:00:00Z,,
# ...

# Run quick validation
python -c "
import pandas as pd
df = pd.read_csv('data/tweets.csv')
print(f'✅ Loaded {len(df)} tweets')
print(f'✅ Columns: {list(df.columns)}')
print(f'✅ Labels: {df.label.value_counts().to_dict()}')
print('Data looks good! Ready for training.')
"
```

## Expected Training Results

With real data (10k+ tweets):
- **Training time**: 1-3 hours (GPU) or 4-8 hours (CPU)
- **Accuracy**: 85-95%
- **F1 Score**: 80-92%
- **Propagation reduction**: 15-40%

## Need Help?

Common issues and solutions:

**"No column detected"** → Use explicit column names:
```bash
python scripts/format_existing_data.py \
    --input data.csv \
    --text-col "your_text_column" \
    --label-col "your_label_column"
```

**"Out of memory"** → Reduce batch size in `configs/config.yaml`:
```yaml
train:
  batch_size: 4  # Reduce from 8
```

**"Twitter API error"** → Check your bearer token:
```bash
echo $TWITTER_BEARER_TOKEN  # Should show your token
```

That's it! Your PhishGuard system is ready to detect phishing with real data.
