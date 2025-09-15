# PhishGuard Troubleshooting Guide

Common issues and their solutions when working with PhishGuard.

## 🚨 Installation Issues

### Problem: Package Installation Fails

**Error**: `pip install -r requirements.txt` fails

**Solutions**:
```bash
# 1. Update pip first
pip install --upgrade pip

# 2. Install in chunks if memory limited
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.0
pip install -r requirements.txt

# 3. Use conda for better dependency resolution
conda env create -f environment.yml
```

### Problem: CUDA/GPU Setup Issues

**Error**: `RuntimeError: CUDA out of memory` or `No CUDA device found`

**Solutions**:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU mode in config
# Edit configs/config.yaml:
model:
  device: "cpu"
  fallback_model: "distilbert-base-uncased"

train:
  batch_size: 4
  fp16: false  # Disable mixed precision for CPU
```

## 🔧 Configuration Issues

### Problem: Model Loading Fails

**Error**: `OSError: Can't load tokenizer/model`

**Solutions**:
```bash
# 1. Check model name
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"

# 2. Use fallback model
# Edit configs/config.yaml:
model:
  model_name_or_path: "distilbert-base-uncased"  # Use working model
  peft: null  # Disable LoRA for simpler models

# 3. Check HuggingFace access token (for gated models)
huggingface-cli login
```

### Problem: Configuration Validation Errors

**Error**: `KeyError: 'required_field'` or `Invalid configuration`

**Solutions**:
```bash
# 1. Validate your config
python -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
    print('Config loaded successfully')
    print(f'Required fields present: {all(k in cfg for k in [\"model\", \"train\", \"data\"])}')
"

# 2. Reset to default config
cp configs/config.yaml.backup configs/config.yaml  # If you have backup
# Or regenerate demo data
python scripts/generate_demo_data.py --reset-config
```

## 📊 Data Issues

### Problem: Data Loading Errors

**Error**: `FileNotFoundError: data/tweets.csv not found`

**Solutions**:
```bash
# 1. Generate demo data
python scripts/generate_demo_data.py --tweets 5000 --users 1000

# 2. Check file exists and format
ls -la data/
head -n 5 data/tweets.csv

# 3. Verify data format
python -c "
import pandas as pd
df = pd.read_csv('data/tweets.csv')
print(f'Columns: {list(df.columns)}')
print(f'Shape: {df.shape}')
required = ['text', 'label', 'user_id']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f'❌ Missing columns: {missing}')
else:
    print('✅ All required columns present')
"
```

### Problem: Data Format Issues

**Error**: `ValueError: Labels must be 0 or 1` or `Column 'text' not found`

**Solutions**:
```bash
# 1. Check actual column names
head -n 1 data/tweets.csv

# 2. Use explicit column mapping
python scripts/format_existing_data.py \
    --input your_data.csv \
    --output data/tweets.csv \
    --text-col "actual_text_column" \
    --label-col "actual_label_column"

# 3. Manual data inspection
python -c "
import pandas as pd
df = pd.read_csv('data/tweets.csv')
print('Sample data:')
print(df.head())
print(f'Label distribution: {df.label.value_counts().to_dict()}')
print(f'Text lengths: min={df.text.str.len().min()}, max={df.text.str.len().max()}')
"
```

## 🏋️ Training Issues

### Problem: Out of Memory During Training

**Error**: `RuntimeError: CUDA out of memory` or `Killed (memory issue)`

**Solutions**:
```bash
# 1. Reduce batch size
# Edit configs/config.yaml:
train:
  batch_size: 2  # Reduce from 8
  gradient_checkpointing: true
  fp16: true  # If using GPU

# 2. Use smaller model
model:
  model_name_or_path: "distilbert-base-uncased"
  peft: null

# 3. Reduce data size temporarily
python scripts/generate_demo_data.py --tweets 1000 --users 200
```

### Problem: Training Converges to Poor Results

**Error**: Accuracy stuck at ~50%, Loss not decreasing

**Solutions**:
```bash
# 1. Check data quality
python -c "
import pandas as pd
df = pd.read_csv('data/tweets.csv')
print(f'Class balance: {df.label.value_counts(normalize=True).to_dict()}')
print(f'Sample phishing: {df[df.label==1].text.iloc[0][:100]}...')
print(f'Sample legitimate: {df[df.label==0].text.iloc[0][:100]}...')
"

# 2. Adjust learning rate
# Edit configs/config.yaml:
train:
  lr: 2e-5  # Try different rates: 1e-5, 2e-5, 5e-5

# 3. Check loss weights
loss:
  lambda_cls: 1.0
  lambda_adv: 0.1  # Reduce adversarial weight
  mu_prop: 0.1     # Reduce propagation weight
```

### Problem: Training is Very Slow

**Error**: Training taking many hours, slow progress

**Solutions**:
```bash
# 1. Enable optimizations
# Edit configs/config.yaml:
train:
  fp16: true  # Mixed precision
  gradient_checkpointing: false  # Disable if memory allows
  compile_model: true  # PyTorch 2.0+ only

# 2. Use smaller dataset for testing
python scripts/generate_demo_data.py --tweets 1000

# 3. Check GPU utilization
nvidia-smi  # Should show high GPU usage
watch -n 1 nvidia-smi
```

## 🌐 Graph/Propagation Issues

### Problem: Social Graph Construction Fails

**Error**: `NetworkX graph empty` or `No edges found`

**Solutions**:
```bash
# 1. Check edges data
ls -la data/edges.csv
head -n 5 data/edges.csv

# 2. Generate edges if missing
python -c "
from propagation.graph import construct_social_graph
import pandas as pd
tweets_df = pd.read_csv('data/tweets.csv')
G = construct_social_graph(tweets_df, None, {})
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
"

# 3. Use demo data with edges
python scripts/generate_demo_data.py --tweets 5000 --users 1000 --generate-edges
```

### Problem: Intervention Selection Fails

**Error**: `ValueError: No candidates for intervention` or poor propagation results

**Solutions**:
```bash
# 1. Check candidate selection
python -c "
from propagation.intervene import pick_candidates
import pandas as pd
tweets_df = pd.read_csv('data/tweets.csv')
candidates = pick_candidates(tweets_df['user_id'], topk=50)
print(f'Found {len(candidates)} candidates: {candidates[:5]}')
"

# 2. Adjust intervention parameters
# Edit configs/config.yaml:
propagation:
  budget: 10           # Reduce intervention budget
  topk_candidates: 50  # Reduce candidate pool
  ic_samples: 50       # Reduce simulation samples for speed
```

## 🔍 Debugging Tools

### Check System Resources
```bash
# Memory usage
free -h
ps aux --sort=-%mem | head

# GPU usage (if available)
nvidia-smi
watch -n 1 nvidia-smi

# Disk space
df -h
```

### Validate Installation
```bash
# Test core imports
python -c "
try:
    import torch
    import transformers
    import networkx
    import pandas as pd
    print('✅ All core packages imported successfully')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Test model loading
python -c "
from models.llama_classifier import PhishGuardClassifier
try:
    model = PhishGuardClassifier('distilbert-base-uncased')
    print('✅ Model loads successfully')
except Exception as e:
    print(f'❌ Model loading error: {e}')
"
```

### Enable Debug Logging
```python
# Add to your script for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configs/config.yaml:
logging:
  level: DEBUG
  file: "debug.log"
```

## ⚡ Performance Issues

### Problem: Slow Inference/Prediction

**Error**: API responses taking >5 seconds per request

**Solutions**:
```bash
# 1. Enable model optimizations
python -c "
import torch
from models.llama_classifier import PhishGuardClassifier

model = PhishGuardClassifier('distilbert-base-uncased')
# Enable compilation (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model)
    print('✅ Model compilation enabled')
"

# 2. Reduce model complexity
# Edit configs/config.yaml:
model:
  model_name_or_path: "distilbert-base-uncased"  # Faster than LLaMA
  max_length: 256  # Reduce from 512

# 3. Batch processing for multiple texts
# Process multiple requests together instead of one-by-one
```

### Problem: High Memory Usage

**Error**: System running out of RAM, process killed

**Solutions**:
```bash
# 1. Monitor memory usage
python -c "
import psutil
import torch
from models.llama_classifier import PhishGuardClassifier

print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f}GB')
model = PhishGuardClassifier('distilbert-base-uncased')
print(f'RAM after model load: {psutil.virtual_memory().available / 1e9:.1f}GB')
"

# 2. Use smaller batch sizes
# Edit configs/config.yaml:
train:
  batch_size: 4  # Reduce from 8 or 16

# 3. Clear cache regularly
python -c "
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"
```

## 🗄️ MLflow Issues

### Problem: MLflow Tracking Server Not Starting

**Error**: `mlflow ui` fails or connection refused

**Solutions**:
```bash
# 1. Check if port is available
lsof -ti:5000
netstat -tulpn | grep :5000

# 2. Start on different port
mlflow ui --port 5001

# 3. Check MLflow installation
pip show mlflow
pip install --upgrade mlflow

# 4. Initialize tracking directory
mkdir -p mlruns
mlflow ui --backend-store-uri ./mlruns
```

### Problem: MLflow Experiments Not Logging

**Error**: No experiments showing in MLflow UI

**Solutions**:
```bash
# 1. Check MLflow tracking URI
python -c "
import mlflow
print(f'Tracking URI: {mlflow.get_tracking_uri()}')
"

# 2. Verify experiment creation
python -c "
import mlflow
mlflow.set_experiment('PhishGuard_Test')
with mlflow.start_run():
    mlflow.log_param('test', 'value')
    print('✅ MLflow logging works')
"

# 3. Check file permissions
ls -la mlruns/
chmod -R 755 mlruns/
```

## 🐳 Docker Issues

### Problem: Docker Build Fails

**Error**: `docker build` fails with package installation errors

**Solutions**:
```bash
# 1. Check Dockerfile syntax
docker build --no-cache -t phishguard:debug .

# 2. Use more specific base image
# Edit Dockerfile:
FROM python:3.9-slim  # Instead of just python:3.9

# 3. Add debugging steps
# Add to Dockerfile:
RUN pip install --no-cache-dir torch transformers
RUN pip install --no-cache-dir -r requirements.txt
```

### Problem: Container Exits Immediately

**Error**: Container starts then immediately exits

**Solutions**:
```bash
# 1. Check container logs
docker logs <container_id>

# 2. Run interactively for debugging
docker run -it --entrypoint /bin/bash phishguard:latest

# 3. Check if model files exist
docker run -it phishguard:latest ls -la models/

# 4. Override command for testing
docker run -p 8080:8080 phishguard:latest python -c "print('Container works')"
```

## 🌐 Network/API Issues

### Problem: API Connection Refused

**Error**: `Connection refused` when calling API endpoints

**Solutions**:
```bash
# 1. Check if service is running
curl http://localhost:8080/health
netstat -tulpn | grep :8080

# 2. Check firewall settings
sudo ufw status
sudo ufw allow 8080

# 3. Verify API server logs
tail -f phishguard.log

# 4. Test with different host
# Change from localhost to 0.0.0.0 in API configuration
```

### Problem: Slow API Response Times

**Error**: API taking >10 seconds to respond

**Solutions**:
```bash
# 1. Profile API performance
curl -w "@curl-format.txt" -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "test message"}'

# Create curl-format.txt:
echo "     time_namelookup:  %{time_namelookup}\\n
          time_connect:  %{time_connect}\\n
       time_appconnect:  %{time_appconnect}\\n
      time_pretransfer:  %{time_pretransfer}\\n
         time_redirect:  %{time_redirect}\\n
    time_starttransfer:  %{time_starttransfer}\\n
                       ----------\\n
            time_total:  %{time_total}\\n" > curl-format.txt

# 2. Optimize model inference
# Use smaller batches, enable caching, or upgrade hardware
```

## 🔄 Integration Issues

### Problem: Twitter API Data Collection Fails

**Error**: `tweepy.errors.Unauthorized` or rate limit errors

**Solutions**:
```bash
# 1. Check API credentials
echo $TWITTER_BEARER_TOKEN
python -c "
import tweepy
import os
client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
print('✅ Twitter API connection works')
"

# 2. Handle rate limiting
# Edit scripts/collect_twitter_data.py to add delays:
import time
time.sleep(1)  # Add delay between requests

# 3. Use sample data instead
python scripts/generate_demo_data.py --tweets 10000 --users 2000
```

### Problem: Real Dataset Integration Fails

**Error**: Format errors when using custom datasets

**Solutions**:
```bash
# 1. Check dataset format
head -n 5 your_dataset.csv
python -c "
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print(f'Columns: {list(df.columns)}')
print(f'Sample: {df.iloc[0].to_dict()}')
"

# 2. Use format script with explicit mapping
python scripts/format_existing_data.py \
    --input your_dataset.csv \
    --output data/tweets.csv \
    --text-col "message" \
    --label-col "is_malicious" \
    --user-col "author_id"

# 3. Manual format conversion
python -c "
import pandas as pd
df = pd.read_csv('your_dataset.csv')
formatted = pd.DataFrame({
    'text': df['your_text_column'],
    'label': df['your_label_column'].astype(int),
    'user_id': df['your_user_column'],
    'timestamp': pd.Timestamp.now().isoformat()
})
formatted.to_csv('data/tweets.csv', index=False)
print('✅ Manual formatting complete')
"
```

## 🧪 Testing and Validation

### Quick System Test

```bash
# Complete system validation
python -c "
import sys
print(f'Python: {sys.version}')

# Test imports
try:
    import torch, transformers, networkx, pandas as pd, yaml
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)

# Test model loading
try:
    from models.llama_classifier import PhishGuardClassifier
    model = PhishGuardClassifier('distilbert-base-uncased')
    print('✅ Model loading works')
except Exception as e:
    print(f'❌ Model loading failed: {e}')

# Test data loading
try:
    df = pd.read_csv('data/tweets.csv')
    print(f'✅ Data loaded: {len(df)} rows')
except Exception as e:
    print(f'❌ Data loading failed: {e}')

print('System validation complete!')
"
```

### Performance Benchmark

```bash
# Benchmark training speed
python -c "
import time
from training.train import run

start_time = time.time()
try:
    results = run('configs/config.yaml', eval_only=True)
    duration = time.time() - start_time
    print(f'✅ Training validation: {duration:.1f}s')
    print(f'   Accuracy: {results.get(\"test_metrics\", {}).get(\"accuracy\", \"N/A\")}')
except Exception as e:
    print(f'❌ Training failed: {e}')
"
```

## 📞 Getting Help

### Diagnostic Information to Collect

When reporting issues, include:

```bash
# System information
python --version
pip list | grep -E "(torch|transformers|networkx)"
nvidia-smi --query-gpu=name,memory.total --format=csv  # If GPU

# Configuration
head -n 20 configs/config.yaml

# Data status
ls -la data/
head -n 3 data/tweets.csv

# Error logs
tail -n 50 phishguard.log  # Or wherever your logs are
```

### Where to Get Help

1. **Check existing documentation**:
   - [API Reference](API.md)
   - [Deployment Guide](DEPLOYMENT.md)
   - [Data Integration Guide](DATA_INTEGRATION_GUIDE.md)

2. **Search GitHub issues** for similar problems

3. **Create a new issue** with:
   - Complete error message and traceback
   - System information (from above)
   - Steps to reproduce
   - Configuration files used

4. **Join community discussions** for general questions

---

**Still having trouble?** Don't hesitate to open a detailed issue on GitHub. The PhishGuard community is here to help! 🛠️

Remember: Most issues are environment-related and can be resolved by checking dependencies, configurations, and system resources first.
