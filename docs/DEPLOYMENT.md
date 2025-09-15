# PhishGuard Deployment Guide

Guide for deploying PhishGuard in production environments.

## 🚀 Quick Deployment Options

### Option 1: Standalone Python Service

```bash
# 1. Install PhishGuard
pip install -r requirements.txt

# 2. Train or download model
python -m training.train --config configs/config.yaml
# Or download pre-trained model to models/

# 3. Run inference server
python -c "
from models.llama_classifier import PhishGuardClassifier
import torch

model = PhishGuardClassifier('distilbert-base-uncased')
print('Model loaded successfully')

# Save for production use
torch.save(model.state_dict(), 'models/phishguard_production.pth')
"
```

### Option 2: Docker Deployment

Create a **Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m phishguard
USER phishguard

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from models.llama_classifier import PhishGuardClassifier; print('OK')" || exit 1

# Run application
CMD ["python", "-m", "training.train", "--config", "configs/config.yaml", "--serve"]
```

Build and run:
```bash
# Build image
docker build -t phishguard:latest .

# Run container
docker run -p 8080:8080 phishguard:latest

# With GPU support (if available)
docker run --gpus all -p 8080:8080 phishguard:latest
```

### Option 3: Cloud Platform Deployment

#### AWS EC2
```bash
# 1. Launch EC2 instance (t3.large or p3.2xlarge for GPU)
# 2. Install dependencies
sudo apt update
sudo apt install -y python3-pip docker.io

# 3. Clone and deploy
git clone <your-repo>
cd phishguard-scaffold
sudo docker build -t phishguard .
sudo docker run -d -p 80:8080 --restart unless-stopped phishguard
```

## ⚙️ Production Configuration

### Environment Setup

Create **.env** file:
```bash
# Model configuration
MODEL_PATH=/app/models/phishguard_production.pth
MODEL_DEVICE=cpu  # or cuda if GPU available
BATCH_SIZE=16

# API settings
API_HOST=0.0.0.0
API_PORT=8080
API_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/phishguard.log

# Security
ENABLE_AUTH=false  # Set to true for API key authentication
API_KEY=your-secret-key-here
```

### Production Config File

Create **configs/production.yaml**:
```yaml
model:
  model_name_or_path: "distilbert-base-uncased"  # CPU-friendly for production
  peft: null  # Disable for faster inference
  max_length: 512
  device: "cpu"  # Change to "cuda" if GPU available

train:
  batch_size: 16  # Adjust based on memory
  fp16: false     # Disable for CPU, enable for GPU

inference:
  batch_size: 32        # Larger batches for inference efficiency
  max_workers: 4        # Parallel processing workers
  cache_enabled: true   # Enable result caching
  cache_ttl: 3600      # Cache TTL in seconds

api:
  host: "0.0.0.0"
  port: 8080
  timeout: 30
  rate_limit: "100/minute"

logging:
  level: "INFO"
  format: "json"
  file: "/app/logs/phishguard.log"
```

## 🔧 Simple REST API

Create **api/simple_server.py**:
```python
#!/usr/bin/env python3
"""
Simple REST API for PhishGuard phishing detection.
"""

from flask import Flask, request, jsonify
import torch
import logging
from models.llama_classifier import PhishGuardClassifier
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model (do this once at startup)
MODEL_PATH = os.getenv("MODEL_PATH", "models/phishguard_production.pth")
DEVICE = os.getenv("MODEL_DEVICE", "cpu")

try:
    model = PhishGuardClassifier("distilbert-base-uncased")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"Loaded model from {MODEL_PATH}")
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if text contains phishing."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data["text"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "Invalid text input"}), 400
        
        # Tokenize and predict
        inputs = model.tokenize([text], max_length=512)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(outputs.logits, dim=-1)
            
        # Extract results
        phishing_prob = probs[0, 1].item()
        is_phishing = phishing_prob > 0.5
        
        return jsonify({
            "is_phishing": bool(is_phishing),
            "confidence": float(phishing_prob),
            "text": text[:100] + "..." if len(text) > 100 else text
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict multiple texts at once."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty list"}), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({"error": "Maximum 100 texts per batch"}), 400
        
        # Process in batches
        results = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = model.tokenize(batch_texts, max_length=512)
            
            with torch.no_grad():
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(outputs.logits, dim=-1)
                
                for j, text in enumerate(batch_texts):
                    phishing_prob = probs[j, 1].item()
                    results.append({
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "is_phishing": phishing_prob > 0.5,
                        "confidence": float(phishing_prob)
                    })
        
        return jsonify({
            "predictions": results,
            "total": len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8080))
    host = os.getenv("API_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting PhishGuard API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
```

### Using the API

```bash
# Start the server
python api/simple_server.py

# Test health check
curl http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "URGENT: Click here to claim your prize: bit.ly/fake123"}'

# Batch prediction
curl -X POST http://localhost:8080/batch_predict \
    -H "Content-Type: application/json" \
    -d '{
        "texts": [
            "Check out this amazing deal!",
            "URGENT: Your account will be suspended!",
            "Having coffee with friends"
        ]
    }'
```

## 📊 Monitoring & Logging

### Simple Logging Setup

Create **utils/logging_config.py**:
```python
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "phishguard.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
```

### Performance Monitoring

Add to your API server:
```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor API performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed in {duration:.3f}s: {e}")
            raise
    return wrapper

# Apply to endpoints
@app.route("/predict", methods=["POST"])
@monitor_performance
def predict():
    # ... prediction logic
```

## 🔒 Basic Security

### API Key Authentication

```python
from functools import wraps
import os

API_KEY = os.getenv("API_KEY")

def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            return func(*args, **kwargs)  # Skip auth if no key set
        
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        
        return func(*args, **kwargs)
    return wrapper

# Apply to protected endpoints
@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    # ... prediction logic
```

Usage:
```bash
# With authentication
curl -X POST http://localhost:8080/predict \
    -H "Authorization: Bearer your-secret-key-here" \
    -H "Content-Type: application/json" \
    -d '{"text": "suspicious message"}'
```

### Rate Limiting

```python
from collections import defaultdict
import time

# Simple in-memory rate limiting
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests per minute

def rate_limit():
    client_ip = request.remote_addr
    now = time.time()
    minute_ago = now - 60
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if req_time > minute_ago
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    # Record request
    request_counts[client_ip].append(now)
    return None

@app.before_request
def before_request():
    if request.endpoint in ['predict', 'batch_predict']:
        limit_response = rate_limit()
        if limit_response:
            return limit_response
```

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Fails**
```bash
# Check model file
ls -la models/
python -c "import torch; print(torch.load('models/phishguard_production.pth', map_location='cpu'))"

# Use fallback
export MODEL_PATH=""  # Forces default model loading
```

**2. Out of Memory**
```bash
# Reduce batch size
export BATCH_SIZE=8

# Or force CPU mode
export MODEL_DEVICE=cpu
```

**3. Port Already in Use**
```bash
# Find process using port
lsof -ti:8080
kill -9 <process_id>

# Or use different port
export API_PORT=8081
```

**4. Slow Inference**
```python
# Enable model optimizations in production config
model:
  device: "cpu"
  compile_model: true  # PyTorch 2.0+
  
inference:
  batch_size: 32  # Larger batches for better throughput
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed system check
python -c "
import torch
import psutil
from models.llama_classifier import PhishGuardClassifier

print(f'Python/PyTorch: {torch.__version__}')
print(f'Memory: {psutil.virtual_memory().percent}% used')
print(f'CPU: {psutil.cpu_percent()}% used')

try:
    model = PhishGuardClassifier('distilbert-base-uncased')
    print('✅ Model loading works')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

## 📈 Performance Optimization

### Production Optimizations

```python
# Enable model compilation (PyTorch 2.0+)
model = torch.compile(model, mode="default")

# Use CPU optimizations
torch.set_num_threads(4)  # Adjust based on CPU cores

# Cache frequent predictions
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(text_hash):
    # Implement caching logic
    pass
```

### Memory Management

```python
import gc
import torch

def cleanup_memory():
    """Clean up memory periodically."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call periodically or after batch processing
```

## 🐳 Docker Compose Setup

Create **docker-compose.yml**:
```yaml
version: '3.8'
services:
  phishguard:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_DEVICE=cpu
      - BATCH_SIZE=16
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

Start services:
```bash
docker-compose up -d
```

## 🌐 Production Considerations

### Scaling
- Use multiple API instances behind a load balancer
- Implement Redis caching for frequent predictions
- Consider using Kubernetes for auto-scaling

### Security
- Always use HTTPS in production
- Implement proper API key management
- Add request validation and sanitization
- Monitor for abuse and implement rate limiting

### Monitoring
- Set up health checks and alerting
- Monitor prediction latency and accuracy
- Track resource usage and costs
- Log all requests for audit trails

---

This guide covers basic deployment scenarios. For advanced production setups with load balancing, auto-scaling, and comprehensive monitoring, consider using container orchestration platforms like Kubernetes or cloud-native services.

Need help with deployment? Check our [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue!
