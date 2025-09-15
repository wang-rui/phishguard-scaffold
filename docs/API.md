# PhishGuard API Reference

Complete API documentation for using PhishGuard components programmatically.

## 🔧 Core Components

### PhishGuardClassifier

The main model class for phishing detection with LLaMA/transformer support.

```python
from models.llama_classifier import PhishGuardClassifier

# Initialize with LLaMA (requires GPU/sufficient memory)
model = PhishGuardClassifier(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    num_labels=2,
    peft_cfg={
        "peft": "lora",
        "lora_r": 16,
        "fallback_model": "distilbert-base-uncased"
    }
)

# Initialize with CPU-friendly model
model = PhishGuardClassifier(
    model_name_or_path="distilbert-base-uncased",
    num_labels=2
)
```

#### Key Methods

**`tokenize(texts, max_length)`**
```python
# Tokenize text for model input
texts = ["Check this suspicious link: bit.ly/fake123", "Normal tweet here"]
inputs = model.tokenize(texts, max_length=512)
# Returns: {"input_ids": tensor, "attention_mask": tensor}
```

**`forward(input_ids, attention_mask, labels=None, return_embeddings=False)`**
```python
# Get model predictions
outputs = model(inputs["input_ids"], inputs["attention_mask"])
logits = outputs.logits  # Shape: [batch_size, 2]
probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
```

**`extract_phishing_features(input_ids, attention_mask)`**
```python
# Get detailed features for analysis
features = model.extract_phishing_features(inputs["input_ids"], inputs["attention_mask"])
# Returns: {
#   "embeddings": tensor,           # Semantic embeddings
#   "attention_scores": tensor,     # Token attention weights  
#   "phishing_probability": tensor, # P(phishing)
#   "risk_score": tensor           # User risk assessment
# }
```

## 🌐 Graph Operations

### Social Network Construction

```python
from propagation.graph import construct_social_graph, ic_spread
import pandas as pd

# Load your data
tweets_df = pd.read_csv("data/tweets.csv")
edges_df = pd.read_csv("data/edges.csv")  # Optional

# Build social network graph
graph = construct_social_graph(
    tweets_df=tweets_df,
    edges_df=edges_df,     # Optional: explicit social connections
    cfg=config             # Configuration dictionary
)
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### Propagation Simulation

```python
# Simulate information spread using Independent Cascade
seed_users = ["user_123", "user_456"]  # Users posting phishing content
expected_spread = ic_spread(
    G=graph,
    seeds=seed_users,
    samples=100,      # Monte Carlo samples for accuracy
    max_steps=10      # Maximum diffusion steps
)
print(f"Expected spread: {expected_spread:.1f} users influenced")
```

### Intervention Selection

```python
from propagation.intervene import greedy_minimize_spread, pick_candidates

# Select candidate users for intervention
candidates = pick_candidates(
    df_users=tweets_df["user_id"],
    topk=100,              # Top 100 most active users
    G=graph,               # Use network structure
    risk_scores=risk_dict  # User risk scores from model
)

# Find optimal intervention nodes
intervention_nodes = greedy_minimize_spread(
    G=graph,
    budget=10,                    # Maximum interventions allowed
    risk=user_risk_scores,        # Risk scores per user
    candidates=candidates,
    samples=50                    # Samples for spread estimation
)
print(f"Selected {len(intervention_nodes)} nodes for intervention")
```

## 🚀 Training API

### Basic Training

```python
from training.train import run

# Run complete training pipeline
results = run(
    config_path="configs/config.yaml",
    eval_only=False  # Set to True for evaluation only
)

# Access training results
print(f"Final accuracy: {results['test_metrics']['accuracy']:.3f}")
print(f"F1-score: {results['test_metrics']['f1']:.3f}")
print(f"Intervention nodes: {results.get('intervention_nodes', [])}")
```

### MLflow Tracking

```python
from training.train_mlflow import MLflowPhishGuardTrainer
import yaml

# Load configuration
with open("configs/mlflow_config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize trainer with experiment tracking
trainer = MLflowPhishGuardTrainer(config)

# Run training with automatic logging
results = trainer.train()

# Access MLflow information
print(f"Experiment ID: {trainer.experiment_id}")
print(f"Run ID: {trainer.mlflow_run_id}")
```

### Hyperparameter Optimization

```python
from training.ray_tune_hyperparams import run_hyperparameter_optimization
from ray import tune

# Define search space
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "lambda_adv": tune.uniform(0.1, 0.5),
    "mu_prop": tune.uniform(0.1, 0.3),
    "batch_size": tune.choice([4, 8, 16])
}

# Run optimization
results, best_trial = run_hyperparameter_optimization(
    config_path="configs/mlflow_config.yaml",
    search_space=search_space,
    num_samples=20,        # Number of trials
    max_num_epochs=5,      # Epochs per trial
    gpus_per_trial=0.5     # GPU allocation
)

print(f"Best configuration: {best_trial.config}")
print(f"Best F1 score: {best_trial.metrics['f1']:.4f}")
```

## 📊 Data Processing API

### Dataset Loading

```python
from data.dataset import load_and_split, enhanced_preprocessing
import yaml

# Load configuration
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Load and split dataset with preprocessing
split_data = load_and_split("data/tweets.csv", config)

# Access different splits
train_df = split_data.train
val_df = split_data.val
test_df = split_data.test

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### Custom Data Formatting

```python
# Format existing dataset to PhishGuard format
import subprocess

result = subprocess.run([
    "python", "scripts/format_existing_data.py",
    "--input", "your_dataset.csv",
    "--output", "data/tweets.csv", 
    "--text-col", "tweet_content",
    "--label-col", "is_phishing"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Data formatting successful")
else:
    print(f"❌ Error: {result.stderr}")
```

## 🔬 Evaluation API

### Classification Metrics

```python
from eval.metrics import compute_cls_metrics
import numpy as np

# Compute comprehensive metrics
metrics = compute_cls_metrics(
    labels=true_labels,        # Ground truth labels (0/1)
    preds=predicted_labels,    # Predicted labels (0/1)
    probs=prediction_probs     # Prediction probabilities [batch_size, 2]
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

### Intervention Impact Analysis

```python
from propagation.intervene import evaluate_intervention_impact

# Evaluate intervention effectiveness
impact = evaluate_intervention_impact(
    G=social_graph,
    intervention_nodes=selected_nodes,
    risk_scores=user_risk_dict,
    samples=100
)

print(f"Baseline spread: {impact['baseline_spread']:.1f} users")
print(f"Spread with intervention: {impact['intervened_spread']:.1f} users") 
print(f"Spread reduction: {impact['relative_reduction']:.1%}")
print(f"Cost effectiveness: {impact['cost_effectiveness']:.2f} reduction per intervention")
```

## ⚙️ Configuration Management

### Loading Configurations

```python
import yaml

# Load YAML configuration
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Access configuration sections
model_config = config["model"]
train_config = config["train"] 
loss_config = config["loss"]

print(f"Model: {model_config['model_name_or_path']}")
print(f"Batch size: {train_config['batch_size']}")
print(f"Loss weights: λ_adv={loss_config['lambda_adv']}, μ_prop={loss_config['mu_prop']}")
```

### Configuration Schema

Essential configuration parameters:

```yaml
model:
  model_name_or_path: "meta-llama/Llama-2-7b-hf"  # or "distilbert-base-uncased"
  fallback_model: "distilbert-base-uncased"        # CPU fallback
  peft: "lora"                                     # Parameter-efficient fine-tuning
  lora_r: 16                                       # LoRA rank
  max_length: 512                                  # Input sequence length

train:
  batch_size: 8                                    # Training batch size
  num_epochs: 5                                    # Training epochs
  lr: 1e-4                                         # Learning rate
  fp16: true                                       # Mixed precision training

loss:
  lambda_cls: 1.0                                  # Classification loss weight
  lambda_adv: 0.3                                  # Adversarial robustness weight
  mu_prop: 0.2                                     # Propagation control weight

propagation:
  ic_samples: 100                                  # Independent Cascade samples
  budget: 20                                       # Intervention budget
  topk_candidates: 200                             # Candidate pool size

data:
  text_col: "text"                                 # Text column name
  label_col: "label"                               # Label column name
  split: {train: 0.8, val: 0.1, test: 0.1}       # Dataset split ratios
```

## 🛠️ Utility Functions

### Risk Assessment

```python
from propagation.intervene import risk_from_logits, advanced_risk_assessment

# Simple risk calculation from model outputs
user_risks = risk_from_logits(user_ids, model_logits)

# Advanced risk assessment with network analysis
comprehensive_risks = advanced_risk_assessment(
    user_ids=user_ids,
    logits=model_logits,
    G=social_graph,
    tweets_df=tweets_dataframe  # Optional: for behavioral analysis
)
```

### Graph Analysis

```python
from propagation.graph import compute_influence_scores

# Compute user influence in network
influence_scores = compute_influence_scores(
    G=social_graph,
    user_risk=user_risk_dict
)

# Find most influential users
top_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most influential users:")
for user_id, score in top_influencers:
    print(f"  {user_id}: {score:.3f}")
```

## 💡 Usage Examples

### Complete Detection Pipeline

```python
import torch
import pandas as pd
from models.llama_classifier import PhishGuardClassifier
from propagation.graph import construct_social_graph, ic_spread
from propagation.intervene import greedy_minimize_spread, pick_candidates

# 1. Load data
tweets_df = pd.read_csv("data/tweets.csv")

# 2. Initialize model
model = PhishGuardClassifier("distilbert-base-uncased")

# 3. Get predictions
texts = tweets_df["text"].tolist()
inputs = model.tokenize(texts, max_length=512)
outputs = model(inputs["input_ids"], inputs["attention_mask"])
predictions = torch.argmax(outputs.logits, dim=-1)

# 4. Build social graph
graph = construct_social_graph(tweets_df, None, {})

# 5. Identify high-risk users
risk_scores = {}
for i, user_id in enumerate(tweets_df["user_id"]):
    risk_scores[user_id] = torch.softmax(outputs.logits[i], dim=-1)[1].item()

# 6. Select intervention candidates
candidates = pick_candidates(tweets_df["user_id"], topk=50, G=graph, risk_scores=risk_scores)

# 7. Optimize interventions
interventions = greedy_minimize_spread(graph, budget=10, risk=risk_scores, candidates=candidates)

print(f"Detected {predictions.sum().item()} phishing tweets")
print(f"Selected {len(interventions)} users for intervention: {interventions}")
```

### Batch Processing

```python
def process_batch(texts, model, batch_size=32):
    """Process texts in batches for memory efficiency."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = model.tokenize(batch_texts, max_length=512)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(outputs.logits, dim=-1)
            results.extend(probs[:, 1].tolist())  # Phishing probabilities
    
    return results

# Usage
phishing_probs = process_batch(tweet_texts, model, batch_size=16)
```

## 🚨 Error Handling

### Common Exception Patterns

```python
try:
    model = PhishGuardClassifier("meta-llama/Llama-2-7b-hf")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Automatic fallback handled internally
    model = PhishGuardClassifier("distilbert-base-uncased")

try:
    results = run("configs/config.yaml")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Reducing batch size due to memory constraints")
        # Modify config and retry
    else:
        raise e
```

## 📚 Data Format Requirements

### tweets.csv
```csv
text,label,user_id,timestamp,parent_user_id,url
"Suspicious message",1,user123,2024-01-01T00:00:00Z,,https://malicious.com
"Normal tweet",0,user456,2024-01-01T01:00:00Z,,
"RT @user123: Suspicious message",1,user789,2024-01-01T02:00:00Z,user123,
```

**Required columns:**
- `text`: Tweet content (string)
- `label`: 0=legitimate, 1=phishing (integer)
- `user_id`: Unique user identifier (string)
- `timestamp`: ISO format timestamp (string)

**Optional columns:**
- `parent_user_id`: For retweets/replies (string)
- `url`: Extracted URLs (string)

### edges.csv  
```csv
src,dst,weight,timestamp
user123,user456,0.15,2024-01-01T00:30:00Z
user456,user789,0.22,2024-01-01T01:00:00Z
```

**Required columns:**
- `src`: Source user ID (string)
- `dst`: Destination user ID (string) 
- `weight`: Influence probability [0,1] (float)

---

For additional examples and advanced usage patterns, see the Jupyter notebooks and documentation in the repository.

Need help? Check out our [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.
