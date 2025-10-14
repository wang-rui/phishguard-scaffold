"""
Constants for PhishGuard Framework.

This module contains all magic numbers and hardcoded values used throughout
the codebase for better maintainability and configuration.
"""

# Model Configuration
DEFAULT_FALLBACK_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_EMBEDDING_DIM = 768

# Data Processing
MIN_TEXT_LENGTH_DEFAULT = 10
MAX_TEXT_LENGTH_DEFAULT = 512
MIN_SAMPLES_THRESHOLD = 100

# Training Defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1

# Loss Weights
DEFAULT_LAMBDA_CLS = 1.0
DEFAULT_LAMBDA_ADV = 0.3
DEFAULT_MU_PROP = 0.2

# Adversarial Training
DEFAULT_ADV_EPSILON = 1e-2
DEFAULT_ADV_STEPS = 3
DEFAULT_ADV_TEMPERATURE = 1.0
ADV_EPSILON_FALLBACK_MULTIPLIER = 0.01  # Used when adversarial pass fails

# Propagation & Graph
DEFAULT_IC_SAMPLES = 100
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_INTERVENTION_BUDGET = 20
DEFAULT_TOPK_CANDIDATES = 200
DEFAULT_EDGE_WEIGHT_THRESHOLD = 0.01
DEFAULT_RISK_THRESHOLD = 0.7
INFLUENCE_DECAY_FACTOR = 0.9  # Decay factor for influence over time/steps
MIN_EDGE_WEIGHT = 0.0
MAX_EDGE_WEIGHT = 1.0

# Intervention Algorithm
HIGH_RISK_SEED_LIMIT = 20  # Maximum seeds to consider for baseline spread
HIGH_RISK_THRESHOLD = 0.5  # Minimum risk score to be considered high-risk
RISK_BIAS_MULTIPLIER = 2.0  # Weight for risky nodes in intervention selection
INFLUENCE_WEIGHT_MULTIPLIER = 1.0  # Weight for influential nodes

# Data Splits
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1

# Centrality Weights (for influence calculation)
PAGERANK_WEIGHT = 0.5
BETWEENNESS_WEIGHT = 0.3
OUT_DEGREE_WEIGHT = 0.2

# Risk Assessment Weights
MODEL_PREDICTION_WEIGHT = 0.5
NETWORK_INFLUENCE_WEIGHT = 0.3
BEHAVIORAL_RISK_WEIGHT = 0.2

# Behavioral Risk Components
URL_RATIO_WEIGHT = 0.5
ACTIVITY_FREQ_WEIGHT = 0.3
POST_COUNT_WEIGHT = 0.2

# Network Influence Components (in advanced_risk_assessment)
PAGERANK_INFLUENCE_WEIGHT = 0.4
BETWEENNESS_INFLUENCE_WEIGHT = 0.3
EIGENVECTOR_INFLUENCE_WEIGHT = 0.2
DEGREE_INFLUENCE_WEIGHT = 0.1

# Intervention Candidate Selection
ACTIVITY_SCORE_WEIGHT = 0.3
CENTRALITY_SCORE_WEIGHT = 0.4
RISK_SCORE_WEIGHT = 0.3

# Performance Thresholds
MAX_BATCH_SIZE_WARNING = 128
MAX_EPOCHS_WARNING = 100
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2
MIN_MAX_LENGTH = 32
MAX_MAX_LENGTH = 2048
MIN_LORA_R = 1
MAX_LORA_R = 256
MIN_TRAIN_SPLIT = 0.5

# Logging
MLFLOW_BATCH_LOG_INTERVAL = 100  # Log batch metrics every N batches

# Validation
SPLIT_TOLERANCE = 1e-6  # Tolerance for split sum validation

