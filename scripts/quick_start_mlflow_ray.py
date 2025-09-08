#!/usr/bin/env python3
"""
Quick start script for PhishGuard with MLflow and Ray integration.
This script provides a simple way to get started with experiment tracking and hyperparameter tuning.
"""

import os
import argparse
import logging
import yaml
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['mlflow', 'ray', 'torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {missing_packages}")
        logger.info("Install them with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required packages are installed")
    return True

def setup_data_and_config():
    """Setup data and configuration files."""
    
    # Check if data exists
    data_files = ['data/tweets.csv', 'data/edges.csv']
    missing_data = [f for f in data_files if not os.path.exists(f)]
    
    if missing_data:
        logger.warning(f"⚠️  Missing data files: {missing_data}")
        logger.info("Generating demo data...")
        
        try:
            result = subprocess.run([
                "python", "scripts/generate_demo_data.py", 
                "--tweets", "5000", 
                "--users", "1000"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Demo data generated successfully")
            else:
                logger.error(f"❌ Failed to generate demo data: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error generating demo data: {e}")
            return False
    else:
        logger.info("✅ Data files found")
    
    # Check if MLflow config exists
    if not os.path.exists('configs/mlflow_config.yaml'):
        logger.error("❌ MLflow configuration file not found")
        return False
    
    logger.info("✅ Configuration files ready")
    return True

def run_single_mlflow_experiment():
    """Run a single experiment with MLflow tracking."""
    logger.info("🚀 Running single MLflow experiment...")
    
    try:
        result = subprocess.run([
            "python", "-m", "training.train_mlflow",
            "--config", "configs/mlflow_config.yaml",
            "--experiment-name", "PhishGuard_QuickStart",
            "--run-name", "quick_start_run"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ MLflow experiment completed successfully!")
            logger.info("🌐 View results: mlflow ui")
            return True
        else:
            logger.error(f"❌ Experiment failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running experiment: {e}")
        return False

def run_ray_hyperparameter_tuning():
    """Run hyperparameter tuning with Ray Tune."""
    logger.info("🔍 Running hyperparameter tuning with Ray Tune...")
    
    try:
        result = subprocess.run([
            "python", "-m", "training.ray_tune_hyperparams",
            "--config", "configs/mlflow_config.yaml",
            "--num-samples", "5",  # Small number for quick start
            "--max-epochs", "3",
            "--gpus-per-trial", "0.25"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Hyperparameter tuning completed!")
            logger.info("📁 Results saved in ./ray_results/")
            logger.info("🏆 Best configuration saved as configs/best_config_ray.yaml")
            return True
        else:
            logger.error(f"❌ Hyperparameter tuning failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running hyperparameter tuning: {e}")
        return False

def run_experiment_comparison():
    """Run multiple experiments for comparison."""
    logger.info("📊 Running experiment comparison...")
    
    try:
        result = subprocess.run([
            "python", "scripts/run_mlflow_experiments.py",
            "--experiment-type", "lr",
            "--config", "configs/mlflow_config.yaml"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Experiment comparison completed!")
            logger.info("📈 Compare results in MLflow UI")
            return True
        else:
            logger.error(f"❌ Experiment comparison failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running experiment comparison: {e}")
        return False

def start_mlflow_ui():
    """Start MLflow UI for viewing results."""
    logger.info("🌐 Starting MLflow UI...")
    logger.info("Access at: http://localhost:5000")
    logger.info("Press Ctrl+C to stop")
    
    try:
        subprocess.run(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
    except KeyboardInterrupt:
        logger.info("MLflow UI stopped")

def create_custom_config():
    """Create a custom configuration file."""
    config_path = "configs/custom_config.yaml"
    
    logger.info(f"📝 Creating custom configuration: {config_path}")
    
    # Load base config
    with open("configs/mlflow_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Customize for quick start
    config['train']['num_epochs'] = 2  # Faster training
    config['train']['batch_size'] = 4   # Smaller batch size
    config['mlflow']['experiment_name'] = "PhishGuard_Custom"
    config['ray_tune']['num_samples'] = 3  # Fewer trials
    
    # Save custom config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✅ Custom configuration created: {config_path}")
    return config_path

def show_help():
    """Show detailed help and usage examples."""
    help_text = """
🎯 PhishGuard MLflow & Ray Quick Start Guide

This script helps you get started with MLflow experiment tracking and Ray hyperparameter tuning.

Available Commands:
─────────────────

1. Setup Environment:
   python scripts/quick_start_mlflow_ray.py --setup

2. Run Single Experiment with MLflow:
   python scripts/quick_start_mlflow_ray.py --single-experiment

3. Run Hyperparameter Tuning with Ray:
   python scripts/quick_start_mlflow_ray.py --tune-hyperparams

4. Run Multiple Experiments for Comparison:
   python scripts/quick_start_mlflow_ray.py --compare-experiments

5. Start MLflow UI:
   python scripts/quick_start_mlflow_ray.py --start-ui

6. Create Custom Configuration:
   python scripts/quick_start_mlflow_ray.py --create-config

Complete Workflow Example:
──────────────────────────

# 1. Setup environment and data
python scripts/quick_start_mlflow_ray.py --setup

# 2. Run a quick experiment
python scripts/quick_start_mlflow_ray.py --single-experiment

# 3. Start MLflow UI (in another terminal)
python scripts/quick_start_mlflow_ray.py --start-ui

# 4. Run hyperparameter tuning
python scripts/quick_start_mlflow_ray.py --tune-hyperparams

# 5. Compare different configurations
python scripts/quick_start_mlflow_ray.py --compare-experiments

Direct Usage (Advanced):
────────────────────────

# MLflow training with custom config
python -m training.train_mlflow --config configs/mlflow_config.yaml

# Ray hyperparameter tuning
python -m training.ray_tune_hyperparams --num-samples 10

# Run experiment grid search
python scripts/run_mlflow_experiments.py --experiment-type lr

Key Features:
─────────────

✨ MLflow Integration:
   - Automatic experiment tracking
   - Parameter and metric logging
   - Model versioning and registry
   - Web UI for result visualization

🚀 Ray Integration:
   - Distributed hyperparameter tuning
   - Early stopping with ASHA scheduler
   - GPU resource management
   - Parallel trial execution

📊 Experiment Management:
   - Grid search automation
   - Result comparison tools
   - Configuration templates
   - Reproducible experiments

For more information, see the documentation or visit:
- MLflow: https://mlflow.org/
- Ray: https://docs.ray.io/
    """
    
    print(help_text)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Quick start script for PhishGuard with MLflow and Ray",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--setup", action="store_true",
                       help="Setup environment and check dependencies")
    
    parser.add_argument("--single-experiment", action="store_true",
                       help="Run a single MLflow experiment")
    
    parser.add_argument("--tune-hyperparams", action="store_true",
                       help="Run hyperparameter tuning with Ray")
    
    parser.add_argument("--compare-experiments", action="store_true",
                       help="Run multiple experiments for comparison")
    
    parser.add_argument("--start-ui", action="store_true",
                       help="Start MLflow UI")
    
    parser.add_argument("--create-config", action="store_true",
                       help="Create a custom configuration file")
    
    parser.add_argument("--help-detailed", action="store_true",
                       help="Show detailed help and usage examples")
    
    args = parser.parse_args()
    
    if args.help_detailed:
        show_help()
        return
    
    if not any(vars(args).values()):
        logger.info("No action specified. Use --help for options or --help-detailed for examples")
        return
    
    logger.info("🎯 PhishGuard MLflow & Ray Quick Start")
    logger.info("=====================================")
    
    if args.setup:
        logger.info("🔧 Setting up environment...")
        if not check_dependencies():
            return
        if not setup_data_and_config():
            return
        logger.info("✅ Environment setup complete!")
    
    if args.create_config:
        create_custom_config()
    
    if args.single_experiment:
        if not os.path.exists('configs/mlflow_config.yaml'):
            logger.error("❌ Please run --setup first")
            return
        run_single_mlflow_experiment()
    
    if args.tune_hyperparams:
        if not os.path.exists('configs/mlflow_config.yaml'):
            logger.error("❌ Please run --setup first")
            return
        run_ray_hyperparameter_tuning()
    
    if args.compare_experiments:
        if not os.path.exists('configs/mlflow_config.yaml'):
            logger.error("❌ Please run --setup first")
            return
        run_experiment_comparison()
    
    if args.start_ui:
        start_mlflow_ui()
    
    logger.info("\n🎉 Quick start complete!")
    logger.info("📚 For more advanced usage, see the documentation")

if __name__ == "__main__":
    main()
