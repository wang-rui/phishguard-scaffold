#!/usr/bin/env python3
"""
Utility script to run MLflow experiments with different configurations.
This script helps automate running multiple experiments with different hyperparameters.
"""

import os
import yaml
import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowExperimentRunner:
    """Run multiple MLflow experiments with different configurations."""
    
    def __init__(self, base_config_path: str = "configs/mlflow_config.yaml"):
        self.base_config_path = base_config_path
        self.base_config = self.load_config(base_config_path)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def create_experiment_configs(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Create experiment configurations from parameter grid."""
        configs = []
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in itertools.product(*values):
            config = self.base_config.copy()
            
            for key, value in zip(keys, combination):
                # Handle nested keys (e.g., "train.lr")
                keys_path = key.split('.')
                current_dict = config
                
                for nested_key in keys_path[:-1]:
                    if nested_key not in current_dict:
                        current_dict[nested_key] = {}
                    current_dict = current_dict[nested_key]
                
                current_dict[keys_path[-1]] = value
            
            configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config: Dict[str, Any], experiment_name: str, run_name: str) -> bool:
        """Run a single MLflow experiment."""
        try:
            # Save temporary config
            temp_config_path = f"configs/temp_{run_name}.yaml"
            config['mlflow']['experiment_name'] = experiment_name
            config['mlflow']['run_name'] = run_name
            
            self.save_config(config, temp_config_path)
            
            # Run training
            cmd = [
                "python", "-m", "training.train_mlflow",
                "--config", temp_config_path,
                "--experiment-name", experiment_name,
                "--run-name", run_name
            ]
            
            logger.info(f"Starting experiment: {run_name}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Experiment {run_name} completed successfully")
                success = True
            else:
                logger.error(f"❌ Experiment {run_name} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                success = False
            
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Exception in experiment {run_name}: {e}")
            return False
    
    def run_experiments(self, param_grid: Dict[str, List[Any]], 
                       experiment_name: str = "PhishGuard_GridSearch") -> List[str]:
        """Run multiple experiments with parameter grid."""
        
        configs = self.create_experiment_configs(param_grid)
        logger.info(f"Generated {len(configs)} experiment configurations")
        
        successful_runs = []
        failed_runs = []
        
        for i, config in enumerate(configs):
            run_name = f"run_{i:03d}"
            
            # Add parameter info to run name
            param_info = []
            for key, values in param_grid.items():
                keys_path = key.split('.')
                current_dict = config
                for nested_key in keys_path:
                    current_dict = current_dict[nested_key]
                param_info.append(f"{key.replace('.', '_')}_{current_dict}")
            
            detailed_run_name = f"{run_name}_" + "_".join(param_info[:3])  # Limit length
            
            success = self.run_single_experiment(config, experiment_name, detailed_run_name)
            
            if success:
                successful_runs.append(detailed_run_name)
            else:
                failed_runs.append(detailed_run_name)
            
            # Brief pause between experiments
            time.sleep(2)
        
        logger.info(f"\n📊 Experiment Summary:")
        logger.info(f"✅ Successful runs: {len(successful_runs)}")
        logger.info(f"❌ Failed runs: {len(failed_runs)}")
        
        if failed_runs:
            logger.info(f"Failed runs: {failed_runs}")
        
        return successful_runs

def run_learning_rate_experiments():
    """Example: Run experiments with different learning rates."""
    runner = MLflowExperimentRunner()
    
    param_grid = {
        "train.lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "train.batch_size": [8, 16],
        "loss.lambda_adv": [0.1, 0.3, 0.5]
    }
    
    return runner.run_experiments(
        param_grid=param_grid,
        experiment_name="PhishGuard_LearningRate_Study"
    )

def run_loss_weights_experiments():
    """Example: Run experiments with different loss weights."""
    runner = MLflowExperimentRunner()
    
    param_grid = {
        "loss.lambda_cls": [0.8, 1.0, 1.2],
        "loss.lambda_adv": [0.1, 0.2, 0.3, 0.4],
        "loss.mu_prop": [0.1, 0.2, 0.3]
    }
    
    return runner.run_experiments(
        param_grid=param_grid,
        experiment_name="PhishGuard_LossWeights_Study"
    )

def run_adversarial_experiments():
    """Example: Run experiments with different adversarial parameters."""
    runner = MLflowExperimentRunner()
    
    param_grid = {
        "adversarial.epsilon": [0.05, 0.1, 0.15, 0.2],
        "adversarial.alpha": [0.005, 0.01, 0.015, 0.02],
        "adversarial.steps": [1, 2, 3, 5]
    }
    
    return runner.run_experiments(
        param_grid=param_grid,
        experiment_name="PhishGuard_Adversarial_Study"
    )

def run_model_architecture_experiments():
    """Example: Run experiments with different model configurations."""
    runner = MLflowExperimentRunner()
    
    param_grid = {
        "model.peft": ["lora", None],
        "model.lora_r": [8, 16, 32, 64],
        "train.lr": [1e-4, 5e-4],  # Adjust LR based on model size
        "train.batch_size": [4, 8]  # Smaller batches for larger models
    }
    
    return runner.run_experiments(
        param_grid=param_grid,
        experiment_name="PhishGuard_Architecture_Study"
    )

def start_mlflow_ui():
    """Start MLflow UI server."""
    try:
        logger.info("Starting MLflow UI...")
        logger.info("Access the MLflow UI at: http://localhost:5000")
        
        subprocess.run(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
        
    except KeyboardInterrupt:
        logger.info("MLflow UI stopped.")
    except Exception as e:
        logger.error(f"Failed to start MLflow UI: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run MLflow experiments for PhishGuard")
    
    parser.add_argument("--experiment-type", 
                       choices=["lr", "loss-weights", "adversarial", "architecture", "custom"],
                       default="lr",
                       help="Type of experiment to run")
    
    parser.add_argument("--config", type=str, default="configs/mlflow_config.yaml",
                       help="Base configuration file")
    
    parser.add_argument("--start-ui", action="store_true",
                       help="Start MLflow UI instead of running experiments")
    
    parser.add_argument("--custom-grid", type=str,
                       help="Path to custom parameter grid YAML file")
    
    args = parser.parse_args()
    
    if args.start_ui:
        start_mlflow_ui()
        return
    
    if args.experiment_type == "lr":
        successful_runs = run_learning_rate_experiments()
    elif args.experiment_type == "loss-weights":
        successful_runs = run_loss_weights_experiments()
    elif args.experiment_type == "adversarial":
        successful_runs = run_adversarial_experiments()
    elif args.experiment_type == "architecture":
        successful_runs = run_model_architecture_experiments()
    elif args.experiment_type == "custom" and args.custom_grid:
        # Load custom parameter grid
        with open(args.custom_grid, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        param_grid = custom_config.get('param_grid', {})
        experiment_name = custom_config.get('experiment_name', 'PhishGuard_Custom')
        
        runner = MLflowExperimentRunner(args.config)
        successful_runs = runner.run_experiments(param_grid, experiment_name)
    else:
        logger.error("Invalid experiment type or missing custom grid file")
        return
    
    logger.info(f"\n🎉 All experiments completed!")
    logger.info(f"Successful runs: {len(successful_runs)}")
    logger.info(f"View results in MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()
