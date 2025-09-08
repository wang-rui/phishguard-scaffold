import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import yaml
import mlflow
from functools import partial
import tempfile
import logging

from training.train_mlflow import MLflowPhishGuardTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_phishguard_ray(config_dict, base_config, checkpoint_dir=None):
    """Training function for Ray Tune with MLflow integration."""
    
    # Create config from base config and tune config
    config = TrainingConfig(**{**base_config.__dict__, **config_dict})
    
    # Set unique run name for this trial
    config.run_name = f"ray_trial_{tune.get_trial_id()}"
    
    # Initialize trainer
    trainer = MLflowPhishGuardTrainer(config)
    
    try:
        # Setup model and data
        trainer.setup_model_and_data()
        
        # Setup training components
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(trainer.train_loader) * config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        # Training loop with Ray Tune reporting
        for epoch in range(config.num_epochs):
            # Train epoch
            epoch_metrics = trainer.train_epoch(optimizer, scheduler, scaler, epoch)
            
            # Validate
            val_metrics, _, _, _ = trainer.evaluate(trainer.val_loader, "validation")
            
            # Report to Ray Tune
            tune.report(
                epoch=epoch,
                loss=val_metrics['loss'],
                accuracy=val_metrics['accuracy'],
                f1=val_metrics['f1'],
                precision=val_metrics.get('precision', 0),
                recall=val_metrics.get('recall', 0),
                auc=val_metrics.get('auc', 0),
                train_loss=epoch_metrics['epoch_loss']
            )
            
            # Save checkpoint if requested
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_metrics': val_metrics
                }, checkpoint_path)
        
        # Final test evaluation
        test_metrics, _, _, _ = trainer.evaluate(trainer.test_loader, "test")
        
        # Report final test metrics
        tune.report(
            final_test_accuracy=test_metrics['accuracy'],
            final_test_f1=test_metrics['f1'],
            final_test_auc=test_metrics.get('auc', 0)
        )
        
    except Exception as e:
        logger.error(f"Training failed in Ray trial: {e}")
        # Report failure to Ray
        tune.report(loss=float('inf'), accuracy=0.0, f1=0.0)

def run_hyperparameter_optimization(base_config_path="configs/config.yaml", 
                                   num_samples=20, 
                                   max_num_epochs=10,
                                   gpus_per_trial=0.25):
    """Run hyperparameter optimization using Ray Tune."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load base configuration
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        base_config = TrainingConfig(**config_dict)
    else:
        base_config = TrainingConfig()
    
    # Set MLflow experiment for hyperparameter tuning
    base_config.experiment_name = "PhishGuard_Hyperparameter_Tuning"
    
    # Define hyperparameter search space
    search_space = {
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([4, 8, 16, 32]),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "warmup_steps": tune.choice([50, 100, 200, 500]),
        
        # Loss function weights (joint optimization)
        "lambda_cls": tune.uniform(0.5, 1.5),
        "lambda_adv": tune.uniform(0.1, 0.5),
        "mu_prop": tune.uniform(0.1, 0.5),
        
        # Adversarial training parameters
        "adv_epsilon": tune.uniform(0.05, 0.2),
        "adv_alpha": tune.uniform(0.005, 0.02),
        "adv_steps": tune.choice([1, 2, 3, 5]),
        
        # Model architecture (if using PEFT)
        "lora_r": tune.choice([8, 16, 32, 64]) if base_config.peft else 16,
        
        # Propagation parameters
        "ic_samples": tune.choice([50, 100, 200]),
        "budget": tune.choice([10, 20, 30, 50]),
        "topk_candidates": tune.choice([100, 200, 300, 500])
    }
    
    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2
    )
    
    # Configure reporter for progress tracking
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "f1", "precision", "recall", "auc"],
        parameter_columns=["lr", "batch_size", "lambda_cls", "lambda_adv", "mu_prop"],
        max_progress_rows=10
    )
    
    # Configure Ray Tune
    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        num_samples=num_samples,
    )
    
    run_config = ray.train.RunConfig(
        name="phishguard_hyperparameter_search",
        progress_reporter=reporter,
        storage_path="./ray_results",
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_score_attribute="f1",
            checkpoint_score_order="max",
            num_to_keep=3
        )
    )
    
    # Create the trainable function
    trainable = tune.with_parameters(train_phishguard_ray, base_config=base_config)
    
    # Configure resources
    if torch.cuda.is_available():
        resources = {"cpu": 2, "gpu": gpus_per_trial}
    else:
        resources = {"cpu": 4}
    
    # Run hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization with {num_samples} trials...")
    logger.info(f"Search space: {search_space}")
    
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=resources),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config
    )
    
    results = tuner.fit()
    
    # Get best trial
    best_trial = results.get_best_result("f1", "max")
    
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation F1: {best_trial.metrics['f1']:.4f}")
    logger.info(f"Best trial final test F1: {best_trial.metrics.get('final_test_f1', 'N/A')}")
    
    # Log best configuration to MLflow
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("PhishGuard_Hyperparameter_Tuning").experiment_id):
        mlflow.log_params(best_trial.config)
        mlflow.log_metrics({
            "best_val_f1": best_trial.metrics['f1'],
            "best_val_accuracy": best_trial.metrics['accuracy'],
            "best_test_f1": best_trial.metrics.get('final_test_f1', 0),
            "best_test_accuracy": best_trial.metrics.get('final_test_accuracy', 0)
        })
    
    # Save best config
    best_config_path = "configs/best_config_ray.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump({**base_config.__dict__, **best_trial.config}, f)
    
    logger.info(f"Best configuration saved to {best_config_path}")
    
    return results, best_trial

def distributed_training_example(config_path="configs/config.yaml"):
    """Example of distributed training using Ray Train."""
    
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig
    
    def train_func(config_dict):
        """Training function for Ray Train distributed training."""
        import torch.distributed as dist
        
        # Initialize distributed training
        dist.init_process_group(backend="nccl")
        
        # Load configuration
        config = TrainingConfig(**config_dict)
        config.run_name = f"distributed_training_{ray.train.get_context().get_world_rank()}"
        
        # Create trainer
        trainer = MLflowPhishGuardTrainer(config)
        
        # Run training
        results = trainer.train()
        
        return results
    
    # Load base configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = TrainingConfig().__dict__
    
    # Configure distributed training
    scaling_config = ScalingConfig(
        num_workers=torch.cuda.device_count() if torch.cuda.is_available() else 2,
        use_gpu=torch.cuda.is_available()
    )
    
    # Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config_dict,
        scaling_config=scaling_config
    )
    
    # Run distributed training
    result = trainer.fit()
    
    logger.info(f"Distributed training completed: {result}")
    return result

def main():
    """Main function for Ray Tune hyperparameter optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter optimization for PhishGuard")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Base configuration file")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of hyperparameter trials")
    parser.add_argument("--max-epochs", type=int, default=5, help="Maximum epochs per trial")
    parser.add_argument("--gpus-per-trial", type=float, default=0.25, help="GPU fraction per trial")
    parser.add_argument("--distributed", action="store_true", help="Run distributed training example")
    
    args = parser.parse_args()
    
    if args.distributed:
        logger.info("Running distributed training example...")
        distributed_training_example(args.config)
    else:
        logger.info("Running hyperparameter optimization...")
        results, best_trial = run_hyperparameter_optimization(
            base_config_path=args.config,
            num_samples=args.num_samples,
            max_num_epochs=args.max_epochs,
            gpus_per_trial=args.gpus_per_trial
        )
        
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best configuration achieved F1 score: {best_trial.metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
