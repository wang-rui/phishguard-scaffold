import os, yaml, random, numpy as np, pandas as pd, torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Optional
import mlflow
import mlflow.pytorch

# Enhanced imports for the PhishGuard framework
from data.dataset import load_and_split
from models.llama_classifier import PhishGuardClassifier, TextClassifier
from training.adversarial import compute_adversarial_loss, adversarial_perturbation, kl_divergence_with_logits
from eval.metrics import compute_cls_metrics
from propagation.graph import (
    load_graph, construct_social_graph, greedy_minimize_spread, 
    ic_spread, compute_propagation_loss
)
from propagation.intervene import (
    advanced_risk_assessment, pick_candidates, evaluate_intervention_impact
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishGuardDataset(Dataset):
    """Enhanced dataset class for PhishGuard framework."""
    
    def __init__(self, df, tokenizer, text_col, label_col, max_length, user_id_col="user_id"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.user_id_col = user_id_col
        self.max_length = max_length
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[self.text_col])
        
        enc = self.tok(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(row[self.label_col], dtype=torch.long),
            'user_id': str(row.get(self.user_id_col, f'user_{i}'))
        }

@dataclass 
class TrainingConfig:
    """Enhanced training configuration with MLflow integration."""
    # Model config
    model_name_or_path: str = "distilbert-base-uncased"
    fallback_model: str = "distilbert-base-uncased"
    peft: Optional[str] = None
    lora_r: int = 16
    max_length: int = 512
    
    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Joint optimization weights
    lambda_cls: float = 1.0
    lambda_adv: float = 0.3
    mu_prop: float = 0.2
    
    # Adversarial training
    adv_epsilon: float = 0.1
    adv_alpha: float = 0.01
    adv_steps: int = 3
    
    # Propagation modeling
    ic_samples: int = 100
    budget: int = 20
    topk_candidates: int = 200
    
    # MLflow config
    experiment_name: str = "PhishGuard"
    run_name: Optional[str] = None
    tracking_uri: str = "mlruns"  # Local tracking
    
    # Data paths
    tweets_path: str = "data/tweets.csv"
    edges_path: str = "data/edges.csv"
    output_dir: str = "runs"

class MLflowPhishGuardTrainer:
    """Enhanced PhishGuard trainer with MLflow integration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        
    def setup_model_and_data(self):
        """Setup model, tokenizer, and data loaders."""
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        try:
            self.model = PhishGuardClassifier(
                self.config.model_name_or_path,
                peft_cfg={'peft': self.config.peft, 'lora_r': self.config.lora_r} if self.config.peft else None
            )
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name_or_path}: {e}")
            logger.info(f"Falling back to {self.config.fallback_model}")
            self.model = TextClassifier(self.config.fallback_model)
        
        self.model = self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        
        # Load and split data
        train_df, val_df, test_df = load_and_split(
            self.config.tweets_path,
            text_col='text',
            label_col='label',
            test_size=0.2,
            val_size=0.1
        )
        
        # Create datasets
        self.train_dataset = PhishGuardDataset(
            train_df, self.tokenizer, 'text', 'label', self.config.max_length
        )
        self.val_dataset = PhishGuardDataset(
            val_df, self.tokenizer, 'text', 'label', self.config.max_length
        )
        self.test_dataset = PhishGuardDataset(
            test_df, self.tokenizer, 'text', 'label', self.config.max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # Load social graph for propagation modeling
        try:
            self.graph = load_graph(self.config.edges_path)
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except:
            logger.warning("Could not load social graph, propagation loss will be disabled")
            self.graph = None
        
        logger.info(f"Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
    
    def train_epoch(self, optimizer, scheduler, scaler, epoch):
        """Train for one epoch with MLflow logging."""
        self.model.train()
        total_loss = 0
        cls_losses = []
        adv_losses = []
        prop_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Classification loss
                cls_loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
                cls_losses.append(cls_loss.item())
                
                # Adversarial loss
                adv_loss = compute_adversarial_loss(
                    self.model, batch, self.config.adv_epsilon, self.config.adv_steps
                )
                adv_losses.append(adv_loss.item())
                
                # Propagation loss
                prop_loss = torch.tensor(0.0, device=self.device)
                if self.graph is not None:
                    try:
                        risk_scores = torch.softmax(logits, dim=-1)[:, 1]  # Phishing probability
                        prop_loss = compute_propagation_loss(
                            self.graph, batch['user_id'], risk_scores, self.config.ic_samples
                        )
                        prop_losses.append(prop_loss.item())
                    except Exception as e:
                        logger.debug(f"Propagation loss computation failed: {e}")
                
                # Joint loss
                total_batch_loss = (
                    self.config.lambda_cls * cls_loss + 
                    self.config.lambda_adv * adv_loss + 
                    self.config.mu_prop * prop_loss
                )
                
                total_loss += total_batch_loss.item()
            
            # Backward pass
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'adv': f'{adv_loss.item():.4f}',
                'prop': f'{prop_loss.item():.4f}' if prop_losses else '0.0000'
            })
            
            # Log batch metrics to MLflow
            if step % 100 == 0:
                mlflow.log_metric("batch_loss", total_batch_loss.item(), step=epoch * len(self.train_loader) + step)
                mlflow.log_metric("batch_cls_loss", cls_loss.item(), step=epoch * len(self.train_loader) + step)
                mlflow.log_metric("batch_adv_loss", adv_loss.item(), step=epoch * len(self.train_loader) + step)
                if prop_losses:
                    mlflow.log_metric("batch_prop_loss", prop_loss.item(), step=epoch * len(self.train_loader) + step)
        
        # Log epoch metrics
        epoch_metrics = {
            'epoch_loss': total_loss / len(self.train_loader),
            'epoch_cls_loss': np.mean(cls_losses),
            'epoch_adv_loss': np.mean(adv_losses),
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        if prop_losses:
            epoch_metrics['epoch_prop_loss'] = np.mean(prop_losses)
        
        for metric_name, metric_value in epoch_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)
        
        return epoch_metrics
    
    def evaluate(self, data_loader, dataset_name="val"):
        """Evaluate model with MLflow logging."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_user_ids = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_user_ids.extend(batch['user_id'])
        
        # Compute metrics
        metrics = compute_cls_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(data_loader)
        
        # Log evaluation metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)
        
        return metrics, all_preds, all_labels, all_user_ids
    
    def train(self):
        """Main training loop with MLflow tracking."""
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.run_name):
            # Log configuration
            mlflow.log_params({
                "model_name": self.config.model_name_or_path,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "lambda_cls": self.config.lambda_cls,
                "lambda_adv": self.config.lambda_adv,
                "mu_prop": self.config.mu_prop,
                "peft": self.config.peft,
                "lora_r": self.config.lora_r if self.config.peft else None,
                "max_length": self.config.max_length,
                "device": str(self.device)
            })
            
            # Setup model and data
            self.setup_model_and_data()
            
            # Setup training components
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
            
            total_steps = len(self.train_loader) * self.config.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
            
            scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)
            
            best_val_f1 = 0
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                
                # Train
                epoch_metrics = self.train_epoch(optimizer, scheduler, scaler, epoch)
                logger.info(f"Train metrics: {epoch_metrics}")
                
                # Validate
                val_metrics, val_preds, val_labels, val_user_ids = self.evaluate(
                    self.val_loader, "validation"
                )
                logger.info(f"Validation metrics: {val_metrics}")
                
                # Save best model
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    
                    # Save model to MLflow
                    mlflow.pytorch.log_model(
                        self.model,
                        "best_model",
                        signature=mlflow.models.signature.infer_signature(
                            {
                                "input_ids": torch.randint(0, 1000, (1, self.config.max_length)),
                                "attention_mask": torch.ones((1, self.config.max_length))
                            },
                            torch.randn(1, 2)
                        )
                    )
                    
                    logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            
            # Final evaluation on test set
            logger.info("Final evaluation on test set...")
            test_metrics, test_preds, test_labels, test_user_ids = self.evaluate(
                self.test_loader, "test"
            )
            logger.info(f"Test metrics: {test_metrics}")
            
            # Intervention analysis if graph is available
            if self.graph is not None:
                logger.info("Performing intervention analysis...")
                try:
                    # Get risk scores from test predictions
                    self.model.eval()
                    risk_scores = {}
                    
                    with torch.no_grad():
                        for i, batch in enumerate(self.test_loader):
                            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in batch.items()}
                            
                            outputs = self.model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask']
                            )
                            
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                            probs = torch.softmax(logits, dim=-1)[:, 1]  # Phishing probability
                            
                            for user_id, prob in zip(batch['user_id'], probs):
                                risk_scores[user_id] = prob.item()
                    
                    # Perform intervention analysis
                    candidates = pick_candidates(self.graph, risk_scores, self.config.topk_candidates)
                    intervention_nodes = greedy_minimize_spread(
                        self.graph, candidates, risk_scores, self.config.budget
                    )
                    
                    impact = evaluate_intervention_impact(
                        self.graph, intervention_nodes, risk_scores
                    )
                    
                    # Log intervention results
                    mlflow.log_metrics({
                        "intervention_nodes_count": len(intervention_nodes),
                        "spread_reduction": impact.get('relative_reduction', 0),
                        "intervention_effectiveness": impact.get('effectiveness', 0)
                    })
                    
                    logger.info(f"Intervention impact: {impact}")
                    
                except Exception as e:
                    logger.error(f"Intervention analysis failed: {e}")
            
            # Log final summary metrics
            mlflow.log_metric("final_test_accuracy", test_metrics['accuracy'])
            mlflow.log_metric("final_test_f1", test_metrics['f1'])
            mlflow.log_metric("final_test_auc", test_metrics.get('auc', 0))
            
            logger.info("Training completed!")
            return test_metrics

def main():
    """Main function to run MLflow-enhanced training."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--experiment-name', type=str, default='PhishGuard')
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line args
    config.experiment_name = args.experiment_name
    config.run_name = args.run_name
    
    # Run training
    trainer = MLflowPhishGuardTrainer(config)
    results = trainer.train()
    
    print(f"Training completed! Final results: {results}")

if __name__ == "__main__":
    main()
