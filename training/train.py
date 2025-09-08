import os
import yaml
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import logging

# Enhanced imports for the PhishGuard framework
from data.dataset import load_and_split
from models.llama_classifier import PhishGuardClassifier  # Backward compatibility
from training.adversarial import adversarial_perturbation, kl_divergence_with_logits
from eval.metrics import compute_cls_metrics
from propagation.graph import (
    load_graph, greedy_minimize_spread
)
from propagation.intervene import (
    pick_candidates, risk_from_logits
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishGuardDataset(Dataset):
    """Enhanced dataset class for PhishGuard framework.
    
    Includes additional metadata needed for propagation modeling and intervention.
    """
    
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
        
        # Enhanced tokenization
        enc = self.tok(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row[self.label_col]))
        item["user_id"] = str(row.get(self.user_id_col, ""))
        
        # Additional metadata for propagation analysis
        item["timestamp"] = row.get("timestamp", "")
        item["url"] = row.get("url", "")
        item["parent_user_id"] = row.get("parent_user_id", "")
        
        return item

# Backward compatibility
TxtDS = PhishGuardDataset

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def run(cfg_path: str, eval_only: bool=False):
    """Enhanced PhishGuard training with joint optimization framework.
    
    Implements the complete research architecture with:
    - LLaMA-based semantic classification
    - Adversarial training for robustness
    - Propagation control via social graph analysis
    - Targeted intervention strategies
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    logger.info("=== PhishGuard Framework Training ===")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Output directory: {cfg['output_dir']}")
    
    # Enhanced data loading with preprocessing
    logger.info("Loading and preprocessing data...")
    split = load_and_split(cfg["data"]["tweets_csv"], cfg)
    logger.info(f"Data split - Train: {len(split.train)}, Val: {len(split.val)}, Test: {len(split.test)}")
    
    # Initialize enhanced PhishGuard model
    logger.info(f"Initializing PhishGuard model: {cfg['model']['model_name_or_path']}")
    model = PhishGuardClassifier(
        cfg["model"]["model_name_or_path"], 
        num_labels=2, 
        peft_cfg=cfg["model"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Enhanced datasets with metadata
    train_ds = PhishGuardDataset(
        split.train, model.tokenizer, 
        cfg["data"]["text_col"], cfg["data"]["label_col"], 
        cfg["model"]["max_length"], cfg["data"]["user_id_col"]
    )
    val_ds = PhishGuardDataset(
        split.val, model.tokenizer,
        cfg["data"]["text_col"], cfg["data"]["label_col"], 
        cfg["model"]["max_length"], cfg["data"]["user_id_col"]
    )
    test_ds = PhishGuardDataset(
        split.test, model.tokenizer,
        cfg["data"]["text_col"], cfg["data"]["label_col"], 
        cfg["model"]["max_length"], cfg["data"]["user_id_col"]
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader  = DataLoader(test_ds, batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = len(train_loader) * cfg["train"]["num_epochs"]
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(total_steps*cfg["train"]["warmup_ratio"]), num_training_steps=total_steps)

    def evaluate(dl):
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=None)
                logits = out.logits
                preds = torch.argmax(logits, dim=-1)
                ys.extend(labels.cpu().numpy().tolist())
                yh.extend(preds.cpu().numpy().tolist())
        return compute_cls_metrics(ys, yh)

    if not eval_only:
        model.train()
        for epoch in range(cfg["train"]["num_epochs"]):
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # clean loss
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                cls_loss = out.loss

                # adversarial KL term
                adv_delta = adversarial_perturbation(model, {"input_ids": input_ids, "attention_mask": attn},
                                                    epsilon=cfg["loss"]["adv_eps"], steps=cfg["loss"]["adv_steps"])
                logits_clean = model.model(input_ids=input_ids, attention_mask=attn).logits
                logits_pert  = model.model(inputs_embeds=model.model.get_input_embeddings()(input_ids)+adv_delta,
                                           attention_mask=attn).logits
                adv_loss = kl_divergence_with_logits(logits_clean, logits_pert)

                # propagation loss proxy: encourage lower risk on frequent spreaders (batch proxy)
                # (In a full setup, compute sigma(S) on graph; here we use a differentiable proxy.)
                prop_loss = torch.relu(logits_clean[:,1]).mean()

                loss = cls_loss + cfg["loss"]["lambda_adv"] * adv_loss + cfg["loss"]["mu_prop"] * prop_loss
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()

            m = evaluate(val_loader)
            print("Val:", m)

    # Final evaluation + simple intervention demo (if edges exist)
    m_test = evaluate(test_loader)
    print("Test:", m_test)

    # Build graph & choose interventions (optional)
    if os.path.exists(cfg["data"]["edges_csv"]):
        import pandas as pd
        G = load_graph(cfg["data"]["edges_csv"])
        # Estimate user risk from model logits on validation set
        model.eval()
        user_ids = []
        logits_all = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=None)
                logits_all.append(out.logits)
                user_ids.extend(batch["user_id"] if isinstance(batch["user_id"], list) else [batch["user_id"]] * input_ids.size(0))
        if len(logits_all):
            logits = torch.cat(logits_all, dim=0)
            risk = risk_from_logits(user_ids, logits)
            df_users = pd.Series(user_ids)
            cand = pick_candidates(df_users, topk=int(cfg["propagation"]["topk_candidates"]))
            chosen = greedy_minimize_spread(G, budget=int(cfg["propagation"]["budget"]), risk=risk, candidates=cand,
                                            samples=int(cfg["propagation"]["ic_samples"]))
            print("Chosen intervention nodes:", chosen)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    run(args.config, eval_only=args.eval_only)
