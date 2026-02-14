#!/usr/bin/env python3
"""
Push a trained PhishGuard checkpoint to the Hugging Face Hub.

Prerequisites:
  1. Install: pip install huggingface_hub
  2. Log in: huggingface-cli login   (or set HF_TOKEN env var)

Usage:
  python scripts/push_to_hub.py --repo_id your-username/phishguard-distilbert
  python scripts/push_to_hub.py --repo_id your-username/phishguard-distilbert --checkpoint runs/phishguard_exp/best_model.pt
"""

import argparse
import os
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Push PhishGuard model to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hub repo id, e.g. username/phishguard-distilbert",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/phishguard_exp/best_model.pt",
        help="Path to checkpoint (best_model.pt or model.pt)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train first with: "
            "python -m training.train --config configs/config_mac_distilbert.yaml"
        )

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface_hub. "
            "Then log in: huggingface-cli login"
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Create repo if needed (idempotent)
    create_repo(args.repo_id, private=args.private, exist_ok=True)

    # 1. Upload the checkpoint (state_dict + config inside)
    upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="model.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )

    # 2. Save and upload config as YAML for easy viewing
    config = ckpt.get("config")
    if config:
        config_path = "runs/phishguard_exp/config_snapshot.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.yaml",
            repo_id=args.repo_id,
            repo_type="model",
        )
        if os.path.isfile(config_path):
            os.remove(config_path)

    # 3. Upload a model card README
    model_name = config.get("model", {}).get("model_name_or_path", "unknown") if config else "unknown"
    readme = f"""---
license: mit
tags:
  - phishing-detection
  - text-classification
  - distilbert
  - phishguard
---

# PhishGuard – Phishing detection

Phishing classifier from the [PhishGuard](https://github.com/your-org/phishguard-scaffold) framework (Joint Semantic Detection & Propagation Control).

## Model

- **Base:** {model_name}
- **Task:** Binary text classification (phishing vs legitimate)
- **Checkpoint:** Trained with PhishGuard (classification + adversarial + propagation losses)

## Usage

Install the project and dependencies, then load this checkpoint:

```python
import torch
from models.llama_classifier import PhishGuardClassifier

# Download from Hub (after cloning the PhishGuard repo)
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="{args.repo_id}", filename="model.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=True)
cfg = ckpt["config"]

model = PhishGuardClassifier(
    cfg["model"]["model_name_or_path"],
    num_labels=2,
    peft_cfg=cfg["model"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Infer
inputs = model.tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=cfg["model"]["max_length"])
with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
probs = torch.softmax(logits, dim=-1)
# probs[0, 1] = P(phishing)
```

## Training

Trained with `training.train` and config in `config.yaml` in this repo.
"""
    readme_path = "runs/phishguard_exp/README.md"
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, "w") as f:
        f.write(readme)
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    if os.path.isfile(readme_path):
        os.remove(readme_path)

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    print("  - model.pt")
    if config:
        print("  - config.yaml")
    print("  - README.md")


if __name__ == "__main__":
    main()
