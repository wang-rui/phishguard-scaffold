#!/usr/bin/env python3
"""
Run PhishGuard phishing detection on text.

Usage:
  # Single text
  python -m inference.run_inference --checkpoint runs/phishguard_exp/best_model.pt --text "Claim your prize now: https://bit.ly/xyz"

  # From a CSV (must have a text column)
  python -m inference.run_inference --checkpoint runs/phishguard_exp/best_model.pt --input data/tweets.csv --output predictions.csv --text-col text
"""

import argparse
import os
import sys

import torch

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.llama_classifier import PhishGuardClassifier


def load_model(checkpoint_path: str, device: str = None):
    """Load PhishGuard from a saved checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    model = PhishGuardClassifier(
        cfg["model"]["model_name_or_path"],
        num_labels=2,
        peft_cfg=cfg["model"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, cfg, device


def predict(model, cfg, device, texts, batch_size=32):
    """Predict phishing probability for a list of texts. Returns list of dicts with label and score."""
    max_length = int(cfg["model"].get("max_length", 256))
    tokenizer = model.tokenizer
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch)):
            p_phish = probs[j, 1].item()
            label = 1 if p_phish >= 0.5 else 0
            results.append({"label": label, "phishing_probability": p_phish})
    return results


def main():
    parser = argparse.ArgumentParser(description="PhishGuard inference")
    parser.add_argument("--checkpoint", type=str, default="runs/phishguard_exp/best_model.pt")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--input", type=str, help="Input CSV with a text column")
    parser.add_argument("--output", type=str, help="Output CSV with predictions")
    parser.add_argument("--text-col", type=str, default="text")
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("Provide --text or --input CSV")

    model, cfg, device = load_model(args.checkpoint)

    if args.text:
        results = predict(model, cfg, device, [args.text])
        r = results[0]
        print(f"Label: {'phishing' if r['label'] == 1 else 'legitimate'}")
        print(f"P(phishing): {r['phishing_probability']:.4f}")
        return

    # CSV mode
    import pandas as pd
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not in CSV. Columns: {list(df.columns)}")
    texts = df[args.text_col].astype(str).tolist()
    results = predict(model, cfg, device, texts)
    df["predicted_label"] = [r["label"] for r in results]
    df["phishing_probability"] = [r["phishing_probability"] for r in results]
    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
