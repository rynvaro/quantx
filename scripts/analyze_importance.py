#!/usr/bin/env python3
"""Analyze feature importance from trained XGBoost models."""

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_model(model_dir: Path, direction: str) -> None:
    """Extract and display feature importance from XGBoost model."""
    model_path = model_dir / "model_cls.json"
    feature_cols_path = model_dir / "feature_cols.txt"

    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        return
    if not feature_cols_path.exists():
        print(f"ERROR: {feature_cols_path} not found")
        return

    # Load feature names
    with open(feature_cols_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # Load model and extract importance
    with open(model_path) as f:
        model_json = json.load(f)

    # XGBoost JSON format: feature_importances in learner.feature_names/feature_types
    # We'll use get_score() equivalent from JSON
    importance_dict = {}

    # Parse trees to count feature splits (gain-based importance)
    trees = model_json.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", [])

    for tree in trees:
        for node in tree.get("split_indices", []):
            if node >= 0 and node < len(feature_names):
                importance_dict[feature_names[node]] = importance_dict.get(feature_names[node], 0) + 1

    if not importance_dict:
        print("WARNING: Could not extract importance from JSON. Using simple feature count.")
        # Fallback: just list features
        importance_dict = {name: 1 for name in feature_names}

    # Sort by importance
    importance_df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in sorted(importance_dict.items(), key=lambda x: -x[1])
    ])

    print(f"\n{'='*60}")
    print(f"Feature Importance: {direction.upper()} model")
    print(f"{'='*60}")
    print(f"Total features: {len(feature_names)}")
    print(f"\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    # Save to CSV
    output_path = model_dir / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull importance saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--models-dir", type=Path, default=Path("models/v1"))
    args = parser.parse_args()

    if args.direction in ["long", "both"]:
        analyze_model(args.models_dir / "long", "long")

    if args.direction in ["short", "both"]:
        analyze_model(args.models_dir / "short", "short")


if __name__ == "__main__":
    main()
