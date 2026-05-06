#!/usr/bin/env python3
"""Analyze feature importance from trained XGBoost or LightGBM models."""

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_model(model_dir: Path, direction: str) -> None:
    """Extract and display feature importance from XGBoost or LightGBM model."""
    feature_cols_path = model_dir / "feature_cols.txt"
    lgb_path = model_dir / "model_cls.lgb"
    xgb_path = model_dir / "model_cls.json"

    if not feature_cols_path.exists():
        print(f"ERROR: {feature_cols_path} not found")
        return

    with open(feature_cols_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    if lgb_path.exists():
        # ── LightGBM path ──────────────────────────────────────────
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(lgb_path))
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        # Map generic Column_N names back to real feature names via index
        if names and names[0].startswith("Column_"):
            names = [
                feature_names[int(n.split("_")[1])] if n.startswith("Column_") else n
                for n in names
            ]
        importance_df = (
            pd.DataFrame({"feature": names, "importance": gains})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    elif xgb_path.exists():
        # ── XGBoost path ───────────────────────────────────────────
        with open(xgb_path) as f:
            model_json = json.load(f)
        importance_dict: dict[str, int] = {}
        trees = (model_json.get("learner", {})
                           .get("gradient_booster", {})
                           .get("model", {})
                           .get("trees", []))
        for tree in trees:
            for node in tree.get("split_indices", []):
                if 0 <= node < len(feature_names):
                    importance_dict[feature_names[node]] = (
                        importance_dict.get(feature_names[node], 0) + 1
                    )
        if not importance_dict:
            importance_dict = {name: 1 for name in feature_names}
        importance_df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(importance_dict.items(), key=lambda x: -x[1])
        ])
    else:
        print(f"ERROR: no model file found in {model_dir}")
        return

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
