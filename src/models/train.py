"""Structure and parameter learning for the latent sentiment BBN."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.pkl"


def build_structure() -> DiscreteBayesianNetwork:
    return DiscreteBayesianNetwork(
        [
            ("firm_size", "engagement"),
            ("engagement", "latent_sentiment"),
        ]
    )


def fit_model(df: pd.DataFrame) -> DiscreteBayesianNetwork:
    model = build_structure()
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model


def save_model(model: DiscreteBayesianNetwork, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def main() -> None:
    # Minimal bootstrap data until dbt marts feed training.
    df = pd.DataFrame(
        {
            "firm_size": ["small", "large", "small", "large"] * 25,
            "engagement": ["low", "high", "high", "low"] * 25,
            "latent_sentiment": ["negative", "positive", "neutral", "positive"] * 25,
        }
    )
    model = fit_model(df)
    out = save_model(model)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
