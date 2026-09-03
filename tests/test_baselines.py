import pandas as pd

from models.baselines import (
    RANDOM_FOREST_BASELINE,
    XGBOOST_BASELINE,
    BaselineConfig,
    predict_all_baselines,
    train_discriminative_baselines,
)
from models.feature_contract import TARGET_COLUMN


def test_discriminative_baselines_fit_and_predict_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "marketing_channel_clean": [
                "paid_search",
                "paid_search",
                "organic",
                "organic",
                "referral",
                "referral",
                "direct",
                "direct",
            ],
            "web_session_band": [
                "deep",
                "deep",
                "short",
                "short",
                "medium",
                "deep",
                "short",
                "medium",
            ],
            TARGET_COLUMN: [
                "true",
                "true",
                "false",
                "false",
                "true",
                "true",
                "false",
                "false",
            ],
        }
    )

    models = train_discriminative_baselines(
        frame,
        config=BaselineConfig(
            random_forest_estimators=5,
            xgboost_estimators=3,
            xgboost_max_depth=2,
        ),
    )
    predictions = predict_all_baselines(models, frame)

    assert set(models) == {RANDOM_FOREST_BASELINE, XGBOOST_BASELINE}
    assert set(predictions) == {RANDOM_FOREST_BASELINE, XGBOOST_BASELINE}
    for probabilities in predictions.values():
        assert len(probabilities) == len(frame)
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)
