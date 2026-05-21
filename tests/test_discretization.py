import math

import polars as pl

from models.discretization import (
    BinSpec,
    apply_discretization,
    fit_discretization_config,
    load_discretization_config,
    save_discretization_config,
)


def test_fit_apply_and_reload_discretization_config(tmp_path) -> None:
    df = pl.DataFrame(
        {
            "lead_score_clean": [10.0, 45.0, 70.0, 95.0, None, math.nan],
            "web_session_seconds_clean": [5.0, 30.0, 120.0, 300.0, None, 900.0],
        }
    )
    specs = [
        BinSpec(
            source_column="lead_score_clean",
            output_column="lead_score_band",
            labels=["low", "medium", "high"],
            strategy="fixed",
            upper_bounds=[40.0, 80.0],
        ),
        BinSpec(
            source_column="web_session_seconds_clean",
            output_column="web_session_band",
            labels=["short", "engaged", "deep"],
            strategy="quantile",
        ),
    ]

    config = fit_discretization_config(df, specs)
    out = apply_discretization(df, config)

    assert out["lead_score_band"].to_list() == [
        "low",
        "medium",
        "medium",
        "high",
        "missing",
        "missing",
    ]
    assert set(out["web_session_band"].drop_nulls()) <= {
        "short",
        "engaged",
        "deep",
        "missing",
    }

    config_path = tmp_path / "discretization_config.json"
    save_discretization_config(config, config_path)
    reloaded = load_discretization_config(config_path)

    assert apply_discretization(df, reloaded).select(
        "lead_score_band",
        "web_session_band",
    ).equals(out.select("lead_score_band", "web_session_band"))
