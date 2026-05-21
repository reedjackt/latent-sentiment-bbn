"""Feature availability rules that guard against training-time leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

ScoringMoment = Literal["lead_capture", "post_engagement"]

SCORING_TIMESTAMP_COLUMN = "captured_at"
TARGET_COLUMN = "demo_requested_bool"

CAPTURE_TIME_EVIDENCE_COLUMNS = (
    "marketing_channel_clean",
    "campaign_tier_clean",
    "region_clean",
    "employee_size_bucket",
    "job_title_clean",
    "web_session_band",
)
POST_CAPTURE_EVIDENCE_COLUMNS = ("replied_within_7d_bool",)
PROXY_SCORE_EVIDENCE_COLUMNS = ("lead_score_band",)

CAPTURE_TIME_TRAINING_COLUMNS = CAPTURE_TIME_EVIDENCE_COLUMNS + (TARGET_COLUMN,)
EXTENDED_FUNNEL_TRAINING_COLUMNS = (
    CAPTURE_TIME_EVIDENCE_COLUMNS
    + POST_CAPTURE_EVIDENCE_COLUMNS
    + PROXY_SCORE_EVIDENCE_COLUMNS
    + (TARGET_COLUMN,)
)


@dataclass(frozen=True)
class FeatureAvailabilityContract:
    """Rules describing which columns are legitimate evidence at score time."""

    scoring_timestamp_column: str = SCORING_TIMESTAMP_COLUMN
    target_column: str = TARGET_COLUMN
    scoring_moment: ScoringMoment = "lead_capture"
    capture_time_evidence_columns: tuple[str, ...] = CAPTURE_TIME_EVIDENCE_COLUMNS
    post_capture_evidence_columns: tuple[str, ...] = POST_CAPTURE_EVIDENCE_COLUMNS
    proxy_score_evidence_columns: tuple[str, ...] = PROXY_SCORE_EVIDENCE_COLUMNS

    @property
    def default_training_columns(self) -> tuple[str, ...]:
        return self.capture_time_evidence_columns + (self.target_column,)

    @property
    def allowed_evidence_columns(self) -> tuple[str, ...]:
        if self.scoring_moment == "post_engagement":
            return (
                self.capture_time_evidence_columns
                + self.post_capture_evidence_columns
                + self.proxy_score_evidence_columns
            )
        return self.capture_time_evidence_columns


DEFAULT_FEATURE_CONTRACT = FeatureAvailabilityContract()


def validate_feature_availability(
    columns: Iterable[str],
    contract: FeatureAvailabilityContract = DEFAULT_FEATURE_CONTRACT,
    *,
    allow_post_capture_features: bool = False,
    allow_proxy_score_features: bool = False,
) -> None:
    """Reject evidence columns that would not be known at the scoring timestamp."""
    selected_columns = tuple(columns)
    evidence_columns = tuple(
        column for column in selected_columns if column != contract.target_column
    )
    allowed = set(contract.capture_time_evidence_columns)
    blocked_reasons: dict[str, str] = {}

    if allow_post_capture_features:
        allowed.update(contract.post_capture_evidence_columns)
    else:
        blocked_reasons.update(
            {
                column: "post-capture behavior is not known at lead capture"
                for column in contract.post_capture_evidence_columns
            }
        )

    if allow_proxy_score_features:
        allowed.update(contract.proxy_score_evidence_columns)
    else:
        blocked_reasons.update(
            {
                column: "proxy scores must be proven available before score time"
                for column in contract.proxy_score_evidence_columns
            }
        )

    disallowed = [column for column in evidence_columns if column not in allowed]
    if disallowed:
        details = [
            f"{column} ({blocked_reasons.get(column, 'not in the feature contract')})"
            for column in disallowed
        ]
        raise ValueError(
            "Training columns violate the feature availability contract: "
            + ", ".join(details)
        )

