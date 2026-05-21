"""Constrained structure learning for the latent sentiment BBN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from networkx.algorithms.dag import is_directed_acyclic_graph
from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.estimators import BIC, ExpectationMaximization, HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork

from models.feature_contract import (
    CAPTURE_TIME_TRAINING_COLUMNS,
    EXTENDED_FUNNEL_TRAINING_COLUMNS,
    TARGET_COLUMN,
)

LATENT_NODE = "latent_sentiment"
LATENT_STATES = ("negative", "neutral", "positive")
LATENT_CARDINALITY = len(LATENT_STATES)
OUTCOME_COLUMN = TARGET_COLUMN
DEBUG_LATENT_COLUMN = "True_Consumer_Sentiment"

EXOGENOUS_COLUMNS = (
    "marketing_channel_clean",
    "campaign_tier_clean",
    "region_clean",
    "employee_size_bucket",
    "job_title_clean",
)
SIGNAL_COLUMNS = (
    "web_session_band",
    "replied_within_7d_bool",
    "lead_score_band",
)
TRAINING_COLUMNS = CAPTURE_TIME_TRAINING_COLUMNS
EXTENDED_TRAINING_COLUMNS = EXTENDED_FUNNEL_TRAINING_COLUMNS
LATENT_CHILDREN = SIGNAL_COLUMNS + (OUTCOME_COLUMN,)
PLAUSIBLE_SIGNAL_EDGES = (
    ("web_session_band", "replied_within_7d_bool"),
    ("web_session_band", "lead_score_band"),
    ("web_session_band", OUTCOME_COLUMN),
    ("replied_within_7d_bool", "lead_score_band"),
    ("replied_within_7d_bool", OUTCOME_COLUMN),
    ("lead_score_band", OUTCOME_COLUMN),
)


Edge = tuple[str, str]


@dataclass(frozen=True)
class StructureConstraints:
    """Edge constraints passed to pgmpy structure learning."""

    required_edges: frozenset[Edge]
    forbidden_edges: frozenset[Edge]
    search_space: frozenset[Edge]
    latent_edges: frozenset[Edge]


@dataclass(frozen=True)
class StructureLearningConfig:
    """Runtime knobs for hill-climb structure learning and EM fitting."""

    max_indegree: int = 3
    tabu_length: int = 100
    max_iter: int = 1_000_000
    epsilon: float = 1e-4
    show_progress: bool = True
    em_max_iter: int = 100
    em_atol: float = 1e-8
    em_seed: int = 42
    latent_cardinality: int = LATENT_CARDINALITY


def prepare_observed_training_frame(
    df: pd.DataFrame,
    columns: Iterable[str] = TRAINING_COLUMNS,
) -> pd.DataFrame:
    """Select observed BBN columns and coerce all states to pgmpy-friendly strings."""
    selected_columns = tuple(columns)
    _validate_training_columns(df, selected_columns)

    observed = df.loc[:, selected_columns].copy()
    for column in selected_columns:
        observed[column] = (
            observed[column]
            .astype("object")
            .where(observed[column].notna(), "missing")
            .map(_normalize_state)
        )
    return observed


def build_structure_constraints(columns: Iterable[str] = TRAINING_COLUMNS) -> StructureConstraints:
    """Build whitelist, blacklist, and post-learning latent edges."""
    nodes = tuple(columns)
    node_set = set(nodes)
    search_space = _build_search_space(nodes)
    all_directed_pairs = {
        (source, target)
        for source in nodes
        for target in nodes
        if source != target
    }
    forbidden_edges = all_directed_pairs - search_space
    latent_edges = {
        (LATENT_NODE, child)
        for child in LATENT_CHILDREN
        if child in node_set
    }
    return StructureConstraints(
        required_edges=frozenset(),
        forbidden_edges=frozenset(forbidden_edges),
        search_space=frozenset(search_space),
        latent_edges=frozenset(latent_edges),
    )


def learn_observed_dag(
    observed_df: pd.DataFrame,
    config: StructureLearningConfig | None = None,
) -> tuple[DAG, float, StructureConstraints]:
    """Run constrained hill-climb structure search with BIC over observed nodes only."""
    config = config or StructureLearningConfig()
    constraints = build_structure_constraints(observed_df.columns)
    expert_knowledge = ExpertKnowledge(
        forbidden_edges=constraints.forbidden_edges,
        required_edges=constraints.required_edges,
        search_space=constraints.search_space,
    )
    score = BIC(observed_df)
    dag = HillClimbSearch(observed_df).estimate(
        scoring_method=score,
        expert_knowledge=expert_knowledge,
        max_indegree=config.max_indegree,
        tabu_length=config.tabu_length,
        max_iter=config.max_iter,
        epsilon=config.epsilon,
        show_progress=config.show_progress,
    )
    dag.add_nodes_from(observed_df.columns)
    _validate_learned_edges(dag.edges(), constraints)
    return dag, float(score.score(dag)), constraints


def build_latent_model(
    observed_dag: DAG,
    constraints: StructureConstraints | None = None,
) -> DiscreteBayesianNetwork:
    """Insert the hidden latent node and its expert-specified child edges."""
    constraints = constraints or build_structure_constraints(observed_dag.nodes())
    edges = set(observed_dag.edges()) | set(constraints.latent_edges)
    model = DiscreteBayesianNetwork(edges, latents={LATENT_NODE})
    model.add_nodes_from(observed_dag.nodes())
    model.add_node(LATENT_NODE)
    if not is_directed_acyclic_graph(model):
        raise ValueError("Latent structure introduced a cycle")
    return model


def fit_latent_parameters(
    model: DiscreteBayesianNetwork,
    observed_df: pd.DataFrame,
    config: StructureLearningConfig | None = None,
) -> DiscreteBayesianNetwork:
    """Estimate final CPTs for observed and hidden variables with EM."""
    config = config or StructureLearningConfig()
    if LATENT_NODE not in model.nodes():
        raise ValueError(f"Model is missing latent node: {LATENT_NODE}")
    if LATENT_NODE in observed_df.columns:
        raise ValueError(f"{LATENT_NODE} must not be present in observed training data")

    cpds = ExpectationMaximization(model, observed_df).get_parameters(
        latent_card={LATENT_NODE: config.latent_cardinality},
        apply_smoothing=True,
        max_iter=config.em_max_iter,
        atol=config.em_atol,
        seed=config.em_seed,
        show_progress=config.show_progress,
    )
    model.add_cpds(*cpds)
    model.check_model()
    return model


def learn_structure_and_fit_parameters(
    df: pd.DataFrame,
    config: StructureLearningConfig | None = None,
    columns: Iterable[str] = TRAINING_COLUMNS,
) -> tuple[DiscreteBayesianNetwork, float, StructureConstraints]:
    """Full structure-learning path from observed training data to fitted latent BBN."""
    observed_df = prepare_observed_training_frame(df, columns)
    observed_dag, bic_score, constraints = learn_observed_dag(observed_df, config)
    model = build_latent_model(observed_dag, constraints)
    fitted = fit_latent_parameters(model, observed_df, config)
    return fitted, bic_score, constraints


def _build_search_space(nodes: tuple[str, ...]) -> set[Edge]:
    """Whitelist observed edges without forcing all signal through the latent node.

    Exogenous context can affect engagement, proxy scores, and conversion directly:
    a campaign tier or job title changes exposure and buying authority even when two
    leads share the same hidden sentiment. Signal-to-signal edges are kept explicit
    and directional so the learned DAG can capture plausible funnels while
    ``max_indegree`` still bounds parent complexity during hill-climb search.
    """
    node_set = set(nodes)
    exogenous = tuple(column for column in EXOGENOUS_COLUMNS if column in node_set)
    signals = tuple(column for column in SIGNAL_COLUMNS if column in node_set)
    search_space: set[Edge] = set()

    downstream = signals + ((OUTCOME_COLUMN,) if OUTCOME_COLUMN in node_set else ())
    for source in exogenous:
        for target in downstream:
            search_space.add((source, target))

    for edge in PLAUSIBLE_SIGNAL_EDGES:
        if edge[0] in node_set and edge[1] in node_set:
            search_space.add(edge)
    return search_space


def _validate_training_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if df.empty:
        raise ValueError("Training data is empty")
    if DEBUG_LATENT_COLUMN in df.columns:
        raise ValueError(f"{DEBUG_LATENT_COLUMN} must not be used for BBN training")
    if LATENT_NODE in df.columns:
        raise ValueError(f"{LATENT_NODE} is hidden and must not be in training data")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")


def _validate_learned_edges(
    edges: Iterable[Edge],
    constraints: StructureConstraints,
) -> None:
    forbidden = set(edges) & set(constraints.forbidden_edges)
    if forbidden:
        raise ValueError(f"Learned forbidden edges: {sorted(forbidden)}")


def _normalize_state(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
