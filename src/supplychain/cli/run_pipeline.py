"""CLI do pipeline completo de 4 fases."""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from supplychain import config
from supplychain.common.logging import setup_logging
from supplychain.common.graph_io import load_graph
from supplychain.cli._common import add_graphml_arg

logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run the full supply-chain risk analysis pipeline (Phases 1-4).",
    )
    add_graphml_arg(p)
    p.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Which phases to run (default: all).",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Phase 4 (OSV API validation) — useful for offline testing.",
    )
    p.add_argument(
        "--approximate-betweenness",
        action="store_true",
        help="Use approximate betweenness centrality (faster). "
             "Only relevant when --structural-metric betweenness.",
    )
    p.add_argument(
        "--structural-metric",
        choices=["none", "katz", "betweenness"],
        default=config.CENTRALITY_STRUCTURAL_METRIC,
        help=f"Phase 2 third structural component of C(v) "
             f"(default: {config.CENTRALITY_STRUCTURAL_METRIC}).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=config.ICM_N_TRIALS,
        help=f"Monte Carlo trials per seed for Phase 3 (default: {config.ICM_N_TRIALS}).",
    )
    p.add_argument(
        "--top-k-seeds",
        type=int,
        default=config.ICM_TOP_K_SEEDS,
        help=f"Number of top seeds for cascade simulation (default: {config.ICM_TOP_K_SEEDS}).",
    )
    p.add_argument(
        "--advisories-cache",
        default=None,
        help="Path to cached OSV advisories CSV (skip API calls in Phase 4).",
    )
    return p.parse_args()


def _run_phase_1(graph) -> pd.DataFrame:
    """Fase 1 — Risk Scoring: computa R(v) para todos os nós."""
    from supplychain.pipeline.risk import RiskScorer

    logger.info("FASE 1 — Risk Scoring")

    scorer = RiskScorer(graph)
    dest = scorer.export_csv()

    top20 = scorer.top_k(20)
    logger.info(
        "Top-20 por R(v):\n%s",
        top20[["risk_rank", "package", "risk_score", "in_degree_raw", "reach_raw"]]
        .to_string(index=False),
    )

    return scorer.compute()


def _run_phase_2(graph, risk_df, approximate=False,
                 structural_metric=config.CENTRALITY_STRUCTURAL_METRIC) -> pd.DataFrame:
    """Fase 2 — Centrality Metrics: computa C(v) e S(v)."""
    from supplychain.pipeline.centrality import CentralityAnalyzer

    logger.info("FASE 2 — Centrality Metrics")

    risk_scores = dict(zip(risk_df["package"], risk_df["risk_score"]))

    analyzer = CentralityAnalyzer(
        graph,
        risk_scores=risk_scores,
        structural_metric=structural_metric,
        betweenness_exact=not approximate,
    )
    dest = analyzer.export_csv()

    top20 = analyzer.top_k(20)
    display_cols = ["criticality_rank", "package", "criticality_index"]
    if "combined_score" in top20.columns:
        display_cols.append("combined_score")
    logger.info(
        "Top-20 por criticidade:\n%s",
        top20[display_cols].to_string(index=False),
    )

    return analyzer.compute()


def _run_phase_3(graph, risk_df, centrality_df, n_trials=config.ICM_N_TRIALS,
                 top_k_seeds=config.ICM_TOP_K_SEEDS) -> pd.DataFrame:
    """Fase 3 — Cascade Simulation: ICM Monte Carlo."""
    from supplychain.pipeline.cascade import CascadeSimulator

    logger.info("FASE 3 — Cascade Simulation (ICM)")

    risk_scores = dict(zip(risk_df["package"], risk_df["risk_score"]))

    if "combined_score" in centrality_df.columns:
        combined_scores = dict(zip(centrality_df["package"], centrality_df["combined_score"]))
    else:
        combined_scores = dict(zip(centrality_df["package"], centrality_df["criticality_index"]))

    sim = CascadeSimulator(graph, risk_scores)
    results_df = sim.run_top_k_seeds(
        combined_scores=combined_scores,
        k=top_k_seeds,
        n_trials=n_trials,
    )
    dest = sim.export_results(results_df)

    logger.info(
        "Top-10 blast radii:\n%s",
        results_df.head(10)[[
            "blast_rank", "seed", "mean_blast_radius",
            "ci_95_lower", "ci_95_upper", "max_blast_radius",
        ]].to_string(index=False),
    )

    return results_df


def _run_phase_4(graph, risk_df, advisories_cache=None) -> dict:
    """Fase 4 — OSV Validation: busca dados e roda testes estatísticos."""
    from supplychain.validation.validator import OSVValidator

    logger.info("FASE 4 — OSV.dev Validation")

    validator = OSVValidator(risk_df, graph)

    if advisories_cache and Path(advisories_cache).exists():
        logger.info("Carregando advisories em cache de %s", advisories_cache)
        gt_df = pd.read_csv(advisories_cache)
    else:
        gt_df = validator.fetch_advisories()
        validator.export_advisories(gt_df)

    results = validator.run_validation(ground_truth_df=gt_df)
    validator.export_validation_report(results)

    return results


def main():
    """Roda o pipeline completo de análise de risco em supply-chain."""
    setup_logging()

    args = _parse_args()
    phases = set(args.phases)

    if args.skip_validation:
        phases.discard(4)

    logger.info("Pipeline iniciando — fases: %s", sorted(phases))

    t0 = time.time()

    graph = load_graph(args.graphml)

    risk_df = None
    centrality_df = None

    if 1 in phases:
        risk_df = _run_phase_1(graph)
    elif config.RISK_SCORES_CSV.exists():
        logger.info("Carregando risk scores existentes de %s", config.RISK_SCORES_CSV)
        risk_df = pd.read_csv(config.RISK_SCORES_CSV)
    else:
        logger.error(
            "Fase 1 não selecionada e risk_scores.csv não encontrado. "
            "Não é possível continuar."
        )
        return

    if 2 in phases:
        centrality_df = _run_phase_2(
            graph, risk_df,
            approximate=args.approximate_betweenness,
            structural_metric=args.structural_metric,
        )
    elif config.CENTRALITY_SCORES_CSV.exists():
        logger.info("Carregando centrality scores existentes de %s", config.CENTRALITY_SCORES_CSV)
        centrality_df = pd.read_csv(config.CENTRALITY_SCORES_CSV)
    else:
        logger.warning(
            "Fase 2 não selecionada e centrality_scores.csv não encontrado. "
            "Fase 3 usará risk scores para seleção de seeds."
        )

    if 3 in phases:
        if centrality_df is None:
            centrality_df = risk_df[["package", "risk_score"]].copy()
            centrality_df["criticality_index"] = centrality_df["risk_score"]

        _run_phase_3(
            graph, risk_df, centrality_df,
            n_trials=args.n_trials,
            top_k_seeds=args.top_k_seeds,
        )

    if 4 in phases:
        _run_phase_4(
            graph, risk_df,
            advisories_cache=args.advisories_cache,
        )

    elapsed = time.time() - t0
    logger.info(
        "Pipeline concluído — tempo total: %.1f segundos (%.1f min)",
        elapsed, elapsed / 60,
    )


if __name__ == "__main__":
    main()
