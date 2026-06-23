"""CLI para a Fase 3: Cascade Simulation."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from supplychain import config
from supplychain.common.logging import setup_logging
from supplychain.common.graph_io import load_graph
from supplychain.cli._common import add_graphml_arg, add_output_arg

logger = logging.getLogger(__name__)


def main():
    """Roda simulação ICM de cascata no grafo de dependências PyPI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Roda simulação ICM de cascata no grafo de dependências PyPI.",
    )
    add_graphml_arg(parser)
    parser.add_argument(
        "--risk-scores",
        default=str(config.RISK_SCORES_CSV),
        help="Caminho para risk_scores.csv.",
    )
    parser.add_argument(
        "--centrality-scores",
        default=str(config.CENTRALITY_SCORES_CSV),
        help="Caminho para centrality_scores.csv (para seleção de seeds por S(v)).",
    )
    add_output_arg(
        parser,
        help_text="Caminho do CSV de saída (padrão: data/simulation_results.csv).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.ICM_TOP_K_SEEDS,
        help=f"Número de seeds a simular (padrão: {config.ICM_TOP_K_SEEDS}).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=config.ICM_N_TRIALS,
        help=f"Trials Monte Carlo por seed (padrão: {config.ICM_N_TRIALS}).",
    )
    args = parser.parse_args()

    graph = load_graph(args.graphml)

    risk_path = Path(args.risk_scores)
    if not risk_path.exists():
        logger.error("Risk scores não encontrados: %s", risk_path)
        return
    risk_df = pd.read_csv(risk_path)
    risk_scores = dict(zip(risk_df["package"], risk_df["risk_score"]))
    logger.info("Risk scores carregados: %d pacotes.", len(risk_scores))

    centrality_path = Path(args.centrality_scores)
    if centrality_path.exists():
        cent_df = pd.read_csv(centrality_path)
        if "combined_score" in cent_df.columns:
            combined_scores = dict(zip(cent_df["package"], cent_df["combined_score"]))
        else:
            combined_scores = dict(zip(cent_df["package"], cent_df["criticality_index"]))
        logger.info("Usando combined scores para seleção de seeds.")
    else:
        logger.warning(
            "Centrality scores não encontrados (%s) — usando R(v) para seeds.",
            centrality_path,
        )
        combined_scores = risk_scores

    from supplychain.pipeline.cascade import CascadeSimulator
    sim = CascadeSimulator(graph, risk_scores)
    results_df = sim.run_top_k_seeds(
        combined_scores=combined_scores,
        k=args.top_k,
        n_trials=args.n_trials,
    )
    dest = sim.export_results(results_df, output_path=args.output)

    print(f"\n{'=' * 100}")
    print(f"  ICM Cascade Simulation Results — Top-{args.top_k} Seeds × {args.n_trials} Trials")
    print(f"{'=' * 100}")
    display_cols = [
        "blast_rank", "seed", "combined_score",
        "mean_blast_radius", "std_blast_radius",
        "ci_95_lower", "ci_95_upper", "max_blast_radius", "mean_depth",
    ]
    print(results_df.head(20)[display_cols].to_string(index=False))
    print(f"\nFull results exported to: {dest}\n")


if __name__ == "__main__":
    main()
