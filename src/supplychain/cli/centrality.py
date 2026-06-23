"""CLI para a Fase 2: Centrality Analysis."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from supplychain import config
from supplychain.common.logging import setup_logging
from supplychain.common.graph_io import load_graph
from supplychain.cli._common import add_graphml_arg, add_output_arg, add_top_k_arg

logger = logging.getLogger(__name__)


def main():
    """Computa métricas de centralidade para o grafo de dependências PyPI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Computa métricas de centralidade para o grafo de dependências PyPI.",
    )
    add_graphml_arg(parser)
    parser.add_argument(
        "--risk-scores",
        default=str(config.RISK_SCORES_CSV),
        help="Caminho para risk_scores.csv (para computar S(v)).",
    )
    add_output_arg(
        parser,
        help_text="Caminho do CSV de saída (padrão: data/centrality_scores.csv).",
    )
    parser.add_argument(
        "--approximate",
        action="store_true",
        help="Usa betweenness aproximada (mais rápido). Só relevante com --structural-metric betweenness.",
    )
    parser.add_argument(
        "--structural-metric",
        choices=["none", "katz", "betweenness"],
        default=config.CENTRALITY_STRUCTURAL_METRIC,
        help="Terceiro componente estrutural de C(v) (padrão: %(default)s).",
    )
    add_top_k_arg(parser, default=20, help_text="Imprime top-k pacotes mais críticos.")
    args = parser.parse_args()

    graph = load_graph(args.graphml)

    risk_scores = None
    risk_path = Path(args.risk_scores)
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        risk_scores = dict(zip(risk_df["package"], risk_df["risk_score"]))
        logger.info("Risk scores carregados: %d pacotes.", len(risk_scores))
    else:
        logger.warning(
            "Arquivo de risk scores não encontrado (%s) — S(v) não será computado.",
            risk_path,
        )

    from supplychain.pipeline.centrality import CentralityAnalyzer
    analyzer = CentralityAnalyzer(
        graph,
        risk_scores=risk_scores,
        structural_metric=args.structural_metric,
        betweenness_exact=not args.approximate,
    )
    dest = analyzer.export_csv(output_path=args.output)

    top = analyzer.top_k(args.top_k)
    print(f"\n{'=' * 90}")
    print(f"  Top-{args.top_k} Most Critical Packages")
    print(f"{'=' * 90}")

    display_cols = ["criticality_rank", "package", "criticality_index"]
    if "combined_score" in top.columns:
        display_cols.append("combined_score")
    display_cols.extend(["structural", "pagerank"])

    print(top[display_cols].to_string(index=False))
    print(f"\nFull results exported to: {dest}\n")


if __name__ == "__main__":
    main()
