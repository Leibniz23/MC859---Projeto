"""CLI para a Fase 1: Risk Scoring."""

import argparse
import logging

from supplychain import config
from supplychain.common.logging import setup_logging
from supplychain.common.graph_io import load_graph
from supplychain.cli._common import add_graphml_arg, add_output_arg, add_top_k_arg

logger = logging.getLogger(__name__)


def main():
    """Computa scores de risco sistêmico para o grafo de dependências PyPI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Computa scores de risco sistêmico para o grafo de dependências PyPI.",
    )
    add_graphml_arg(parser)
    add_output_arg(parser, help_text="Caminho do CSV de saída (padrão: data/risk_scores.csv).")
    add_top_k_arg(
        parser,
        default=20,
        help_text="Imprime os top-k pacotes de maior risco (padrão: 20).",
    )
    args = parser.parse_args()

    graph = load_graph(args.graphml)

    from supplychain.pipeline.risk import RiskScorer
    scorer = RiskScorer(graph)
    dest = scorer.export_csv(output_path=args.output)

    top = scorer.top_k(args.top_k)
    print(f"\n{'=' * 80}")
    print(f"  Top-{args.top_k} Highest-Risk Packages")
    print(f"{'=' * 80}")
    print(
        top[["risk_rank", "package", "risk_score", "in_degree_raw", "reach_raw", "usage_norm"]]
        .to_string(index=False)
    )
    print(f"\nFull results exported to: {dest}\n")


if __name__ == "__main__":
    main()
