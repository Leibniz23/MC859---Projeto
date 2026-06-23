"""CLI para coleta de métricas iniciais do grafo."""

import argparse
import logging
import sys
from pathlib import Path

from supplychain.common.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Analisa um GraphML de dependências PyPI e produz plots de métricas."""
    setup_logging()

    p = argparse.ArgumentParser(
        description="Analisa um grafo de dependências PyPI e produz um relatório de métricas iniciais.",
    )
    p.add_argument(
        "--graphml",
        default="data/dependency_graph.graphml",
        help="Caminho para o arquivo GraphML de entrada (padrão: dependency_graph.graphml).",
    )
    p.add_argument(
        "--output-dir",
        default="data/analysis_results",
        help="Diretório para os gráficos e relatório de saída.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI para as figuras salvas (padrão: 200, mínimo: 150).",
    )
    args = p.parse_args()

    _collect_initial_metrics(
        graphml_path=args.graphml,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


def _collect_initial_metrics(graphml_path="dependency_graph.graphml", output_dir=".", dpi=200):
    """Roda o pipeline completo de métricas iniciais."""
    from supplychain.graph.analyzer import GraphAnalyzer
    from supplychain.graph.plotter import GraphPlotter

    output_dir = Path(output_dir)

    logger.info("Iniciando coleta de métricas do grafo de dependências PyPI...")

    try:
        analyzer = GraphAnalyzer(graphml_path)
    except FileNotFoundError:
        logger.error("Arquivo de entrada não encontrado: %s", graphml_path)
        sys.exit(1)

    topography = analyzer.basic_topography()
    degrees = analyzer.degree_sequences()
    scc_info = analyzer.scc_analysis()
    wcc_info = analyzer.wcc_analysis()

    plotter = GraphPlotter(output_dir=output_dir, dpi=dpi)

    degree_plot = plotter.plot_degree_distribution(
        in_degrees=degrees["in"],
        out_degrees=degrees["out"],
    )

    scc_plot = None
    if scc_info["total_sccs"] > 1:
        scc_plot = plotter.plot_scc_distribution(
            size_distribution=scc_info["size_distribution"],
            largest_scc_size=scc_info["largest_scc_size"],
        )

    wcc_plot = None
    if wcc_info["total_wccs"] > 1:
        wcc_plot = plotter.plot_wcc_distribution(
            size_distribution=wcc_info["size_distribution"],
            largest_wcc_size=wcc_info["largest_wcc_size"],
        )

    logger.info("Pipeline concluído! Gráficos -> %s", output_dir)


if __name__ == "__main__":
    main()
