"""CLI para construção do grafo de dependências."""

import argparse
import json
import logging

from supplychain import config
from supplychain.common.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Constrói o grafo de dependências PyPI a partir de BigQuery ou CSVs locais."""
    setup_logging()

    p = argparse.ArgumentParser(
        description="Constrói o grafo de dependências PyPI a partir de árvores completas.",
    )
    p.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Modo de ingestão: 'online' executa extração ao vivo via BigQuery; "
             "'offline' lê CSVs em cache (padrão: offline).",
    )
    p.add_argument(
        "--gcp-project",
        default=config.GCP_PROJECT_ID,
        help="Projeto de faturamento GCP para consultas BigQuery.",
    )
    p.add_argument(
        "--export",
        metavar="ARQUIVO",
        default=None,
        help="Caminho para o arquivo GraphML exportado (padrão: data/dependency_graph.graphml).",
    )
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="Executa a extração e salva os CSVs, mas pula a construção do grafo.",
    )
    args = p.parse_args()

    from supplychain.graph.builder import GraphBuilder, IngestionMode
    from supplychain.extraction.extractor import Extractor

    if args.extract_only:
        logger.info("Executando apenas extração...")
        ext = Extractor(gcp_project=args.gcp_project)

        seeds = ext.extract_download_counts()
        ext.extract_dependencies(seeds=seeds)
        ext.extract_usage()

        logger.info("Extração concluída. Arquivos CSV salvos em data/.")
        return

    if args.mode == "online":
        ext = Extractor(gcp_project=args.gcp_project)
        builder = GraphBuilder(mode=IngestionMode.ONLINE, extractor=ext)
    else:
        builder = GraphBuilder(mode=IngestionMode.OFFLINE)

    stats = builder.summary()
    logger.info("Resumo do grafo:\n%s", json.dumps(stats, indent=2))

    out = builder.export_graphml(output_path=args.export)
    logger.info("Concluído. Grafo gravado em %s", out)


if __name__ == "__main__":
    main()
