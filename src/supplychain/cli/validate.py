"""CLI para a Fase 4: OSV Validation."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from supplychain import config
from supplychain.common.logging import setup_logging
from supplychain.common.graph_io import load_graph
from supplychain.cli._common import add_graphml_arg, add_output_dir_arg

logger = logging.getLogger(__name__)


def main():
    """Valida o modelo de risco/criticidade contra dados OSV.dev."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Valida o modelo de risco/criticidade contra dados OSV.dev.",
    )
    add_graphml_arg(parser)
    parser.add_argument(
        "--risk-scores",
        default=str(config.RISK_SCORES_CSV),
        help="Caminho para risk_scores.csv.",
    )
    parser.add_argument(
        "--advisories-cache",
        default=None,
        help="Caminho para CSV de advisories em cache (evita chamadas à API).",
    )
    parser.add_argument(
        "--skip-api-calls",
        action="store_true",
        help="Reutiliza osv_advisories.csv em cache (sem chamadas de rede).",
    )
    add_output_dir_arg(
        parser,
        default=str(config.DATA_DIR),
        help_text="Diretório de saída para os resultados.",
    )
    parser.add_argument(
        "--group-a-size",
        type=int,
        default=config.OSV_GROUP_A_SIZE,
        help=f"Número de pacotes de alto risco no Grupo A (padrão: {config.OSV_GROUP_A_SIZE}).",
    )
    args = parser.parse_args()

    advisories_cache = args.advisories_cache
    if args.skip_api_calls and advisories_cache is None:
        advisories_cache = str(config.OSV_ADVISORIES_CSV)

    graph = load_graph(args.graphml)

    risk_path = Path(args.risk_scores)
    if not risk_path.exists():
        logger.error("Risk scores não encontrados: %s", risk_path)
        return
    risk_df = pd.read_csv(risk_path)

    from supplychain.validation.validator import OSVValidator
    validator = OSVValidator(risk_df, graph, group_a_size=args.group_a_size)

    if advisories_cache and Path(advisories_cache).exists():
        logger.info("Carregando advisories em cache de %s", advisories_cache)
        gt_df = pd.read_csv(advisories_cache)
    else:
        gt_df = validator.fetch_advisories()
        validator.export_advisories(gt_df)

    results = validator.run_validation(ground_truth_df=gt_df)

    validator.export_validation_report(results)

    print(f"\n{'=' * 70}")
    print("  VALIDATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(json.dumps(
        {k: v for k, v in results.items() if k not in ("cvss_sensitivity", "bootstrap_ci")},
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
