"""Fragmentos argparse reutilizados pelos entry-points do CLI."""

import argparse
from supplychain import config


def add_graphml_arg(parser: argparse.ArgumentParser, default=None):
    """Adiciona --graphml ao parser."""
    parser.add_argument(
        "--graphml",
        default=default if default is not None else str(config.GRAPH_OUTPUT),
        help="Caminho para o arquivo GraphML de entrada (padrão: %(default)s).",
    )


def add_output_arg(parser: argparse.ArgumentParser, help_text="Caminho do CSV de saída."):
    """Adiciona --output ao parser."""
    parser.add_argument(
        "--output",
        default=None,
        help=help_text,
    )


def add_output_dir_arg(parser: argparse.ArgumentParser, default=None, help_text="Diretório de saída."):
    """Adiciona --output-dir ao parser."""
    parser.add_argument(
        "--output-dir",
        default=default if default is not None else str(config.DATA_DIR),
        help=help_text,
    )


def add_top_k_arg(parser: argparse.ArgumentParser, default=20, help_text=None):
    """Adiciona --top-k ao parser."""
    parser.add_argument(
        "--top-k",
        type=int,
        default=default,
        help=help_text or f"Número de pacotes top a imprimir (padrão: {default}).",
    )


def add_risk_scores_arg(parser: argparse.ArgumentParser):
    """Adiciona --risk-scores ao parser."""
    parser.add_argument(
        "--risk-scores",
        default=str(config.RISK_SCORES_CSV),
        help="Caminho para risk_scores.csv (padrão: %(default)s).",
    )
