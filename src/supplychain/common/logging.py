"""Configuração de logging compartilhada."""

import logging


def setup_logging() -> None:
    """Configura o logger raiz com o formato padrão do projeto."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
