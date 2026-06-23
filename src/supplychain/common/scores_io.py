"""Helpers de I/O para os CSVs de scores de cada fase."""

from pathlib import Path

import pandas as pd


def export_scores(df: pd.DataFrame, path, float_format: str) -> Path:
    """Salva df em CSV e retorna o Path resolvido."""
    df.to_csv(path, index=False, float_format=float_format)
    return Path(path)


def load_scores(path) -> pd.DataFrame:
    """Lê um CSV de scores gerado por qualquer fase do pipeline."""
    return pd.read_csv(path)
