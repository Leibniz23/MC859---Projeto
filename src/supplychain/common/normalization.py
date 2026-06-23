"""Funcoes de normalizacao usadas antes de combinar scores."""

import numpy as np


def log_minmax_normalise(values):
    """log(1+x) seguido de min-max para [0, 1].

    O log comprime as caudas pesadas (distribuicoes power-law) pra que os
    poucos hubs gigantes nao dominem o score.
    """
    log_vals = np.log1p(values.astype(np.float64))
    v_min, v_max = log_vals.min(), log_vals.max()
    if v_max > v_min:
        return (log_vals - v_min) / (v_max - v_min)
    return np.zeros_like(log_vals)


def minmax_normalise(values):
    """Min-max simples pra [0, 1] (pra valores que ja sao bem comportados)."""
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        return (values - v_min) / (v_max - v_min)
    return np.zeros_like(values)
