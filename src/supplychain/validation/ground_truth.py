"""Constrói o ground truth a partir dos dados brutos de advisories OSV."""

import logging

import numpy as np
import pandas as pd

from supplychain.validation.cvss import extract_cvss

logger = logging.getLogger(__name__)


def build_ground_truth(advisories: dict) -> pd.DataFrame:
    """Constrói DataFrame de ground truth a partir dos advisories OSV.

    Para cada pacote computa:
    - vuln_count: número de advisories distintos
    - cvss_max: maior score CVSS entre todos os advisories
    - has_vuln: 1 se tem algum advisory
    - has_high_vuln: 1 se tem algum advisory com CVSS >= 7.0
    - advisory_ids: ids separados por ponto-e-vírgula
    """
    rows = []
    cvss_found = 0
    cvss_missing = 0

    for package, vulns in advisories.items():
        vuln_count = len(vulns)
        advisory_ids = [v.get("id", "unknown") for v in vulns]

        cvss_scores = []
        for v in vulns:
            score = extract_cvss(v)
            if score is not None:
                cvss_scores.append(score)
                cvss_found += 1
            else:
                cvss_missing += 1

        cvss_max = max(cvss_scores) if cvss_scores else np.nan
        has_vuln = 1 if vuln_count > 0 else 0
        has_high_vuln = (
            1 if has_vuln and not np.isnan(cvss_max) and cvss_max >= 7.0
            else 0
        )

        rows.append({
            "package": package,
            "vuln_count": vuln_count,
            "cvss_max": cvss_max,
            "has_vuln": has_vuln,
            "has_high_vuln": has_high_vuln,
            "advisory_ids": ";".join(advisory_ids),
        })

    logger.info(
        "Ground truth: %d pacotes, CVSS encontrado=%d, ausente=%d",
        len(rows), cvss_found, cvss_missing,
    )

    return pd.DataFrame(rows)
