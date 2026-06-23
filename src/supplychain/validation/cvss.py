"""Parsing de vetores CVSS e extração de score para advisories OSV.

Estratégia em 3 camadas:
  Camada 1: severity[].score — vetor CVSS parseado matematicamente (V3 > V4 > V2)
  Camada 2: database_specific.severity — label textual mapeado para midpoints FIRST/NVD
  Camada 3: sem informação -> retorna None
"""

import math
import re

from supplychain import config

# Tabelas de pesos do CVSS v3.1 (constantes da especificação)
_AV_MAP = {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.20}
_AC_MAP = {'L': 0.77, 'H': 0.44}
_PR_MAP_U = {'N': 0.85, 'L': 0.62, 'H': 0.27}   # scope unchanged
_PR_MAP_C = {'N': 0.85, 'L': 0.68, 'H': 0.50}   # scope changed
_UI_MAP = {'N': 0.85, 'R': 0.62}
_IMPACT_MAP = {'N': 0.00, 'L': 0.22, 'H': 0.56}


def _parse_cvss_vector(score_str: str):
    """Parseia um vetor CVSS e retorna o base score, ou None se não reconhecido.

    Suporta CVSS v2, v3.x e v4.0. Para v3.x usa as fórmulas exatas da spec.
    """
    # CVSS v3.x: formato CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
    v3_match = re.match(r'CVSS:[23]\.[01]/(.*)', score_str)
    if v3_match:
        metrics_str = v3_match.group(1)
        metrics = {}
        for part in metrics_str.split('/'):
            if ':' in part:
                k, v = part.split(':', 1)
                metrics[k] = v

        try:
            scope = metrics.get('S', 'U')
            av = _AV_MAP.get(metrics.get('AV', 'N'), 0.85)
            ac = _AC_MAP.get(metrics.get('AC', 'L'), 0.77)
            if scope == 'U':
                pr = _PR_MAP_U.get(metrics.get('PR', 'N'), 0.85)
            else:
                pr = _PR_MAP_C.get(metrics.get('PR', 'N'), 0.85)
            ui = _UI_MAP.get(metrics.get('UI', 'N'), 0.85)
            c = _IMPACT_MAP.get(metrics.get('C', 'H'), 0.56)
            i = _IMPACT_MAP.get(metrics.get('I', 'H'), 0.56)
            a = _IMPACT_MAP.get(metrics.get('A', 'H'), 0.56)

            exploitability = 8.22 * av * ac * pr * ui

            iss = 1 - (1 - c) * (1 - i) * (1 - a)
            if scope == 'U':
                impact = 6.42 * iss
            else:
                impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15

            if impact <= 0:
                return 0.0

            if scope == 'U':
                base = min(exploitability + impact, 10.0)
            else:
                base = min(1.08 * (exploitability + impact), 10.0)

            # arredondamento para cima para 1 decimal (conforme spec CVSS)
            base = math.ceil(base * 10) / 10
            return round(min(base, 10.0), 1)

        except Exception:
            pass

    # CVSS v4.0 — aproximação por lookup de severidade
    v4_match = re.match(r'CVSS:4\.0/(.*)', score_str)
    if v4_match:
        metrics_str = v4_match.group(1)
        metrics = {}
        for part in metrics_str.split('/'):
            if ':' in part:
                k, v = part.split(':', 1)
                metrics[k] = v

        high_impact = any(
            metrics.get(k, 'N') == 'H'
            for k in ('VC', 'VI', 'VA', 'SC', 'SI', 'SA')
        )
        av_network = metrics.get('AV', 'N') == 'N'
        ac_low = metrics.get('AC', 'L') == 'L'

        if high_impact and av_network and ac_low:
            return 8.0
        elif high_impact and av_network:
            return 7.0
        elif high_impact:
            return 6.5
        else:
            return 4.0

    return None


def extract_cvss(advisory: dict):
    """Extrai o score CVSS de um advisory OSV usando estratégia de 3 camadas.

    Retorna float em [0, 10] ou None se não houver info de severidade.
    """
    # Camada 1: vetor CVSS (prefere V3 > V4 > V2)
    severity_list = advisory.get("severity", [])
    v3_score = None
    v4_score = None
    v2_score = None

    for sev in severity_list:
        sev_type = sev.get("type", "")
        score_str = sev.get("score", "")
        if not score_str:
            continue
        parsed = _parse_cvss_vector(score_str)
        if parsed is not None:
            if "V3" in sev_type:
                v3_score = parsed
            elif "V4" in sev_type:
                v4_score = parsed
            elif "V2" in sev_type:
                v2_score = parsed

    if v3_score is not None:
        return v3_score
    if v4_score is not None:
        return v4_score
    if v2_score is not None:
        return v2_score

    # Camada 2: label textual em database_specific.severity
    db_specific = advisory.get("database_specific", {})
    severity_label = db_specific.get("severity", "")
    if isinstance(severity_label, str) and severity_label.strip():
        label_lower = severity_label.strip().lower()
        if label_lower in config.SEVERITY_LABEL_MAP:
            return config.SEVERITY_LABEL_MAP[label_lower]

    # Camada 3: sem info de severidade
    return None
