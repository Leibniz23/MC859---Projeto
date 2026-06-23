"""Cliente REST para a API OSV.dev."""

import logging
import time

import requests

from supplychain import config

logger = logging.getLogger(__name__)


class OSVClient:
    """Consulta a API OSV.dev para advisories de pacotes PyPI.

    Implementa batch querying em dois passos:
      Passo 1 — querybatch para identificar pacotes com algum advisory (rápido).
      Passo 2 — /v1/query individual para os pacotes afetados, obtendo detalhes
                completos incluindo severidade/CVSS.
    """

    def __init__(
        self,
        api_url=config.OSV_API_URL,
        batch_url=config.OSV_BATCH_URL,
        request_delay=config.OSV_REQUEST_DELAY,
        max_retries=config.OSV_MAX_RETRIES,
        batch_size=config.OSV_BATCH_SIZE,
    ):
        self._api_url = api_url
        self._batch_url = batch_url
        self._delay = request_delay
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "pypi-dependency-risk-model/1.0",
        })

    def query_single(self, package_name: str) -> list:
        """Consulta OSV para todos os advisories de um pacote PyPI."""
        payload = {
            "package": {
                "name": package_name,
                "ecosystem": "PyPI",
            }
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.post(self._api_url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                return data.get("vulns", [])
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited para %s, aguardando %ds (tentativa %d/%d)",
                        package_name, wait, attempt, self._max_retries,
                    )
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    logger.warning(
                        "Erro de servidor %d para %s (tentativa %d/%d)",
                        resp.status_code, package_name, attempt, self._max_retries,
                    )
                    time.sleep(1)
                else:
                    logger.error("HTTP %d para %s: %s", resp.status_code, package_name, e)
                    return []
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Request falhou para %s: %s (tentativa %d/%d)",
                    package_name, e, attempt, self._max_retries,
                )
                time.sleep(1)

        logger.error("Todas as tentativas esgotadas para %s", package_name)
        return []

    def query_batch(self, package_names: list) -> dict:
        """Consulta OSV para múltiplos pacotes com detalhes completos.

        Passo 1: querybatch identifica pacotes com algum advisory.
        Passo 2: query individual para obter dados completos de CVSS dos afetados.
        """
        results = {}

        # Passo 1: batch para identificar pacotes com advisories
        logger.info(
            "Passo 1: identificando pacotes vulneráveis em batch (%d total)...",
            len(package_names),
        )
        has_vulns = []

        for batch_start in range(0, len(package_names), self._batch_size):
            batch = package_names[batch_start:batch_start + self._batch_size]
            queries = [
                {"package": {"name": pkg, "ecosystem": "PyPI"}}
                for pkg in batch
            ]

            for attempt in range(1, self._max_retries + 1):
                try:
                    resp = self._session.post(
                        self._batch_url,
                        json={"queries": queries},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    batch_results = data.get("results", [])

                    for pkg, res in zip(batch, batch_results):
                        vulns = res.get("vulns", [])
                        results[pkg] = []
                        if len(vulns) > 0:
                            has_vulns.append(pkg)

                    break
                except requests.exceptions.HTTPError as e:
                    if resp.status_code == 429:
                        wait = 2 ** attempt + 1
                        logger.warning(
                            "Batch rate limited, aguardando %ds (tentativa %d/%d)",
                            wait, attempt, self._max_retries,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Batch HTTP %d (tentativa %d/%d): %s",
                            resp.status_code, attempt, self._max_retries, e,
                        )
                        time.sleep(1)
                except requests.exceptions.RequestException as e:
                    logger.warning(
                        "Batch request falhou (tentativa %d/%d): %s",
                        attempt, self._max_retries, e,
                    )
                    time.sleep(1)
            else:
                # se o batch falhou, assume que todos podem ter vulns
                for pkg in batch:
                    has_vulns.append(pkg)

            time.sleep(self._delay)

            if (batch_start + self._batch_size) % 500 == 0:
                logger.info(
                    "Progresso OSV: %d / %d pacotes",
                    min(batch_start + self._batch_size, len(package_names)),
                    len(package_names),
                )

        logger.info(
            "Passo 1 concluído: %d / %d pacotes com advisories — buscando detalhes...",
            len(has_vulns), len(package_names),
        )

        # Passo 2: detalhes completos para os pacotes afetados
        for i, pkg in enumerate(has_vulns):
            results[pkg] = self.query_single(pkg)
            time.sleep(self._delay)
            if (i + 1) % 50 == 0:
                logger.info(
                    "Passo 2: %d / %d pacotes vulneráveis processados",
                    i + 1, len(has_vulns),
                )

        logger.info(
            "Query OSV completa: %d pacotes com dados de advisory.",
            len([p for p, v in results.items() if len(v) > 0]),
        )

        return results
