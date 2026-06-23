"""Extrai dados de dependências e downloads do PyPI via BigQuery."""

import logging
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from supplychain import config

logger = logging.getLogger(__name__)


class Extractor:
    """Coleta dados de dependências e usage do PyPI via BigQuery.

    Fase 1: query em pypi.file_downloads para selecionar os top-N seeds.
    Fase 2: query em deps_dev_v1.DependenciesLatest para todas as arestas
            diretas do PyPI, depois BFS a partir dos seeds para resolver
            as árvores de dependência completas.
    """

    def __init__(self, gcp_project=None):
        self._gcp_project = gcp_project or config.GCP_PROJECT_ID
        self._bq_client = None
        self._download_counts = None

    @property
    def bq_client(self) -> bigquery.Client:
        """BigQuery client com inicialização lazy."""
        if self._bq_client is None:
            self._bq_client = bigquery.Client(project=self._gcp_project)
            logger.info("BigQuery client iniciado (project=%s)", self._gcp_project)
        return self._bq_client

    def extract_download_counts(self, n_seeds=config.NUM_SEED_PACKAGES,
                                lookback_days=config.DOWNLOAD_LOOKBACK_DAYS,
                                seeds_output=None) -> pd.DataFrame:
        """Query de downloads e seleção dos top-n seeds.

        Cacheia os counts de todos os pacotes para uso posterior em extract_usage().
        Retorna DataFrame com colunas [package, total_downloads].
        """
        dest = Path(seeds_output or config.SEEDS_CSV)
        if dest.exists():
            logger.info("Cache de seeds encontrado em %s. Pulando query.", dest)
            seeds = pd.read_csv(dest)
            self._download_counts = seeds
            return seeds

        logger.info(
            "Querying BigQuery para download counts (lookback=%d dias, seeds=%d)...",
            lookback_days, n_seeds,
        )

        query = f"""
        SELECT
            project                 AS package,
            COUNT(*)                AS total_downloads
        FROM `bigquery-public-data.pypi.file_downloads`
        WHERE
            DATE(timestamp) BETWEEN
                DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
                AND CURRENT_DATE()
        GROUP BY project
        ORDER BY total_downloads DESC
        """

        try:
            all_downloads = self.bq_client.query(query).to_dataframe()
        except Exception:
            logger.exception("Query de download counts falhou")
            raise

        all_downloads["package"] = all_downloads["package"].str.strip().str.lower()

        all_downloads = (
            all_downloads
            .groupby("package", as_index=False)["total_downloads"]
            .sum()
            .sort_values("total_downloads", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("Download counts obtidos para %d pacotes.", len(all_downloads))

        self._download_counts = all_downloads

        seeds = all_downloads.head(n_seeds).copy()

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        seeds.to_csv(dest, index=False)
        logger.info("Top-%d seeds salvos -> %s", n_seeds, dest)

        return seeds

    def extract_dependencies(self, seeds=None, output_path=None) -> pd.DataFrame:
        """Resolve árvores de dependência via BFS a partir dos seeds.

        1. Se seeds não fornecido, chama extract_download_counts().
        2. Query BigQuery para todas as arestas diretas do PyPI.
        3. BFS a partir dos seeds (ilimitado em profundidade).
        4. Persiste o subgrafo resultante em dependencies.csv.
        """
        dest = Path(output_path or config.DEPENDENCIES_CSV)
        if dest.exists():
            logger.info("Cache de dependências encontrado em %s. Pulando BFS.", dest)
            return pd.read_csv(dest)

        if seeds is None:
            seeds = self.extract_download_counts()

        seed_packages = set(seeds["package"].str.strip().str.lower())
        logger.info("Seeds disponíveis: %d pacotes", len(seed_packages))

        all_edges_df = self._query_all_pypi_dependencies()
        closure_df, visited_nodes = self._bfs_from_seeds(all_edges_df, seed_packages)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        closure_df.to_csv(dest, index=False)
        logger.info(
            "Árvores resolvidas — %d nós, %d arestas -> %s",
            len(visited_nodes), len(closure_df), dest,
        )

        return closure_df

    def _query_all_pypi_dependencies(self) -> pd.DataFrame:
        """Query BigQuery para todas as arestas diretas do PyPI (versão mais recente).

        Usa VersionInfo.Ordinal para identificar a versão mais recente de cada pacote
        (IsRelease=TRUE filtra pre-releases). Retorna DataFrame com [source, target].
        """
        dest = config.DATA_DIR / "all_pypi_edges.csv"
        if dest.exists():
            logger.info("Cache de arestas encontrado em %s. Pulando query.", dest)
            df = pd.read_csv(dest)
            df = df.dropna(subset=['source', 'target'])
            return df

        logger.info("Querying BigQuery para todas as arestas PyPI...")

        query = """
        WITH latest_versions AS (
            SELECT Name, Version
            FROM `bigquery-public-data.deps_dev_v1.PackageVersionsLatest`
            WHERE
                System = 'PYPI'
                AND VersionInfo.IsRelease = TRUE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY Name
                ORDER BY VersionInfo.Ordinal DESC
            ) = 1
        )
        SELECT DISTINCT
            LOWER(d.Name)            AS source,
            LOWER(d.Dependency.Name) AS target
        FROM `bigquery-public-data.deps_dev_v1.DependenciesLatest` AS d
        INNER JOIN latest_versions AS lv
            ON  d.Name    = lv.Name
            AND d.Version = lv.Version
        WHERE
            d.System            = 'PYPI'
            AND d.Dependency.System = 'PYPI'
        """

        try:
            df = self.bq_client.query(query).to_dataframe()
        except Exception:
            logger.exception("Query de dependências falhou")
            raise

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.dropna(subset=['source', 'target'], inplace=True)
        df.to_csv(dest, index=False)
        logger.info("Arestas brutas salvas em %s (%d arestas)", dest, len(df))

        return df

    def _bfs_from_seeds(self, all_edges_df: pd.DataFrame, seed_packages: set):
        """BFS a partir dos seeds seguindo arestas de dependência (source -> target).

        Retorna (DataFrame com [source, target], conjunto de nós visitados).
        """
        adj = defaultdict(set)
        for src, tgt in zip(all_edges_df["source"], all_edges_df["target"]):
            adj[src].add(tgt)

        logger.info(
            "Adjacência construída: %d nós fonte, %d arestas. BFS de %d seeds...",
            len(adj), len(all_edges_df), len(seed_packages),
        )

        visited = set()
        queue = deque()
        edges = []

        for pkg in seed_packages:
            visited.add(pkg)
            queue.append(pkg)

        depth_stats = defaultdict(int)
        current_depth = 0
        nodes_at_current_depth = len(queue)
        nodes_processed_at_depth = 0

        while queue:
            pkg = queue.popleft()
            nodes_processed_at_depth += 1

            for dep in adj.get(pkg, set()):
                edges.append((pkg, dep))
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
                    depth_stats[current_depth + 1] = (
                        depth_stats.get(current_depth + 1, 0) + 1
                    )

            if nodes_processed_at_depth >= nodes_at_current_depth:
                if depth_stats.get(current_depth + 1, 0) > 0:
                    logger.info(
                        "BFS profundidade %d: %d novos nós (total: %d nós, %d arestas)",
                        current_depth + 1,
                        depth_stats.get(current_depth + 1, 0),
                        len(visited), len(edges),
                    )
                current_depth += 1
                nodes_at_current_depth = depth_stats.get(current_depth, 0)
                nodes_processed_at_depth = 0

        logger.info(
            "BFS concluído — profundidade max: %d, nós: %d, arestas: %d",
            max(depth_stats.keys()) if depth_stats else 0,
            len(visited), len(edges),
        )

        closure_df = pd.DataFrame(edges, columns=["source", "target"])
        return closure_df, visited

    def extract_usage(self, packages=None, dependencies_csv=None,
                      output_path=None) -> pd.DataFrame:
        """Gera usage_stats.csv a partir dos download counts em cache.

        Pacotes sem count recebem total_downloads=0.
        """
        dest = Path(output_path or config.USAGE_STATS_CSV)
        if dest.exists():
            logger.info("Cache de usage encontrado em %s. Pulando geração.", dest)
            return pd.read_csv(dest)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        if packages is None:
            dep_path = dependencies_csv or str(config.DEPENDENCIES_CSV)
            logger.info("Carregando lista de pacotes de %s", dep_path)
            deps_df = pd.read_csv(dep_path)
            packages = set(deps_df["source"]).union(set(deps_df["target"]))

        logger.info("Gerando usage_stats para %d pacotes...", len(packages))

        if self._download_counts is None:
            logger.info("Download counts não cacheados — rodando extract_download_counts()...")
            self.extract_download_counts()

        dl_map = dict(
            zip(self._download_counts["package"], self._download_counts["total_downloads"])
        )

        records = [
            {"package": pkg, "total_downloads": int(dl_map.get(pkg, 0))}
            for pkg in sorted(packages)
        ]

        df = pd.DataFrame(records)

        with_downloads = df[df["total_downloads"] > 0]
        logger.info(
            "Cobertura de downloads: %d / %d pacotes (%.1f%%) com count > 0",
            len(with_downloads), len(df),
            100.0 * len(with_downloads) / len(df) if len(df) > 0 else 0,
        )

        df.to_csv(dest, index=False)
        logger.info("Usage stats salvos -> %s (%d pacotes)", dest, len(df))
        return df
