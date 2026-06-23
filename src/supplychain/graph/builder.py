"""Constrói o grafo de dependências e exporta como GraphML."""

import logging
from enum import Enum, auto
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from supplychain import config

logger = logging.getLogger(__name__)


class IngestionMode(Enum):
    ONLINE = auto()   # roda Extractor ao vivo
    OFFLINE = auto()  # lê CSVs do disco


class GraphBuilder:
    """Constrói um DiGraph com atributos de uso normalizados.

    Em modo OFFLINE lê os CSVs de dependências e usage do disco.
    Em modo ONLINE roda o Extractor via BigQuery.
    """

    def __init__(
        self,
        mode=IngestionMode.OFFLINE,
        extractor=None,
        dependencies_csv=None,
        usage_csv=None,
    ):
        self.graph = nx.DiGraph()

        self._dep_path = Path(dependencies_csv) if dependencies_csv else config.DEPENDENCIES_CSV
        self._usage_path = Path(usage_csv) if usage_csv else config.USAGE_STATS_CSV

        if mode is IngestionMode.ONLINE:
            if extractor is None:
                raise ValueError("Extractor é obrigatório quando mode=ONLINE.")
            logger.info("Modo ONLINE — rodando extração ao vivo...")
            self._deps_df = extractor.extract_dependencies()
            self._usage_df = extractor.extract_usage()
        else:
            logger.info("Modo OFFLINE — lendo CSVs do disco...")
            if not self._dep_path.exists():
                raise FileNotFoundError(f"CSV de dependências não encontrado: {self._dep_path}")
            if not self._usage_path.exists():
                raise FileNotFoundError(f"CSV de usage não encontrado: {self._usage_path}")
            self._deps_df = pd.read_csv(self._dep_path)
            self._usage_df = pd.read_csv(self._usage_path)

        self._build()

    def _build(self):
        """Popula self.graph com nós, arestas e atributos normalizados."""
        self._add_edges()
        self._attach_usage_attributes()
        logger.info(
            "Grafo construído — %d nós, %d arestas",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
        )

    def _add_edges(self):
        """Adiciona arestas source -> target a partir do CSV de dependências."""
        for _, row in self._deps_df.iterrows():
            src = str(row["source"]).strip().lower()
            tgt = str(row["target"]).strip().lower()
            self.graph.add_edge(src, tgt)

    def _attach_usage_attributes(self):
        """Normaliza downloads com log10(1+x) + min-max e atribui aos nós.

        NOTA: usa log base 10 (np.log10), diferente do log natural de
        common/normalization.py. Mantido assim pra preservar os valores
        numéricos originais do grafo.

        Atributos por nó:
          downloads_raw  — total de downloads (int)
          usage_log      — log10(1 + downloads)
          usage_norm     — valor normalizado em [0, 1]
        """
        usage_map = {}
        for _, row in self._usage_df.iterrows():
            pkg = str(row["package"]).strip().lower()
            usage_map[pkg] = int(row.get("total_downloads", 0))

        raw_values = np.array(
            [usage_map.get(n, 0) for n in self.graph.nodes()], dtype=np.float64
        )

        log_values = np.log10(1.0 + raw_values)

        v_min, v_max = log_values.min(), log_values.max()
        if v_max - v_min > 0:
            norm_values = (log_values - v_min) / (v_max - v_min)
        else:
            norm_values = np.zeros_like(log_values)

        for node, raw, log_val, norm in zip(
            self.graph.nodes(), raw_values, log_values, norm_values
        ):
            self.graph.nodes[node]["downloads_raw"] = int(raw)
            self.graph.nodes[node]["usage_log"] = float(log_val)
            self.graph.nodes[node]["usage_norm"] = float(norm)

    def export_graphml(self, output_path=None) -> Path:
        """Salva o grafo em GraphML e retorna o caminho."""
        dest = Path(output_path) if output_path else config.GRAPH_OUTPUT
        dest.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, str(dest))
        logger.info("Grafo exportado -> %s", dest)
        return dest

    def summary(self) -> dict:
        """Retorna um resumo estatístico básico do grafo."""
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_in_degree": np.mean(in_degrees) if in_degrees else 0,
            "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "weakly_connected_components": nx.number_weakly_connected_components(self.graph),
        }
