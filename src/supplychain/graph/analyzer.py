"""Análise estrutural do grafo de dependências."""

import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Análise topológica de um grafo de dependências PyPI."""

    def __init__(self, graphml_path):
        self._path = Path(graphml_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Arquivo GraphML não encontrado: {self._path}")

        logger.info("Carregando grafo de %s ...", self._path)
        self.graph = nx.read_graphml(str(self._path))
        logger.info(
            "Grafo carregado — %d nós, %d arestas",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
        )

    def basic_topography(self) -> dict:
        """Métricas básicas de topologia: graus, densidade, etc."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        in_degrees = np.array([d for _, d in self.graph.in_degree()])
        out_degrees = np.array([d for _, d in self.graph.out_degree()])

        metrics = {
            "nodes": n_nodes,
            "edges": n_edges,
            "avg_in_degree": float(np.mean(in_degrees)),
            "avg_out_degree": float(np.mean(out_degrees)),
            "max_in_degree": int(np.max(in_degrees)),
            "min_in_degree": int(np.min(in_degrees)),
            "max_out_degree": int(np.max(out_degrees)),
            "min_out_degree": int(np.min(out_degrees)),
            "density": nx.density(self.graph),
        }

        logger.info(
            "Topografia: %d nós, %d arestas, avg_in=%.2f, max_in=%d",
            n_nodes, n_edges, metrics["avg_in_degree"], metrics["max_in_degree"],
        )
        return metrics

    def degree_sequences(self) -> dict:
        """Retorna arrays de in-degree e out-degree de todos os nós."""
        return {
            "in": np.array([d for _, d in self.graph.in_degree()]),
            "out": np.array([d for _, d in self.graph.out_degree()]),
        }

    def scc_analysis(self) -> dict:
        """Identifica e caracteriza todas as SCCs (algoritmo de Tarjan, O(V+E))."""
        sccs = list(nx.strongly_connected_components(self.graph))
        scc_sizes = [len(c) for c in sccs]

        size_dist = Counter(scc_sizes)
        largest_size = max(scc_sizes)
        n_nodes = self.graph.number_of_nodes()
        largest_pct = (largest_size / n_nodes) * 100.0
        isolated = size_dist.get(1, 0)

        logger.info(
            "SCC: %d SCCs, maior=%d nós (%.2f%%), triviais=%d",
            len(sccs), largest_size, largest_pct, isolated,
        )

        return {
            "total_sccs": len(sccs),
            "largest_scc_size": largest_size,
            "largest_scc_pct": largest_pct,
            "size_distribution": size_dist,
            "isolated_nodes": isolated,
        }

    def wcc_analysis(self) -> dict:
        """Identifica e caracteriza todas as WCCs (grafo tratado como não-direcionado)."""
        wccs = list(nx.weakly_connected_components(self.graph))
        wcc_sizes = [len(c) for c in wccs]

        size_dist = Counter(wcc_sizes)
        largest_size = max(wcc_sizes)
        n_nodes = self.graph.number_of_nodes()
        largest_pct = (largest_size / n_nodes) * 100.0
        isolated = size_dist.get(1, 0)

        logger.info(
            "WCC: %d WCCs, maior=%d nós (%.2f%%), triviais=%d",
            len(wccs), largest_size, largest_pct, isolated,
        )

        return {
            "total_wccs": len(wccs),
            "largest_wcc_size": largest_size,
            "largest_wcc_pct": largest_pct,
            "size_distribution": size_dist,
            "isolated_nodes": isolated,
        }
