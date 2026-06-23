"""Helpers de I/O para grafos."""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def load_graph(path) -> nx.DiGraph:
    """Lê um arquivo GraphML e retorna o DiGraph."""
    g = nx.read_graphml(path)
    logger.info("Grafo carregado — %d nós, %d arestas", g.number_of_nodes(), g.number_of_edges())
    return g
