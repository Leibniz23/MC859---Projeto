"""Calcula o score de risco sistêmico R(v) para cada pacote do grafo."""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from bitarray import bitarray

from supplychain import config
from supplychain.common.normalization import log_minmax_normalise
from supplychain.common.graph_io import load_graph
from supplychain.common.scores_io import export_scores

logger = logging.getLogger(__name__)

# Pesos padrão: R(v) = alpha*d_in + beta*u + gamma*reach
ALPHA: float = 0.35
BETA: float = 0.25
GAMMA: float = 0.40


class RiskScorer:
    """Computa R(v) = alpha*d_in + beta*u + gamma*reach para todos os nós."""

    def __init__(self, graph: nx.DiGraph, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError(
                f"Pesos devem somar 1.0, got alpha+beta+gamma = {alpha + beta + gamma:.6f}"
            )

        self._graph = graph
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self._nodes = list(graph.nodes())
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}
        self._n = len(self._nodes)

        # cache dos componentes (calculados lazy)
        self._in_degree_raw = None
        self._in_degree_norm = None
        self._usage_norm = None
        self._reach_raw = None
        self._reach_norm = None
        self._risk_scores = None

    def _compute_in_degree(self):
        """In-degree normalizado com log-minmax."""
        self._in_degree_raw = np.array(
            [self._graph.in_degree(n) for n in self._nodes], dtype=np.float64,
        )
        self._in_degree_norm = log_minmax_normalise(self._in_degree_raw)

    def _compute_usage(self):
        """Pega o atributo usage_norm pré-computado pelo GraphBuilder."""
        self._usage_norm = np.array(
            [float(self._graph.nodes[n].get("usage_norm", 0.0)) for n in self._nodes],
            dtype=np.float64,
        )

    def _compute_transitive_reach(self):
        """Calcula o blast radius via bitarray DP em ordem topológica.

        Para cada nó v, reach[v] é um bitarray de tamanho |V| onde o bit i
        fica setado se o nó i transitivemente depende de v. O algoritmo
        processa em ordem topológica e propaga reach[v] |= reach[u] para cada
        predecessor u de v. Grafos com ciclos são condensados antes.
        """
        n = self._n
        node_to_idx = self._node_to_idx

        logger.info("Calculando transitive reach (%d nós)...", n)

        is_dag = nx.is_directed_acyclic_graph(self._graph)

        if is_dag:
            work_graph = self._graph
            orig_to_super = None
            work_nodes = self._nodes
            super_to_members_map = None
        else:
            condensed = nx.condensation(self._graph)
            orig_to_super = {}
            super_to_members_map = {}
            for super_id in condensed.nodes():
                members = list(condensed.nodes[super_id]["members"])
                super_to_members_map[super_id] = members
                for member in members:
                    orig_to_super[member] = super_id
            non_trivial = [s for s in super_to_members_map.values() if len(s) > 1]
            if non_trivial:
                logger.warning(
                    "Grafo tem ciclos: condensando %d SCCs não-triviais (maior=%d nós).",
                    len(non_trivial), max(len(s) for s in non_trivial),
                )
            work_graph = condensed
            work_nodes = list(condensed.nodes())

        work_n = len(work_nodes)
        work_node_to_idx = {v: i for i, v in enumerate(work_nodes)}

        topo_order = list(nx.topological_sort(work_graph))

        reach = []
        for w in work_nodes:
            ba = bitarray(n)
            ba.setall(0)
            if is_dag:
                ba[node_to_idx[w]] = 1
            else:
                for member in super_to_members_map[w]:
                    if member in node_to_idx:
                        ba[node_to_idx[member]] = 1
            reach.append(ba)

        for v in topo_order:
            v_w_idx = work_node_to_idx[v]
            for u in work_graph.predecessors(v):
                u_w_idx = work_node_to_idx[u]
                reach[v_w_idx] |= reach[u_w_idx]

        if is_dag:
            self._reach_raw = np.array(
                [reach[work_node_to_idx[v]].count() - 1 for v in self._nodes],
                dtype=np.float64,
            )
        else:
            reach_arr = np.zeros(n, dtype=np.float64)
            for orig_v in self._nodes:
                super_id = orig_to_super[orig_v]
                w_idx = work_node_to_idx[super_id]
                reach_arr[node_to_idx[orig_v]] = reach[w_idx].count() - 1
            self._reach_raw = reach_arr

        self._reach_norm = log_minmax_normalise(self._reach_raw)
        logger.info(
            "Transitive reach: max=%d, media=%.2f",
            int(self._reach_raw.max()), self._reach_raw.mean(),
        )

    def compute(self) -> pd.DataFrame:
        """Calcula R(v) para todos os nós e retorna DataFrame ordenado."""
        if self._risk_scores is not None:
            return self._build_dataframe()

        logger.info("Computando scores de risco R(v)...")
        self._compute_in_degree()
        self._compute_usage()
        self._compute_transitive_reach()

        self._risk_scores = (
            self._alpha * self._in_degree_norm
            + self._beta * self._usage_norm
            + self._gamma * self._reach_norm
        )

        logger.info(
            "Scores computados — max=%.4f, media=%.4f",
            self._risk_scores.max(), self._risk_scores.mean(),
        )
        return self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Monta o DataFrame de saída a partir dos arrays em cache."""
        df = pd.DataFrame({
            "package": self._nodes,
            "in_degree_raw": self._in_degree_raw.astype(int),
            "in_degree_norm": self._in_degree_norm,
            "usage_norm": self._usage_norm,
            "reach_raw": self._reach_raw.astype(int),
            "reach_norm": self._reach_norm,
            "risk_score": self._risk_scores,
        })
        df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        df["risk_rank"] = df.index + 1
        return df

    def export_csv(self, output_path=None) -> Path:
        """Computa (se necessário) e exporta os scores para CSV."""
        dest = Path(output_path) if output_path else config.RISK_SCORES_CSV
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = self.compute()
        out = export_scores(df, dest, "%.6f")
        logger.info("Risk scores exportados -> %s (%d pacotes)", out, len(df))
        return out

    def get_scores_dict(self) -> dict:
        """Retorna {package: risk_score} para uso downstream."""
        if self._risk_scores is None:
            self.compute()
        return dict(zip(self._nodes, self._risk_scores.tolist()))

    def top_k(self, k: int = 20) -> pd.DataFrame:
        """Retorna os k pacotes com maior R(v)."""
        return self.compute().head(k)
