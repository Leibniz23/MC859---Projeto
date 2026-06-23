"""Calcula métricas de centralidade e índice de criticidade C(v)."""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from supplychain import config
from supplychain.common.normalization import minmax_normalise
from supplychain.common.graph_io import load_graph
from supplychain.common.scores_io import export_scores

logger = logging.getLogger(__name__)


class CentralityAnalyzer:
    """Computa in-degree, métrica estrutural (opcional) e PageRank, combinando em C(v).

    Com risk_scores fornecido, também computa S(v) = lambda*R(v) + (1-lambda)*C(v).
    """

    _VALID_STRUCTURAL = ("none", "katz", "betweenness")

    def __init__(
        self,
        graph: nx.DiGraph,
        risk_scores=None,
        structural_metric=config.CENTRALITY_STRUCTURAL_METRIC,
        w_degree=None,
        w_structural=None,
        w_pagerank=None,
        pagerank_damping=config.PAGERANK_DAMPING,
        betweenness_exact=config.BETWEENNESS_EXACT,
        k_pivots=config.BETWEENNESS_K_PIVOTS,
        katz_alpha=config.KATZ_ALPHA,
        katz_beta=config.KATZ_BETA,
        combined_lambda=config.COMBINED_LAMBDA,
    ):
        structural_metric = structural_metric.lower()
        if structural_metric not in self._VALID_STRUCTURAL:
            raise ValueError(
                f"structural_metric deve ser um de {self._VALID_STRUCTURAL}, "
                f"recebido {structural_metric!r}"
            )
        self._structural_metric = structural_metric

        # seleção de pesos: modo "none" não tem termo estrutural
        if structural_metric == "none":
            self._w1 = config.CENTRALITY_W_DEGREE_NB if w_degree is None else w_degree
            self._w2 = 0.0
            self._w3 = config.CENTRALITY_W_PAGERANK_NB if w_pagerank is None else w_pagerank
            weight_sum = self._w1 + self._w3
        else:
            self._w1 = config.CENTRALITY_W_DEGREE if w_degree is None else w_degree
            self._w2 = config.CENTRALITY_W_STRUCTURAL if w_structural is None else w_structural
            self._w3 = config.CENTRALITY_W_PAGERANK if w_pagerank is None else w_pagerank
            weight_sum = self._w1 + self._w2 + self._w3

        if not np.isclose(weight_sum, 1.0):
            raise ValueError(
                f"Pesos de centralidade devem somar 1.0, got {weight_sum:.6f}"
            )

        self._graph = graph
        self._risk_scores = risk_scores
        self._damping = pagerank_damping
        self._exact_betweenness = betweenness_exact
        self._k_pivots = k_pivots
        self._katz_alpha = katz_alpha
        self._katz_beta = katz_beta
        self._lambda = combined_lambda

        self._nodes = list(graph.nodes())
        self._n = len(self._nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._nodes)}

        # cache
        self._in_degree_centrality = None
        self._out_degree_centrality = None
        self._structural = None
        self._pagerank = None
        self._criticality = None
        self._combined = None

    def _compute_degree_centrality(self):
        """In-degree e out-degree centrality normalizados por (|V|-1)."""
        in_cent = nx.in_degree_centrality(self._graph)
        out_cent = nx.out_degree_centrality(self._graph)

        self._in_degree_centrality = np.array(
            [in_cent[n] for n in self._nodes], dtype=np.float64
        )
        self._out_degree_centrality = np.array(
            [out_cent[n] for n in self._nodes], dtype=np.float64
        )

    def _compute_structural(self):
        """Despacha para betweenness, katz ou zeros dependendo da config."""
        if self._structural_metric == "none":
            self._structural = np.zeros(self._n, dtype=np.float64)
        elif self._structural_metric == "betweenness":
            self._compute_betweenness()
        else:
            self._compute_katz()

    def _compute_betweenness(self):
        """Betweenness via algoritmo de Brandes. Lento (O(V*E)) e degenerado no quasi-DAG."""
        if self._exact_betweenness:
            logger.info("Computando betweenness exata (V=%d, E=%d)...", self._n,
                        self._graph.number_of_edges())
            bc = nx.betweenness_centrality(self._graph, normalized=True, weight=None)
        else:
            logger.info("Computando betweenness aproximada (k=%d pivots)...", self._k_pivots)
            bc = nx.betweenness_centrality(
                self._graph, k=self._k_pivots, normalized=True, weight=None, seed=42,
            )

        self._structural = np.array([bc[n] for n in self._nodes], dtype=np.float64)

    def _compute_katz(self):
        """Katz centrality via solução direta do sistema linear (numpy solver)."""
        alphas = [self._katz_alpha, self._katz_alpha / 2, self._katz_alpha / 5, 0.01]
        kc = None
        for a in alphas:
            try:
                logger.info("Computando Katz centrality (alpha=%.4f)...", a)
                kc = nx.katz_centrality_numpy(
                    self._graph, alpha=a, beta=self._katz_beta, normalized=True
                )
                self._katz_alpha = a
                break
            except Exception as exc:
                logger.warning("Katz falhou com alpha=%.4f (%s); tentando menor.", a, exc)

        if kc is None:
            logger.error("Katz falhou para todos os alphas; usando zeros.")
            self._structural = np.zeros(self._n, dtype=np.float64)
            return

        self._structural = np.array([kc[n] for n in self._nodes], dtype=np.float64)

    def _compute_pagerank(self):
        """PageRank por iteração de potência."""
        logger.info("Computando PageRank (damping=%.2f)...", self._damping)
        pr = nx.pagerank(self._graph, alpha=self._damping, max_iter=200, tol=1e-8)
        self._pagerank = np.array([pr[n] for n in self._nodes], dtype=np.float64)

    def compute(self) -> pd.DataFrame:
        """Computa todas as métricas e retorna DataFrame com C(v) (e S(v) se disponível)."""
        if self._criticality is not None:
            return self._build_dataframe()

        logger.info("Computando métricas de centralidade...")
        self._compute_degree_centrality()
        self._compute_structural()
        self._compute_pagerank()

        in_deg_norm = minmax_normalise(self._in_degree_centrality)
        struct_norm = minmax_normalise(self._structural)
        pr_norm = minmax_normalise(self._pagerank)

        self._criticality = (
            self._w1 * in_deg_norm
            + self._w2 * struct_norm
            + self._w3 * pr_norm
        )

        if self._risk_scores is not None:
            risk_arr = np.array(
                [self._risk_scores.get(n, 0.0) for n in self._nodes],
                dtype=np.float64,
            )
            crit_norm = minmax_normalise(self._criticality)
            self._combined = self._lambda * risk_arr + (1 - self._lambda) * crit_norm

        logger.info(
            "Criticality index: max=%.4f, media=%.4f",
            self._criticality.max(), self._criticality.mean(),
        )
        return self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Monta o DataFrame com os arrays em cache."""
        in_deg_norm = minmax_normalise(self._in_degree_centrality)
        struct_norm = minmax_normalise(self._structural)
        pr_norm = minmax_normalise(self._pagerank)

        data = {
            "package": self._nodes,
            "in_degree_centrality": self._in_degree_centrality,
            "out_degree_centrality": self._out_degree_centrality,
            "structural_metric": self._structural_metric,
            "structural": self._structural,
            "pagerank": self._pagerank,
            "in_degree_centrality_norm": in_deg_norm,
            "structural_norm": struct_norm,
            "pagerank_norm": pr_norm,
            "criticality_index": self._criticality,
        }

        if self._combined is not None:
            risk_arr = np.array(
                [self._risk_scores.get(n, 0.0) for n in self._nodes],
                dtype=np.float64,
            )
            data["risk_score"] = risk_arr
            data["combined_score"] = self._combined

        df = pd.DataFrame(data)
        rank_col = "combined_score" if "combined_score" in df.columns else "criticality_index"
        df = df.sort_values(rank_col, ascending=False).reset_index(drop=True)
        df["criticality_rank"] = df.index + 1
        return df

    def export_csv(self, output_path=None) -> Path:
        """Computa (se necessário) e exporta scores para CSV."""
        dest = Path(output_path) if output_path else config.CENTRALITY_SCORES_CSV
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = self.compute()
        out = export_scores(df, dest, "%.8f")
        logger.info("Centrality scores exportados -> %s (%d pacotes)", out, len(df))
        return out

    def get_combined_scores_dict(self) -> dict:
        """Retorna {package: combined_score} (ou criticality se não houver combined)."""
        if self._criticality is None:
            self.compute()
        if self._combined is not None:
            return dict(zip(self._nodes, self._combined.tolist()))
        return dict(zip(self._nodes, self._criticality.tolist()))

    def top_k(self, k: int = 20) -> pd.DataFrame:
        """Retorna os k pacotes mais críticos."""
        return self.compute().head(k)
