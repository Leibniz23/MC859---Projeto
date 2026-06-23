"""Simulação de propagação de vulnerabilidades via Independent Cascade Model (ICM)."""

import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import trange

from supplychain import config
from supplychain.common.graph_io import load_graph
from supplychain.common.scores_io import export_scores

logger = logging.getLogger(__name__)


class CascadeSimulator:
    """Simula propagação de vulnerabilidades pelo ICM no grafo reverso.

    Aresta u->v no grafo original significa "u depende de v", então uma falha
    em v propaga para u. A simulação opera no sentido reverso: para cada aresta
    u->v, montamos reverse_adj[v] = [(u, p), ...] onde p é a probabilidade de
    propagação.

    p(v->u) = p_base + (1 - p_base) * R(v) * usage_norm(u)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        risk_scores: dict,
        p_base=config.ICM_P_BASE,
        seed=config.ICM_RANDOM_SEED,
    ):
        self._graph = graph
        self._risk = risk_scores
        self._p_base = p_base
        self._rng = np.random.default_rng(seed)

        self._reverse_adj = defaultdict(list)
        self._precompute_probabilities()

        logger.info(
            "CascadeSimulator: %d nós, %d arestas, p_base=%.3f",
            graph.number_of_nodes(), graph.number_of_edges(), p_base,
        )

    def _precompute_probabilities(self):
        """Pré-computa probabilidades de propagação para cada aresta reversa."""
        for u, v in self._graph.edges():
            usage_u = float(self._graph.nodes[u].get("usage_norm", 0.0))
            risk_v = self._risk.get(v, 0.0)
            p = self._p_base + (1 - self._p_base) * risk_v * usage_u
            self._reverse_adj[v].append((u, p))

    def simulate_single(self, seed_nodes: set) -> tuple:
        """Roda uma trial ICM a partir de seed_nodes.

        Retorna (conjunto de nós ativados, profundidade da cascata).
        """
        activated = set(seed_nodes)
        newly_activated = set(seed_nodes)
        depth = 0

        while newly_activated:
            next_wave = set()
            for node in newly_activated:
                for neighbor, prob in self._reverse_adj.get(node, []):
                    if neighbor not in activated:
                        if self._rng.random() < prob:
                            next_wave.add(neighbor)
                            activated.add(neighbor)
            newly_activated = next_wave
            if newly_activated:
                depth += 1

        return activated, depth

    def monte_carlo(self, seed_nodes: set, n_trials=config.ICM_N_TRIALS,
                    track_activation_frequency=True) -> dict:
        """Roda N trials ICM e retorna estatísticas do blast radius."""
        sizes = np.empty(n_trials, dtype=np.float64)
        depths = np.empty(n_trials, dtype=np.int32)
        activation_counts = defaultdict(int) if track_activation_frequency else {}

        seed_label = ", ".join(sorted(seed_nodes)[:3])
        if len(seed_nodes) > 3:
            seed_label += "..."

        for i in trange(n_trials, desc=f"ICM [{seed_label}]", leave=False):
            activated, depth = self.simulate_single(seed_nodes)
            sizes[i] = len(activated)
            depths[i] = depth
            if track_activation_frequency:
                for node in activated:
                    activation_counts[node] += 1

        mean_b = float(np.mean(sizes))
        std_b = float(np.std(sizes, ddof=1))
        ci_margin = 1.96 * std_b / np.sqrt(n_trials)

        result = {
            "seed_nodes": sorted(seed_nodes),
            "mean_blast_radius": mean_b,
            "std_blast_radius": std_b,
            "ci_95_lower": max(0.0, mean_b - ci_margin),
            "ci_95_upper": mean_b + ci_margin,
            "max_blast_radius": int(np.max(sizes)),
            "min_blast_radius": int(np.min(sizes)),
            "mean_depth": float(np.mean(depths)),
            "max_depth": int(np.max(depths)),
        }

        if track_activation_frequency:
            result["activation_frequency"] = {
                node: count / n_trials
                for node, count in activation_counts.items()
            }

        return result

    def run_top_k_seeds(self, combined_scores: dict, k=config.ICM_TOP_K_SEEDS,
                        n_trials=config.ICM_N_TRIALS) -> pd.DataFrame:
        """Roda ICM para os k seeds com maior score combinado S(v)."""
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_seeds = [pkg for pkg, _ in ranked[:k]]

        logger.info("Rodando ICM para %d seeds (%d trials cada)...", len(top_seeds), n_trials)

        results = []
        for i, seed_pkg in enumerate(top_seeds):
            mc_result = self.monte_carlo(
                seed_nodes={seed_pkg},
                n_trials=n_trials,
                # só rastreia frequência de ativação para os top-10 (memória)
                track_activation_frequency=(i < 10),
            )

            results.append({
                "seed": seed_pkg,
                "combined_score": combined_scores[seed_pkg],
                "mean_blast_radius": mc_result["mean_blast_radius"],
                "std_blast_radius": mc_result["std_blast_radius"],
                "ci_95_lower": mc_result["ci_95_lower"],
                "ci_95_upper": mc_result["ci_95_upper"],
                "max_blast_radius": mc_result["max_blast_radius"],
                "min_blast_radius": mc_result["min_blast_radius"],
                "mean_depth": mc_result["mean_depth"],
                "max_depth": mc_result["max_depth"],
            })

        df = pd.DataFrame(results)
        df = df.sort_values("mean_blast_radius", ascending=False).reset_index(drop=True)
        df["blast_rank"] = df.index + 1

        logger.info(
            "Simulação concluída — maior blast radius: %s (E[B]=%.1f)",
            df.iloc[0]["seed"], df.iloc[0]["mean_blast_radius"],
        )
        return df

    def export_results(self, results_df: pd.DataFrame, output_path=None) -> Path:
        """Exporta resultados da simulação para CSV."""
        dest = Path(output_path) if output_path else config.SIMULATION_RESULTS_CSV
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = export_scores(results_df, dest, "%.4f")
        logger.info("Resultados exportados -> %s (%d seeds)", out, len(results_df))
        return out
