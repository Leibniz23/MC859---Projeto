"""Validação estatística do modelo contra dados OSV.dev.

Três camadas de análise:

Camada 1 — Correlação (Grupo B apenas)
  Teste 1: Spearman rho entre R(v) e vuln_count
  Teste 2: Correlação parcial controlando por popularidade

Camada 2 — Enriquecimento (Grupo A vs. Grupo B)
  Teste 3: Precision@k para k em {10, 25, 50, 100}
  Teste 4: Lift@k com base rate do Grupo B
  Teste 5: Mann-Whitney U

Camada 3 — Robustez
  Teste 6: Sensibilidade a thresholds CVSS
  Teste 7: Bootstrap CI para Precision@k e Lift@k
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from supplychain import config
from supplychain.common.scores_io import export_scores
from supplychain.validation.ground_truth import build_ground_truth
from supplychain.validation.osv_client import OSVClient

logger = logging.getLogger(__name__)


class ValidationAnalyzer:
    """Executa a análise estatística de três camadas."""

    def __init__(
        self,
        risk_scores_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        group_a_packages: list,
        group_b_packages: list,
        usage_norm=None,
        reach_norm=None,
        in_degree_norm=None,
    ):
        self._risk_df = risk_scores_df.set_index("package")
        self._gt_df = ground_truth_df.set_index("package")
        self._group_a = group_a_packages
        self._group_b = group_b_packages
        self._usage_norm = usage_norm or {}
        self._reach_norm = reach_norm or {}
        self._in_degree_norm = in_degree_norm or {}

        self._merged_a = self._merge_group(group_a_packages)
        self._merged_b = self._merge_group(group_b_packages)

        logger.info(
            "ValidationAnalyzer: Grupo A=%d, Grupo B=%d pacotes",
            len(self._merged_a), len(self._merged_b),
        )

    def _merge_group(self, packages: list) -> pd.DataFrame:
        """Junta risk scores e ground truth para um grupo de pacotes."""
        pkgs_with_data = [
            p for p in packages
            if p in self._risk_df.index and p in self._gt_df.index
        ]
        risk_subset = self._risk_df.loc[pkgs_with_data, ["risk_score"]].copy()
        gt_subset = self._gt_df.loc[pkgs_with_data].copy()
        return risk_subset.join(gt_subset, how="inner")

    def test_spearman(self) -> dict:
        """Teste 1: Spearman rho no Grupo B. Critério: rho > 0.3, p < 0.01."""
        df = self._merged_b.dropna(subset=["vuln_count"])

        if len(df) < 10:
            logger.warning("Poucos dados para Spearman (%d)", len(df))
            return {"rho": np.nan, "p_value": np.nan, "n": len(df), "passed": False}

        rho, p_val = stats.spearmanr(df["risk_score"], df["vuln_count"])

        passed = rho > 0.3 and p_val < 0.01
        logger.info(
            "Teste 1 — Spearman rho (Grupo B): rho=%.4f, p=%.2e, n=%d -> %s",
            rho, p_val, len(df), "PASS" if passed else "FAIL",
        )
        return {"rho": rho, "p_value": p_val, "n": len(df), "passed": passed}

    def test_kendall(self) -> dict:
        """Kendall tau no Grupo B (mais robusto a empates que Spearman)."""
        df = self._merged_b.dropna(subset=["vuln_count"])

        if len(df) < 10:
            return {"tau": np.nan, "p_value": np.nan, "n": len(df), "passed": False}

        tau, p_val = stats.kendalltau(df["risk_score"], df["vuln_count"])

        passed = tau > 0.2 and p_val < 0.01
        logger.info(
            "Suplementar — Kendall tau (Grupo B): tau=%.4f, p=%.2e, n=%d -> %s",
            tau, p_val, len(df), "PASS" if passed else "FAIL",
        )
        return {"tau": tau, "p_value": p_val, "n": len(df), "passed": passed}

    def test_partial_correlation(self) -> dict:
        """Teste 2: Correlação parcial controlando por popularidade.

        R_topo(v) = (alpha/(alpha+gamma)) * d_in + (gamma/(alpha+gamma)) * reach
        Mede se a topologia prediz vulns independentemente de popularidade.
        Critério: rho_partial > 0.15, p < 0.05
        """
        df = self._merged_b.dropna(subset=["vuln_count"]).copy()

        if not self._usage_norm or not self._in_degree_norm or not self._reach_norm:
            logger.warning("Faltam dados de usage/in_degree/reach para correlação parcial.")
            return {
                "rho_partial": np.nan, "p_value": np.nan, "n": 0,
                "r_topo_vuln": np.nan, "r_topo_usage": np.nan,
                "r_usage_vuln": np.nan, "passed": False,
            }

        alpha = config.RISK_ALPHA
        gamma = config.RISK_GAMMA
        w_in = alpha / (alpha + gamma)
        w_reach = gamma / (alpha + gamma)

        r_topo = []
        usage = []
        vuln = []
        valid_pkgs = []

        for pkg in df.index:
            if pkg in self._in_degree_norm and pkg in self._reach_norm and pkg in self._usage_norm:
                r_t = w_in * self._in_degree_norm[pkg] + w_reach * self._reach_norm[pkg]
                r_topo.append(r_t)
                usage.append(self._usage_norm[pkg])
                vuln.append(df.loc[pkg, "vuln_count"])
                valid_pkgs.append(pkg)

        if len(valid_pkgs) < 10:
            logger.warning("Poucos pacotes válidos para correlação parcial (%d)", len(valid_pkgs))
            return {
                "rho_partial": np.nan, "p_value": np.nan, "n": len(valid_pkgs),
                "r_topo_vuln": np.nan, "r_topo_usage": np.nan,
                "r_usage_vuln": np.nan, "passed": False,
            }

        r_topo_arr = np.array(r_topo)
        usage_arr = np.array(usage)
        vuln_arr = np.array(vuln)

        rho_tv, _ = stats.spearmanr(r_topo_arr, vuln_arr)
        rho_tu, _ = stats.spearmanr(r_topo_arr, usage_arr)
        rho_uv, _ = stats.spearmanr(usage_arr, vuln_arr)

        # rho(X,Y|Z) = (rho(X,Y) - rho(X,Z)*rho(Z,Y)) / sqrt((1-rho(X,Z)^2)(1-rho(Z,Y)^2))
        denominator = np.sqrt((1 - rho_tu ** 2) * (1 - rho_uv ** 2))
        if denominator > 1e-10:
            rho_partial = (rho_tv - rho_tu * rho_uv) / denominator
        else:
            logger.warning("Denominador degenerado na correlação parcial")
            rho_partial = np.nan

        n = len(valid_pkgs)
        if not np.isnan(rho_partial) and n > 3:
            z = 0.5 * np.log((1 + rho_partial) / (1 - rho_partial + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            z_stat = abs(z) / se
            p_val = 2 * (1 - stats.norm.cdf(z_stat))
        else:
            p_val = np.nan

        passed = (
            not np.isnan(rho_partial)
            and rho_partial > 0.15
            and not np.isnan(p_val)
            and p_val < 0.05
        )

        logger.info(
            "Teste 2 — Correlação parcial (Grupo B): rho_partial=%.4f, p=%.2e, n=%d -> %s",
            rho_partial, p_val if not np.isnan(p_val) else 0, n,
            "PASS" if passed else "FAIL",
        )

        return {
            "rho_partial": rho_partial,
            "p_value": p_val,
            "n": n,
            "r_topo_vuln": rho_tv,
            "r_topo_usage": rho_tu,
            "r_usage_vuln": rho_uv,
            "passed": passed,
        }

    def test_precision_at_k(self, vuln_column="has_high_vuln") -> dict:
        """Teste 3: Precision@k para k em {10, 25, 50, 100}.

        Precision@k = |{v in top-k : vuln(v) = 1}| / k
        """
        df = self._merged_a.copy()
        df = df.sort_values("risk_score", ascending=False)

        results = {}
        for k in config.PRECISION_K_VALUES:
            top_k = df.head(k)
            if vuln_column in top_k.columns:
                n_vuln = int(top_k[vuln_column].sum())
                precision = n_vuln / k if k > 0 else 0.0
            else:
                n_vuln = 0
                precision = 0.0

            results[str(k)] = {
                "precision": precision,
                "count_vulnerable": n_vuln,
                "k": k,
            }
            logger.info(
                "  Precision@%d (%s): %.2f%% (%d/%d)",
                k, vuln_column, precision * 100, n_vuln, k,
            )

        return results

    def test_lift_at_k(self, vuln_column="has_high_vuln") -> dict:
        """Teste 4: Lift@k = Precision@k / base_rate(Grupo B).

        Base rate do Grupo B para evitar bias do denominador.
        Critério: Lift@25 > 3.
        """
        df_b = self._merged_b.copy()
        if vuln_column in df_b.columns:
            base_rate = float(df_b[vuln_column].mean())
        else:
            base_rate = 0.0

        # Wilson CI para a base rate
        n_b = len(df_b)
        if n_b > 0 and 0 < base_rate < 1:
            z = 1.96
            denom = 1 + z ** 2 / n_b
            centre = (base_rate + z ** 2 / (2 * n_b)) / denom
            margin = z * np.sqrt(base_rate * (1 - base_rate) / n_b + z ** 2 / (4 * n_b ** 2)) / denom
            base_rate_ci = (centre - margin, centre + margin)
        else:
            base_rate_ci = (0.0, 0.0)

        logger.info(
            "Base rate (%s) Grupo B: %.4f (95%% Wilson CI: [%.4f, %.4f])",
            vuln_column, base_rate, base_rate_ci[0], base_rate_ci[1],
        )

        precision_results = self.test_precision_at_k(vuln_column=vuln_column)

        results = {}
        for k_str, prec_data in precision_results.items():
            precision = prec_data["precision"]
            lift = precision / base_rate if base_rate > 0 else np.inf
            results[k_str] = {
                "lift": lift,
                "precision": precision,
                "base_rate": base_rate,
                "base_rate_ci_lower": base_rate_ci[0],
                "base_rate_ci_upper": base_rate_ci[1],
                "k": prec_data["k"],
            }
            logger.info(
                "  Lift@%s (%s): %.2fx (precision=%.2f%%, base=%.2f%%)",
                k_str, vuln_column, lift, precision * 100, base_rate * 100,
            )

        return results

    def test_mann_whitney(self) -> dict:
        """Teste 5: Mann-Whitney U comparando vuln_count entre Grupo A e B.

        Não-paramétrico, adequado para contagens com distribuição assimétrica.
        Critério: p < 0.01.
        """
        vuln_a = self._merged_a["vuln_count"].dropna().values
        vuln_b = self._merged_b["vuln_count"].dropna().values

        if len(vuln_a) < 5 or len(vuln_b) < 5:
            logger.warning("Dados insuficientes para Mann-Whitney")
            return {
                "statistic": np.nan, "p_value": np.nan,
                "n_a": len(vuln_a), "n_b": len(vuln_b),
                "median_a": np.nan, "median_b": np.nan,
                "passed": False,
            }

        # alternative='greater': esperamos que Grupo A tenha MAIS vulns
        stat, p_val = stats.mannwhitneyu(vuln_a, vuln_b, alternative="greater")

        passed = p_val < 0.01
        logger.info(
            "Teste 5 — Mann-Whitney U: U=%.1f, p=%.2e, median_A=%.0f, median_B=%.0f -> %s",
            stat, p_val, np.median(vuln_a), np.median(vuln_b),
            "PASS" if passed else "FAIL",
        )

        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "n_a": len(vuln_a),
            "n_b": len(vuln_b),
            "median_a": float(np.median(vuln_a)),
            "median_b": float(np.median(vuln_b)),
            "mean_a": float(np.mean(vuln_a)),
            "mean_b": float(np.mean(vuln_b)),
            "passed": passed,
        }

    def test_cvss_sensitivity(self) -> dict:
        """Teste 6: Sensibilidade a thresholds CVSS.

        Repete Precision@25 e Lift@25 para vários thresholds CVSS.
        """
        results = {}

        for name, threshold in config.CVSS_THRESHOLDS.items():
            vuln_col = f"has_{name}_vuln"

            for df in [self._merged_a, self._merged_b]:
                if name == "any":
                    df[vuln_col] = (df["vuln_count"] > 0).astype(int)
                else:
                    df[vuln_col] = (
                        (df["vuln_count"] > 0) & (df["cvss_max"] >= threshold)
                    ).astype(int)

            lift_results = self.test_lift_at_k(vuln_column=vuln_col)

            results[name] = {
                "threshold": threshold,
                "lift_25": lift_results.get("25", {}).get("lift", np.nan),
                "precision_25": lift_results.get("25", {}).get("precision", np.nan),
                "base_rate": lift_results.get("25", {}).get("base_rate", np.nan),
            }

            for df in [self._merged_a, self._merged_b]:
                if vuln_col in df.columns:
                    df.drop(columns=[vuln_col], inplace=True)

        stable_count = sum(
            1 for r in results.values()
            if not np.isnan(r["lift_25"]) and r["lift_25"] > 3
        )
        logger.info(
            "Teste 6 — CVSS sensibilidade: %d / %d thresholds com Lift@25 > 3",
            stable_count, len(results),
        )

        return results

    def test_bootstrap_ci(self, vuln_column="has_high_vuln",
                          n_resamples=config.BOOTSTRAP_N_RESAMPLES) -> dict:
        """Teste 7: Bootstrap CI para Precision@k e Lift@k (10k resamples).

        Reamostras o Grupo B para estimar a distribuição da base rate.
        Critério: limite inferior do CI para Lift@25 > 1.5.
        """
        rng = np.random.default_rng(42)

        df_b = self._merged_b.copy()
        if vuln_column not in df_b.columns:
            logger.warning("Coluna %s não encontrada no Grupo B", vuln_column)
            return {}

        vuln_b = df_b[vuln_column].values

        boot_base_rates = np.empty(n_resamples)
        for i in range(n_resamples):
            sample = rng.choice(vuln_b, size=len(vuln_b), replace=True)
            boot_base_rates[i] = sample.mean()

        base_rate_mean = float(boot_base_rates.mean())
        base_rate_ci = (
            float(np.percentile(boot_base_rates, 2.5)),
            float(np.percentile(boot_base_rates, 97.5)),
        )

        precision_results = self.test_precision_at_k(vuln_column=vuln_column)

        results = {}
        for k_str, prec_data in precision_results.items():
            precision = prec_data["precision"]

            valid_rates = boot_base_rates[boot_base_rates > 0]
            if len(valid_rates) > 0:
                boot_lifts = precision / valid_rates
                lift_mean = float(np.mean(boot_lifts))
                lift_ci = (
                    float(np.percentile(boot_lifts, 2.5)),
                    float(np.percentile(boot_lifts, 97.5)),
                )
            else:
                lift_mean = np.nan
                lift_ci = (np.nan, np.nan)

            results[k_str] = {
                "lift_mean": lift_mean,
                "lift_ci_lower": lift_ci[0],
                "lift_ci_upper": lift_ci[1],
                "precision": precision,
                "base_rate_mean": base_rate_mean,
                "base_rate_ci_lower": base_rate_ci[0],
                "base_rate_ci_upper": base_rate_ci[1],
            }

            logger.info(
                "  Bootstrap Lift@%s: mean=%.2f, 95%% CI=[%.2f, %.2f]",
                k_str, lift_mean, lift_ci[0], lift_ci[1],
            )

        return results

    def run_all_tests(self) -> dict:
        """Executa a análise completa de três camadas e retorna todos os resultados."""
        logger.info("Iniciando análise de validação estatística...")

        results = {}

        logger.info("Camada 1: Correlação (Grupo B)")
        results["spearman"] = self.test_spearman()
        results["kendall"] = self.test_kendall()
        results["partial_correlation"] = self.test_partial_correlation()

        logger.info("Camada 2: Enriquecimento (Grupo A vs. B)")
        results["precision_at_k"] = self.test_precision_at_k()
        results["lift_at_k"] = self.test_lift_at_k()
        results["mann_whitney"] = self.test_mann_whitney()

        logger.info("Camada 3: Robustez")
        results["cvss_sensitivity"] = self.test_cvss_sensitivity()
        results["bootstrap_ci"] = self.test_bootstrap_ci()

        tests_passed = 0
        tests_total = 0
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "passed" in test_result:
                tests_total += 1
                if test_result["passed"]:
                    tests_passed += 1
                    logger.info("  PASS: %s", test_name)
                else:
                    logger.info("  FAIL: %s", test_name)

        logger.info("Resultado: %d / %d testes principais passaram.", tests_passed, tests_total)

        return results


class OSVValidator:
    """Orquestrador end-to-end: busca dados OSV, constrói ground truth e roda testes."""

    def __init__(
        self,
        risk_scores_df: pd.DataFrame,
        graph: nx.DiGraph,
        group_a_size=config.OSV_GROUP_A_SIZE,
        group_b_size=config.OSV_GROUP_B_SIZE,
        random_seed=42,
    ):
        self._risk_df = risk_scores_df
        self._graph = graph
        self._rng = np.random.default_rng(random_seed)

        sorted_df = risk_scores_df.sort_values("risk_score", ascending=False)
        self._group_a = sorted_df.head(group_a_size)["package"].tolist()

        remaining = sorted_df.iloc[group_a_size:]["package"].tolist()
        sample_size = min(group_b_size, len(remaining))
        self._group_b = list(
            self._rng.choice(remaining, size=sample_size, replace=False)
        )

        logger.info(
            "Grupos: A (top-%d) = %d pacotes, B (controle) = %d pacotes",
            group_a_size, len(self._group_a), len(self._group_b),
        )

        self._usage_norm = {}
        self._in_degree_norm = {}
        self._reach_norm = {}
        for _, row in risk_scores_df.iterrows():
            pkg = row["package"]
            self._usage_norm[pkg] = float(row.get("usage_norm", 0.0))
            self._in_degree_norm[pkg] = float(row.get("in_degree_norm", 0.0))
            self._reach_norm[pkg] = float(row.get("reach_norm", 0.0))

    def fetch_advisories(self) -> pd.DataFrame:
        """Consulta OSV.dev para todos os pacotes do Grupo A e B."""
        client = OSVClient()
        all_packages = self._group_a + self._group_b

        logger.info("Consultando OSV.dev para %d pacotes...", len(all_packages))
        advisories = client.query_batch(all_packages)

        gt_df = build_ground_truth(advisories)

        logger.info(
            "Ground truth: %d com vuln, %d com vuln alta (CVSS >= 7.0)",
            int(gt_df["has_vuln"].sum()), int(gt_df["has_high_vuln"].sum()),
        )

        return gt_df

    def run_validation(self, ground_truth_df=None) -> dict:
        """Roda o pipeline completo de validação."""
        if ground_truth_df is None:
            ground_truth_df = self.fetch_advisories()

        analyzer = ValidationAnalyzer(
            risk_scores_df=self._risk_df,
            ground_truth_df=ground_truth_df,
            group_a_packages=self._group_a,
            group_b_packages=self._group_b,
            usage_norm=self._usage_norm,
            reach_norm=self._reach_norm,
            in_degree_norm=self._in_degree_norm,
        )

        return analyzer.run_all_tests()

    def export_advisories(self, ground_truth_df: pd.DataFrame, output_path=None) -> Path:
        """Exporta o ground truth de advisories para CSV."""
        dest = Path(output_path) if output_path else config.OSV_ADVISORIES_CSV
        dest.parent.mkdir(parents=True, exist_ok=True)
        ground_truth_df.to_csv(dest, index=False)
        logger.info("Advisories exportados -> %s (%d pacotes)", dest, len(ground_truth_df))
        return dest

    def export_validation_report(self, results: dict, output_path=None) -> Path:
        """Exporta um resumo dos resultados de validação como CSV."""
        dest = Path(output_path) if output_path else config.VALIDATION_REPORT_CSV
        dest.parent.mkdir(parents=True, exist_ok=True)

        rows = []

        sp = results.get("spearman", {})
        rows.append({
            "test": "Spearman rho (Group B)",
            "metric": "rho",
            "value": sp.get("rho", np.nan),
            "p_value": sp.get("p_value", np.nan),
            "n": sp.get("n", 0),
            "passed": sp.get("passed", False),
        })

        kt = results.get("kendall", {})
        rows.append({
            "test": "Kendall tau (Group B)",
            "metric": "tau",
            "value": kt.get("tau", np.nan),
            "p_value": kt.get("p_value", np.nan),
            "n": kt.get("n", 0),
            "passed": kt.get("passed", False),
        })

        pc = results.get("partial_correlation", {})
        rows.append({
            "test": "Partial correlation",
            "metric": "rho_partial",
            "value": pc.get("rho_partial", np.nan),
            "p_value": pc.get("p_value", np.nan),
            "n": pc.get("n", 0),
            "passed": pc.get("passed", False),
        })

        mw = results.get("mann_whitney", {})
        rows.append({
            "test": "Mann-Whitney U",
            "metric": "U_statistic",
            "value": mw.get("statistic", np.nan),
            "p_value": mw.get("p_value", np.nan),
            "n": mw.get("n_a", 0) + mw.get("n_b", 0),
            "passed": mw.get("passed", False),
        })

        for k_str, lift_data in results.get("lift_at_k", {}).items():
            rows.append({
                "test": f"Lift@{k_str}",
                "metric": "lift",
                "value": lift_data.get("lift", np.nan),
                "p_value": np.nan,
                "n": lift_data.get("k", 0),
                "passed": lift_data.get("lift", 0) > 3 if k_str == "25" else np.nan,
            })

        df = pd.DataFrame(rows)
        export_scores(df, dest, "%.6f")
        logger.info("Relatório de validação exportado -> %s", dest)
        return dest
