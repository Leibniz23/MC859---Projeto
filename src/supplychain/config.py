"""
Constantes de configuração do pipeline de análise de dependências PyPI.

Para autenticação GCP local, rode:
    gcloud auth application-default login
e garanta que o projeto mc859-projeto tenha a BigQuery API habilitada.
"""

from pathlib import Path

# GCP — projeto de faturamento para as queries BigQuery
GCP_PROJECT_ID: str = "mc859-projeto"

# Quantos pacotes sementes selecionar por volume de downloads
NUM_SEED_PACKAGES: int = 5_000

# Janela de tempo (dias) para a query de downloads
DOWNLOAD_LOOKBACK_DAYS: int = 30

# Caminhos de arquivo
# config.py fica em src/supplychain/, então a raiz do repo é parents[2]
DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
SEEDS_CSV: Path = DATA_DIR / "seeds.csv"
DEPENDENCIES_CSV: Path = DATA_DIR / "dependencies.csv"
USAGE_STATS_CSV: Path = DATA_DIR / "usage_stats.csv"
GRAPH_OUTPUT: Path = DATA_DIR / "dependency_graph.graphml"

# Saídas das fases 1-4
RISK_SCORES_CSV: Path = DATA_DIR / "risk_scores.csv"
CENTRALITY_SCORES_CSV: Path = DATA_DIR / "centrality_scores.csv"
SIMULATION_RESULTS_CSV: Path = DATA_DIR / "simulation_results.csv"
OSV_ADVISORIES_CSV: Path = DATA_DIR / "osv_advisories.csv"
VALIDATION_REPORT_CSV: Path = DATA_DIR / "validation_report.csv"

# Fase 1 — pesos de R(v) = alpha*d_in + beta*u + gamma*reach
RISK_ALPHA: float = 0.35
RISK_BETA: float = 0.25
RISK_GAMMA: float = 0.40

# Fase 2 — métrica estrutural de C(v)
# "none" é recomendado: betweenness é degenerado neste quasi-DAG (~97% dos nós
# ficam em zero) e é O(V*E). "katz" é alternativa; "betweenness" só pra reproduzir
# resultado antigo.
CENTRALITY_STRUCTURAL_METRIC: str = "none"

# Pesos com 3 termos (degree + structural + pagerank)
CENTRALITY_W_DEGREE: float = 0.25
CENTRALITY_W_STRUCTURAL: float = 0.35
CENTRALITY_W_PAGERANK: float = 0.40

# Pesos sem termo estrutural ("none"): C(v) = w_degree*d_in + w_pagerank*PR
CENTRALITY_W_DEGREE_NB: float = 0.40
CENTRALITY_W_PAGERANK_NB: float = 0.60

# PageRank
PAGERANK_DAMPING: float = 0.85

# Katz centrality — alpha deve ser < 1/lambda_max
KATZ_ALPHA: float = 0.1
KATZ_BETA: float = 1.0

# Betweenness — só usado quando CENTRALITY_STRUCTURAL_METRIC == "betweenness"
BETWEENNESS_EXACT: bool = True
BETWEENNESS_K_PIVOTS: int = 500

# Score combinado: S(v) = lambda*R(v) + (1-lambda)*C(v)
COMBINED_LAMBDA: float = 0.5

# Fase 3 — simulação ICM
ICM_P_BASE: float = 0.01
ICM_N_TRIALS: int = 1_000
ICM_TOP_K_SEEDS: int = 100
ICM_RANDOM_SEED: int = 42

# Fase 4 — validação OSV
OSV_API_URL: str = "https://api.osv.dev/v1/query"
OSV_BATCH_URL: str = "https://api.osv.dev/v1/querybatch"
OSV_GROUP_A_SIZE: int = 1_000
OSV_GROUP_B_SIZE: int = 1_000
OSV_REQUEST_DELAY: float = 0.1
OSV_MAX_RETRIES: int = 3
OSV_BATCH_SIZE: int = 100
BOOTSTRAP_N_RESAMPLES: int = 10_000
PRECISION_K_VALUES: list[int] = [10, 25, 50, 100]
CVSS_THRESHOLDS: dict[str, float] = {
    "any": 0.1,
    "medium": 4.0,
    "high": 7.0,
    "critical": 9.0,
}
# Mapeamento label textual -> score CVSS aproximado (midpoints FIRST/NVD v3.x)
SEVERITY_LABEL_MAP: dict[str, float] = {
    "low": 2.0,
    "moderate": 5.0,
    "medium": 5.0,
    "high": 7.5,
    "critical": 9.5,
}
