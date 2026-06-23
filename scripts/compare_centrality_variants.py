"""Compara variantes de centralidade (betweenness vs none vs katz) contra o ground truth OSV.

Para cada ranking candidato reporta Precision@k e Lift@k no universo avaliável
(os 2000 pacotes consultados no OSV), usando base rate do Grupo B reconstruído
com np.random.default_rng(42) — igual ao validator.

Run:  python scripts/compare_centrality_variants.py
"""
import numpy as np
import pandas as pd

from supplychain import config

KS = [10, 25, 50, 100]
DATA = config.DATA_DIR


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(DATA / "osv_advisories.csv").set_index("package")
    gt["has_vuln"] = gt["has_vuln"].astype(int)
    gt["has_high_vuln"] = gt["has_high_vuln"].astype(int)
    return gt


def base_rates(gt: pd.DataFrame) -> dict:
    """Reconstrói o Grupo B para estimar base rates não-enviesadas."""
    risk = pd.read_csv(DATA / "risk_scores.csv").sort_values("risk_score", ascending=False)
    remaining = risk.iloc[config.OSV_GROUP_A_SIZE:]["package"].tolist()
    rng = np.random.default_rng(42)
    size = min(config.OSV_GROUP_B_SIZE, len(remaining))
    group_b = list(rng.choice(remaining, size=size, replace=False))
    gb = gt.reindex(group_b).dropna(subset=["has_vuln"])
    return {
        "any": float(gb["has_vuln"].mean()),
        "high": float(gb["has_high_vuln"].mean()),
        "n_group_b": len(gb),
    }


def evaluate(ranking: pd.Series, gt: pd.DataFrame, br: dict) -> pd.DataFrame:
    """ranking: Series indexada por pacote -> score (maior = mais crítico)."""
    evaluable = ranking[ranking.index.isin(gt.index)].sort_values(ascending=False)
    rows = []
    for k in KS:
        topk = evaluable.head(k).index
        hv = gt.loc[topk, "has_high_vuln"].sum()
        av = gt.loc[topk, "has_vuln"].sum()
        rows.append({
            "k": k,
            "prec_high": hv / k,
            "lift_high": (hv / k) / br["high"] if br["high"] else np.nan,
            "prec_any": av / k,
            "lift_any": (av / k) / br["any"] if br["any"] else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    gt = load_ground_truth()
    br = base_rates(gt)
    print(f"Evaluable universe: {len(gt)} packages")
    print(f"Base rate (Group B, n={br['n_group_b']}): "
          f"any={br['any']:.4f}, high={br['high']:.4f}\n")

    risk = pd.read_csv(DATA / "risk_scores.csv").set_index("package")["risk_score"]

    variants = {
        "betweenness": DATA / "centrality_scores.betweenness_baseline.csv",
        "none": DATA / "centrality_none.csv",
        "katz": DATA / "centrality_katz.csv",
    }
    cent = {name: pd.read_csv(path).set_index("package") for name, path in variants.items()}

    rankings = {"R(v) [Phase-4 target]": risk}
    for name, df in cent.items():
        rankings[f"S(v) {name}"] = df["combined_score"]
        rankings[f"C(v) {name}"] = df["criticality_index"]

    for label, series in rankings.items():
        print("=" * 70)
        print(f"  {label}")
        print("=" * 70)
        print(evaluate(series, gt, br).to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))
        print()

    # rastreia onde google-auth fica em cada ranking (artefato conhecido)
    print("=" * 70)
    print("  google-auth rank by criticality_index C(v) (artifact tracker)")
    print("=" * 70)
    for name, df in cent.items():
        ranked = df["criticality_index"].sort_values(ascending=False).reset_index()
        pos = ranked.index[ranked["package"] == "google-auth"]
        rank = int(pos[0]) + 1 if len(pos) else None
        print(f"  {name:12s}: rank {rank}  "
              f"(google-auth has_high_vuln={int(gt.loc['google-auth','has_high_vuln']) if 'google-auth' in gt.index else 'NA'})")


if __name__ == "__main__":
    main()
