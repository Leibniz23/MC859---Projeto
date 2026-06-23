"""Gera plots de distribuição de graus, SCCs e WCCs."""

import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# estilo visual
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# paleta de cores
_CLR_IN = "#1B9E77"
_CLR_OUT = "#D95F02"
_CLR_MEAN = "#7570B3"
_CLR_GIANT = "#E7298A"
_CLR_SCATTER = "#66A61E"
_CLR_WCC = "#3182BD"
_CLR_WCC_GIANT = "#A63603"


class GraphPlotter:
    """Salva plots de análise do grafo em disco."""

    def __init__(self, output_dir=".", dpi=200):
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._dpi = max(dpi, 150)

    def plot_degree_distribution(self, in_degrees, out_degrees,
                                 filename="degree_distribution.png") -> Path:
        """Histograma log-log de in/out-degree com bins logarítmicos."""
        fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(14, 5.5))

        self._plot_single_degree(ax_in, in_degrees, "In-Degree", _CLR_IN)
        self._plot_single_degree(ax_out, out_degrees, "Out-Degree", _CLR_OUT)

        fig.suptitle(
            "Degree Distribution (Log-Log Scale with Logarithmic Binning)",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        dest = self._out / filename
        fig.savefig(str(dest), dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Plot de grau salvo -> %s", dest)
        return dest

    @staticmethod
    def _plot_single_degree(ax, degrees, label, colour):
        """Desenha o histograma de grau em ax."""
        positive = degrees[degrees > 0]

        if len(positive) == 0:
            ax.text(0.5, 0.5, "No non-zero values", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{label} Distribution")
            return

        d_min, d_max = positive.min(), positive.max()

        if d_min == d_max:
            bins = np.array([d_min - 0.5, d_max + 0.5])
        else:
            n_bins = max(15, int(np.log2(len(positive))))
            bins = np.logspace(np.log10(d_min), np.log10(d_max + 1), n_bins + 1)

        ax.hist(
            positive, bins=bins, density=False,
            color=colour, edgecolor="white", linewidth=0.6,
            alpha=0.85, label=f"{label} (n={len(positive)})",
        )

        mean_val = np.mean(degrees)
        ax.axvline(
            mean_val, color=_CLR_MEAN, linestyle="--", linewidth=1.5,
            label=f"Mean = {mean_val:.2f}",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label} (k)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} Distribution")
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

        n_zeros = int(np.sum(degrees == 0))
        if n_zeros > 0:
            ax.annotate(
                f"{n_zeros} nodes with k=0",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
            )

    def plot_scc_distribution(self, size_distribution: Counter, largest_scc_size: int,
                              filename="scc_size_distribution.png"):
        """Scatter plot log-log de tamanho vs frequência de SCCs."""
        if len(size_distribution) <= 1:
            logger.info("Apenas %d tamanho(s) de SCC — pulando plot.", len(size_distribution))
            return None

        sizes = np.array(sorted(size_distribution.keys()))
        counts = np.array([size_distribution[s] for s in sizes])

        fig, ax = plt.subplots(figsize=(9, 6))

        mask_regular = sizes != largest_scc_size
        ax.scatter(
            sizes[mask_regular], counts[mask_regular],
            s=50, color=_CLR_SCATTER, edgecolors="white", linewidths=0.5,
            zorder=3, label="SCC Sizes",
        )

        mask_giant = sizes == largest_scc_size
        if np.any(mask_giant):
            giant_count = counts[mask_giant][0]
            ax.scatter(
                [largest_scc_size], [giant_count],
                s=180, color=_CLR_GIANT, edgecolors="black", linewidths=1.2,
                zorder=4, marker="*", label="Giant SCC",
            )
            ax.annotate(
                f"Giant SCC\nsize = {largest_scc_size:,} nodes\ncount = {giant_count}",
                xy=(largest_scc_size, giant_count),
                xytext=(largest_scc_size * 0.3, giant_count * 3),
                fontsize=9, fontweight="bold", color=_CLR_GIANT,
                arrowprops=dict(arrowstyle="->", color=_CLR_GIANT, lw=1.5,
                                connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_CLR_GIANT, alpha=0.9),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SCC Size (k vertices)")
        ax.set_ylabel("Frequency (number of SCCs with size k)")
        ax.set_title("Strongly Connected Component — Size Distribution", fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        dest = self._out / filename
        fig.savefig(str(dest), dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Plot de SCC salvo -> %s", dest)
        return dest

    def plot_wcc_distribution(self, size_distribution: Counter, largest_wcc_size: int,
                              filename="wcc_size_distribution.png"):
        """Scatter plot log-log de tamanho vs frequência de WCCs."""
        if len(size_distribution) <= 1:
            logger.info("Apenas %d tamanho(s) de WCC — pulando plot.", len(size_distribution))
            return None

        sizes = np.array(sorted(size_distribution.keys()))
        counts = np.array([size_distribution[s] for s in sizes])

        fig, ax = plt.subplots(figsize=(9, 6))

        mask_regular = sizes != largest_wcc_size
        ax.scatter(
            sizes[mask_regular], counts[mask_regular],
            s=50, color=_CLR_WCC, edgecolors="white", linewidths=0.5,
            zorder=3, label="WCC Sizes",
        )

        mask_giant = sizes == largest_wcc_size
        if np.any(mask_giant):
            giant_count = counts[mask_giant][0]
            ax.scatter(
                [largest_wcc_size], [giant_count],
                s=180, color=_CLR_WCC_GIANT, edgecolors="black", linewidths=1.2,
                zorder=4, marker="*", label="Giant WCC",
            )
            ax.annotate(
                f"Giant WCC\nsize = {largest_wcc_size:,} nodes\ncount = {giant_count}",
                xy=(largest_wcc_size, giant_count),
                xytext=(largest_wcc_size * 0.3, giant_count * 3),
                fontsize=9, fontweight="bold", color=_CLR_WCC_GIANT,
                arrowprops=dict(arrowstyle="->", color=_CLR_WCC_GIANT, lw=1.5,
                                connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_CLR_WCC_GIANT, alpha=0.9),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("WCC Size (k vertices)")
        ax.set_ylabel("Frequency (number of WCCs with size k)")
        ax.set_title("Weakly Connected Component — Size Distribution", fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        dest = self._out / filename
        fig.savefig(str(dest), dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Plot de WCC salvo -> %s", dest)
        return dest
