#!/usr/bin/env python3
"""Render the controlled search-efficiency results with scienceplots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT.parent.parent / "repro_search_efficiency" / "search_efficiency_summary.csv"
IMAGES = ROOT / "images"

ORDER = [
    ("uct_llm", "UCT + aligned reward", "#0B6E4F"),
    ("uct_rule", "UCT + rule proxy", "#C84C09"),
    ("uniform_llm", "Uniform + aligned reward", "#3C6997"),
]


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    with plt.style.context(["science", "ieee", "no-latex", "grid"]):
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), constrained_layout=True)

        for key, label, color in ORDER:
            sub = df[df["algorithm"] == key].sort_values("T")
            axes[0].plot(
                sub["T"],
                sub["mean_true_utility"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=color,
                label=label,
                zorder=3,
            )
            axes[1].plot(
                sub["T"],
                sub["mean_cumulative_regret"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=color,
                label=label,
                zorder=3,
            )

        for ax in axes:
            ax.axvline(8, linestyle="--", linewidth=1.0, color="#666666")
            ax.set_xticks([1, 2, 4, 8, 12, 16])

        axes[0].set_title("Selected Strategy Utility")
        axes[0].set_xlabel("Simulations per turn (T)")
        axes[0].set_ylabel("Mean true utility")
        axes[0].set_ylim(0.72, 0.90)

        axes[1].set_title("Cumulative Search Regret")
        axes[1].set_xlabel("Simulations per turn (T)")
        axes[1].set_ylabel("Mean cumulative regret")
        axes[1].set_ylim(0.0, 2.5)

        axes[1].legend(loc="upper left", fontsize=6.7, frameon=True)
        axes[0].annotate("T=8", xy=(8, 0.723), xytext=(8.4, 0.728), fontsize=6.8)

        pdf_path = IMAGES / "search_efficiency_science.pdf"
        png_path = IMAGES / "search_efficiency_science.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"wrote {pdf_path}")
        print(f"wrote {png_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
