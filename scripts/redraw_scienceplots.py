from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"


def savefig(fig, stem: str) -> None:
    pdf_path = IMAGES / f"{stem}.pdf"
    png_path = IMAGES / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def plot_edge_deployment() -> None:
    labels = ["FP16\n(A100)", "INT4\n(Smartphone)", "INT4\n(Edge Hub)"]
    memory_gb = np.array([14.0, 4.2, 4.2])
    tok_s = np.array([45.0, 12.0, 22.0])
    prof = np.array([2.95, 2.82, 2.88])

    x = np.arange(len(labels))
    width = 0.34

    with plt.style.context(["science", "ieee", "no-latex", "grid"]):
        fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
        bars_mem = ax1.bar(
            x - width / 2,
            memory_gb,
            width=width,
            color="#4C78A8",
            label="Memory (GB)",
            zorder=3,
        )
        bars_speed = ax1.bar(
            x + width / 2,
            tok_s,
            width=width,
            color="#F58518",
            label="Speed (tok/s)",
            zorder=3,
        )
        ax1.set_ylabel("Memory / Speed")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 50)

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            prof,
            color="#54A24B",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Professionalism",
        )
        ax2.set_ylabel("Professionalism")
        ax2.set_ylim(2.7, 3.0)

        for bar in list(bars_mem) + list(bars_speed):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        for xi, yi in zip(x, prof):
            ax2.text(xi, yi + 0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=7)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=7)
        ax1.set_title("Edge Deployment Trade-off")
        fig.tight_layout()
        savefig(fig, "edge_deployment_science")
        plt.close(fig)


def plot_component_ablation() -> None:
    labels = [
        "Full",
        "w/o Persona",
        "w/o RAG",
        "w/o LenNorm",
        "w/o Margin",
        "w/o CoT",
        "Rule-based",
    ]
    rouge_l = np.array([29.21, 27.84, 28.15, 28.67, 28.93, 27.52, 26.89])
    bleu4 = np.array([17.23, 14.56, 15.12, 16.08, 16.44, 14.23, 13.47])
    prof = np.array([2.95, 2.71, 2.78, 2.83, 2.87, 2.68, 2.59])

    x = np.arange(len(labels))

    with plt.style.context(["science", "ieee", "no-latex", "grid"]):
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.0), sharex=True)
        palette = ["#4C78A8", "#F58518", "#54A24B"]
        metrics = [
            (axes[0], rouge_l, "ROUGE-L", palette[0]),
            (axes[1], bleu4, "BLEU-4", palette[1]),
            (axes[2], prof, "Professionalism", palette[2]),
        ]

        for ax, values, ylabel, color in metrics:
            ax.bar(x, values, color=color, zorder=3)
            ax.set_ylabel(ylabel)
            for xi, yi in zip(x, values):
                ax.text(xi, yi + (0.18 if ylabel != "Professionalism" else 0.02), f"{yi:.2f}", ha="center", va="bottom", fontsize=6.5)

        axes[0].set_title("Component Ablation")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=20, ha="right")
        fig.tight_layout()
        savefig(fig, "component_ablation_science")
        plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_edge_deployment()
    plot_component_ablation()
