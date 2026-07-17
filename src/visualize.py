import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, TOP_N
from src.utils import logger, Timer, ensure_dir


def plot_top_spreaders_bar(influence_df, top_n=TOP_N):
    """
    Horizontal bar chart of top spreaders
    and their influence scores.
    """
    with Timer("Plotting top spreaders bar chart"):
        ensure_dir(OUTPUT_DIR)

        top = influence_df.head(top_n).copy()
        top = top.sort_values("influence_score", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            top["node_id"],
            top["influence_score"],
            color=plt.cm.Reds(
                np.linspace(0.4, 0.9, len(top))
            )
        )

        # add value labels on bars
        for bar, val in zip(bars, top["influence_score"]):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", fontsize=9
            )

        ax.set_xlabel("Influence Score", fontsize=11)
        ax.set_title(
            f"DeceptRank — Top {top_n} Super-Spreaders",
            fontsize=13, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

        path = os.path.join(OUTPUT_DIR, "top_spreaders.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Bar chart saved → {path}")