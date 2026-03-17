import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from src.config import OUTPUT_DIR, TOP_N
from src.utils import logger, Timer, ensure_dir


def plot_spreader_network(G, influence_df, cluster_map, top_n=TOP_N):
    """
    Draw the network graph highlighting
    top spreaders and their communities.
    """
    with Timer("Plotting spreader network"):
        ensure_dir(OUTPUT_DIR)

        # get top n nodes
        top_nodes = set(influence_df.head(top_n)["node_id"].tolist())

        # use full graph but highlight top nodes
        pos    = nx.spring_layout(G, seed=42, k=1.5)
        scores = influence_df.set_index("node_id")["influence_score"]

        # node sizes based on influence score
        node_sizes = []
        for n in G.nodes():
            s = scores.get(n, 0.01)
            node_sizes.append(s * 3000 + 100)

        # node colors based on cluster
        colors = []
        cmap   = plt.cm.tab10
        for n in G.nodes():
            c = cluster_map.get(n, 0)
            colors.append(cmap(c / 10))

        # edge colors based on weight
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w        = max(edge_weights) if edge_weights else 1
        edge_colors  = [plt.cm.Reds(w / max_w) for w in edge_weights]

        fig, ax = plt.subplots(figsize=(14, 10))

        # draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=15,
            width=1.2,
            alpha=0.7
        )

        # draw all nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=colors,
            alpha=0.85
        )

        # draw labels only for top spreaders
        top_labels = {n: n for n in G.nodes() if n in top_nodes}
        nx.draw_networkx_labels(
            G, pos, labels=top_labels,
            ax=ax, font_size=8,
            font_weight="bold"
        )

        ax.set_title(
            f"DeceptRank — Top {top_n} Misinformation Super-Spreaders",
            fontsize=14, fontweight="bold", pad=20
        )
        ax.axis("off")

        # legend for communities
        unique_clusters = set(cluster_map.values())
        for c in unique_clusters:
            ax.scatter([], [], color=cmap(c / 10),
                       label=f"Community {c}", s=80)
        ax.legend(loc="upper left", fontsize=9)

        path = os.path.join(OUTPUT_DIR, "network_map.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Network map saved → {path}")


def plot_umap_clusters(nodes, embedding2d, cluster_map, influence_df):
    """
    Plot UMAP 2D projection of node embeddings
    colored by community, sized by influence score.
    """
    with Timer("Plotting UMAP clusters"):
        ensure_dir(OUTPUT_DIR)

        scores = influence_df.set_index("node_id")["influence_score"]
        sizes  = np.array([scores.get(n, 0.01) * 500 + 30 for n in nodes])
        colors = [cluster_map.get(n, 0) for n in nodes]

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            embedding2d[:, 0],
            embedding2d[:, 1],
            c=colors,
            cmap="tab10",
            s=sizes,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5
        )

        # label top spreaders
        top_nodes = set(influence_df.head(TOP_N)["node_id"].tolist())
        for i, node in enumerate(nodes):
            if node in top_nodes:
                ax.annotate(
                    node,
                    (embedding2d[i, 0], embedding2d[i, 1]),
                    fontsize=7,
                    fontweight="bold",
                    xytext=(4, 4),
                    textcoords="offset points"
                )

        plt.colorbar(scatter, ax=ax, label="Community")
        ax.set_title(
            "DeceptRank — Node Embeddings (UMAP)\nSize = influence score",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("UMAP dimension 1")
        ax.set_ylabel("UMAP dimension 2")

        path = os.path.join(OUTPUT_DIR, "umap_clusters.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"UMAP plot saved → {path}")


def plot_scalability(results):
    """
    Plot time taken at each graph size
    to show scalability of the pipeline.
    """
    with Timer("Plotting scalability chart"):
        ensure_dir(OUTPUT_DIR)

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "DeceptRank — Scalability Evaluation",
            fontsize=14, fontweight="bold"
        )

        stages = [
            ("time_preprocess", "Preprocessing",      "steelblue"),
            ("time_walks",      "Walk Generation",     "coral"),
            ("time_train",      "Embedding Training",  "mediumseagreen"),
        ]

        for ax, (col, title, color) in zip(axes, stages):
            ax.plot(
                df["n_nodes"], df[col],
                "o-", color=color,
                linewidth=2, markersize=8
            )
            ax.fill_between(
                df["n_nodes"], df[col],
                alpha=0.15, color=color
            )
            ax.set_xlabel("Graph size (nodes)", fontsize=10)
            ax.set_ylabel("Time (seconds)",     fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        path = os.path.join(OUTPUT_DIR, "scalability.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Scalability chart saved → {path}")


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