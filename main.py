import os
from src.utils import logger, Timer, save_csv, print_top_spreaders, ensure_dir
from src.config import OUTPUT_DIR, INTERACTIONS_CSV
from src.spark_session import get_spark, stop_spark
from src.hdfs_upload import run_hdfs_upload
from src.preprocess import run_preprocessing
from src.graph_builder import (
    build_networkx_graph,
    compute_pagerank,
    get_weighted_outdegree,
    print_graph_stats
)
from src.node2vec_runner import run_node2vec
from src.influence_scorer import run_influence_scoring, get_top_spreaders
from src.clustering import run_clustering
from src.visualize import (
    plot_spreader_network,
    plot_umap_clusters,
    plot_top_spreaders_bar,
    plot_scalability
)
from src.scalability import run_scalability_tests


def main():
    print("\n")
    print("=" * 55)
    print("   DeceptRank — Misinformation Super-Spreader Detector")
    print("=" * 55)
    print()

    ensure_dir(OUTPUT_DIR)

    # ── Step 1: HDFS Upload ───────────────────────────────
    logger.info("STEP 1 — HDFS Upload")
    hdfs_available = run_hdfs_upload()

    # decide data path based on hdfs availability
    if hdfs_available:
        from src.config import HDFS_HOST, HDFS_INPUT_DIR
        data_path = HDFS_HOST + HDFS_INPUT_DIR + "interactions.csv"
    else:
        data_path = INTERACTIONS_CSV
        logger.info(f"Using local file: {data_path}")

    # ── Step 2: PySpark Preprocessing ────────────────────
    logger.info("STEP 2 — Preprocessing")
    spark   = get_spark()
    edge_df = run_preprocessing(spark, path=data_path)

    # ── Step 3: Graph Construction ────────────────────────
    logger.info("STEP 3 — Graph Construction")
    G        = build_networkx_graph(edge_df)
    pagerank = compute_pagerank(G)
    outdegree = get_weighted_outdegree(G)
    print_graph_stats(G)

    # ── Step 4: Node2Vec Embeddings ───────────────────────
    logger.info("STEP 4 — Node2Vec Embeddings")
    model = run_node2vec(G)

    # ── Step 5: Influence Scoring ─────────────────────────
    logger.info("STEP 5 — Influence Scoring")
    influence_df = run_influence_scoring(model, pagerank, outdegree)

    # print top spreaders to terminal
    print_top_spreaders(influence_df)

    # save ranked list to CSV
    save_csv(influence_df, "spreaders_ranked.csv", OUTPUT_DIR)

    # ── Step 6: Clustering ────────────────────────────────
    logger.info("STEP 6 — Clustering")
    cluster_map, nodes, embedding2d, bridge_df = run_clustering(
        model, G, influence_df
    )

    # save bridge nodes
    save_csv(bridge_df, "bridge_nodes.csv", OUTPUT_DIR)

    # ── Step 7: Visualizations ────────────────────────────
    logger.info("STEP 7 — Visualizations")
    plot_spreader_network(G, influence_df, cluster_map)
    plot_umap_clusters(nodes, embedding2d, cluster_map, influence_df)
    plot_top_spreaders_bar(influence_df)

    # ── Step 8: Scalability Tests ─────────────────────────
    logger.info("STEP 8 — Scalability Tests")
    scalability_results = run_scalability_tests()
    plot_scalability(scalability_results)

    # ── Done ──────────────────────────────────────────────
    stop_spark(spark)

    print("\n" + "="*55)
    print("   DeceptRank — Pipeline Complete!")
    print(f"   Results saved to: {OUTPUT_DIR}")
    print("="*55 + "\n")


if __name__ == "__main__":
    with Timer("Full DeceptRank Pipeline"):
        main()