import time
import random
import networkx as nx
import pandas as pd

from src.config import SCALABILITY_SIZES, OUTPUT_DIR
from src.node2vec_runner import generate_walks, train_embeddings
from src.utils import logger, Timer, ensure_dir, save_csv


def generate_synthetic_graph(n_nodes, edge_prob=0.15, fake_ratio=0.7):
    """
    Generate a random directed graph for
    scalability testing.
    n_nodes    = number of users
    edge_prob  = probability of edge between any two nodes
    fake_ratio = fraction of edges that are fake
    """
    with Timer(f"Generating synthetic graph ({n_nodes} nodes)"):
        G = nx.erdos_renyi_graph(
            n=n_nodes,
            p=edge_prob,
            directed=True,
            seed=42
        )

        # assign random weights based on fake ratio
        for u, v in G.edges():
            is_fake = random.random() < fake_ratio
            G[u][v]["weight"] = 1.0 if is_fake else 0.2

    logger.info(
        f"Synthetic graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def run_single_scalability_test(n_nodes):
    """
    Run full pipeline on a synthetic graph
    of given size and record time for each stage.
    """
    logger.info(f"\n--- Scalability test: {n_nodes} nodes ---")
    result = {"n_nodes": n_nodes}

    # stage 1 — graph generation (simulates preprocessing)
    t0 = time.time()
    G  = generate_synthetic_graph(n_nodes)
    result["time_preprocess"] = round(time.time() - t0, 2)

    # stage 2 — walk generation
    t1    = time.time()
    walks = generate_walks(G, num_walks=5, walk_length=20)
    result["time_walks"] = round(time.time() - t1, 2)

    # stage 3 — embedding training
    t2    = time.time()
    model = train_embeddings(walks, embedding_dim=32, epochs=2)
    result["time_train"] = round(time.time() - t2, 2)

    # total time
    result["time_total"] = round(
        result["time_preprocess"] +
        result["time_walks"] +
        result["time_train"], 2
    )

    logger.info(
        f"n={n_nodes} | "
        f"preprocess={result['time_preprocess']}s | "
        f"walks={result['time_walks']}s | "
        f"train={result['time_train']}s | "
        f"total={result['time_total']}s"
    )
    return result


def run_scalability_tests(sizes=SCALABILITY_SIZES):
    """Run scalability tests across all graph sizes."""
    logger.info("=== Starting Scalability Tests ===")

    results = []
    for n in sizes:
        result = run_single_scalability_test(n)
        results.append(result)

    # save results to csv
    ensure_dir(OUTPUT_DIR)
    df = pd.DataFrame(results)
    save_csv(df, "scalability_results.csv", OUTPUT_DIR)

    # print summary table
    print("\n" + "="*60)
    print("  SCALABILITY RESULTS")
    print("="*60)
    print(f"  {'Nodes':<10} {'Preprocess':>12} {'Walks':>10} "
          f"{'Train':>10} {'Total':>10}")
    print("-"*60)
    for _, row in df.iterrows():
        print(f"  {int(row['n_nodes']):<10} "
              f"{row['time_preprocess']:>10}s "
              f"{row['time_walks']:>9}s "
              f"{row['time_train']:>9}s "
              f"{row['time_total']:>9}s")
    print("="*60 + "\n")

    logger.info("=== Scalability Tests Complete ===")
    return results