import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import N_CLUSTERS
from src.utils import logger, Timer


def get_embeddings_matrix(model):
    """
    Extract all node embeddings into a
    numpy matrix for clustering.
    """
    nodes   = list(model.wv.index_to_key)
    vectors = np.array([model.wv[n] for n in nodes])
    return nodes, vectors


def cluster_nodes(model, n_clusters=N_CLUSTERS):
    """
    Run KMeans on node embeddings to find
    communities of similar users.
    """
    with Timer("Clustering nodes"):
        nodes, vectors = get_embeddings_matrix(model)

        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = km.fit_predict(vectors)

        # build a dict: node → cluster label
        cluster_map = dict(zip(nodes, labels.tolist()))

    logger.info(f"Clustered {len(nodes)} nodes into {n_clusters} communities")

    # print community sizes
    for i in range(n_clusters):
        size = sum(1 for v in cluster_map.values() if v == i)
        logger.info(f"  Community {i} : {size} users")

    return cluster_map


def get_umap_projection(model):
    """
    Reduce embeddings to 2D using UMAP
    for visualization purposes.
    Falls back to PCA if UMAP not available.
    """
    nodes, vectors = get_embeddings_matrix(model)

    try:
        import umap
        with Timer("UMAP 2D projection"):
            reducer     = umap.UMAP(n_components=2, random_state=42)
            embedding2d = reducer.fit_transform(vectors)
        logger.info("UMAP projection complete")

    except Exception as e:
        logger.warning(f"UMAP failed ({e}), falling back to PCA")
        from sklearn.decomposition import PCA
        with Timer("PCA 2D projection"):
            pca         = PCA(n_components=2, random_state=42)
            embedding2d = pca.fit_transform(vectors)
        logger.info("PCA projection complete")

    return nodes, embedding2d


def find_bridge_nodes(G, cluster_map, influence_df):
    """
    Bridge nodes = high influence users who
    connect two or more different communities.
    These are the most dangerous spreaders.
    """
    with Timer("Finding bridge nodes"):
        bridge_scores = []

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors or node not in cluster_map:
                continue

            node_cluster = cluster_map[node]

            # count how many different communities this node connects to
            neighbor_clusters = set(
                cluster_map[n] for n in neighbors
                if n in cluster_map
            )

            # bridge score = number of communities connected
            n_communities = len(neighbor_clusters)

            bridge_scores.append({
                "node_id":      node,
                "own_cluster":  node_cluster,
                "n_communities_connected": n_communities
            })

        bridge_df = pd.DataFrame(bridge_scores)

        # merge with influence scores
        bridge_df = bridge_df.merge(
            influence_df[["node_id", "influence_score"]],
            on="node_id",
            how="left"
        )

        # final bridge score = influence x communities connected
        bridge_df["bridge_score"] = (
            bridge_df["influence_score"] *
            bridge_df["n_communities_connected"]
        )

        bridge_df = bridge_df.sort_values("bridge_score", ascending=False)
        bridge_df = bridge_df.reset_index(drop=True)
        bridge_df.index += 1

    logger.info(f"Bridge node analysis complete for {len(bridge_df)} nodes")
    return bridge_df


def run_clustering(model, G, influence_df):
    """Full clustering pipeline."""
    logger.info("=== Starting Clustering ===")

    cluster_map       = cluster_nodes(model)
    nodes, embedding2d = get_umap_projection(model)
    bridge_df         = find_bridge_nodes(G, cluster_map, influence_df)

    logger.info("=== Clustering Complete ===")
    return cluster_map, nodes, embedding2d, bridge_df