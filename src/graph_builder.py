import networkx as nx
from pyspark.sql import functions as F

from src.utils import logger, Timer


def build_networkx_graph(edge_df):
    """
    Convert Spark edge DataFrame to a
    NetworkX directed weighted graph.
    """
    with Timer("Building NetworkX graph"):

        # convert spark df to pandas for networkx
        edge_pd = edge_df.select(
            "src_user_id",
            "dst_user_id",
            "edge_weight"
        ).toPandas()

        G = nx.DiGraph()

        for _, row in edge_pd.iterrows():
            G.add_edge(
                row["src_user_id"],
                row["dst_user_id"],
                weight=float(row["edge_weight"])
            )

    logger.info(f"Graph nodes : {G.number_of_nodes()}")
    logger.info(f"Graph edges : {G.number_of_edges()}")
    return G


def compute_pagerank(G, reset_prob=0.15, max_iter=100):
    """Compute PageRank score for every node."""
    with Timer("Computing PageRank"):
        pr = nx.pagerank(
            G,
            alpha=reset_prob,
            max_iter=max_iter,
            weight="weight"
        )
    logger.info(f"PageRank computed for {len(pr)} nodes")
    return pr


def get_weighted_outdegree(G):
    """
    Compute weighted out-degree for each node.
    This captures how much fake content each user pushed out.
    """
    outdegree = {
        node: sum(data["weight"] for _, _, data in G.out_edges(node, data=True))
        for node in G.nodes()
    }
    return outdegree


def sample_graph(G, n_nodes):
    """
    Sample top n_nodes by out-degree for scalability testing.
    """
    top_nodes = sorted(
        G.nodes(),
        key=lambda n: G.out_degree(n),
        reverse=True
    )[:n_nodes]

    subG = G.subgraph(top_nodes).copy()
    logger.info(f"Sampled graph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
    return subG


def print_graph_stats(G):
    """Print basic graph statistics."""
    print("\n" + "="*50)
    print("  GRAPH STATISTICS")
    print("="*50)
    print(f"  Nodes         : {G.number_of_nodes()}")
    print(f"  Edges         : {G.number_of_edges()}")
    print(f"  Is directed   : {G.is_directed()}")

    if G.number_of_nodes() > 0:
        top = sorted(
            G.out_degree(weight="weight"),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print(f"\n  Top 5 nodes by weighted out-degree:")
        for node, deg in top:
            print(f"    {node:<15} outdegree: {deg:.3f}")
    print("="*50 + "\n")