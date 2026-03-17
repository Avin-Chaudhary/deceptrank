import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import ALPHA, BETA, GAMMA, TOP_N
from src.utils import logger, Timer


def compute_embedding_norms(model):
    """
    L2 norm of each node's embedding vector.
    Higher norm = more structurally active node.
    """
    norms = {}
    for node in model.wv.index_to_key:
        norms[node] = float(np.linalg.norm(model.wv[node]))
    return norms


def build_score_dataframe(model, pagerank, outdegree):
    """
    Combine embedding norm, pagerank and
    weighted outdegree into one dataframe.
    """
    with Timer("Building score dataframe"):

        emb_norms = compute_embedding_norms(model)
        nodes     = list(model.wv.index_to_key)

        rows = []
        for node in nodes:
            rows.append({
                "node_id":          node,
                "emb_norm":         emb_norms.get(node, 0.0),
                "pagerank":         pagerank.get(node, 0.0),
                "weighted_outdegree": outdegree.get(node, 0.0),
            })

        df = pd.DataFrame(rows)

    return df


def normalize_and_score(df):
    """
    Normalize all three metrics to 0-1 range
    then combine into one influence score.
    """
    with Timer("Computing influence scores"):

        scaler = MinMaxScaler()

        # normalize the three columns
        df[["emb_norm_n", "pagerank_n", "outdegree_n"]] = scaler.fit_transform(
            df[["emb_norm", "pagerank", "weighted_outdegree"]]
        )

        # weighted composite score
        # gamma is highest because outdegree
        # best captures misinformation spread volume
        df["influence_score"] = (
            ALPHA * df["emb_norm_n"] +
            BETA  * df["pagerank_n"] +
            GAMMA * df["outdegree_n"]
        )

        # sort highest score first
        df = df.sort_values("influence_score", ascending=False)
        df = df.reset_index(drop=True)
        df.index += 1  # rank starts from 1

    logger.info(f"Scored {len(df)} nodes")
    return df


def get_top_spreaders(df, top_n=TOP_N):
    """Return top N spreaders."""
    return df.head(top_n)


def run_influence_scoring(model, pagerank, outdegree):
    """Full influence scoring pipeline."""
    logger.info("=== Starting Influence Scoring ===")

    df = build_score_dataframe(model, pagerank, outdegree)
    df = normalize_and_score(df)

    logger.info("=== Influence Scoring Complete ===")
    return df