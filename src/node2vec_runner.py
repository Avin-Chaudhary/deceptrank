import random
from gensim.models import Word2Vec

from src.config import (
    WALK_LENGTH,
    NUM_WALKS,
    P, Q,
    EMBEDDING_DIM,
    WINDOW_SIZE,
    EPOCHS,
    WORKERS
)
from src.utils import logger, Timer


def biased_walk(G, start_node, walk_length, p, q):
    """
    Single biased random walk from start_node.
    Prefers high weight edges.
    p = return parameter (low p = stay local)
    q = in-out parameter (low q = explore far)
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))

        if not neighbors:
            break

        if len(walk) == 1:
            # first step — pick neighbor by edge weight
            weights = [G[cur][nbr]["weight"] for nbr in neighbors]
            total   = sum(weights)
            probs   = [w / total for w in weights]
            walk.append(random.choices(neighbors, weights=probs)[0])

        else:
            prev = walk[-2]
            weights = []

            for nbr in neighbors:
                edge_w = G[cur][nbr].get("weight", 1.0)

                if nbr == prev:
                    # return to previous node — penalize by p
                    weights.append(edge_w / p)

                elif G.has_edge(prev, nbr):
                    # neighbor of both cur and prev — BFS like
                    weights.append(edge_w)

                else:
                    # far away node — penalize by q
                    weights.append(edge_w / q)

            total = sum(weights)
            probs = [w / total for w in weights]
            walk.append(random.choices(neighbors, weights=probs)[0])

    return walk


def generate_walks(G, num_walks=NUM_WALKS, walk_length=WALK_LENGTH, p=P, q=Q):
    """
    Generate all random walks for all nodes.
    Each node gets num_walks walks starting from it.
    """
    with Timer("Generating random walks"):
        nodes = list(G.nodes())
        all_walks = []

        for walk_num in range(num_walks):
            # shuffle nodes each round for better coverage
            random.shuffle(nodes)
            for node in nodes:
                walk = biased_walk(G, node, walk_length, p, q)
                all_walks.append([str(n) for n in walk])

        logger.info(f"Generated {len(all_walks)} walks "
                    f"({num_walks} walks x {len(nodes)} nodes)")
    return all_walks


def train_embeddings(walks,
                     embedding_dim=EMBEDDING_DIM,
                     window=WINDOW_SIZE,
                     epochs=EPOCHS,
                     workers=WORKERS):
    """
    Train Word2Vec skip-gram on the random walks.
    Each walk is treated like a sentence.
    Each node is treated like a word.
    """
    with Timer("Training Node2Vec embeddings"):
        model = Word2Vec(
            sentences=walks,
            vector_size=embedding_dim,
            window=window,
            min_count=1,
            sg=1,           # skip-gram
            workers=workers,
            epochs=epochs,
            seed=42
        )

    logger.info(f"Embedding dim : {embedding_dim}")
    logger.info(f"Vocab size    : {len(model.wv)} nodes embedded")
    return model


def run_node2vec(G):
    """Full Node2Vec pipeline — walks + embeddings."""
    logger.info("=== Starting Node2Vec ===")

    walks = generate_walks(G)
    model = train_embeddings(walks)

    logger.info("=== Node2Vec Complete ===")
    return model


def get_most_similar(model, node, topn=5):
    """Find nodes with most similar embedding to given node."""
    if node not in model.wv:
        logger.warning(f"Node {node} not found in embeddings")
        return []
    return model.wv.most_similar(node, topn=topn)