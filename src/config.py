import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")
SYNTHETIC_DIR   = os.path.join(BASE_DIR, "data", "synthetic")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")

INTERACTIONS_CSV = os.path.join(DATA_DIR, "interactions.csv")

# ─── HDFS ─────────────────────────────────────────────────
HDFS_HOST       = "hdfs://localhost:9000"
HDFS_INPUT_DIR  = "/deceptrank/input/"
HDFS_OUTPUT_DIR = "/deceptrank/output/"

# ─── PySpark ──────────────────────────────────────────────
SPARK_APP_NAME  = "DeceptRank"
SPARK_MASTER    = "local[*]"
SPARK_EXECUTOR_MEMORY = "2g"
SPARK_DRIVER_MEMORY   = "2g"

# ─── Veracity Weights ─────────────────────────────────────
VERACITY_WEIGHTS = {
    "fake":       1.0,
    "unverified": 0.5,
    "real":       0.2
}

# ─── Graph ────────────────────────────────────────────────
MIN_EDGE_WEIGHT = 0.1

# ─── Node2Vec ─────────────────────────────────────────────
WALK_LENGTH     = 30
NUM_WALKS       = 10
P               = 1.0
Q               = 0.5
EMBEDDING_DIM   = 64
WINDOW_SIZE     = 5
EPOCHS          = 3
WORKERS         = 2

# ─── Influence Score Weights ──────────────────────────────
ALPHA = 0.3   # embedding norm weight
BETA  = 0.3   # pagerank weight
GAMMA = 0.4   # weighted outdegree weight

# ─── Clustering ───────────────────────────────────────────
N_CLUSTERS      = 3

# ─── Scalability Test Sizes ───────────────────────────────
SCALABILITY_SIZES = [100, 500, 1000]

# ─── Top N spreaders to show ──────────────────────────────
TOP_N = 10