import os
import time
import logging

# ─── Logger setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DeceptRank")


# ─── Timer ────────────────────────────────────────────────
class Timer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        logger.info(f"STARTED  → {self.label}")
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self.start, 2)
        logger.info(f"FINISHED → {self.label} in {self.elapsed}s")


# ─── Ensure output folder exists ──────────────────────────
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ─── Print ranked results nicely ──────────────────────────
def print_top_spreaders(df, top_n=10):
    print("\n" + "="*50)
    print(f"  TOP {top_n} MISINFORMATION SUPER-SPREADERS")
    print("="*50)
    top = df.head(top_n)
    for i, row in top.iterrows():
        print(f"  #{i+1:02d}  {row['node_id']:<15} score: {row['influence_score']:.4f}")
    print("="*50 + "\n")


# ─── Save dataframe to output folder ──────────────────────
def save_csv(df, filename, output_dir):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved → {path}")
    return path