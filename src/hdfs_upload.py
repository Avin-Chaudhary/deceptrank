import os
import subprocess

from src.config import (
    HDFS_HOST,
    HDFS_INPUT_DIR,
    HDFS_OUTPUT_DIR,
    INTERACTIONS_CSV
)
from src.utils import logger, Timer


def run_hdfs_command(cmd):
    """Run HDFS command through WSL."""
    full_cmd = f"wsl hdfs dfs {cmd}"
    logger.info(f"Running: {full_cmd}")

    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        logger.info(f"Success: {result.stdout.strip()}")
        return True
    else:
        logger.warning(f"HDFS command failed: {result.stderr.strip()}")
        return False


def create_hdfs_dirs():
    """Create input and output directories on HDFS."""
    with Timer("Creating HDFS directories"):
        run_hdfs_command(f"-mkdir -p {HDFS_INPUT_DIR}")
        run_hdfs_command(f"-mkdir -p {HDFS_OUTPUT_DIR}")
    logger.info("HDFS directories ready")


def upload_to_hdfs(local_path, hdfs_path):
    """Upload a local file to HDFS."""
    with Timer(f"Uploading {os.path.basename(local_path)} to HDFS"):
        # remove existing file first to avoid errors
        run_hdfs_command(f"-rm -f {hdfs_path}")
        success = run_hdfs_command(f"-put {local_path} {hdfs_path}")

    if success:
        logger.info(f"Uploaded → {hdfs_path}")
    else:
        logger.warning(f"Upload failed for {local_path}")

    return success


def download_from_hdfs(hdfs_path, local_path):
    """Download results from HDFS to local."""
    with Timer(f"Downloading from HDFS → {local_path}"):
        success = run_hdfs_command(f"-get {hdfs_path} {local_path}")

    if success:
        logger.info(f"Downloaded → {local_path}")
    else:
        logger.warning(f"Download failed for {hdfs_path}")

    return success


def list_hdfs_dir(hdfs_path):
    """List contents of an HDFS directory."""
    run_hdfs_command(f"-ls {hdfs_path}")


def check_hdfs_available():
    """Check if HDFS is running by connecting to its web port."""
    import socket
    logger.info("Checking HDFS availability...")
    try:
        sock = socket.create_connection(("localhost", 9870), timeout=3)
        sock.close()
        logger.info("HDFS is running and reachable")
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        logger.warning(
            "HDFS not reachable — running in local mode only.\n"
            "Start Hadoop with: start-dfs.sh in WSL"
        )
        return False


def run_hdfs_upload():
    """
    Full HDFS upload pipeline.
    Checks if HDFS is available first.
    If not available, skips silently — 
    pipeline will use local files instead.
    """
    logger.info("=== Starting HDFS Upload ===")

    hdfs_available = check_hdfs_available()

    if not hdfs_available:
        logger.info("Skipping HDFS upload — using local files")
        return False

    create_hdfs_dirs()

    # upload interactions CSV
    hdfs_csv_path = HDFS_INPUT_DIR + "interactions.csv"
    upload_to_hdfs(INTERACTIONS_CSV, hdfs_csv_path)

    list_hdfs_dir(HDFS_INPUT_DIR)

    logger.info("=== HDFS Upload Complete ===")
    return True