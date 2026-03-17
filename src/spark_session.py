from pyspark.sql import SparkSession
from src.config import (
    SPARK_APP_NAME,
    SPARK_MASTER,
    SPARK_EXECUTOR_MEMORY,
    SPARK_DRIVER_MEMORY
)
from src.utils import logger


def get_spark():
    logger.info("Starting PySpark session...")

    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    # suppress noisy spark logs
    spark.sparkContext.setLogLevel("ERROR")

    logger.info(f"PySpark version: {spark.version}")
    logger.info(f"Master: {SPARK_MASTER}")

    return spark


def stop_spark(spark):
    logger.info("Stopping PySpark session...")
    spark.stop()
    logger.info("PySpark stopped.")