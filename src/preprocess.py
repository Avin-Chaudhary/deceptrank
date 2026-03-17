import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

from src.config import VERACITY_WEIGHTS, MIN_EDGE_WEIGHT, INTERACTIONS_CSV
from src.utils import logger, Timer


# ─── Schema for CSV ───────────────────────────────────────
SCHEMA = StructType([
    StructField("tweet_id",         StringType(), True),
    StructField("src_user_id",      StringType(), True),
    StructField("dst_user_id",      StringType(), True),
    StructField("interaction_type", StringType(), True),
    StructField("veracity",         StringType(), True),
    StructField("timestamp",        LongType(),   True),
])


def load_interactions(spark, path=None):
    """Load interactions CSV into a Spark DataFrame."""
    path = path or INTERACTIONS_CSV

    with Timer("Loading interactions CSV"):
        df = spark.read.csv(
            path,
            header=True,
            schema=SCHEMA
        )

    total = df.count()
    logger.info(f"Loaded {total} interactions from {path}")
    return df


def clean_interactions(df):
    """Remove nulls, duplicates, and unknown veracity."""
    with Timer("Cleaning interactions"):

        # drop rows where critical fields are null
        df = df.dropna(subset=["src_user_id", "dst_user_id", "veracity"])

        # remove self loops — user interacting with themselves
        df = df.filter(F.col("src_user_id") != F.col("dst_user_id"))

        # remove duplicates
        df = df.dropDuplicates(["tweet_id"])

        # keep only known veracity values
        df = df.filter(F.col("veracity").isin(["fake", "real", "unverified"]))

    logger.info(f"After cleaning: {df.count()} interactions remain")
    return df


def assign_veracity_weights(df):
    """Add a numeric weight column based on veracity label."""
    with Timer("Assigning veracity weights"):
        df = df.withColumn(
            "veracity_weight",
            F.when(F.col("veracity") == "fake",       VERACITY_WEIGHTS["fake"])
             .when(F.col("veracity") == "unverified",  VERACITY_WEIGHTS["unverified"])
             .otherwise(                               VERACITY_WEIGHTS["real"])
        )
    return df


def aggregate_edges(df):
    """
    Aggregate multiple interactions between same
    src and dst into one weighted edge.
    Final weight = sum of veracity weights + log of frequency.
    """
    with Timer("Aggregating edges"):
        edge_df = df.groupBy("src_user_id", "dst_user_id").agg(
            F.sum("veracity_weight").alias("veracity_score"),
            F.count("*").alias("interaction_count"),
            F.first("interaction_type").alias("interaction_type")
        ).withColumn(
            "edge_weight",
            F.col("veracity_score") + F.log1p(F.col("interaction_count"))
        ).filter(
            F.col("edge_weight") >= MIN_EDGE_WEIGHT
        )

    logger.info(f"Total unique edges: {edge_df.count()}")
    return edge_df


def run_preprocessing(spark, path=None):
    """Run full preprocessing pipeline, return clean edge DataFrame."""
    logger.info("=== Starting Preprocessing ===")

    df       = load_interactions(spark, path)
    df       = clean_interactions(df)
    df       = assign_veracity_weights(df)
    edge_df  = aggregate_edges(df)

    logger.info("=== Preprocessing Complete ===")
    return edge_df