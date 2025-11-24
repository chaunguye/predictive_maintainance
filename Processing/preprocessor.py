from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# --- CONFIGURATION ---
KAFKA_TOPIC = "data-sensors"
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"

def main():
    # 1. Initialize Spark Session
    # IMPORTANT: We must download the Kafka connector JAR here
    spark = SparkSession.builder \
        .appName("Predictive Maintenance Consumer") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .getOrCreate()

    # Set log level to WARN to avoid console spam
    spark.sparkContext.setLogLevel("WARN")

    # 2. Define the Schema
    schema = StructType([
        StructField("timestamp", StringType(), True),
        StructField("file", StringType(), True),
        # Bearing data fields
        StructField("b1", DoubleType(), True),
        StructField("b2", DoubleType(), True),
        StructField("b3", DoubleType(), True),
        StructField("b4", DoubleType(), True),
    ])

    print("--- Spark Started. Listening to Kafka... ---")

    # 3. Read Stream from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # 4. Parse JSON
    # Kafka sends data in binary 'value' column. We cast to String -> JSON Struct
    parsed_stream = raw_stream \
        .selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # 5. Output to Console (for debugging)
    # We use "append" mode to see new rows as they arrive
    query = parsed_stream.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()