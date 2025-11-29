from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, first
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import joblib
import pyspark.sql.functions as F
import pandas as pd


# --- CONFIGURATION ---
KAFKA_TOPIC = "data-sensors"
KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"

JDBC_URL = "jdbc:postgresql://postgresql:5432/predictive_maintenance"
JDBC_PROPS = {
    "user": "bigdatauser",
    "password": "group5",
    "driver": "org.postgresql.Driver"
}

def main():
    # 1. Initialize Spark Session
    # IMPORTANT: We must download the Kafka connector JAR here
    spark = SparkSession.builder \
        .appName("Predictive Maintenance Consumer") \
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

    # Load ML Models
    bearing_model_path = "/app/models/bearing_rul_sklearn.pkl" 

    print(f"[STREAM] Loading bearing model from {bearing_model_path}")
    bearing_bundle = joblib.load(bearing_model_path)
    bearing_model = bearing_bundle["model"]
    bearing_feature_cols = bearing_bundle["feature_cols"]
    print(f"[STREAM] Bearing model loaded. Features: {bearing_feature_cols}")

    def predict_bearing_batch(df, batch_id: int):
        if df.rdd.isEmpty():
            return
        print(f"[STREAM] Processing bearing batch {batch_id} with {df.count()} records")
        print(df.show(5, truncate=False))

        pdf = df.toPandas()
        if pdf.empty:
            return
        

        for bearing_idx in range(1, 5):
            input_batch = []
            prefix = f"b{bearing_idx}_"
            for _, row in pdf.iterrows():
                sample = {}
                for tf in bearing_feature_cols:
                    col_name = f"{prefix}{tf}"
                    sample[tf] = row.get(col_name, 0.0)
                input_batch.append(sample)
            X_bearing = pd.DataFrame(input_batch)[bearing_feature_cols].values
            pdf[f"b{bearing_idx}_rul"] = bearing_model.predict(X_bearing)

        # OUTPUT: Print to Docker Logs (Replaces format("console"))
        print(f"--- Batch {batch_id} Processed ---")
        # Print just the important columns to keep logs clean
        print(pdf[["timestamp", "b1_rul", "b2_rul", "b3_rul", "b4_rul"]].head(10))

        final_df = spark.createDataFrame(pdf)
        final_df = final_df.withColumn(
            "timestamp", 
            F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss")
        )
        # 2. Write to Postgres
        try:
            final_df.select("timestamp", "b1_max", "b1_p2p", "b1_rms", "b1_rul", "b2_max", "b2_p2p", "b2_rms", "b2_rul", "b3_max", "b3_p2p", "b3_rms", "b3_rul", "b4_max", "b4_p2p", "b4_rms", "b4_rul") \
                .write \
                .mode("append") \
                .jdbc(url=JDBC_URL, table="bearing_predictions", properties=JDBC_PROPS)
            
            print("Successfully wrote to Database!")
        except Exception as e:
            print(f"Database Error: {e}")

    print("--- Spark Started. Listening to Kafka... ---")

    # 3. Read Stream from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()

    # 4. Parse JSON
    # Kafka sends data in binary 'value' column. We cast to String -> JSON Struct
    
    # parsed_stream = raw_stream \
    #     .selectExpr("CAST(value AS STRING)") \
    #     .select(from_json(col("value"), schema).alias("data")) \
    #     .select("data.*")
    
    parsed_stream = raw_stream \
        .select(F.from_json(F.col("value").cast("string"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("event_time", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
    

    feature_stream = parsed_stream \
        .withWatermark("event_time", "30 seconds") \
        .groupBy("event_time") \
        .agg(
            first("timestamp").alias("timestamp"),

        # b1
        F.max(F.abs(col("b1"))).alias("b1_max"),
        (F.max(F.abs(col("b1"))) + F.abs(F.min(col("b1")))).alias("b1_p2p"),
        F.sqrt(F.avg(col("b1") * col("b1"))).alias("b1_rms"),

        # b2
        F.max(F.abs(col("b2"))).alias("b2_max"),
        (F.max(F.abs(col("b2"))) + F.abs(F.min(col("b2")))).alias("b2_p2p"),
        F.sqrt(F.avg(col("b2") * col("b2"))).alias("b2_rms"),

        # b3
        F.max(F.abs(col("b3"))).alias("b3_max"),
        (F.max(F.abs(col("b3"))) + F.abs(F.min(col("b3")))).alias("b3_p2p"),
        F.sqrt(F.avg(col("b3") * col("b3"))).alias("b3_rms"),

        # b4
        F.max(F.abs(col("b4"))).alias("b4_max"),
        (F.max(F.abs(col("b4"))) + F.abs(F.min(col("b4")))).alias("b4_p2p"),
        F.sqrt(F.avg(col("b4") * col("b4"))).alias("b4_rms"),
    )



    # 5. Output to Console (for debugging)
    # We use "append" mode to see new rows as they arrive
    query = feature_stream.writeStream \
        .outputMode("append") \
        .foreachBatch(predict_bearing_batch) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()