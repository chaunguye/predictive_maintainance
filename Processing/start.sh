#!/bin/bash

echo "Waiting 30 seconds for Kafka to be ready..."
# This sleep is crucial. Kafka takes time to elect leaders.
# Without this, Spark will crash with "Connection Refused" immediately.
sleep 30

echo "Submitting Spark Streaming Job..."

# Run the job.
# Since your python script has .awaitTermination(), this command will 
# stay running forever, keeping the container alive.
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0 /app/preprocessor.py