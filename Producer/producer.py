import os
import time
import pandas as pd
import json
from kafka import KafkaProducer

# --- CONFIGURATION ---
# Path to data
DATA_DIR = "/app/NASA_Bearing_Data/2nd_test/2nd_test" 
TOPIC_NAME = "data-sensor-2"

# Simulate speed (0 = max speed, 1 = 1 second wait between files)

FILE_INTERVAL = 10

def get_sorted_files(directory):
    """Returns a list of filenames sorted by time (filename is the timestamp)"""
    files = [f for f in os.listdir(directory)]
    files.sort() # The format YYYY.MM.DD... sorts correctly as strings
    return files

def run_producer():
    print("--- Setting up Kafka Producer ---")
    # Retrying connection in case Kafka is still starting up
    producer = None
    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers='kafka:9092', # 'kafka' is the service name in docker-compose
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                
                request_timeout_ms=120000, 
                max_block_ms=120000, 
                batch_size=32768,
                linger_ms=50,
                delivery_timeout_ms=130000
            )
            print("Connected to Kafka!")
        except Exception as e:
            print("Waiting for Kafka...", e)
            time.sleep(5)

    #Give Kafka 10 seconds to finish waking up / creating topics
    print("Warming up for 10 seconds...")
    time.sleep(10)

    # Get all files
    files = get_sorted_files(DATA_DIR)
    print(f"Found {len(files)} files. Starting stream...")

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        
        # 1. Read the file (Tab separated, no header)
        # Set 2 has 4 columns: Bearing 1, 2, 3, 4
        try:
            df = pd.read_csv(file_path, sep='\t', header=None)
            # Rename columns for clarity (Set 2 has 4 bearings)
            df.columns = ['b1', 'b2', 'b3', 'b4']
        except Exception as e:
            print(f"Skipping bad file {filename}: {e}")
            continue

        # 2. Convert filename to a timestamp string
        # Filename format: 2004.02.12.10.32.39 -> 2004-02-12 10:32:39
        timestamp_str = filename.replace('.', '-', 3).replace('.', ':', 3).replace('-', '.', 1) # Quick fix or keep as is

        print(f"Streaming file: {filename} ({len(df)} rows)...")

        # 3. Send Data
        # Instead of Sending row-by-row (True Simulation - High Load)
        # We batch this. For now just send the whole file as one "batch" message
        
        # Let's loop through rows but let Kafka batch them (High throughput)
        for index, row in df.iterrows():
            message = {
                "timestamp": timestamp_str,
                "file": filename,
                "b1": row['b1'],
                "b2": row['b2'],
                "b3": row['b3'],
                "b4": row['b4']
            }
            producer.send(TOPIC_NAME, message)
        
        # Flush ensures all 20k messages are sent before we sleep
        producer.flush()
        
        # Wait for the next "second" of data (Real-time simulation)
        time.sleep(FILE_INTERVAL)

if __name__ == "__main__":
    run_producer()