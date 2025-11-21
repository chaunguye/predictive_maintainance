import os
import time
import pandas as pd
import json
from kafka import KafkaProducer

# --- CONFIGURATION ---
# Path to data
DATA_DIR = "/app/NASA_Bearing_Data/1nd_test/1nd_test" 
TOPIC_NAME = "data-sensors"

# Simulate speed (0 = max speed, 30 = 30 second wait between files)
# In the original dataset, the gap between files is 10 minutes, but for demo purposes we use 30 seconds

FILE_INTERVAL = 30  # seconds between files

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
                
                request_timeout_ms=120000,  # wait for the broker to respond
                max_block_ms=120000,  # if buffer is full, max time wait before raising error
                batch_size=32768,  # batch size of a message 
                linger_ms=50,        # time to wait before sending if a batch is not full
                delivery_timeout_ms=130000 # total time to deliver a message
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
        # Set 1 has 8 columns: Bearing 1a, 1b, 2a, 2b, 3a, 3b, 4a, 4b
        try:
            df = pd.read_csv(file_path, sep='\t', header=None)
            
            df.columns = ['b1a', 'b1b', 'b2a', 'b2b', 'b3a', 'b3b', 'b4a', 'b4b']
        except Exception as e:
            print(f"Skipping bad file {filename}: {e}")
            continue

        # 2. Convert filename to a timestamp string
        # Filename format: 2004.02.12.10.32.39 -> 2004-02-12 10:32:39
        timestamp_str = filename.replace('.', '-', 3).replace('.', ':', 3).replace('-', '.', 1) # Quick fix or keep as is

        print(f"Streaming file: {filename} ({len(df)} rows)...")
        
        # Loop through rows and send each as a message
        for index, row in df.iterrows():
            message = {
                "timestamp": timestamp_str,
                "file": filename,
                "b1": row['b1a'],
                "b2": row['b2a'],
                "b3": row['b3a'],
                "b4": row['b4a']
            }
            producer.send(TOPIC_NAME, message)
        
        # Flush ensures all 20k messages are sent before we sleep
        producer.flush()
        
        # Wait for the next "second" of data (Real-time simulation)
        time.sleep(FILE_INTERVAL)

if __name__ == "__main__":
    run_producer()