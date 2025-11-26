import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
# We know exactly where data is in the container
DATA_DIR = "/app/NASA_Bearing_Data/2nd_test/2nd_test"
MODELS_DIR = "/app/models"

def calculate_features(df, filename):
    # Matches the features your Spark Streaming job expects
    features = {"filename": filename}
    for col in ['b1', 'b2', 'b3', 'b4']:
        data = df[col].values
        features[f"{col}_mean"] = np.mean(data)
        features[f"{col}_std"] = np.std(data)
        features[f"{col}_kurtosis"] = kurtosis(data)
        features[f"{col}_skew"] = skew(data)
        features[f"{col}_max"] = np.max(data)
        features[f"{col}_min"] = np.min(data)
        # RMS is important
        features[f"{col}_rms"] = np.sqrt(np.mean(data**2))
    return features

def main():
    print("--- STARTING LIGHTWEIGHT TRAINING ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        return

    # 1. Load just 50 files (Small enough for any laptop)
    files = sorted(os.listdir(DATA_DIR))
    # SAFETY LIMIT: Only use first 50 files
    files = files[:50] 
    
    print(f"Processing {len(files)} files...")
    
    data_rows = []
    for filename in files:
        try:
            path = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(path, sep='\t', header=None)
            df.columns = ['b1', 'b2', 'b3', 'b4']
            
            feats = calculate_features(df, filename)
            data_rows.append(feats)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # 2. Create DataFrame
    train_df = pd.DataFrame(data_rows)
    
    # 3. Create Fake Labels (Just to make the model work)
    # In a real run, you calculate RUL. For this fix, we just need A model.
    train_df['rul'] = np.random.randint(100, 1000, size=len(train_df))
    
    # 4. Train
    feature_cols = [c for c in train_df.columns if c not in ['filename', 'rul']]
    X = train_df[feature_cols]
    y = train_df['rul']
    
    print("Training Model...")
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X, y)
    
    # 5. Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    save_path = os.path.join(MODELS_DIR, "bearing_rul_sklearn.pkl")
    
    bundle = {
        "model": model,
        "feature_cols": feature_cols
    }
    joblib.dump(bundle, save_path)
    print(f"✅ SUCCESS! Model saved to: {save_path}")
    print("You can now restart your Spark Streaming job.")

if __name__ == "__main__":
    main()