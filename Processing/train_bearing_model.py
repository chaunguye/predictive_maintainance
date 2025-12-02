import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


# ---------------------------------------------------------
# Fixed Docker-style paths (no host fallback)
# ---------------------------------------------------------

DATA_ROOT = Path("/app/NASA_Bearing_Data")

DATASET_PATHS = [
    DATA_ROOT / "1st_test" / "1st_test",
    DATA_ROOT / "2nd_test" / "2nd_test",
    DATA_ROOT / "3rd_test" / "4th_test",
]

MODELS_DIR = Path("/app/models")


# -----------------------------
# Feature engineering helpers
# -----------------------------

def _shannon_entropy(values: np.ndarray, bins: int = 500) -> float:
    """Simple Shannon entropy implementation."""
    if values.ndim != 1:
        values = values.ravel()
    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts.astype(float) / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def calculate_rms(df: pd.DataFrame) -> np.ndarray:
    """Root mean square per column."""
    return np.sqrt((df.values ** 2).sum(axis=0) / len(df))


def calculate_p2p(df: pd.DataFrame) -> np.ndarray:
    """Peak-to-peak (max - min) per column."""
    return (df.max().abs() + df.min().abs()).values


def calculate_entropy(df: pd.DataFrame) -> np.ndarray:
    """Entropy per column using simple histogram-based Shannon entropy."""
    ent = []
    for col in df.columns:
        ent.append(_shannon_entropy(df[col].values, bins=500))
    return np.array(ent)


def calculate_clearance(df: pd.DataFrame) -> np.ndarray:
    """Clearance factor per column (as in the notebook)."""
    result = []
    for col in df.columns:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col])) ** 2
        result.append(r)
    return np.array(result)


def time_features(dataset_path: Path) -> pd.DataFrame:
    """
    Compute rich time-domain features per file, using 4 channels b1..b4.

    For each file:
      - If it has 8 columns (1st_test): use columns 0,2,4,6 (b1a,b2a,b3a,b4a)
      - If it has 4 columns (2nd/3rd_test): use all four as b1..b4
    """
    dataset_path = Path(dataset_path)
    base_cols = ["b1", "b2", "b3", "b4"]
    rows = []

    for filename in sorted(os.listdir(dataset_path)):
        file_path = dataset_path / filename
        if not file_path.is_file():
            continue

        raw = pd.read_csv(file_path, sep="\t", header=None)

        # Match producer behaviour:
        # - 1st_test: 8 cols -> use columns 0, 2, 4, 6
        # - 2nd/3rd_test: 4 cols -> use all
        if raw.shape[1] == 8:
            raw4 = raw.iloc[:, [0, 2, 4, 6]].copy()
        elif raw.shape[1] == 4:
            raw4 = raw.iloc[:, :4].copy()
        else:
            print(f"[BEARING] Skipping {file_path}: unexpected number of columns ({raw.shape[1]})")
            continue

        raw4.columns = base_cols

        # ---- time-domain features on b1..b4 ----
        mean_abs = np.array(raw4.abs().mean())
        std = np.array(raw4.std())
        skew = np.array(raw4.skew())
        kurtosis = np.array(raw4.kurtosis())
        ent = calculate_entropy(raw4)
        rms = calculate_rms(raw4)
        max_abs = np.array(raw4.abs().max())
        p2p = calculate_p2p(raw4)
        crest = max_abs / rms
        clear = calculate_clearance(raw4)
        shape = rms / mean_abs
        impulse = max_abs / mean_abs

        feats = {}
        for i, col in enumerate(base_cols):
            feats[f"{col}_mean"] = mean_abs[i]
            feats[f"{col}_std"] = std[i]
            feats[f"{col}_skew"] = skew[i]
            feats[f"{col}_kurtosis"] = kurtosis[i]
            feats[f"{col}_entropy"] = ent[i]
            feats[f"{col}_rms"] = rms[i]
            feats[f"{col}_max"] = max_abs[i]
            feats[f"{col}_p2p"] = p2p[i]
            feats[f"{col}_crest"] = crest[i]
            feats[f"{col}_clearance"] = clear[i]
            feats[f"{col}_shape"] = shape[i]
            feats[f"{col}_impulse"] = impulse[i]

        feats["filename"] = filename
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame(rows).set_index("filename")

    # enforce time index from filename
    try:
        data.index = pd.to_datetime(data.index, format="%Y.%m.%d.%H.%M.%S")
        data = data.sort_index()
    except Exception:
        data = data.sort_index()

    return data


def build_rul_all_bearings(
    features: pd.DataFrame,
    selected_features=("max", "p2p", "rms", "std", "kurtosis"),
) -> pd.DataFrame:
    """
    Create a supervised dataset with one row per (time, bearing).

    For each timestamp:
      - compute RUL as cycles until end of test
      - for each bearing b1..b4, create one row with:
          [max, p2p, rms] for that bearing, and label 'rul'.

    We assume the rig's RUL applies to all four bearings at that time.
    """
    if features.empty:
        return pd.DataFrame(columns=[*selected_features, "rul"])

    df = features.copy().sort_index()
    df["cycle"] = np.arange(1, len(df) + 1)
    df["rul"] = len(df) - df["cycle"] + 1

    rows = []
    for bearing_idx in range(1, 5):
        prefix = f"b{bearing_idx}_"
        for _, row in df.iterrows():
            sample = {}
            for tf in selected_features:
                col_name = f"{prefix}{tf}"
                sample[tf] = row.get(col_name, 0.0)
            sample["rul"] = row["rul"]
            rows.append(sample)

    return pd.DataFrame(rows)


def main():
    print(f"[BEARING] Using DATA_ROOT = {DATA_ROOT}")

    # Ensure all dataset paths exist
    for path in DATASET_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"[BEARING] Dataset path does not exist: {path}")

    # 1) Compute features for each test set
    set_feats = []
    for i, path in enumerate(DATASET_PATHS, start=1):
        print(f"[BEARING] Computing time features from {path}")
        set_feats.append(time_features(path))

    # 2) Build supervised datasets (1 row per bearing per time)
    sel_feats = ("max", "p2p", "rms", "std", "kurtosis")
    supervised_dfs = [
        build_rul_all_bearings(feats, selected_features=sel_feats) for feats in set_feats
    ]
    supervised_dfs = [df for df in supervised_dfs if not df.empty]

    if not supervised_dfs:
        raise RuntimeError("[BEARING] No supervised data built from any dataset.")

    # 3) Align feature columns and concatenate
    feature_cols_sets = [set(c for c in df.columns if c != "rul") for df in supervised_dfs]
    all_feature_cols = sorted(set.union(*feature_cols_sets))

    def align(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in all_feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[all_feature_cols + ["rul"]]

    aligned = [align(df) for df in supervised_dfs]
    all_data = pd.concat(aligned, ignore_index=True)
    print(f"[BEARING] Final supervised dataset shape: {all_data.shape}")
    print(f"[BEARING] Feature columns: {all_feature_cols}")

    # 4) Train scikit-learn model
    X = all_data[all_feature_cols].values
    y = all_data["rul"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    print("[BEARING] Training GradientBoostingRegressor (scikit-learn)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    print(f"[BEARING] Test RMSE (scikit-learn): {rmse:.3f} cycles")

    # 5) Save model to /app/models/bearing_rul_sklearn.pkl
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "bearing_rul_sklearn.pkl"
    bundle = {
        "model": model,
        "feature_cols": all_feature_cols,  # should be ["max", "p2p", "rms"]
    }
    joblib.dump(bundle, model_path)
    print(f"[BEARING] Saved scikit-learn model to {model_path}")


if __name__ == "__main__":
    main()