# import os
# from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path


# -----------------------------
# Feature engineering helpers
# -----------------------------

DATA_ROOT = "/app/NASA_Bearing_Data"  
MODELS_DIR = "/app/models"  
dataset_path = ["/app/NASA_Bearing_Data/1st_test/1st_test",
                    "/app/NASA_Bearing_Data/2nd_test/2nd_test",
                    "/app/NASA_Bearing_Data/3rd_test/4th_test"]



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


def time_features(dataset_path) -> pd.DataFrame:
    """
    Compute rich time-domain features per file, using 4 channels: b1, b2, b3, b4.

    For each file:
      - If it has 8 columns (1st_test): use columns 0,2,4,6 (b1a,b2a,b3a,b4a)
      - If it has 4 columns (2nd/3rd_test): use all four as b1..b4
    """
    time_feats = [
        "mean",
        "std",
        "skew",
        "kurtosis",
        "entropy",
        "rms",
        "max",
        "p2p",
        "crest",
        "clearance",
        "shape",
        "impulse",
    ]

    base_cols = ["b1", "b2", "b3", "b4"]
    rows = []

    for filename in sorted(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path):
            continue

        raw = pd.read_csv(file_path, sep="\t", header=None)

        # Match the producer behaviour:
        #  - 1st_test: 8 cols -> use a-channels: 0,2,4,6
        #  - 2nd/3rd_test: 4 cols -> use all
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
        # fallback: keep as-is if parsing fails
        data = data.sort_index()

    return data


def build_rul_supervised(
    features: pd.DataFrame,
    bearing_prefixes,
    selected_features=("max", "p2p", "rms"),
) -> pd.DataFrame:
    """
    Create a supervised (features, RUL) dataset from the aggregated time features.

    Parameters
    ----------
    features : DataFrame
        Output of time_features() with columns like 'b3_max', 'b3_p2p', ...
    bearing_prefixes : list[str]
        Channel prefixes, e.g. ["b3_"] or ["b4_"].
    selected_features : tuple of str
        Subset of time features to keep per bearing
        (mirrors the notebook: max, p2p, rms).
    """
    if features.empty:
        return pd.DataFrame(columns=["rul"])

    df = features.copy().sort_index()
    df["cycle"] = np.arange(1, len(df) + 1)
    # simple RUL in "cycles until end"
    df["rul"] = len(df) - df["cycle"] + 1

    cols = []
    for prefix in bearing_prefixes:
        for tf in selected_features:
            col_name = f"{prefix}{tf}"
            if col_name in df.columns:
                cols.append(col_name)

    supervised = df[cols + ["rul"]].copy()
    return supervised


def main():
    print(f"[BEARING] Computing time features from {dataset_path[0]}")
    set1_feats = time_features(dataset_path[0])

    print(f"[BEARING] Computing time features from {dataset_path[1]}")
    set2_feats = time_features(dataset_path[1])

    print(f"[BEARING] Computing time features from {dataset_path[2]}")
    set3_feats = time_features(dataset_path[2])

    # Failing bearings per test:
    #  - 1st test: bearing 3 -> "b3_"
    #  - 2nd test: bearing 4 -> "b4_"
    #  - 3rd test: bearing 3 -> "b3_"
    set1_supervised = build_rul_supervised(
        set1_feats,
        bearing_prefixes=["b3_"],
        selected_features=("max", "p2p", "rms"),
    )
    set2_supervised = build_rul_supervised(
        set2_feats,
        bearing_prefixes=["b4_"],
        selected_features=("max", "p2p", "rms"),
    )
    set3_supervised = build_rul_supervised(
        set3_feats,
        bearing_prefixes=["b3_"],
        selected_features=("max", "p2p", "rms"),
    ) if not set3_feats.empty else pd.DataFrame(columns=["rul"])

    # Align feature columns across sets (fill missing with zeros)
    supervised_dfs = [df for df in [set1_supervised, set2_supervised, set3_supervised] if not df.empty]
    if not supervised_dfs:
        raise RuntimeError("[BEARING] No supervised data could be built from the bearing sets.")

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

    # -----------------------------
    # scikit-learn model training
    # -----------------------------
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

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Older scikit-learn versions don't support squared=False,
    # so compute RMSE manually.
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5


    # Save model under repo_root/models/bearing_rul_sklearn.pkl
    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "bearing_rul_sklearn.pkl"

    bundle = {
        "model": model,
        "feature_cols": all_feature_cols,
    }
    joblib.dump(bundle, model_path)


if __name__ == "__main__":
    main()


        
