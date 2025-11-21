"""anomaly_ml.py

Create silver labels, sample for human verification, and train a simple ML
anomaly classifier. This script supports processing multiple anomaly files per
country (ensemble, per_model, sarima) and a batch mode that scans the
`outputs/anomalies/` directory.

Usage examples:
  - Process a single country + source:
      python -m src.anomaly.anomaly_ml --country DK --source ensemble

  - Batch process all anomaly files found under `outputs/anomalies/`:
      python -m src.anomaly.anomaly_ml --batch

Outputs placed under `outputs/anomaly_ml/` and `outputs/checkpoints/`.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
import yaml

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    average_precision_score,
)

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def load_anomaly_file(path: Path, model_name: str = None) -> pd.DataFrame:
    df = (
        pd.read_csv(path, parse_dates=["timestamp"])
        if path.exists()
        else pd.DataFrame()
    )
    if df.empty:
        raise FileNotFoundError(f"Anomaly file not found or empty: {path}")

    if model_name is not None and "model_type" in df.columns:
        df = df[df["model_type"] == model_name].copy()

    if "resid" not in df.columns and ("y_true" in df.columns and "yhat" in df.columns):
        df["resid"] = df["y_true"] - df["yhat"]

    if "z_resid" not in df.columns:
        df["z_resid"] = np.nan

    return df


def make_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["y_true", "yhat", "lo", "hi", "z_resid"]:
        if c not in df.columns:
            df[c] = np.nan

    abs_z = df["z_resid"].abs()
    outside_pi = (df["y_true"] < df["lo"]) | (df["y_true"] > df["hi"])

    pos = (abs_z >= 3.5) | (outside_pi & (abs_z >= 2.5))
    neg = (abs_z < 1.0) & (~outside_pi)

    df["silver_label"] = np.nan
    df.loc[pos.fillna(False), "silver_label"] = 1
    df.loc[neg.fillna(False), "silver_label"] = 0

    return df


def featurize(df: pd.DataFrame, lags: int = 48) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)
    feat["resid"] = df.get("resid", pd.Series(0, index=df.index)).fillna(0)

    for i in range(1, lags + 1):
        feat[f"resid_lag_{i}"] = df["resid"].shift(i).fillna(0)

    feat["resid_roll_mean_24"] = df["resid"].rolling(24, min_periods=1).mean().fillna(0)
    feat["resid_roll_std_24"] = df["resid"].rolling(24, min_periods=1).std().fillna(0)

    ts = (
        pd.to_datetime(df["timestamp"])
        if "timestamp" in df.columns
        else pd.to_datetime(pd.Series(index=df.index))
    )
    feat["hour"] = ts.dt.hour.fillna(0).astype(int)
    feat["dow"] = ts.dt.dayofweek.fillna(0).astype(int)

    feat = pd.get_dummies(feat, columns=["hour"], prefix="h", drop_first=True)

    if "lo" in df.columns and "hi" in df.columns:
        feat["pi_width"] = (df["hi"] - df["lo"]).fillna(0)

    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
    return feat


def train_and_eval(X, y):
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes to train classifier")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if HAS_LGBM:
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, verbose=-1
        )
    else:
        print("⚠ LightGBM unavailable → fallback to Logistic Regression")
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)

    y_scores = clf.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    best_f1 = 0.0
    best_thr = 0.5
    if thresholds is not None and len(thresholds) > 0:
        for thr in np.unique(np.clip(thresholds, 0, 1)):
            preds = (y_scores >= thr).astype(int)
            p = precision_score(y_test, preds, zero_division=0)
            if p >= 0.80:
                f = f1_score(y_test, preds, zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_thr = thr

    preds_best = (y_scores >= best_thr).astype(int)
    f1 = f1_score(y_test, preds_best, zero_division=0)
    p = precision_score(y_test, preds_best, zero_division=0)

    metrics = {
        "pr_auc": float(pr_auc),
        "f1_at_p0.8": float(best_f1),
        "best_threshold": float(best_thr),
        "f1": float(f1),
        "precision": float(p),
    }
    return clf, metrics


def process_anomaly_file(path: Path, out_root: Path, model_name: str = None):
    cc = path.stem.split("_")[0]
    print(f"\n--- Processing {path.name} (country={cc})")
    df = load_anomaly_file(path, model_name)
    df = make_silver_labels(df)

    out_labels = (
        out_root / f"anomaly_labels_silver_{cc}_{path.stem.split('_',1)[1]}.csv"
    )
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_labels, index=False)
    print(f"Saved silver labels → {out_labels} (total={len(df)})")

    labeled = df[df["silver_label"].notna()].copy()
    if labeled.empty or len(labeled) < 20:
        print(
            "Not enough labeled examples to train (need >=20). Skipping training for this file."
        )
        return

    pos = labeled[labeled["silver_label"] == 1]
    neg = labeled[labeled["silver_label"] == 0]
    npos = min(len(pos), 50)
    nneg = min(len(neg), 50)
    sample = pd.concat(
        [
            pos.sample(n=npos, random_state=1) if npos > 0 else pos.iloc[0:0],
            neg.sample(n=nneg, random_state=1) if nneg > 0 else neg.iloc[0:0],
        ]
    )
    sample_path = (
        out_root / f"anomaly_labels_verified_{cc}_{path.stem.split('_',1)[1]}.csv"
    )
    sample.to_csv(sample_path, index=False)
    print(f"Saved human-verification sample → {sample_path} (pos={npos}, neg={nneg})")

    X = featurize(labeled)
    y = labeled["silver_label"].astype(int).values

    clf, metrics = train_and_eval(X.values, y)

    ckpt_dir = out_root.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / f"{cc}_anomaly_clf_{path.stem.split('_',1)[1]}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    eval_path = out_root / f"anomaly_ml_eval_{cc}_{path.stem.split('_',1)[1]}.json"
    with open(eval_path, "w") as f:
        json.dump(
            {"country": cc, "file": str(path.name), "metrics": metrics}, f, indent=2
        )

    print(f"Trained classifier saved → {model_path}")
    print(f"Eval saved → {eval_path}")


def find_anomaly_files(anomalies_dir: Path):
    files = list(anomalies_dir.glob("*_anomalies*.csv"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", help="Country code (e.g. DK)")
    parser.add_argument(
        "--source",
        choices=["ensemble", "sarima", "per_model"],
        help="choose source type when using --country",
    )
    parser.add_argument(
        "--model-name", help="when source is per_model, optionally filter by model_type"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all anomaly files found under outputs/anomalies/",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        anomalies_dir = Path(cfg.get("output_anomalies", "outputs/anomalies"))
        out_root = Path(cfg.get("output_anomaly_ml", "outputs/anomaly_ml"))
    else:
        anomalies_dir = Path("outputs/anomalies")
        out_root = Path("outputs/anomaly_ml")

    out_root.mkdir(parents=True, exist_ok=True)

    if args.batch:
        files = find_anomaly_files(anomalies_dir)
        if not files:
            print(f"No anomaly files found in {anomalies_dir}")
            return
        for f in files:
            try:
                process_anomaly_file(f, out_root)
            except Exception as e:
                print(f"Error processing {f.name}: {e}")
    else:
        if not args.country:
            print("Either --batch or --country must be provided")
            sys.exit(1)

        cc = args.country

        if args.source == "ensemble":
            p = anomalies_dir / f"{cc}_anomalies_ensemble.csv"
        elif args.source == "sarima":
            p = anomalies_dir / f"{cc}_anomalies_sarima.csv"
        elif args.source == "per_model":
            p = anomalies_dir / f"{cc}_anomalies_per_model.csv"
        else:

            p = anomalies_dir / f"{cc}_anomalies.csv"

        if not p.exists():
            print(f"Requested anomaly file not found: {p}")
            sys.exit(2)

        process_anomaly_file(p, out_root, model_name=args.model_name)


if __name__ == "__main__":
    main()
