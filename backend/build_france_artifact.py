"""Train and package the shared France wind model without committing its CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from main import FEATURES, WINDOW, fit_model, forecast, make_features, parse_csv


def scaler_state(scaler) -> dict:
    return {
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to intermittent-renewables-production-france.csv")
    args = parser.parse_args()

    raw = args.csv.read_bytes()
    history, source = parse_csv(raw)
    model, x_scaler, y_scaler, valid, metrics = fit_model(make_features(history))
    future = forecast(model, x_scaler, y_scaler, history, valid, 168)
    observed = [
        {"datetime": row.datetime.isoformat(), "value": float(row.Production)}
        for row in history.tail(WINDOW).itertuples()
    ]

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "france-wind-lstm.pt"
    results_path = artifact_dir / "france-wind-results.json"
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "window": WINDOW,
        "features": FEATURES,
        "x_scaler": scaler_state(x_scaler),
        "y_scaler": scaler_state(y_scaler),
    }
    metadata = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "source": source,
        "rows_used": len(history),
        "trained_through": history["datetime"].iloc[-1].isoformat(),
        "metrics": metrics,
        "observed": observed,
        "forecast": future,
    }

    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    temporary_results = results_path.with_suffix(".json.tmp")
    torch.save(checkpoint, temporary_checkpoint)
    temporary_results.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    temporary_checkpoint.replace(checkpoint_path)
    temporary_results.replace(results_path)

    print(f"Saved model: {checkpoint_path}")
    print(f"Saved results: {results_path}")
    print(f"Dataset SHA-256: {metadata['dataset_sha256']}")
    print(f"Source: {source}; rows: {len(history)}; best epoch: {metrics['epochs']}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

