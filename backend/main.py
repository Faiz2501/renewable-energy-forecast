from __future__ import annotations

import io
import hashlib
import json
import math
import os
import logging
import base64
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

WINDOW = 24
FEATURES = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "lag_1", "lag_24", "lag_168", "roll_24_mean", "roll_24_std"]
TARGET = "Production"
DATE = "Date and Hour"
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

app = FastAPI(title="Renewables Forecast API", version="1.0.0")
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
trained_cache: dict = {}
forecast_jobs: dict = {}
active_jobs: dict = {}
jobs_lock = threading.Lock()
job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast")


class LSTMForecast(nn.Module):
    def __init__(self, input_size: int = len(FEATURES), hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(1)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW, axis=0)[:-1]
        self.x = torch.from_numpy(np.ascontiguousarray(np.moveaxis(windows, -1, 1), dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y[WINDOW:], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
PRETRAINED_MODEL_PATH = ARTIFACT_DIR / "france-wind-lstm.pt"
PRETRAINED_RESULTS_PATH = ARTIFACT_DIR / "france-wind-results.json"
pretrained_artifact = None
pretrained_model = None


def load_pretrained_artifact():
    global pretrained_artifact, pretrained_model
    if not PRETRAINED_RESULTS_PATH.is_file():
        return
    try:
        with PRETRAINED_RESULTS_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        model = LSTMForecast().to(DEVICE)
        checkpoint_parts = sorted(ARTIFACT_DIR.glob("france-wind-lstm-*.b64"))
        if checkpoint_parts:
            encoded_checkpoint = "".join(part.read_text(encoding="ascii").strip() for part in checkpoint_parts)
            checkpoint_bytes = base64.b64decode(encoded_checkpoint, validate=True)
        elif PRETRAINED_MODEL_PATH.is_file():
            checkpoint_bytes = PRETRAINED_MODEL_PATH.read_bytes()
        else:
            raise FileNotFoundError("The saved France model checkpoint is missing.")
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=DEVICE, weights_only=True)
        if checkpoint.get("window") != WINDOW or checkpoint.get("features") != FEATURES:
            raise ValueError("The saved France model uses an incompatible feature layout.")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        if not metadata.get("dataset_sha256") or not metadata.get("forecast") or not metadata.get("metrics"):
            raise ValueError("The saved France forecast artifact is incomplete.")
        pretrained_artifact, pretrained_model = metadata, model
        logging.info("Loaded saved France wind model and forecast for dataset %s", metadata["dataset_sha256"])
    except Exception:
        logging.exception("Could not load the saved France wind forecast artifact")


load_pretrained_artifact()


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    hour = result["datetime"].dt.hour
    day = result["datetime"].dt.dayofyear
    values = result[TARGET]
    result["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    result["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    result["sin_doy"] = np.sin(2 * np.pi * day / 365)
    result["cos_doy"] = np.cos(2 * np.pi * day / 365)
    result["lag_1"] = values.shift(1)
    result["lag_24"] = values.shift(24)
    result["lag_168"] = values.shift(168)
    result["roll_24_mean"] = values.rolling(24).mean()
    result["roll_24_std"] = values.rolling(24).std()
    return result


def parse_csv(raw: bytes) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc
    missing = [col for col in (DATE, TARGET) if col not in frame.columns]
    if missing:
        raise HTTPException(400, f"CSV must contain columns named {DATE!r} and {TARGET!r}.")
    frame["datetime"] = pd.to_datetime(frame[DATE], utc=True, errors="coerce").dt.tz_convert(None)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    selected_source = "All sources"
    if "Source" in frame.columns:
        source = frame["Source"].mode()
        if not source.empty:
            selected_source = str(source.iloc[0])
            frame = frame[frame["Source"] == source.iloc[0]]
    frame = frame.reset_index(drop=True)
    if len(frame) < 400:
        raise HTTPException(400, "Upload at least 400 timestamped records so the 70/20/10 split leaves enough rows in each section.")
    return frame[["datetime", TARGET]], selected_source


def fit_model(features: pd.DataFrame, on_epoch=None):
    valid = features.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    if len(valid) < 3 * (WINDOW + 1):
        raise HTTPException(400, "Not enough rows remain after feature generation for train, validation, and test windows.")
    test_n = int(len(valid) * 0.10)
    val_n = int(len(valid) * 0.20)
    train_n = len(valid) - val_n - test_n
    train_df = valid.iloc[:train_n]
    val_df = valid.iloc[train_n:train_n + val_n]
    test_df = valid.iloc[train_n + val_n:]
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_train = x_scaler.fit_transform(train_df[FEATURES].values)
    y_train = y_scaler.fit_transform(train_df[[TARGET]].values).ravel()
    x_val = x_scaler.transform(val_df[FEATURES].values)
    y_val = y_scaler.transform(val_df[[TARGET]].values).ravel()
    x_test = x_scaler.transform(test_df[FEATURES].values)
    y_test = y_scaler.transform(test_df[[TARGET]].values).ravel()
    train_ds = WindowDataset(x_train, y_train)
    val_ds = WindowDataset(x_val, y_val)
    test_ds = WindowDataset(x_test, y_test)
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise HTTPException(400, "The CSV is too short to form all three 24-hour windows after the chronological split.")
    torch.manual_seed(42)
    model = LSTMForecast().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, 251):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(losses))
        scheduler.step(val_loss)
        epoch_train_loss = train_loss / len(train_ds)
        print(f"epoch {epoch}/250: train={epoch_train_loss:.6f} val={val_loss:.6f}", flush=True)
        if on_epoch:
            on_epoch(epoch, epoch_train_loss, val_loss)
        if val_loss < best - 1e-8:
            best, best_state, stale, best_epoch = val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            stale += 1
            if epoch - best_epoch >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            test_preds.extend(model(xb).numpy().tolist())
            test_true.extend(yb.numpy().tolist())
    predicted = y_scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).ravel()
    actual = y_scaler.inverse_transform(np.array(test_true).reshape(-1, 1)).ravel()
    mse = mean_squared_error(actual, predicted)
    return model, x_scaler, y_scaler, valid, {
        "mse": float(mse),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "epochs": best_epoch,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(actual),
    }


def forecast(model, x_scaler, y_scaler, history: pd.DataFrame, valid: pd.DataFrame, horizon: int):
    scaled = x_scaler.transform(valid[FEATURES].to_numpy())[-WINDOW:]
    values = history[TARGET].astype(float).tolist()
    next_date = pd.Timestamp(history["datetime"].iloc[-1])
    result = []
    for _ in range(horizon):
        with torch.no_grad():
            scaled_y = model(torch.tensor(scaled[None, :, :], dtype=torch.float32)).item()
        value = float(y_scaler.inverse_transform(np.array([[scaled_y]])).ravel()[0])
        next_date += timedelta(hours=1)
        result.append({"datetime": next_date.isoformat(), "value": value})
        values.append(value)
        hour, day = next_date.hour, next_date.dayofyear
        recent = np.asarray(values, dtype=float)
        row = pd.DataFrame([{
            "sin_hour": np.sin(2 * np.pi * hour / 24), "cos_hour": np.cos(2 * np.pi * hour / 24),
            "sin_doy": np.sin(2 * np.pi * day / 365), "cos_doy": np.cos(2 * np.pi * day / 365),
            "lag_1": recent[-1], "lag_24": recent[-24], "lag_168": recent[-168],
            "roll_24_mean": recent[-24:].mean(), "roll_24_std": recent[-24:].std(ddof=1),
        }], columns=FEATURES)
        scaled = np.vstack([scaled[1:], x_scaler.transform(row[FEATURES].to_numpy())])
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": "LSTM", "window_hours": WINDOW}


def run_forecast_job(job_id: str, key: str, raw: bytes, filename: str, horizon: int):
    try:
        with jobs_lock:
            job = forecast_jobs[job_id]
            job["status"] = "preparing"
            job["message"] = "Reading the CSV and preparing hourly features."

        if pretrained_artifact and key == pretrained_artifact["dataset_sha256"]:
            result = {
                "observed": pretrained_artifact["observed"],
                "forecast": pretrained_artifact["forecast"][:horizon],
                "metrics": pretrained_artifact["metrics"],
                "source": pretrained_artifact["source"],
                "filename": filename,
                "rows_used": pretrained_artifact["rows_used"],
                "trained_through": pretrained_artifact["trained_through"],
            }
            with jobs_lock:
                forecast_jobs[job_id].update({"status": "completed", "message": "Saved forecast ready.", "result": result})
            return

        cached = trained_cache.get(key)
        if cached is None:
            history, source = parse_csv(raw)
            features = make_features(history)

            def report_epoch(epoch, train_loss, val_loss):
                with jobs_lock:
                    current = forecast_jobs.get(job_id)
                    if current:
                        current.update({
                            "status": "training",
                            "epoch": epoch,
                            "total_epochs": 250,
                            "train_loss": train_loss,
                            "validation_loss": val_loss,
                            "message": f"Training the LSTM · epoch {epoch} of up to 250",
                        })

            model, x_scaler, y_scaler, valid, metrics = fit_model(features, on_epoch=report_epoch)
            trained_cache.clear()
            trained_cache[key] = (history, model, x_scaler, y_scaler, valid, metrics, source)
        else:
            history, model, x_scaler, y_scaler, valid, metrics, source = cached

        future = forecast(model, x_scaler, y_scaler, history, valid, horizon)
        observed = [{"datetime": row.datetime.isoformat(), "value": float(row.Production)} for row in history.tail(WINDOW).itertuples()]
        result = {"observed": observed, "forecast": future, "metrics": metrics,
                  "source": source, "filename": filename, "rows_used": len(history),
                  "trained_through": history.datetime.iloc[-1].isoformat()}
        with jobs_lock:
            forecast_jobs[job_id].update({"status": "completed", "message": "Forecast ready.", "result": result})
    except HTTPException as exc:
        with jobs_lock:
            forecast_jobs[job_id].update({"status": "failed", "error": str(exc.detail)})
    except Exception as exc:
        logging.exception("Forecast job %s failed", job_id)
        with jobs_lock:
            forecast_jobs[job_id].update({"status": "failed", "error": "The forecast could not be completed. Check the backend logs and try again."})
    finally:
        with jobs_lock:
            for active_key, active_id in list(active_jobs.items()):
                if active_id == job_id:
                    active_jobs.pop(active_key, None)


@app.post("/forecast", status_code=202)
async def create_forecast(file: UploadFile = File(...), horizon: int = Form(24)):
    if horizon < 1 or horizon > 168:
        raise HTTPException(400, "Forecast horizon must be between 1 and 168 hours.")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "CSV upload is limited to 25 MB.")
    key = hashlib.sha256(raw).hexdigest()
    job_key = f"{key}:{horizon}"
    with jobs_lock:
        existing = active_jobs.get(job_key)
        if existing:
            return {"job_id": existing, "status": forecast_jobs[existing]["status"]}
        job_id = str(uuid.uuid4())
        forecast_jobs[job_id] = {"status": "queued", "message": "Forecast queued.", "epoch": 0, "total_epochs": 250}
        active_jobs[job_key] = job_id
    job_executor.submit(run_forecast_job, job_id, key, raw, file.filename, horizon)
    return {"job_id": job_id, "status": "queued", "message": "Forecast queued."}


@app.get("/forecast/{job_id}")
def get_forecast_job(job_id: str):
    with jobs_lock:
        job = forecast_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "This forecast job is no longer available. Please upload the CSV and try again.")
        return dict(job)

