# Current — Renewable Production Forecasts

Current is a web application for exploring short-term hourly wind and solar production forecasts for France. Upload a CSV, choose a forecast horizon, and review the model's held-out test metrics alongside the projected production curve.

**Live app:** [renewable-energy-forecast.vercel.app](https://renewable-energy-forecast.vercel.app/)  
**API:** [renewables-forecast-api.onrender.com](https://renewables-forecast-api.onrender.com) · [Health check](https://renewables-forecast-api.onrender.com/health) · [Interactive API docs](https://renewables-forecast-api.onrender.com/docs)

## How it works

The frontend is a static HTML, CSS, and JavaScript application hosted on Vercel. It sends the selected CSV and forecast horizon to the Python FastAPI service on Render. The API reads and validates the data, trains or loads the model, then returns test metrics and hourly forecast values for the chart.

Forecast requests run as background jobs so the browser does not have to keep a long-running upload request open. `POST /forecast` returns a job ID; the frontend polls `GET /forecast/{job_id}` for training progress and the completed result. The backend's `GET /health` endpoint is used by Render to check service health.

## Model

The forecasting model is a PyTorch **sequence-to-one LSTM** for regression. It takes the previous 24 hourly feature rows and predicts the next hour's production. The network has two LSTM layers with 128 hidden units per layer, 0.2 dropout, and a fully connected output layer that returns one production value. Training uses CPU, a fixed random seed of 42, Adam with a 0.0005 learning rate, mean squared error loss, and batches of 64.

### Features

For each timestamp, the API derives nine inputs:

| Feature | Meaning |
| --- | --- |
| `sin_hour`, `cos_hour` | Cyclical hour-of-day encoding |
| `sin_doy`, `cos_doy` | Cyclical day-of-year encoding |
| `lag_1` | Production one hour earlier |
| `lag_24` | Production 24 hours earlier |
| `lag_168` | Production 168 hours earlier (one week) |
| `roll_24_mean`, `roll_24_std` | Mean and standard deviation over the previous 24 hours |

Both features and the target are MinMax-scaled. Scalers are fitted on the training portion and reused for validation, test, and prediction. The data is split in timestamp order into 70% training, 20% validation, and 10% test data, preserving the forecasting timeline. The best validation checkpoint is kept; training stops after 20 epochs without improvement or at 250 epochs. A learning-rate scheduler reduces the learning rate when validation loss plateaus.

MSE, RMSE, MAE, and R² are calculated from the held-out test segment after predictions are converted back to the CSV's original production units. To forecast multiple hours, the model predicts one hour at a time; each prediction is fed into the lag and rolling features used for the next hour. Forecast horizons range from 1 to 168 hours.

## Shared France wind artifact

The repository includes the trained checkpoint, scalers, test metrics, the last 24 observed values, and a precomputed 168-hour forecast for the supplied France wind CSV. The API compares the SHA-256 hash of the uploaded file's **raw bytes** with the saved dataset hash. A match returns the bundled results without retraining. The CSV itself is not included in this repository.

The bundled artifact was trained on 29,904 hourly wind rows, with the data through **2023-06-30 21:00 UTC**. Its held-out test scores are:

| Metric | Score |
| --- | ---: |
| MSE | 692,804.99 |
| RMSE | 832.35 |
| MAE | 495.87 |
| R² | 0.9471 |

These are results for this specific dataset and this chronological split; they are not a guarantee of accuracy on future data or other production sources. Because matching uses a hash of the file bytes, re-exporting or editing the CSV—even if its values look the same—may result in a different hash and trigger training.

For any other CSV, the API trains the same model and reports epoch and validation-loss progress in the page. A fitted model is cached in backend memory for reuse with the same file while that service process remains running. This cache is temporary and is cleared when the service restarts; only the shared France artifact is bundled with the deployment.

The checkpoint is stored as numbered Base64 text parts in `backend/artifacts/` so it can be transported reliably through GitHub's file APIs. At startup the API joins those parts, decodes the PyTorch checkpoint, and loads its model weights. The result metadata is in `backend/artifacts/france-wind-results.json`.

To regenerate the artifact from a local copy of the source CSV, run this from the `backend` directory:

```bash
python build_france_artifact.py /path/to/intermittent-renewables-production-france.csv
```

This writes the `.pt` checkpoint and result JSON under `backend/artifacts/`. The source CSV stays at the path you supplied and is not copied into the repository.

## CSV format and limits

The CSV must have these columns:

| Column | Required | Description |
| --- | --- | --- |
| `Date and Hour` | Yes | Timestamp for each production record |
| `Production` | Yes | Numeric production value |
| `Source` | No | If present, the API selects the most frequent source in the file |

The API accepts CSV uploads up to 25 MB and requires at least 400 timestamped records. Hourly data is expected. Rows with unreadable timestamps are dropped; records are sorted by timestamp before features and splits are created. The application is configured for the France wind and solar production dataset and does not currently support choosing a source column value in the UI.

Example:

```csv
Date and Hour,Production,Source
2024-01-01 00:00:00+01:00,123.4,Wind
```

## Run locally

On Windows, run `start-local.ps1` from the project root. It creates the backend virtual environment and installs requirements on first use, then starts the API at `http://127.0.0.1:8000` and the static frontend at `http://localhost:3000`.

To start the API manually:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

In another terminal, serve the `frontend` folder with a static file server, then open its local URL. Set `window.CURRENT_API_URL` in `frontend/config.js` to the API URL you are running.

## Deployment

The current production setup is:

| Part | Host | Configuration |
| --- | --- | --- |
| Frontend | Vercel | Static files from the `frontend` directory; connected to the GitHub repository for deployments |
| Backend | Render | Python FastAPI web service defined in `render.yaml`; health check path `/health` |

The frontend's `frontend/config.js` points at the Render API URL. The Render service's `CORS_ORIGINS` variable currently allows browser requests from configured origins; set it to the frontend's exact public origin if restricting CORS. Render Free services can sleep while idle, so the first request after a quiet period may wait for the backend to wake before a cached result is returned.

## Project layout

```text
backend/
  main.py                       FastAPI endpoints, data pipeline, LSTM and forecast jobs
  build_france_artifact.py      Rebuild the shared France wind model and results
  artifacts/                    Shared model checkpoint pieces and forecast metadata
frontend/
  index.html                    Page structure
  app.js                        Upload, progress polling, results and chart behavior
  style.css                     Visual styling
  config.js                     Public API base URL
  vercel.json                   Static-site response configuration
render.yaml                     Render backend service definition
start-local.ps1                 Local Windows development launcher
```

## License

No license has been added yet. All rights remain with the repository owner unless a license is added.

