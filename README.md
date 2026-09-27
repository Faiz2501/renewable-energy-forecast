# Current

An hourly wind and solar production forecast dashboard for France. The web page is a static site; the FastAPI service parses uploaded CSV data, fits the LSTM, reports scores on the final chronological test segment, and produces the requested forecast.

## Input data

CSV files must include `Date and Hour` and `Production` columns. A `Source` column is optional; when present, the most frequent source is forecast. Files are limited to 25 MB. At least 400 timestamped rows are required, and 360 hourly records are recommended.

Example:

```csv
Date and Hour,Production,Source
2024-01-01 00:00:00+01:00,123.4,Wind
```

## Run locally

On Windows, run `start-local.ps1`. It installs the API dependencies into `backend/.venv` the first time and starts the frontend at `http://localhost:3000` and the API at `http://127.0.0.1:8000`.

To start the API manually:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Serve the `frontend` directory with any static web server. The API health check is `/health`; interactive API documentation is at `/docs`.

## Model and metrics

The API builds cyclical hour and day-of-year features, recent and weekly lags, and rolling 24-hour statistics. It uses a two-layer, 128-unit LSTM with a 24-hour input window and a chronological 70/20/10 train, validation, and test split. MAE, RMSE, MSE, and R² are calculated on the held-out test segment in the original production units. The selected model then projects forward one hour at a time.

The first forecast for a new CSV trains the model and can take a long time on a CPU. The service must allow long-running HTTP requests. The fitted model is cached in memory for later forecasts using the same CSV while the API process stays running. Uploaded CSV files are not saved by the API. A restart clears the in-memory cache.

## Deployment

The frontend is a static site and the API is a Python FastAPI service. Deploy the API first, then set `window.CURRENT_API_URL` in `frontend/config.js` to the API's public HTTPS URL. Set `CORS_ORIGINS` on the API to the frontend's public origin. The supplied `render.yaml` describes the API service for Render; connect the repository containing this folder and deploy it as a Blueprint.

