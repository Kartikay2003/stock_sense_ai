from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras
from keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Register custom objects for Keras loading
@keras.saving.register_keras_serializable(package="Custom")
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

app = FastAPI()

# 2. CORS CONFIGURATION: This allows your Vercel site to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocalStockForecaster:
    def __init__(self, seq_len=60):
        self.SEQ_LEN = seq_len
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_file = os.path.join(BASE_DIR, "forecasting.keras")
        self.scaler_file = os.path.join(BASE_DIR, "scaler.pkl")

        if not os.path.exists(self.model_file) or not os.path.exists(self.scaler_file):
            raise FileNotFoundError("Model or Scaler file missing from the directory.")

        # Load model without compiling to avoid issues with custom metrics during init
        self.model = load_model(self.model_file, compile=False)
        self.scaler = joblib.load(self.scaler_file)

    def download_stock_data(self, ticker):
        try:
            # ATTEMPT 1: Try yfinance
            stock = yf.Ticker(ticker)
            raw = stock.history(period="max")

            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty data")

            df = raw.copy()
            df['Price'] = df['Close']

        except Exception as e:
            print(f"yfinance blocked by Yahoo ({e}). Falling back to Stooq API...")

            # ATTEMPT 2: The Stooq Fallback (Completely bypasses Yahoo)
            # Format ticker for Stooq (e.g., AAPL -> AAPL.US)
            stooq_ticker = ticker.upper()
            if "." not in stooq_ticker:
                stooq_ticker += ".US"

            url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"

            # Read directly from the URL using pandas
            df = pd.read_csv(url, index_col="Date", parse_dates=True)

            if df is None or df.empty or 'Close' not in df.columns:
                raise RuntimeError(f"Complete failure: Could not fetch {ticker} from Yahoo or Stooq.")

            df['Price'] = df['Close']

        # Clean and finalize formatting for the AI model
        df['Volume'] = df['Volume'].fillna(0) if 'Volume' in df.columns else 0
        df['Sentiment'] = 0.0
        df = df[['Price', 'Volume', 'Sentiment']]
        df.index = pd.to_datetime(df.index)

        # Ensure it's sorted oldest to newest
        return df.sort_index()

    def _prepare_sequences(self, df):
        values = df[["Price", "Volume", "Sentiment"]].values.astype(float)
        scaled = self.scaler.transform(values)
        X, y = [], []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i-self.SEQ_LEN:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def predict_future(self, ticker, days=10):
        df = self.download_stock_data(ticker)
        X, y = self._prepare_sequences(df)

        split = max(int(len(X) * 0.8), 1)
        X_test, y_test = X[split:], y[split:]

        # Test set predictions for metrics
        pred_scaled_test = self.model.predict(X_test, verbose=0).reshape(-1, 1)
        other_test = X_test[:, -1, 1:]
        inv_pred_test = self.scaler.inverse_transform(np.hstack([pred_scaled_test, other_test]))[:, 0]
        inv_real_test = self.scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), other_test]))[:, 0]

        mae = mean_absolute_error(inv_real_test, inv_pred_test)
        rmse_val = np.sqrt(mean_squared_error(inv_real_test, inv_pred_test))

        if len(inv_real_test) >= 2:
            direction_acc = (np.sum((np.diff(inv_real_test) * np.diff(inv_pred_test)) > 0) / len(np.diff(inv_real_test))) * 100
        else:
            direction_acc = 0.0

        # Recursive future forecasting
        seq = X_test[-1].copy() if len(X_test) else X[-1].copy()
        future_scaled = []
        for _ in range(days):
            p_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
            future_scaled.append(p_scaled)
            # Update sequence: shift values and append new prediction
            seq = np.vstack([seq[1:], np.hstack([[p_scaled], seq[-1, 1:]])])

        # Inverse transform future predictions
        future_other = np.tile(seq[-1, 1:], (days, 1))
        inv_future = self.scaler.inverse_transform(np.hstack([np.array(future_scaled).reshape(-1,1), future_other]))[:, 0]

        future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": np.round(inv_future, 2)})

        metrics = {
            "MAE": float(np.round(mae, 4)),
            "RMSE": float(np.round(rmse_val, 4)),
            "Direction_Accuracy": float(np.round(direction_acc, 2))
        }
        return forecast_df, df, metrics

@app.get("/api/predict")
def predict_stock(ticker: str, days: int = 10, seq_len: int = 60):
    try:
        forecaster = LocalStockForecaster(seq_len=seq_len)
        forecast_df, history_df, metrics = forecaster.predict_future(ticker, days=days)

        # Extract current price safely
        cp = history_df['Price'].iloc[-1]
        current_price = float(cp.iloc[0]) if isinstance(cp, pd.Series) else float(cp)

        # Format historical data (last 30 days)
        hist_data = []
        for date, row in history_df.tail(30).iterrows():
            p = row['Price']
            price_val = float(p.iloc[0]) if isinstance(p, pd.Series) else float(p)
            hist_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": price_val,
                "type": "History"
            })

        # Format forecast data
        forecast_data = [
            {
                "date": row['Date'].strftime("%Y-%m-%d"),
                "price": float(row['Predicted_Price']),
                "type": "Forecast"
            } for _, row in forecast_df.iterrows()
        ]

        return {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "metrics": metrics,
            "chart_data": hist_data + forecast_data
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))