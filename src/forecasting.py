"""Short‑term and long‑term sales forecasting."""
import pandas as pd
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Prepare time‑series dataframe per Store/Dept
# ---------------------------------------------------------------------------

def prepare_series(df: pd.DataFrame, store: int, dept: int):
    ts = (
        df[(df["Store"] == store) & (df["Dept"] == dept)]
        .set_index("Date")["Weekly_Sales"]
        .asfreq("W-MON")
        .fillna(method="ffill")
    )
    return ts

# ---------------------------------------------------------------------------
# SARIMA model
# ---------------------------------------------------------------------------

def train_sarima(ts: pd.Series, order=(1,1,1), seasonal=(1,1,1,52)):
    model = SARIMAX(ts, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

# ---------------------------------------------------------------------------
# Gradient boosting regressor using engineered features
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, params=None):
    params = params or {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str):
    joblib.dump(model, MODEL_DIR / name)

# ---------------------------------------------------------------------------
# Short-term model using SARIMAX
# ---------------------------------------------------------------------------

def train_sarimax(ts: pd.Series):
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    results = model.fit(disp=False)
    return results


def forecast_sarimax(model, steps: int = 12):
    forecast = model.forecast(steps=steps)
    return forecast


def save_model(model, name: str):
    out = MODEL_DIR / name
    joblib.dump(model, out)
    print(f"Model saved to {out}")


# ---------------------------------------------------------------------------
# Long-term model using XGBoost
# ---------------------------------------------------------------------------

def create_features(ts: pd.Series):
    df = ts.reset_index().rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    df["week"] = df["ds"].dt.isocalendar().week
    df["year"] = df["ds"].dt.year
    df["lag1"] = df["y"].shift(1)
    df["lag52"] = df["y"].shift(52)
    df = df.dropna()
    return df


def train_xgb(df: pd.DataFrame):
    features = ["week", "year", "lag1", "lag52"]
    X = df[features]
    y = df["y"]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model


def predict_xgb(model, df: pd.DataFrame):
    features = ["week", "year", "lag1", "lag52"]
    return model.predict(df[features])