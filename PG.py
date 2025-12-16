from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# КОНФИГУРАЦИЯ
@dataclass(frozen=True)
class AppConfig:
    horizon_days: int = 30
    train_split: float = 0.85
    min_extra_points: int = 5


CFG = AppConfig()

# ДАННЫЕ
RAW_DATA = """Month,Milk (1L),Bread (500g),Rice (1kg),Eggs (12),Chicken (1kg),Potatoes (1kg),Apples (1kg)
2025-01,85,45,75,95,300,40,120
2025-02,87,46,78,97,305,42,125
2025-03,90,48,80,100,310,43,130
2025-04,92,50,82,102,315,44,135
2025-05,95,52,85,105,320,45,140
2025-06,97,54,88,108,325,46,145
2025-07,99,55,90,110,330,47,150
2025-08,100,56,92,112,335,48,155
2025-09,101,57,93,113,340,49,160
2025-10,102,58,95,115,345,50,165
2025-11,103,59,96,116,350,51,170
2025-12,105,60,98,118,355,52,175
"""


def load_monthly_df() -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.StringIO(RAW_DATA))
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.set_index("Month").sort_index()
    return df.astype(float)


def monthly_to_daily(df_monthly: pd.DataFrame) -> pd.DataFrame:
    start = df_monthly.index.min()
    end = df_monthly.index.max() + pd.offsets.MonthEnd(0)
    idx = pd.date_range(start=start, end=end, freq="D")
    df_daily = df_monthly.reindex(idx)
    df_daily = df_daily.interpolate(method="time").ffill().bfill()
    df_daily.index.name = "Дата"
    return df_daily


def slice_until_date(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    return df.loc[:dt.normalize()].copy()


# ДАТЫ
DATE_FMT = "%d.%m.%Y"


def parse_ddmmyyyy(s: str) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(s.strip(), format=DATE_FMT).normalize()
    except Exception:
        return None


def fmt_ddmmyyyy(dt: pd.Timestamp) -> str:
    return dt.strftime(DATE_FMT)


# МОДЕЛЬ
@dataclass
class TrainResult:
    model: tf.keras.Model
    scaler: MinMaxScaler
    history: tf.keras.callbacks.History
    mae: float
    rmse: float


def make_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def build_lstm(lookback: int, units: int, dropout: float, lr: float) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    return model


def train_lstm(
    series: np.ndarray,
    lookback: int,
    epochs: int,
    batch_size: int,
    units: int,
    dropout: float,
    lr: float,
    patience: int,
    seed: int
) -> TrainResult:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.astype(np.float32))

    X, y = make_sequences(scaled, lookback)
    split = int(len(X) * CFG.train_split)

    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_lstm(lookback, units, dropout, lr)

    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

# МЕТРИКИ КАЧЕСТВА 
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_true = scaler.inverse_transform(y_val)
    y_val_pred_inv = scaler.inverse_transform(y_val_pred)

    mae = mean_absolute_error(y_val_true, y_val_pred_inv)
    rmse = mean_squared_error(y_val_true, y_val_pred_inv, squared=False)

    return TrainResult(
        model=model,
        scaler=scaler,
        history=history,
        mae=float(mae),
        rmse=float(rmse)
    )

# ПРОГНОЗ
def forecast_days(
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    series: np.ndarray,
    lookback: int,
    days: int
) -> np.ndarray:
    scaled = scaler.transform(series.astype(np.float32))
    window = scaled[-lookback:, 0].copy()
    preds = []

    for _ in range(days):
        x = window.reshape(1, lookback, 1)
        y_hat = model.predict(x, verbose=0)[0, 0]
        preds.append(y_hat)
        window = np.roll(window, -1)
        window[-1] = y_hat

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


# СОХРАНЕНИЕ
def save_artifacts(folder: str, tr: TrainResult, meta: dict):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    tr.model.save(path / "model.keras")
    joblib.dump(tr.scaler, path / "scaler.pkl")

    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# UI
st.set_page_config(page_title="Прогноз цен на продукты (LSTM)", layout="wide")
st.title("Прогнозирование цен на продукты")
st.caption("LSTM • обучение • прогноз • метрики качества • экспорт • сохранение")

df_monthly = load_monthly_df()
df_daily = monthly_to_daily(df_monthly)

with st.sidebar:
    st.header("Настройки")
    product = st.selectbox("Продукт", df_daily.columns)
    epochs = st.slider("Эпохи", 10, 300, 120, 10)
    lookback = st.slider("Lookback (дней)", 7, 120, 60)
    batch = st.selectbox("Batch size", [8, 16, 32], index=1)
    units = st.selectbox("Units", [16, 32, 64], index=1)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2)
    lr = st.selectbox("LR", [1e-4, 1e-3, 3e-3], index=1)
    patience = st.slider("Patience", 0, 20, 10)
    seed = st.number_input("Seed", 0, 10000, 42)

tabs = st.tabs(["Данные", "Прогноз", "Обучение"])

with tabs[0]:
    st.subheader("Исходные данные")
    st.dataframe(df_monthly)

with tabs[1]:
    st.subheader("Дата старта прогноза")
    start_str = st.text_input("Дата (DD.MM.YYYY)", fmt_ddmmyyyy(df_daily.index[-2]))
    start_dt = parse_ddmmyyyy(start_str)

    if start_dt is None:
        st.error("Неверный формат даты")
    else:
        df_cut = slice_until_date(df_daily, start_dt)
        series = df_cut[[product]].values

        tr = train_lstm(
            series, lookback, epochs, batch,
            units, dropout, lr, patience, seed
        )

        forecast = forecast_days(
            tr.model, tr.scaler, series,
            lookback, CFG.horizon_days
        )

        dates = pd.date_range(start=start_dt + pd.Timedelta(days=1),
                              periods=CFG.horizon_days)

        df_fc = pd.DataFrame({
            "Дата": dates,
            "Прогноз, ₽": forecast
        })

        st.line_chart(df_fc.set_index("Дата"))

        csv = df_fc.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV", csv, "forecast.csv")

with tabs[2]:
    st.subheader("Метрики качества (валидация)")
    st.metric("MAE", f"{tr.mae:.2f} ₽")
    st.metric("RMSE", f"{tr.rmse:.2f} ₽")

    st.subheader("График обучения")
    fig, ax = plt.subplots()
    ax.plot(tr.history.history["loss"], label="loss")
    ax.plot(tr.history.history["val_loss"], label="val_loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if st.button("Сохранить модель"):
        save_artifacts(
            "artifacts",
            tr,
            {
                "product": product,
                "date": fmt_ddmmyyyy(start_dt),
                "mae": tr.mae,
                "rmse": tr.rmse
            }
        )
        st.success("Модель сохранена")
