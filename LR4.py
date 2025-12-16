from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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
    df = df.set_index("Month").sort_index().astype(float)
    return df

def monthly_to_daily(df_monthly: pd.DataFrame) -> pd.DataFrame:
    start = df_monthly.index.min()
    end = df_monthly.index.max() + pd.offsets.MonthEnd(0)
    idx = pd.date_range(start=start, end=end, freq="D")
    df_daily = df_monthly.reindex(idx).interpolate(method="time").ffill().bfill()
    df_daily.index.name = "Дата"
    return df_daily


# ПОДГОТВКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ
def minmax_scale(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    scaled = (x - mn) / (mx - mn)
    return scaled.astype(np.float32), mn, mx

def minmax_inverse(s: np.ndarray, mn: float, mx: float) -> np.ndarray:
    return s * (mx - mn) + mn

def make_sequences(series_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(series_scaled)):
        X.append(series_scaled[i - lookback:i])
        y.append(series_scaled[i])
    X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


# МЕТРИКИ
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# МОДЕЛЬ 
def build_model(lookback: int, lr: float = 1e-3) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(8),        
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

# ЗАПУСК
def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    out_dir = Path("lr4_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_daily = monthly_to_daily(load_monthly_df())

    product = "Milk (1L)"
    lookback = 30
    epochs = 10_000         
    batch_size = 16

    # берем один продукт
    series = df_daily[product].values.astype(np.float32).reshape(-1, 1)

    # scale вручную (чтобы не зависеть от sklearn)
    scaled, mn, mx = minmax_scale(series)

    # train/val split (по времени)
    X, y = make_sequences(scaled, lookback)
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(lookback=lookback, lr=1e-3)

    # без EarlyStopping, чтобы дошло до 10k эпох
    t0 = time.perf_counter()
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False
    )
    train_time = time.perf_counter() - t0

    # история loss - CSV
    history_df = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "loss": hist.history["loss"],
        "val_loss": hist.history["val_loss"],
    })
    history_path = out_dir / "history_10k.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8")

    # график loss
    plt.figure(figsize=(10, 4))
    plt.plot(history_df["epoch"], history_df["loss"], label="loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.title("Обучение 10 000 эпох: loss/val_loss")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    fig_path = out_dir / "loss_10k.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close()

    # метрики на валидации в рублях
    y_val_pred = model.predict(X_val, verbose=0).astype(np.float32)
    y_val_true_rub = minmax_inverse(y_val, mn, mx)
    y_val_pred_rub = minmax_inverse(y_val_pred, mn, mx)

    mae_val = mae(y_val_true_rub, y_val_pred_rub)
    rmse_val = rmse(y_val_true_rub, y_val_pred_rub)

    # baseline (naive): предсказание последнего значения окна
    y_bl_scaled = X_val[:, -1, 0].reshape(-1, 1)
    y_bl_rub = minmax_inverse(y_bl_scaled, mn, mx)
    mae_bl = mae(y_val_true_rub, y_bl_rub)
    rmse_bl = rmse(y_val_true_rub, y_bl_rub)

    # время отклика (мс): один шаг и прогноз 30 дней
    x_one = X_val[0:1]
    t1 = time.perf_counter()
    _ = model.predict(x_one, verbose=0)
    one_step_ms = (time.perf_counter() - t1) * 1000.0

    def forecast_30_days():
        window = scaled[-lookback:, 0].copy()
        preds = []
        for _ in range(30):
            x = window.reshape(1, lookback, 1)
            y_hat = float(model.predict(x, verbose=0)[0, 0])
            preds.append(y_hat)
            window[:-1] = window[1:]
            window[-1] = y_hat
        return preds

    t2 = time.perf_counter()
    _ = forecast_30_days()
    forecast30_ms = (time.perf_counter() - t2) * 1000.0

    # сохранение модели
    model_path = out_dir / "model_10k.keras"
    model.save(model_path)

    # печать результатов 
    print("БЕНЧМАРК")
    print("Продукт:", product)
    print("Эпохи:", epochs)
    print(f"Время обучения, c: {train_time:.3f}")
    print(f"MAE, ₽: {mae_val:.4f} | RMSE, ₽: {rmse_val:.4f}")
    print(f"Baseline MAE, ₽: {mae_bl:.4f} | Baseline RMSE, ₽: {rmse_bl:.4f}")
    print(f"Отклик 1 шага, мс: {one_step_ms:.3f}")
    print(f"Прогноз 30 дней, мс: {forecast30_ms:.3f}")
    print("Файлы:")
    print("-", history_path)
    print("-", fig_path)
    print("-", model_path)

if __name__ == "__main__":
    main()

