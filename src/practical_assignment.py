""" 
Data Stream Mining Practical Assignment 1
PA30, T6- Adverse Outdoor Event
Cities: Viana do Castelo Vs Guarda
Business Scenario: Seasonal Outdoor Fair
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
import requests
import re
import pandas as pd
import numpy as np
import sys
import time
import importlib.metadata as importlib_metadata
from sklearn.preprocessing import MinMaxScaler
from river import ensemble, forest, linear_model, naive_bayes, optim, tree

#----------------------------------------------
# 1. Configuration

# coordinates of Viana do Castelo and Guarda
CITY_A = {"name": "Viana do Castelo", "latitude": 41.693, "longitude": -8.832}
CITY_B = {"name": "Guarda", "latitude": 40.537, "longitude": -7.268}

# historical period
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Europe/Lisbon"

# open meteo features
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

#target T6 = T1 (rain) OR T5 (strong wind)
RAIN_THRESHOLD_MM = 0.1   # T1: y = 1 if precipitation >= 0.1 mm
WIND_QUANTILE = 0.80      # T5: threshold = 0.80 quantile of wind in the pooled warm-up

#warm-up fraction 
WARMUP_FRACTION = 0.25

# feature windows (hours)
WINDOW_SHORT = 6
WINDOW_LONG = 24

#default random seed from the template
RANDOM_SEED = 42

# model parameters 
HOEFFDING_GRACE_PERIOD = 50
HOEFFDING_DELTA = 1e-3
HOEFFDING_TAU = 0.10

ENSEMBLE_SIZE = 5
ENSEMBLE_GRACE_PERIOD = 30
ENSEMBLE_DELTA = 1e-3
ENSEMBLE_TAU = 0.10

# setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# open meteo archive url
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORCE_DOWNLOAD = False

#----------------------------------------------
#2. Data Acquisition

def slugify(text: str) -> str:
    """Turn a label into a filename-friendly string."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

def build_raw_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}.json"

def build_open_meteo_params(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: Sequence[str],
    timezone: str,
) -> dict:
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone,
    }

def ensure_output_directories(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

def download_open_meteo(city: dict) -> dict:
    """Download one fixed historical period from Open-Meteo."""
    params = build_open_meteo_params(
        latitude=float(city["latitude"]),
        longitude=float(city["longitude"]),
        start_date=START_DATE,
        end_date=END_DATE,
        hourly_variables=FEATURES,
        timezone=TIMEZONE,
    )
    print(f"      Downloading {city['name']} ({START_DATE} -> {END_DATE})...")
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Open-Meteo request failed: {exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo returned invalid JSON.") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Open-Meteo API error: {payload.get('reason', 'unknown reason')}")
    return payload


def load_or_download_raw(
    raw_path: Path,
    downloader: Callable[[], dict],
    force_download: bool = False,
) -> dict:
    """COLLECT ONCE -> SAVE RAW DATA -> REPLAY MANY TIMES."""
    raw_path = Path(raw_path)
    if raw_path.exists() and not force_download:
        print(f"      Reusing raw snapshot: {raw_path.name}")
        return json.loads(raw_path.read_text(encoding="utf-8"))

    payload = downloader()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      Raw response saved to: {raw_path}")
    return payload


def load_city_payload(city: dict, raw_dir: Path = RAW_DIR) -> tuple[Path, dict]:
    raw_path = raw_dir / build_raw_filename(city["name"], START_DATE, END_DATE)
    payload = load_or_download_raw(
        raw_path,
        downloader=lambda: download_open_meteo(city),
        force_download=FORCE_DOWNLOAD,
    )
    return raw_path, payload

# ----------------------------------------------
# 3. Data Preparation

def build_processed_filename(city_name: str, start_date: str, end_date: str) -> str:
    return f"open_meteo_{slugify(city_name)}_{start_date}_{end_date}.csv"

def prepare_city_frame(payload: dict, city_name: str, features: Sequence[str]) -> pd.DataFrame:
    """Validate one Open-Meteo payload and return a chronological table."""
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Open-Meteo payload is missing hourly.time")

    missing = [feature for feature in features if feature not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo payload is missing required features: {missing}")

    data = {"timestamp": pd.to_datetime(hourly["time"], errors="coerce")}
    for feature in features:
        data[feature] = pd.to_numeric(pd.Series(hourly[feature]), errors="coerce")

    df = pd.DataFrame(data)
    df["city"] = city_name
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", *features, "city"]]


def check_data_quality(df: pd.DataFrame, features: Sequence[str]) -> None:
    """Print the data quality evidence."""
    city = df["city"].iloc[0]
    print(f"\nData quality - {city}")
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("  Missing values per column:")
    print(df[["timestamp", *features]].isna().sum().to_string())
    print(f"  Duplicated timestamps: {df['timestamp'].duplicated().sum()}")

    steps = df["timestamp"].diff().dropna()
    irregular = steps[steps != pd.Timedelta(hours=1)]
    print(f"  Time steps different from 1 hour: {len(irregular)}")

    humidity = df["relative_humidity_2m"]
    print(f"  Humidity outside [0, 100]: {((humidity < 0) | (humidity > 100)).sum()}")
    print(f"  Negative precipitation: {(df['precipitation'] < 0).sum()}")
    print(f"  Negative wind speed: {(df['wind_speed_10m'] < 0).sum()}")

#------------------------------------
# 4. Stream profile for both cities

# 7 day window to make the hourly series readable
ROLLING_WINDOW_HOURS = 24 * 7

def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path

def plot_temporal_profile(
    df: pd.DataFrame,
    features: Sequence[str] = FEATURES,
    window: int = ROLLING_WINDOW_HOURS,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot each weather variable over time with a rolling mean."""
    city = df["city"].iloc[0]
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 10), sharex=True)

    for ax, feature in zip(axes, features):
        ax.plot(df["timestamp"], df[feature], linewidth=0.5, alpha=0.4, label="Hourly")
        ax.plot(
            df["timestamp"],
            df[feature].rolling(window).mean(),
            linewidth=1.6,
            label=f"{window // 24}-day rolling mean",
        )
        ax.set_ylabel(feature)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelbottom=True, labelrotation=90, labelsize=7)

    axes[0].set_title(f"{city} - Weather variables over time (°C, %, mm, km/h)")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time")
    return _save_figure(figures_dir / f"{slugify(city)}_temporal_profile.png")

# descriptive statistics 
def summarise_city(
    df: pd.DataFrame,
    features: Sequence[str] = FEATURES,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Descriptive statistics and timestamps of the extremes."""
    city = df["city"].iloc[0]
    summary = df[list(features)].describe().T.round(2)
    summary["min_at"] = [df.loc[df[f].idxmin(), "timestamp"] for f in features]
    summary["max_at"] = [df.loc[df[f].idxmax(), "timestamp"] for f in features]

    rainy_hours = int((df["precipitation"] >= RAIN_THRESHOLD_MM).sum())
    print(f"\nDescriptive summary - {city}")
    print(summary.to_string())
    print(
        f"  Hours with precipitation >= {RAIN_THRESHOLD_MM} mm: "
        f"{rainy_hours:,} ({rainy_hours / len(df):.1%})"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / f"{slugify(city)}_summary.csv")
    return summary


def top_extremes(df: pd.DataFrame, feature: str, n: int = 5) -> pd.DataFrame:
    """Return the n highest hourly values of one variable."""
    return df.nlargest(n, feature)[["timestamp", feature]]

def plot_hourly_profile(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot the share of rainy hours and the mean wind speed by hour of day."""
    city = df["city"].iloc[0]
    hours = df["timestamp"].dt.hour

    # group the hours into 24 groups and calculate the share of hours with rain and the mean wind speed
    rain_share = (df["precipitation"] >= RAIN_THRESHOLD_MM).groupby(hours).mean()
    wind_mean = df["wind_speed_10m"].groupby(hours).mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(rain_share.index, rain_share.values)
    axes[0].set_ylabel(f"Share of hours with rain (>= {RAIN_THRESHOLD_MM} mm)")
    axes[0].set_title(f"{city} - hour-of-day profile")
    axes[1].plot(wind_mean.index, wind_mean.values, marker="o")
    axes[1].set_ylabel("Mean wind speed (km/h)")
    axes[1].set_xlabel("Hour of day (local time)")
    axes[1].set_xticks(range(24))
    for ax in axes:
        ax.grid(alpha=0.2)
    return _save_figure(figures_dir / f"{slugify(city)}_hourly_profile.png")

def plot_rolling_behaviour(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW_HOURS,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot rolling variability of wind and rolling share of rainy hours"""
    city = df["city"].iloc[0]
    wind = df["wind_speed_10m"]
    rainy = (df["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["timestamp"], wind.rolling(window).mean(), label="Rolling mean")
    axes[0].plot(df["timestamp"], wind.rolling(window).std(), label="Rolling std")
    axes[0].set_ylabel("Wind speed (km/h)")
    axes[0].set_title(f"{city} - {window // 24}-day rolling behaviour")
    axes[0].legend(loc="best")
    axes[1].plot(df["timestamp"], rainy.rolling(window).mean())
    axes[1].set_ylabel(f"Share of rainy hours (>= {RAIN_THRESHOLD_MM} mm)")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelbottom=True, labelrotation=90, labelsize=7)
    return _save_figure(figures_dir / f"{slugify(city)}_rolling_behaviour.png")


def adverse_episode_durations(df: pd.DataFrame) -> pd.Series:
    """Length of each run of consecutive T6 hours (y = 1)."""
    run_id = (df["y"] != df["y"].shift()).cumsum()
    runs = df.groupby(run_id)["y"].agg(["first", "size"])
    return runs.loc[runs["first"] == 1, "size"]


def plot_episode_durations(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Plot the distribution of adverse-episode durations for one city."""
    city = df["city"].iloc[0]
    durations = adverse_episode_durations(df)
    print(f"\nAdverse episodes - {city}")
    print(
        f"  Episodes: {len(durations):,} | mean {durations.mean():.1f} h "
        f"| median {durations.median():.0f} h | longest {durations.max()} h"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=range(1, int(durations.max()) + 2))
    ax.set_xlabel("Episode duration (consecutive hours with T6 = 1)")
    ax.set_ylabel("Number of episodes")
    ax.set_title(f"{city} - duration of adverse episodes")
    ax.grid(alpha=0.2)
    return _save_figure(figures_dir / f"{slugify(city)}_episode_durations.png")

#---------------------------------------
#5. Warm-up and target T6

def get_warmup_end(city_frames: dict[str, pd.DataFrame], warmup_fraction: float) -> pd.Timestamp:
    """Last timestamp of the warm-up, shared by both cities"""
    timestamps = sorted(pd.concat([f["timestamp"] for f in city_frames.values()]).unique())
    n_train_timestamps = int(len(timestamps) * warmup_fraction)
    n_train_timestamps = max(1, min(len(timestamps) - 1, n_train_timestamps))
    return pd.Timestamp(timestamps[n_train_timestamps - 1])

def compute_wind_threshold(
    city_frames: dict[str, pd.DataFrame],
    warmup_end: pd.Timestamp,
    quantile: float,
) -> float:
    """Wind speed quantile over the pooled warm-up of both cities."""
    pooled = pd.concat(
        [f.loc[f["timestamp"] <= warmup_end, "wind_speed_10m"] for f in city_frames.values()]
    )
    return float(pooled.quantile(quantile))


def add_target_t6(df: pd.DataFrame, wind_threshold: float) -> pd.DataFrame:
    """Add T6 target: rain (T1) or strong wind (T5)"""
    df = df.copy()
    df["rain_event"] = (df["precipitation"] >= RAIN_THRESHOLD_MM).astype(int)
    df["wind_event"] = (df["wind_speed_10m"] >= wind_threshold).astype(int)
    df["y"] = df["rain_event"] | df["wind_event"]
    return df


def report_target_frequency(df: pd.DataFrame, warmup_end: pd.Timestamp) -> None:
    """Report of share of rain, strong wind and T6 hours in warm-up and evaluation"""
    city = df["city"].iloc[0]
    in_warmup = df["timestamp"] <= warmup_end
    print(f"\nTarget frequency - {city}")
    for label, mask in (("Warm-up", in_warmup), ("Evaluation", ~in_warmup)):
        part = df[mask]
        print(
            f"  {label}: {len(part):,} hours | rain {part['rain_event'].mean():.1%} "
            f"| strong wind {part['wind_event'].mean():.1%} | T6 {part['y'].mean():.1%}"
        )

#---------------------------------------
# 6. Features (past info only)

# Representation A: previous-state features (t-1)
FEATURES_A = [f"{f}_lag1" for f in FEATURES]
# Representation B: A + recent temporal context (all windows end at t-1)
FEATURES_B = FEATURES_A + [
    "temperature_delta",
    "temperature_mean_6h",
    "wind_std_6h",
    "rain_hours_24h",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in FEATURES:
        df[f"{feature}_lag1"] = df[feature].shift(1)

    temperature = df["temperature_2m"].shift(1)
    df["temperature_delta"] = temperature - df["temperature_2m"].shift(2)
    df["temperature_mean_6h"] = temperature.rolling(WINDOW_SHORT).mean()
    df["wind_std_6h"] = df["wind_speed_10m"].shift(1).rolling(WINDOW_SHORT).std()
    df["rain_hours_24h"] = df["rain_event"].shift(1).rolling(WINDOW_LONG).sum()
    return df


def verify_features(df: pd.DataFrame, i: int = 100) -> None:
    """Recompute the features of one row from earlier hours only and check they match."""
    temperature = df["temperature_2m"]
    wind = df["wind_speed_10m"]
    for feature in FEATURES:
        assert df[f"{feature}_lag1"].iloc[i] == df[feature].iloc[i - 1]
    assert abs(df["temperature_delta"].iloc[i] - (temperature.iloc[i - 1] - temperature.iloc[i - 2])) < 1e-9
    assert abs(df["temperature_mean_6h"].iloc[i] - temperature.iloc[i - WINDOW_SHORT : i].mean()) < 1e-9
    assert abs(df["wind_std_6h"].iloc[i] - wind.iloc[i - WINDOW_SHORT : i].std()) < 1e-9
    assert df["rain_hours_24h"].iloc[i] == df["rain_event"].iloc[i - WINDOW_LONG : i].sum()
#---------------------------------
#7. Models

def build_city_events(df: pd.DataFrame, feature_names: Sequence[str]) -> list[dict]:
    """Turn one city table into chronological events (x = features, y = T6)."""
    return [
        {
            "timestamp": row.timestamp,
            "x": {f: float(getattr(row, f)) for f in feature_names},
            "y": int(row.y),
        }
        for row in df.sort_values("timestamp").itertuples(index=False)
    ]

def split_events(events: Sequence[dict], warmup_end: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    warm = [e for e in events if e["timestamp"] <= warmup_end]
    evaluation = [e for e in events if e["timestamp"] > warmup_end]
    return warm, evaluation


def fit_shared_scaler(warm_events: Sequence[dict], feature_names: Sequence[str]) -> MinMaxScaler:
    X = np.asarray([[e["x"][f] for f in feature_names] for e in warm_events], dtype=float)
    return MinMaxScaler(feature_range=(0.0, 1.0), clip=True).fit(X)


def transform_events(events: Sequence[dict], scaler: MinMaxScaler, feature_names: Sequence[str]) -> list[dict]:
    names = list(feature_names)
    X = np.asarray([[e["x"][f] for f in names] for e in events], dtype=float)
    Xt = scaler.transform(X)
    return [
        {
            "timestamp": e["timestamp"],
            "x": {f"x{i}": float(row[i]) for i in range(len(names))},
            "y": int(e["y"]),
        }
        for e, row in zip(events, Xt)
    ]

def _river_base_tree() -> tree.HoeffdingTreeClassifier:
    return tree.HoeffdingTreeClassifier(
        grace_period=ENSEMBLE_GRACE_PERIOD,
        delta=ENSEMBLE_DELTA,
        tau=ENSEMBLE_TAU,
        leaf_prediction="mc",
    )

def make_models(seed: int = RANDOM_SEED) -> dict:
    return {
        "Gaussian Naive Bayes": naive_bayes.GaussianNB(),
        "Online Logistic Regression": linear_model.LogisticRegression(optimizer=optim.SGD(0.05)),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(
            grace_period=HOEFFDING_GRACE_PERIOD,
            delta=HOEFFDING_DELTA,
            tau=HOEFFDING_TAU,
            leaf_prediction="mc",
        ),
        "Online Bagging": ensemble.BaggingClassifier(
            model=_river_base_tree(), n_models=ENSEMBLE_SIZE, seed=seed
        ),
        "Adaptive Random Forest": forest.ARFClassifier(
            n_models=ENSEMBLE_SIZE,
            max_features="sqrt",
            lambda_value=6,
            leaf_prediction="mc",
            grace_period=ENSEMBLE_GRACE_PERIOD,
            delta=ENSEMBLE_DELTA,
            tau=ENSEMBLE_TAU,
            seed=seed,
        ),
    }

#----------------------------------------------
# 8. Prequential evaluation (PREDICT -> EVALUATE -> LEARN)

def _metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    p1 = tp / (tp + fp) if (tp + fp) else 0.0
    r1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) else 0.0
    p0 = tn / (tn + fn) if (tn + fn) else 0.0
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) else 0.0
    return float(accuracy), float((f1_0 + f1_1) / 2)


class StreamingBinaryEvaluator:
    """Incremental confusion matrix; the positive class (1) is the T6 event."""

    def __init__(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self.cold_starts = 0

    def update(self, y_true: int, y_pred: int | None) -> None:
        if y_pred is None:
            self.cold_starts += 1
            return
        y_true, y_pred = int(y_true), int(y_pred)
        if y_true == 1 and y_pred == 1: self.tp += 1
        elif y_true == 0 and y_pred == 0: self.tn += 1
        elif y_true == 0 and y_pred == 1: self.fp += 1
        elif y_true == 1 and y_pred == 0: self.fn += 1

    def summary(self) -> dict:
        accuracy, macro_f1 = _metrics_from_counts(self.tp, self.tn, self.fp, self.fn)
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp,
            "evaluated": self.tp + self.tn + self.fp + self.fn,
            "cold_starts": self.cold_starts,
        }


def warm_up_model(model, events):
    """Learn the shared warm-up without scoring it."""
    for event in events:
        model.learn_one(event["x"], event["y"])
    return model


def run_prequential(events, model) -> dict:
    """Run one model over the evaluation stream: predict, evaluate, then learn."""
    evaluator = StreamingBinaryEvaluator()
    prediction_seconds = update_seconds = 0.0
    for event in events:
        t0 = time.perf_counter()
        y_pred = model.predict_one(event["x"])            # 1- predict
        prediction_seconds += time.perf_counter() - t0
        evaluator.update(event["y"], y_pred)              # 2- evaluate
        t1 = time.perf_counter()
        model.learn_one(event["x"], event["y"])           # 3- learn
        update_seconds += time.perf_counter() - t1
    summary = evaluator.summary()
    n_events = max(1, len(events))
    summary["prediction_ms/event"] = 1000 * prediction_seconds / n_events
    summary["update_ms/event"] = 1000 * update_seconds / n_events
    return summary


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def save_configuration(wind_threshold: float, warmup_end: pd.Timestamp, results_dir: Path = RESULTS_DIR) -> None:
    """Save every parameter needed to reproduce the experiment."""
    config = {
        "assignment": "PA30 (T6)",
        "city_a": CITY_A["name"],
        "city_b": CITY_B["name"],
        "period": f"{START_DATE} to {END_DATE}",
        "timezone": TIMEZONE,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_end": str(warmup_end),
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "wind_quantile": WIND_QUANTILE,
        "wind_threshold_kmh": round(wind_threshold, 4),
        "window_short_h": WINDOW_SHORT,
        "window_long_h": WINDOW_LONG,
        "random_seed": RANDOM_SEED,
        "hoeffding_grace_period": HOEFFDING_GRACE_PERIOD,
        "hoeffding_delta": HOEFFDING_DELTA,
        "hoeffding_tau": HOEFFDING_TAU,
        "ensemble_size": ENSEMBLE_SIZE,
        "ensemble_grace_period": ENSEMBLE_GRACE_PERIOD,
        "ensemble_delta": ENSEMBLE_DELTA,
        "ensemble_tau": ENSEMBLE_TAU,
        "python": sys.version.split()[0],
    }
    for package in ("river", "scikit-learn", "pandas", "numpy", "matplotlib", "requests"):
        config[f"{package}_version"] = package_version(package)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(config.items()), columns=["parameter", "value"]).to_csv(
        results_dir / "configurations.csv", index=False
    )

#----------------------------------------------
# 9. Results figures 

def plot_model_comparison(metrics: pd.DataFrame, city: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Macro-F1 and event-class recall of the five models, Representation A vs B."""
    data = metrics[metrics["city"] == city]
    models = list(make_models())
    x = np.arange(len(models), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (column, label) in zip(axes, (("macro_f1", "Macro-F1"), ("recall", "Recall (event class)"))):
        for offset, rep in ((-width / 2, "A"), (width / 2, "B")):
            values = data[data["representation"] == rep].set_index("model").loc[models, column]
            ax.bar(x + offset, values.values, width, label=f"Representation {rep}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(label)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="best")
    fig.suptitle(f"{city} - predictive comparison under one common protocol")
    return _save_figure(figures_dir / f"{slugify(city)}_model_comparison.png")


def plot_processing_cost(metrics: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Prediction and update cost per event (mean over cities and representations)."""
    models = list(make_models())
    cost = metrics.groupby("model")[["prediction_ms/event", "update_ms/event"]].mean().loc[models]
    x = np.arange(len(models), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, cost["prediction_ms/event"], width, label="Prediction ms/event")
    ax.bar(x + width / 2, cost["update_ms/event"], width, label="Update ms/event")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Milliseconds per event (log scale)")
    ax.set_title("Computational cost per event - environment dependent")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    return _save_figure(figures_dir / "processing_cost.png")

def plot_monthly_t6_risk(city_frames: dict[str, pd.DataFrame], figures_dir: Path = FIGURES_DIR) -> Path:
    """Original visualization 1: Monthly adverse event (T6) rate comparison between cities."""
    fig, ax = plt.subplots(figsize=(11, 5))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)
    width = 0.38
    colors = {CITY_A["name"]: "#1f77b4", CITY_B["name"]: "#ff7f0e"}

    for offset, (city_name, df) in zip((-width / 2, width / 2), city_frames.items()):
        monthly_rates = df.groupby(df["timestamp"].dt.month)["y"].mean()
        rates = [monthly_rates.get(m, 0.0) * 100 for m in x]
        ax.bar(x + offset, rates, width, label=city_name, color=colors.get(city_name, None))

    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("Hours with adverse event T6 (%)")
    ax.set_xlabel("Month of the Year")
    ax.set_title("Monthly Comparison of Operational Meteorological Risk (T6) - Seasonal Fair")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    return _save_figure(figures_dir / "comparative_monthly_risk.png")

def save_monthly_t6_risk(city_frames: dict[str, pd.DataFrame], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Save and print the monthly T6 rate (%) per city: the numbers behind the monthly risk figure."""
    table = pd.DataFrame(
        {
            city_name: df.groupby(df["timestamp"].dt.month)["y"].mean().mul(100).round(1)
            for city_name, df in city_frames.items()
        }
    )
    table.index.name = "month"
    table.to_csv(Path(results_dir) / "monthly_t6_risk.csv")

    print("\nMonthly T6 risk (%)")
    print(table.to_string())
    for city_name in table.columns:
        low, high = table[city_name].idxmin(), table[city_name].idxmax()
        print(
            f"  {city_name}: lowest month {low} ({table[city_name].min():.1f}%) | "
            f"highest month {high} ({table[city_name].max():.1f}%)"
        )
    return table



def save_target_frequency(city_frames: dict[str, pd.DataFrame], warmup_end: pd.Timestamp) -> None:
    """Save the share of rain, strong wind and T6 hours per city and period."""
    rows = []
    for city, df in city_frames.items():
        in_warmup = df["timestamp"] <= warmup_end
        for period, mask in (("warm-up", in_warmup), ("evaluation", ~in_warmup)):
            part = df[mask]
            rows.append({
                "city": city,
                "period": period,
                "hours": len(part),
                "rain_share": part["rain_event"].mean(),
                "strong_wind_share": part["wind_event"].mean(),
                "t6_share": part["y"].mean(),
            })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "target_frequency.csv", index=False)

#----------------------------------------------
# 10. Main experiment

def run_all_experiments() -> None:
    """Run the experiment, from raw data to results and figures."""
    ensure_output_directories()

    # 1- data: load, prepare and check both cities
    city_frames: dict[str, pd.DataFrame] = {}
    for city in (CITY_A, CITY_B):
        raw_path, payload = load_city_payload(city)
        df = prepare_city_frame(payload, city["name"], FEATURES)
        check_data_quality(df, FEATURES)

        out_csv = PROCESSED_DIR / build_processed_filename(city["name"], START_DATE, END_DATE)
        df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv.name}")
        city_frames[city["name"]] = df

    same_timestamps = city_frames[CITY_A["name"]]["timestamp"].equals(
        city_frames[CITY_B["name"]]["timestamp"]
    )
    print(f"\nBoth cities share the same timestamps: {same_timestamps}")

    # 2- warm-up, wind threshold and target T6
    warmup_end = get_warmup_end(city_frames, WARMUP_FRACTION)
    wind_threshold = compute_wind_threshold(city_frames, warmup_end, WIND_QUANTILE)
    print(f"Warm-up ends at: {warmup_end}")
    print(f"Wind threshold (frozen, quantile {WIND_QUANTILE}): {wind_threshold:.2f} km/h")

    for name in city_frames:
        city_frames[name] = add_target_t6(city_frames[name], wind_threshold)
        report_target_frequency(city_frames[name], warmup_end)
    save_target_frequency(city_frames, warmup_end)

    # 3- features A and B (past info only)
    model_frames: dict[str, pd.DataFrame] = {}
    for name in city_frames:
        featured = add_features(city_frames[name])
        # the first 24 rows have no complete past, so they are dropped
        model_frames[name] = featured.dropna(subset=FEATURES_B)
        verify_features(model_frames[name])
        print(
            f"{name}: {len(model_frames[name]):,} rows with complete features "
            f"({len(featured) - len(model_frames[name])} initial rows dropped)"
        )

    # 4- prequential evaluation: one stream per city and representation
    representations = {"A": FEATURES_A, "B": FEATURES_B}
    results = []
    for name, frame in model_frames.items():
        for rep, cols in representations.items():
            events = build_city_events(frame, cols)
            warm, evaluation = split_events(events, warmup_end)
            # scaler fitted on the warm-up only, then applied to everything
            scaler = fit_shared_scaler(warm, cols)
            warm_t = transform_events(warm, scaler, cols)
            evaluation_t = transform_events(evaluation, scaler, cols)

            # fresh models for every city and representation
            for model_name, model in make_models().items():
                warm_up_model(model, warm_t)
                summary = run_prequential(evaluation_t, model)
                results.append({"city": name, "representation": rep, "model": model_name, **summary})
                print(
                    f"{name} | {rep} | {model_name}: accuracy {summary['accuracy']:.3f} "
                    f"| macro-F1 {summary['macro_f1']:.3f} "
                    f"| precision {summary['precision']:.3f} | recall {summary['recall']:.3f}"
                )

    metrics = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    save_configuration(wind_threshold, warmup_end)
    print("\nSaved: metrics.csv, configurations.csv and target_frequency.csv")

    # 5- results figures
    for city_name in (CITY_A["name"], CITY_B["name"]):
        path = plot_model_comparison(metrics, city_name)
        print(f"Figure saved: {path.name}")
    print(f"Figure saved: {plot_processing_cost(metrics).name}")



    # original visualizations for the assignment
    print(f"Figure saved: {plot_monthly_t6_risk(city_frames).name}")
    save_monthly_t6_risk(city_frames)

    # 6- stream profile of each city 
    for city in (CITY_A, CITY_B):
        df = city_frames[city["name"]]
        summarise_city(df)
        for feature in ("wind_speed_10m", "precipitation"):
            print(f"\nTop 5 hourly values - {city['name']} - {feature}")
            print(top_extremes(df, feature).to_string(index=False))

        figures = [
            plot_temporal_profile(df),
            plot_hourly_profile(df),
            plot_rolling_behaviour(df),
            plot_episode_durations(df),
        ]
        print(f"  Figures saved: {', '.join(path.name for path in figures)}")


def main() -> None:
    np.random.seed(RANDOM_SEED)
    print("Practical Assignment 1 - PA30 Data Stream Mining")
    print(
        f"Python {sys.version.split()[0]} | river {package_version('river')} "
        f"| scikit-learn {package_version('scikit-learn')}"
    )
    run_all_experiments()


if __name__ == "__main__":
    main()