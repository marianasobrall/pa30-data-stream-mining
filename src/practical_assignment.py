""" 
Data Stream Mining Practical Assignment 1
PA30, T6- Adverse Outdoor Event
Cities: Viana do Castelo Vs Guarda
Business Scenario: Seasonal Outdoor Fair
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
import requests
import re
import pandas as pd

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

# target T6: (T1 or T5)strong rain or strong wind
RAIN_THRESHOLD_MM = 0.1   # T1: y = 1 if precipitation >= 0.1 mm
WIND_QUANTILE = 0.80      # T5: threshold = 0.80 quantile of wind in the pooled warm-up

#warm-up fraction 
WARMUP_FRACTION = 0.25

# feature windows (hours)
WINDOW_SHORT = 6
WINDOW_LONG = 24

#default random seed from the template
RANDOM_SEED = 42

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

# 7 day rolling window - make the hourly series readable
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

    #group the hours into 24 groups and calculate the share of hours with rain and the mean wind speed
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
    past = df["temperature_2m"].iloc[i - WINDOW_SHORT : i]
    assert df["temperature_2m_lag1"].iloc[i] == df["temperature_2m"].iloc[i - 1]
    assert abs(df["temperature_mean_6h"].iloc[i] - past.mean()) < 1e-9
    assert df["rain_hours_24h"].iloc[i] == df["rain_event"].iloc[i - WINDOW_LONG : i].sum()


def main() -> None:
    ensure_output_directories()
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

    warmup_end = get_warmup_end(city_frames, WARMUP_FRACTION)
    wind_threshold = compute_wind_threshold(city_frames, warmup_end, WIND_QUANTILE)
    print(f"Warm-up ends at: {warmup_end}")
    print(f"Wind threshold (frozen, quantile {WIND_QUANTILE}): {wind_threshold:.2f} km/h")

    for name in city_frames:
        city_frames[name] = add_target_t6(city_frames[name], wind_threshold)
        report_target_frequency(city_frames[name], warmup_end)


    model_frames: dict[str, pd.DataFrame] = {}
    for name in city_frames:
        featured = add_features(city_frames[name])
        model_frames[name] = featured.dropna(subset=FEATURES_B)
        verify_features(model_frames[name])
        print(f"{name}: {len(model_frames[name]):,} rows with complete features "
              f"({len(featured) - len(model_frames[name])} initial rows dropped)")
        

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
        ]
        print(f"  Figures saved: {', '.join(path.name for path in figures)}")


if __name__ == "__main__":
    main()