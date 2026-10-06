""" 
Data Stream Mining Practical Assignment 1
PA30, T6- Adverse Outdoor Event
Cities: Viana do Castelo Vs Guarda
Business Scenario: Seasonal Outdoor Fair
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence
import json
import requests
import re
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

# target T6: strong rain or strong wind
RAIN_THRESHOLD_MM = 0.1   # T1: y = 1 if precipitation >= 0.1 mm
WIND_QUANTILE = 0.80      # T5: threshold = 0.80 quantile of wind in the pooled warm-up

#warm-up fraction 
WARMUP_FRACTION = 0.25

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

# turn a label into a filename-friendly string
def slugify(text: str) -> str:
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

# download one fixed historical period from Open-Meteo
def download_open_meteo(city: dict) -> dict:
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
            "Could not download Open-Meteo data. Check the connection or reuse "
            "a saved raw snapshot with FORCE_DOWNLOAD=False."
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
def main() -> None:
    ensure_output_directories()
    for city in (CITY_A, CITY_B):
        raw_path, payload = load_city_payload(city)
        n_hours = len(payload["hourly"]["time"])
        print(f"{city['name']}: {n_hours} hourly observations")


if __name__ == "__main__":
    main()