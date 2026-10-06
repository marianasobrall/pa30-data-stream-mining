""" 
Data Stream Mining Practical Assignment 1
PA30, T6- Adverse Outdoor Event
Cities: Viana do Castelo Vs Guarda
Business Scenario: Seasonal Outdoor Fair
"""

from __future__ import annotations

from pathlib import Path

#----------------------------------------------
# 1. Configuration
#----------------------------------------------


#confirmar as coordenadas das cidades!!!
CITY_A = {"name": "Viana do Castelo", "latitude": 41.693, "longitude": -8.832}
CITY_B = {"name": "Guarda", "latitude": 40.537, "longitude": -7.268}

# historical period
START_DATE = "2025-01-01"
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

#warm-up fraction confirmar!!!!
WARMUP_FRACTION = 0.25

#default random seed from the template
RANDOM_SEED = 42

# setup directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# open meteo archive url
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORCE_DOWNLOAD = False

def main() -> None:
    print(f"Cities: {CITY_A['name']} | {CITY_B['name']}")
    print(f"Period: {START_DATE} -> {END_DATE}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"Project root: {BASE_DIR}")


if __name__ == "__main__":
    main()

