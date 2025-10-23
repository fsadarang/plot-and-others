# rain_extract_multi_station_combined.py
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar, ll_to_xy, to_np

# --- User settings ---
WRF_DIRECTORY = "/scratch/riset_2/output/wrfout/analisis/spu_12/28082025_12/3dvar_rv"
WRF_FILES = ['wrfout_d03_2025-08-28_12:00:00']  # list your files here
outdir = os.path.join(WRF_DIRECTORY, "plot")

# Time settings
start_index = 72
num_plots = 144
interval = 6  # adjust for your timestep spacing

# --- List of coordinates (lat, lon) ---
locations = [
    (-6.2956944, 106.4755556),
    (-6.17247, 107.18012),
    (-6.24974, 106.99718),
    (-5.984693, 107.042771),
    (-6.2331047, 106.7931924),
    (-6.116474, 106.584349),
    (-6.131808, 106.957408),
    (-6.30567, 106.96229),
    (-6.072986, 107.143099),
    (-6.16733, 107.04884),
    (-6.12492, 107.06328),
    (-6.3397, 107.03978),
    (-6.22782, 107.09377),
    (-6.127633, 106.3802),
    (-6.552729, 106.538597),
    (-6.37191, 106.82762),
    (-6.49821, 107.12513),
    (-6.55324, 106.74283),
    (-6.20748, 106.8487),
    (-6.3443, 106.799),
    (-6.16666, 106.8809),
    (-6.304318, 106.7774),
    (-6.166699, 106.78),
    (-6.166633, 106.9136),
    (-6.63306, 106.83682),
    (-6.37, 107.18103),
    (-6.06467902, 106.5302122),
    (-6.72252, 106.60391),
    (-6.64443, 106.69378),
    (-6.73858, 106.83864),
    (-6.48438, 106.83848),
    (-6.5782875, 107.1397344),
    (-6.65493, 106.88076),
    (-6.68664, 106.99525),
    (-6.62387, 106.49531),
    (-6.305489, 106.899662),
    (-6.46052, 106.86946),
    (-6.600471, 106.8054),
    (-6.116237, 106.745451),
    (-6.2792, 106.6503),
    (-6.1978, 106.6448),
    (-6.6979, 106.935),
    (-6.26147016, 106.7509349),
    (-6.26248, 106.8971),
    (-6.121074, 106.838433),
    (-6.3025406, 106.7563266),
    (-6.43199, 107.19197),
    (-6.287308, 106.5677384),
    (-6.12523, 106.6581),
    (-6.155415, 106.842308),
    (-6.2651616, 106.74869),
    (-6.71072, 106.95022),
    (-6.172, 106.647),
    (-6.55687, 106.86832),
    (-6.69814, 106.93503),
    (-6.60785, 106.79298),
    (-6.50457, 107.06525),
    (-6.156485, 106.418242),
    (-6.60036, 106.7962),
    (-6.2952608, 106.8194739),
    (-6.174436, 106.732117),
    (-6.356835, 106.890804),
    (-6.15559, 106.84)
]

# --- Station IDs (must match order) ---
station_ids = [
    "30001", "30004", "30005", "30006", "30009", "30011", "30013", "30014", "30015", "30016",
    "30017", "30018", "30019", "150088", "150294", "160033", "14032802", "30010", "STA0026",
    "STA0027", "STA0028", "STA0029", "STA0030", "STA0031", "STA0037", "STA0051", "STA0146",
    "STA0236", "STA0237", "STA0238", "STA0239", "STA0241", "STA0251", "STA0253", "STA0254",
    "STA2041", "STA2043", "STA2045", "STA2046", "STA2047", "STA2048", "STA2053", "STA2062",
    "STA2065", "STA2172", "STA2290", "STA3007", "STA5062", "STA5082", "STA9001", "STA9003",
    "STA9004", "STA9007", "STA9008", "STA9009", "STA9010", "STA9011", "STAL111", "STAL131",
    "STG1011", "STG1012", "STW1031", "STW1071"
]

# --- Load WRF file ---
file_path = os.path.join(WRF_DIRECTORY, WRF_FILES[0])
wrfin = Dataset(file_path)

# --- Collect all stations' data ---
all_data = []

for (lat, lon), station_id in zip(locations, station_ids):
    print(f"Extracting rainfall for {station_id} at lat={lat}, lon={lon}")

    prev_rain = None

    for i in range(start_index, num_plots, interval):
        # Accumulated rain
        rainc = getvar(wrfin, "RAINC", timeidx=i)
        rainnc = getvar(wrfin, "RAINNC", timeidx=i)
        rainsh = getvar(wrfin, "RAINSH", timeidx=i)
        rain_accum = rainc + rainnc + rainsh

        x, y = ll_to_xy(wrfin, lat, lon)
        rain_point = float(to_np(rain_accum[y, x]))

        # Hourly rainfall difference
        rain_hourly = rain_point - prev_rain if prev_rain is not None else 0.0
        prev_rain = rain_point

        # Timestamp
        time_val = getvar(wrfin, "Times", timeidx=i)
        formatted_time = pd.to_datetime(str(to_np(time_val)))

        all_data.append({
            "station_id": station_id,
            "lat": lat,
            "lon": lon,
            "time": formatted_time,
            "rain_mm_per_hr": rain_hourly
        })

# --- Combine into one DataFrame and save ---
df_all = pd.DataFrame(all_data)
df_all.sort_values(["station_id", "time"], inplace=True)

os.makedirs(outdir, exist_ok=True)
out_csv = os.path.join(outdir, "rain_all_stations_combined.csv")
df_all.to_csv(out_csv, index=False)

print(f"\n✅ Combined rainfall file saved to: {out_csv}")

