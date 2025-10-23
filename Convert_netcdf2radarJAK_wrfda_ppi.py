# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:01:06 2025

@author: Abdul Hamid Al Habib
"""

import os
import numpy as np
import netCDF4 as nc
from datetime import datetime

# =========================
# Konfigurasi awal
# =========================
WORKDIR = "/home/riset_2/WRF/WRFDA_Script/"
os.chdir(WORKDIR)

radar_name = "JAK"
radar_lon = 106.646
radar_lat = -6.171
radar_alt = 25

# Cari file NC dan CTL
nczfilelist = [f for f in os.listdir(WORKDIR) if f.endswith("_dbz.nc")]
ncvfilelist = [f for f in os.listdir(WORKDIR) if f.endswith("_radv.nc")]
ctlfilelist = [f for f in os.listdir(WORKDIR) if f.endswith(".ctl")]

# =========================
# Fungsi bantu
# =========================
def read_ctl_coordinates(filename):
    with open(filename, "r") as f:
        all_lines = f.readlines()
    # Ambil hanya baris yang diawali spasi
    get_lines = [ln.strip() for ln in all_lines if ln.startswith(" ")]
    get_lines = [float(x) for x in get_lines]
    half = len(get_lines) // 2
    lonc = np.array(get_lines[:half])
    latc = np.array(get_lines[half:])
    return lonc, latc

# =========================
# Loop semua ctl file
# =========================
for ctlfile in ctlfilelist:
    lonc, latc = read_ctl_coordinates(ctlfile)
    nlonc = len(lonc)
    nlatc = len(latc)

    for ncfname, nc1fname in zip(nczfilelist, ncvfilelist):
        print(f"Reading {ncfname} and {nc1fname} for domain {ctlfile} ...")

        # buat tanggal dari nama file
        #strdate = f"{ncfname[3:7]}-{ncfname[7:9]}-{ncfname[9:11]}_{ncfname[11:13]}:{ncfname[13:15]}:00"
        basename = os.path.basename(ncfname)
        
        # Asumsikan nama file radar = JAK-20250823-234615-PPIVol_dbz.nc
        # atau format serupa dengan YYYYMMDDHHMMSS di posisi tengah
        try:
            parts = basename.split("-")
            date_str = parts[1]  # '20250823'
            time_str = parts[2]  # '234615'
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except Exception:
            # Jika format tidak ada tanda "-", fallback baca langsung dari awal nama file
            date_str = basename[0:8]   # contoh: 20250824
            time_str = basename[8:14]  # contoh: 000425
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        
        # Format waktu sesuai WRF
        strdate = dt.strftime("%Y-%m-%d_%H:%M:%S")

        # buka netCDF
        ds = nc.Dataset(ncfname)
        ds1 = nc.Dataset(nc1fname)

        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]
        lev = ds.variables["lev"][:]

        dbz_array = ds.variables["dBZ"][:]
        rv_array = ds1.variables["radv"][:]

        ds.close()
        ds1.close()

        nlon = len(lon)
        nlat = len(lat)
        nlev = len(lev)

        # domain filter
        idlonstart, idlonend = 0, nlonc
        idlatstart, idlatend = 0, nlatc

        if lonc[0] < lon[0]:
            idlonstart = np.where(lonc > lon[0])[0][0]
        if lonc[-1] > lon[-1]:
            idlonend = np.where(lonc < lon[-1])[0][-1]
        if latc[0] < lat[0]:
            idlatstart = np.where(latc > lat[0])[0][0]
        if latc[-1] > lat[-1]:
            idlatend = np.where(latc < lat[-1])[0][-1]

        linedata = []
        alt = []

        for ilon in lonc[idlonstart:idlonend]:
            for jlat in latc[idlatstart:idlatend]:
                close_lon = np.argmin(np.abs(lon - ilon))
                close_lat = np.argmin(np.abs(lat - jlat))

                # level dengan data valid
                valid_levels = np.where(~np.isnan(dbz_array[close_lon, close_lat,1:]))[0]
                nalt = len(valid_levels)
                if nalt > 0:
                    alt.append(nalt)

                    # header setiap titik
                    linedata.append(
                        f"FM-128 RADAR   {strdate:<19}  "
                        f"{jlat:12.3f}  {ilon:12.3f}  {radar_alt:8.1f}  {nalt:6d}"
                    )

                    for kalt in valid_levels[::-1]:
                        k = kalt + 1  # skip level 0
                        rf_data = dbz_array[close_lon, close_lat, k]
                        rf_qc = 0
                        rf_err = 0.1 * rf_data

                        rv_data = rv_array[close_lon, close_lat, k]
                        rv_qc = 0
                        rv_err = 0.1 * rv_data

                        # perhitungan ketinggian
                        delta_lat = radar_lat - jlat
                        delta_lon = radar_lon - ilon
                        degtometer = 111320
                        Rtopography = np.sqrt(delta_lon**2 + delta_lat**2) * degtometer / 1000.0
                        re = 6370.0
                        scan_angle = lev[k]
                        R = Rtopography / np.cos(np.deg2rad(scan_angle))
                        H = R * np.sin(np.deg2rad(scan_angle)) + (R**2) / (2 * re)
                        H *= 1000.0

                        linedata.append(
                            f"   {H:12.2f}{rv_data:12.3f}{rv_qc:4d}{rv_err:12.3f}  "
                            f"{rf_data:12.3f}{rf_qc:4d}{rf_err:12.3f}"
                        )

        # tulis output file
        #outfname = f"{strdate.replace(':','_')}.{ctlfile}.ob.radar"
        # --- Nama output file ---
        outfname = f"{strdate}.{ctlfile}.ob.radar"
        with open(outfname, "w") as f:
            f.write(f"TOTAL NUMBER =  {1}\n")
            f.write("#-----------------#\n\n")
            f.write(
                f"RADAR  {radar_name:^12}{radar_lon:8.3f}  {radar_lat:8.3f}  "
                f"{radar_alt:8.1f}  {strdate:<19}{len(alt):6d}{max(alt):6d}\n"
            )
            f.write("#-------------------------------------------------------------------------------#\n\n")
            f.write("\n".join(linedata))

        print(f"Saved {outfname}")
