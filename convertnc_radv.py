# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 09:46:51 2025

@author: Abdul Hamid Al Habib
"""

from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import RegularGridInterpolator

from scipy.ndimage import zoom
from skimage.transform import resize
from matplotlib.colors import LightSource
#import datetime
import ephem  # Untuk menghitung posisi matahari

import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
from matplotlib import colors

import wradlib as wrl
import numpy as np
import warnings, math
print("NumPy version:", np.__version__)  
print("Matplotlib version:", matplotlib.__version__)  
print("wradlib version:", wrl.__version__)  
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os

radarFile='/home/riset_2/jak_cband/20250823/JAK-20250823-130613-PPIVol.nc'
# radarFile='E:/voltonc/JAK-20250823-234615-PPIVol.nc'
#f = wrl.util.get_wradlib_data_file(radarFile)
f = radarFile
raw = wrl.io.read_generic_netcdf(f)
# Ambil nama file tanpa ekstensi
basename = os.path.splitext(os.path.basename(radarFile))[0]  # JAK-20250823-234615-PPIVol

# Buat nama output _dbz.nc
output_nc1 = f"/home/riset_2/WRF/WRFDA_Script/{basename}_radv.nc"

radarLon = float(raw['variables']['longitude']['data'])
radarLat = float(raw['variables']['latitude']['data'])
radarAlt = float(raw['variables']['altitude']['data'])
sitecoords=(radarLon,radarLat,radarAlt)

res=250. # resolusi data yang diinginkan dalam meter
resCoords=res/111229. # resolusi data dalam derajat
rmax=250000./111229. # range maksimum
lonMax,lonMin=radarLon+(rmax),radarLon-(rmax) 
latMax,latMin=radarLat+(rmax),radarLat-(rmax)
nGrid=int(np.floor((lonMax-lonMin)/resCoords))+1 # jumlah grid
lonGrid=np.linspace(lonMin,lonMax,nGrid) # grid longitude
latGrid=np.linspace(latMin,latMax,nGrid) # grid latitude           
dataContainer = np.zeros((len(lonGrid),len(latGrid))) # penampung data
tstart = dt.datetime.now()
# define flag option untuk melihat apakah gates vary atau tidak
sweep_start_idx = raw['variables']['sweep_start_ray_index']['data']
sweep_end_idx = raw['variables']['sweep_end_ray_index']['data']
try:
    if raw['gates_vary']=='true':
        ray_n_gates=raw['variables']['ray_n_gates']['data']
        ray_start_index=raw['variables']['ray_start_index']['data']
        flag='true'
    elif raw['gates_vary']=='false':
        flag='false'
except :
    if raw['n_gates_vary']=='true':
        ray_n_gates=raw['variables']['ray_n_gates']['data']
        ray_start_index=raw['variables']['ray_start_index']['data']
        flag='true'
    elif raw['n_gates_vary']=='false':
        flag='false'

# ekstrak waktu radar (end of observation)
try:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%SZ")
except:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%S.%fZ") 

nElevation=np.size(raw['variables']['fixed_angle']['data'])
for i in range(nElevation):
    elevation=float('{0:.1f}'.format(raw['variables']['fixed_angle']['data'][i]))
    print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
    
    # ekstrak azimuth data
    azi = raw['variables']['azimuth']['data'][sweep_start_idx[i]:sweep_end_idx[i]]   
    
    # ekstrak range data dan radar (dBZ) data berdasarkan nilai flag
    r_all = raw['variables']['range']['data'] 
    if flag == 'false':
        data = raw['variables']['VELH']['data'][sweep_start_idx[i]:sweep_end_idx[i], :]
        r = r_all    
    else:              
        data = np.array([])
        n_azi = sweep_end_idx[i]-sweep_start_idx[i]        
        try:
            for ll in range(sweep_start_idx[i],sweep_end_idx[i]):
                data = np.append(data,raw['variables']['VELH']['data'][ray_start_index[ll]:ray_start_index[ll+1]])
            data = data.reshape((n_azi,ray_n_gates[sweep_start_idx[i]]))
        except:
            pass
        r = r_all[0:ray_n_gates[sweep_start_idx[i]]]
    
    # transformasi dari koordinat bola ke koordinat kartesian
    rangeMesh, azimuthMesh =np.meshgrid(r,azi) # meshgrid azimuth dan range
    lonlatalt = wrl.georef.polar.spherical_to_proj(
        rangeMesh, azimuthMesh, elevation, sitecoords
    ) 
    x, y = lonlatalt[:, :, 0], lonlatalt[:, :, 1]
    
    # proses regriding ke data container yang sudah dibuat sebelumnya
    lonMesh, latMesh=np.meshgrid(lonGrid,latGrid)
    gridLatLon = np.vstack((lonMesh.ravel(), latMesh.ravel())).transpose()
    xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
    radius=r[np.size(r)-1]
    center=[x.mean(),y.mean()]
    gridded = wrl.comp.togrid(
        xy, gridLatLon,
        radius, center, data.ravel(),
        wrl.ipol.Linear
    )
    griddedData = np.ma.masked_invalid(gridded).reshape((len(lonGrid), len(latGrid)))
    dataContainer=np.dstack((dataContainer,griddedData))


# print(dataContainer.shape)  # Bentuk (dimensi) array
# print(dataContainer)        # Isi array
# print(dataContainer[0])  # Seluruh lapisan pertama
# print(dataContainer[:, 0, 0])  # Semua lapisan di elemen [0, 0]
# print(dataContainer[0:3, 0:5, 0:5])  # Potongan data (subset)


#===================================================================#

import numpy as np
import datetime as dt
from netCDF4 import Dataset, date2num
import os


# =====================
# Simpan VELH ke NetCDF
# =====================

# dataContainer sudah berbentuk (lon, lat, lev)
# Jika saat ini dataContainer hanya berisi 2D grid per sweep, 
# pastikan kamu stack semua sweep (lev) -> sudah dilakukan pakai np.dstack()

# Pastikan lonGrid, latGrid, dan levs sudah ada
lons = lonGrid.astype(np.float32)
lats = latGrid.astype(np.float32)
levs = raw['variables']['fixed_angle']['data']  # gunakan fixed_angle sebagai lev

# Hapus layer awal dataContainer (yang kosong)
dbz_data = dataContainer[:, :, 1:]  # karena index 0 hasil inisialisasi nol

# Ubah urutan axis agar sesuai format (lat, lon, lev)
# Saat ini dbz_data shape = (nLon, nLat, nLev)
dbz_data = np.transpose(dbz_data, (1, 0, 2))  # jadi (lat, lon, lev)

# Buat file output
output_nc = output_nc1
ncfile = Dataset(output_nc, 'w', format='NETCDF4')

# ===== Global attributes =====
ncfile.Conventions = 'CF-1.6'
ncfile.description = 'Radial Velocity'
ncfile.institution = 'DMU BMKG'
ncfile.title = 'PPI-Radial Velocity Radar'
ncfile.comment = f'Based on {radarFile} file, created by Abdul Hamid Al Habib'

# ===== Dimensions =====
ncfile.createDimension('lon', len(lons))
ncfile.createDimension('lat', len(lats))
ncfile.createDimension('lev', len(levs))
ncfile.createDimension('time', None)

# ===== Variables =====
longitude = ncfile.createVariable('lon', 'f', ('lon',))
longitude.grads_dim = 'x'
longitude.grads_mapping = 'linear'
longitude.grads_size = str(len(lons))
longitude.units = 'degrees_east'
longitude.long_name = 'longitude'
longitude.minimum = str(lons.min())
longitude.maximum = str(lons.max())
longitude.resolution = str(lons[2] - lons[1])
longitude[:] = lons

latitude = ncfile.createVariable('lat', 'f', ('lat',))
latitude.grads_dim = 'y'
latitude.grads_mapping = 'linear'
latitude.grads_size = str(len(lats))
latitude.units = 'degrees_north'
latitude.long_name = 'latitude'
latitude.minimum = str(lats.min())
latitude.maximum = str(lats.max())
latitude.resolution = str(lats[2] - lats[1])
latitude[:] = lats

ttime = ncfile.createVariable('time', 'd', ('time',))
ttime.grads_dim = 't'
ttime.grads_mapping = 'linear'
ttime.units = 'hours since 1-1-1 00:00'
ttime.calendar = 'gregorian'
ttime.long_name = 'time'
dtime = '0001-01-01 00:00:00'  # dummy datetime
strdt = dt.datetime.strptime(dtime, "%Y-%m-%d %H:%M:%S")
ttime[:] = date2num(strdt, units=ttime.units, calendar=ttime.calendar)

levo = ncfile.createVariable('lev', 'd', ('lev',))
levo.grads_dim = 'z'
levo.grads_mapping = 'levels'
levo.grads_size = len(levs)
levo.units = 'meter'
levo.long_name = 'vertical scan angle (degree)'
levo.minimum = str(levs.min())
levo.maximum = str(levs.max())
levo[:] = levs

dbz = ncfile.createVariable('radv', 'd', ('lat', 'lon', 'lev'), zlib=True)
dbz.long_name = 'Radial Velocity'
dbz.units = 'radv'
dbz[:, :, :] = dbz_data  # langsung assign semua data

ncfile.close()

print("All processes took:", dt.datetime.now() - tstart)
print(f"File NetCDF berhasil dibuat: {output_nc}")





  
