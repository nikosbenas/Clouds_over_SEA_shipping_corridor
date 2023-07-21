#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 07:16:13 2023

@author: benas
"""
import datetime
import h5py
import numpy as np
import claas_trends_functions as ctf
import claas3_dictionaries as cdict
from netCDF4 import Dataset
import multiprocessing
import time

# Lats and lons at four corners of region: ul, ur, lr, ll
toplat = -10; bottomlat = -35
leftlon = -10; rightlon = 20 

lats = [toplat, toplat, bottomlat, bottomlat]
lons = [leftlon, rightlon, rightlon, leftlon]
del toplat, bottomlat, leftlon, rightlon

# =============================================================================
# Read CLAAS-3 data
# =============================================================================

# Define bounding box: ul lon, lr lat,lr lon, ul lat
bounding_box = [lons[0], lats[2], lons[2], lats[0]]
# Definitions for reading data:
stride = 1
read_mode = 'stride'
read_cores = 10

# Define variables to read
var_list = ['cdnc_liq']

# Select the time period to analyze CLAAS-3 data
start_year = 2004
end_year = 2022

# Create vector of dates for plotting
dates = [datetime.strptime(str(year) + str(month).zfill(2) + '01', '%Y%m%d')
         for year in range(start_year, end_year + 1) for month in range(1, 13)]

# =============================================================================
# Read some CLAAS data once: lat, lon and VZA, land-sea mask
# =============================================================================

# Read full CLAAS-3 lat and lon arrays, needed to find b.box indices
claas3_l3_aux_data_file = '/data/windows/m/benas/Documents/CMSAF/CLAAS-3/' +\
    'CLAAS-3_trends/claas3_level3_aux_data_005deg.nc'
f = h5py.File(claas3_l3_aux_data_file, 'r')
latData = f['lat'][:]
lonData = f['lon'][:]
landseamask = f['lsm'][:][:]
f.close()
lonData, latData = np.meshgrid(lonData, latData)
lsm = landseamask == 2

istart, iend, jstart, jend = ctf.find_bbox_indices(bounding_box, latData,
                                                   lonData)

lsm = lsm[istart:iend:stride, jstart:jend:stride]

del latData, lonData 

# Read VZA to use as data threshold
f = h5py.File(claas3_l3_aux_data_file, 'r')
# Reading data only for lon0 = 0.0
vzaData = f['satzen'][1, istart:iend:stride, jstart:jend:stride] 
f.close()
vzaMask = vzaData > 70

for var in var_list:
    
    if 'lat_claas' not in locals():
          
          # Use an existing file
        claas3_file = cdict.cdr_folder[2004] + '2004/02/' +\
            cdict.FileNameStart[var] + '20040201000000423SVMSG01MA.nc'
        # Open the file and read the latitude and longitude
        claas3_data = Dataset(claas3_file, 'r')
        lat_claas = claas3_data['lat'][istart:iend:stride]
        lon_claas = claas3_data['lon'][jstart:jend:stride]
        # Create mesh grid
        lon_claas, lat_claas = np.meshgrid(lon_claas, lat_claas)
        
        del claas3_file, claas3_data


# =============================================================================
# Loop over all years and months to read CLAAS data into a 3D array
# =============================================================================

    # Create a list of arguments for each combination of year and month
    args_list = [(year, month, istart, iend, jstart, jend, stride, 
                  cdict.cdr_folder[year], var, lat_claas, lon_claas, vzaMask, 
                  read_mode) for year in range(start_year, end_year + 1) 
                 for month in range(1, 13)]
        
    # Create a multiprocessing pool with the desired number of cores
    pool = multiprocessing.Pool(read_cores)
        
    # Use the pool to call the function with the given arguments in parallel
    start_time = time.time()
    var_data_stack = pool.map(ctf.read_data_parallel, args_list)
    print("Read data with %d cores in %s seconds" % (read_cores, time.time() -
                                                     start_time))
        
    # Close the pool when you're done
    pool.close()
        
    del args_list
        
    # Convert the results to a numpy array
    var_data_stack = np.dstack(var_data_stack)
    var_data_stack.filled(np.nan)
    var_data_stack = var_data_stack.data
    var_data_stack[var_data_stack == -999] = np.nan

# =============================================================================
# OPTIONAL CODE: Calculate time series average per grid cell
# =============================================================================
    var_data_mean = np.nanmean(var_data_stack, axis = 2)
    var_data_nmonths = (100 * (np.nansum(var_data_stack, axis = 2) / 
                               var_data_mean) / var_data_stack.shape[2])
    
    # Keep only sea areas
    var_data_mean = np.where(lsm, var_data_mean, np.nan)
    
    
# =============================================================================
# Create some test maps
# =============================================================================

# Flip ship data and flag upside down to match with CLAAS conventions
lat_reduced = np.flipud(lat_reduced)
flag_reduced = np.flipud(flag_reduced)

outfile = './CDNC_average_all_noland.png'
create_map(lat_claas, lon_claas, var_data_mean, np.nanmin(var_data_mean), 
           np.nanmax(var_data_mean), 'CDNC average', 'cm-3', 'viridis', 'neither', 
            outfile, saveplot = True)

# Mask non-ship-corridor areas
var_data_mean_only_ships = np.where(flag_reduced == 0, np.nan, var_data_mean)
outfile = './CDNC_average_only_ships.png'
create_map(lat_claas, lon_claas, var_data_mean_only_ships, 
           np.min(var_data_mean), np.max(var_data_mean), 
           'CDNC average ships', 'cm-3', 
           'viridis', 'neither', outfile, saveplot = True)

# Mask ship-corridor areas
var_data_mean_no_ships = np.where(flag_reduced == 1, np.nan, var_data_mean)
outfile = './CDNC_average_no_ships.png'
create_map(lat_claas, lon_claas, var_data_mean_no_ships, 
           np.min(var_data_mean), np.max(var_data_mean), 
           'CDNC average no ships', 'cm-3', 
           'viridis', 'neither', outfile, saveplot = True)    