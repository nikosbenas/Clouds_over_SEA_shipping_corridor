#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 07:16:13 2023

@author: benas
"""
from datetime import datetime
import h5py
import numpy as np
import claas_trends_functions as ctf
import claas3_dictionaries as cdict
from netCDF4 import Dataset
import multiprocessing
import time
import sys
sys.path.append('/usr/people/benas/Documents/CMSAF/python_modules/')
from modis_python_functions import map_l3_var as create_map
import copy
import matplotlib.pyplot as plt
import geopy.distance as gd


# Define variables to read
var_list = ['cre_liq']

# Define Shipping Corridor
sc = '5'


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

lsm = np.flipud(lsm)

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
        
        lat_claas = np.flipud(lat_claas)
        
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
# Load shipping corridor data
# =============================================================================    

flag_sc = np.load('flags_shipping_corridor_' + sc + '.npy')

# =============================================================================
# Calculate variable average and std at shipping corridor
# =============================================================================    

var_data_mean_sc = copy.deepcopy(var_data_mean)
var_data_mean_sc[flag_sc == 0] = np.nan

var_data_mean_sc_avg = np.nanmean(var_data_mean_sc)
var_data_mean_sc_std = np.nanstd(var_data_mean_sc)
var_data_mean_sc_count = np.nansum(var_data_mean_sc) / var_data_mean_sc_avg

# =============================================================================
# Calculate variable average and std at shipping corridor surroundings
# =============================================================================  

# Find first and last occurrences of ones (shipping corridor edges) in each 
# line
first_occurrences = np.argmax(flag_sc == 1, axis=1)

# Find the last occurrence of value 1 in each row (by reversing the array 
# along columns)
last_occurrences = np.argmax((flag_sc == 1)[:, ::-1], axis=1)

# Adjust the indices of the last occurrence to account for reversing
last_occurrences = flag_sc.shape[1] - 1 - last_occurrences


# Calculate averages west and east of shipping corridor
var_avg_west = []; var_std_west = []; var_cnt_west = []; dist_avg_west = []
var_avg_east = []; var_std_east = []; var_cnt_east = []; dist_avg_east = []

for d in range(1, 31):
    
    print(d)
    
    data_west = []; dist_west = []
    data_east = []; dist_east = []
    
    for i in range(len(first_occurrences)):
        
        col_w = first_occurrences[i] - d
        col_e = last_occurrences[i] + d
        
        # Column and point at middle of shipping corridor
        d_col = int((last_occurrences[i] + first_occurrences[i])/2)
        p1 = [lat_claas[i, d_col], lon_claas[i, d_col]]
        
        if col_w >  0:
            
            data_west.append(var_data_mean[i, col_w])
            
            # Find distance of west point with middle of ship corridor in m
            pw = [lat_claas[i, col_w], lon_claas[i, col_w]]
            dist_west.append(gd.geodesic(p1, pw).m)
            
                             
        if col_e < var_data_mean.shape[1]:
            
            data_east.append(var_data_mean[i, col_e])
            
            # Find distance of east point with middle of ship corridor in m
            pe = [lat_claas[i, col_e], lon_claas[i, col_e]]
            dist_east.append(gd.geodesic(p1, pe).m)
                             
    var_avg_west.append(np.nanmean(data_west))
    var_std_west.append(np.nanstd(data_west))
    var_cnt_west.append(len(data_west))
    dist_avg_west.append(np.nanmean(dist_west))
    
    var_avg_east.append(np.nanmean(data_east))
    var_std_east.append(np.nanstd(data_east))
    var_cnt_east.append(len(data_east))
    dist_avg_east.append(np.nanmean(dist_east))

    
# Combine averages in an east-to-west row:
sc_widths = (last_occurrences - first_occurrences).astype(float)
sc_widths[sc_widths == float(flag_sc.shape[1] - 1)] = np.nan
sc_width = round(np.nanmean(sc_widths))
    
inside_corr = [var_data_mean_sc_avg] * sc_width
for i in range(len(inside_corr)):
    if i != (len(inside_corr) // 2):
        inside_corr[i] = float('nan')
        
var_avg_all = var_avg_west[::-1] +  inside_corr + var_avg_east

# Average plus 1 sigma
var_avg_plus_std_west = list(np.array(var_avg_west) + np.array(var_std_west))
var_avg_plus_std_east = list(np.array(var_avg_east) + np.array(var_std_east))
var_avg_plus_std_sc = var_data_mean_sc_avg + var_data_mean_sc_std

var_avg_plus_std_all = var_avg_plus_std_west[::-1] + [var_avg_plus_std_sc] *\
    sc_width + var_avg_plus_std_east
    
# Average minus 1 sigma
var_avg_minus_std_west = list(np.array(var_avg_west) - np.array(var_std_west))
var_avg_minus_std_east = list(np.array(var_avg_east) - np.array(var_std_east))
var_avg_minus_std_sc = var_data_mean_sc_avg - var_data_mean_sc_std

var_avg_minus_std_all = var_avg_minus_std_west[::-1] +\
    [var_avg_minus_std_sc] * sc_width + var_avg_minus_std_east    
    

# Plot East-to-West line averages

# x-axis centered on corridor

diff = (dist_avg_east[0] + dist_avg_west[0]) / sc_width
dist_in_corr = np.arange(-dist_avg_west[0], dist_avg_east[0], diff)

x = [-i for i in dist_avg_west[::-1]] + list(dist_in_corr) + dist_avg_east
# Convert to km
x = [i / 1000 for i in x]

fig = plt.figure()

plt.plot(x, var_avg_all, 'o') 
# plt.plot(x, var_avg_plus_std_all)
# plt.plot(x, var_avg_minus_std_all)

plt.title(var + ' around shipping corridor ' + sc)
plt.xlabel('[km] (West to East, centered on corridor)')
plt.ylabel('[' + cdict.varUnits[var] + ']')

fig.savefig('Figures/' + var.upper() + '_around_sc_' + sc + '.png', 
            dpi = 300, bbox_inches = 'tight')

# =============================================================================
# Create some test maps
# =============================================================================

# outfile = './CDNC_average.png'
# create_map(lat_claas, lon_claas, var_data_mean_sc, np.nanmin(var_data_mean), 
#             np.nanmax(var_data_mean), 'CDNC average', 'cm-3', 'viridis', 'neither', 
#             outfile, saveplot = False)


# outfile = './Flag_sc.png'
# create_map(lat_claas, lon_claas, flag_sc, 0, 1, 'Shipping corridor 2', '-', 
#            'Reds', 'neither', outfile, saveplot = False)

# # Mask non-ship-corridor areas
# var_data_mean_only_ships = np.where(flag_reduced == 0, np.nan, var_data_mean)
# outfile = './CDNC_average_only_ships.png'
# create_map(lat_claas, lon_claas, var_data_mean_only_ships, 
#            np.min(var_data_mean), np.max(var_data_mean), 
#            'CDNC average ships', 'cm-3', 
#            'viridis', 'neither', outfile, saveplot = True)

# # Mask ship-corridor areas
# var_data_mean_no_ships = np.where(flag_reduced == 1, np.nan, var_data_mean)
# outfile = './CDNC_average_no_ships.png'
# create_map(lat_claas, lon_claas, var_data_mean_no_ships, 
#            np.min(var_data_mean), np.max(var_data_mean), 
#            'CDNC average no ships', 'cm-3', 
#            'viridis', 'neither', outfile, saveplot = True)    