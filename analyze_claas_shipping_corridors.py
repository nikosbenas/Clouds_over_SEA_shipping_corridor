#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 07:16:13 2023

@author: benas
"""
from datetime import datetime
import h5py
import numpy as np
from netCDF4 import Dataset
import multiprocessing
import time
import sys
sys.path.append('/usr/people/benas/Documents/CMSAF/python_modules/')
from modis_python_functions import map_l3_var as create_map
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
import claas_trends_functions as ctf
import claas3_dictionaries as cdict
import copy
import matplotlib.pyplot as plt
import geopy.distance as gd


# Define variables to read
var_list = ['cdnc_liq']

# Define Shipping Corridor
sc = '2'


# Lats and lons at four corners of region: ul, ur, lr, ll
# toplat = -10; bottomlat = -35
# leftlon = -10; rightlon = 20 
toplat = -10; bottomlat = -20
leftlon = -10; rightlon = 10 

lats = [toplat, toplat, bottomlat, bottomlat]
lons = [leftlon, rightlon, rightlon, leftlon]
del toplat, bottomlat, leftlon, rightlon

# =============================================================================
# Read CLAAS-3 data
# =============================================================================

# Define bounding box: ul lon, lr lat,lr lon, ul lat
bounding_box = [lons[0], lats[2], lons[2], lats[0]]
del lats, lons

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

del latData, lonData, landseamask, bounding_box 

# Read VZA to use as data threshold
f = h5py.File(claas3_l3_aux_data_file, 'r')
# Reading data only for lon0 = 0.0
vzaData = f['satzen'][1, istart:iend:stride, jstart:jend:stride] 
f.close()
vzaMask = vzaData > 70
del vzaData, claas3_l3_aux_data_file

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
# Find shipping corridor edges and center per latitudinal line
# =============================================================================  

# Find first and last occurrences of ones (shipping corridor edges) in each 
# line
first_occurrences = np.argmax(flag_sc == 1, axis=1)

# Find the last occurrence of value 1 in each row (by reversing the array 
# along columns)
last_occurrences = np.argmax((flag_sc == 1)[:, ::-1], axis=1)

# Adjust the indices of the last occurrence to account for reversing
last_occurrences = flag_sc.shape[1] - 1 - last_occurrences  

# Define shipping corridor center array index, lat and lon
sc_centind = np.empty(flag_sc.shape[0])
sc_centlat = np.empty(flag_sc.shape[0])
sc_centlon = np.empty(flag_sc.shape[0])

for i in range(flag_sc.shape[0]):
    
    index = int((last_occurrences[i] + first_occurrences[i])/2)
    
    sc_centind[i] = index
    
    sc_centlat[i] = lat_claas[i][index]
    sc_centlon[i] = lon_claas[i][index]
    
# Calculate the angle between shipping corridor and south-to-north direction

# Perform linear regression to find the line equation
coefficients = np.polyfit(sc_centlon, sc_centlat, 1)
slope = coefficients[0]

# Calculate the angle between the line and south-to-north (meridian) direction
angle_radians = np.arctan(slope)
angle_degrees = np.degrees(angle_radians)

# Ensure the angle is between 0 and 180 degrees
if angle_degrees < 0:
    angle_degrees += 180
    
# This is the angle starting from the x-axis (east). Starting from North:
angle_degrees = angle_degrees - 90
    

# =============================================================================
# Analyze all pixels around center of shipping corridor
# =============================================================================

# List containg all values around center of corridor
var_pxl_all = []; lat_pxl_all = []; lon_pxl_all = []

halfwidth = 50

for i in range(300):#len(first_occurrences)):
    
    if first_occurrences[i] != 0:
        
        # Column at middle of shipping corridor
        d_col = int((last_occurrences[i] + first_occurrences[i])/2)
        
        # Save all points around the center of SC
        if d_col - halfwidth > 0: 
            
            var_pxl_all.append(var_data_mean[i][d_col-halfwidth : d_col+halfwidth])
            lat_pxl_all.append(lat_claas[i][d_col-halfwidth : d_col+halfwidth])
            lon_pxl_all.append(lon_claas[i][d_col-halfwidth : d_col+halfwidth])
        
var_pxl_all = np.array(var_pxl_all)
lat_pxl_all = np.array(lat_pxl_all)
lon_pxl_all = np.array(lon_pxl_all)

distances = np.full_like(lat_pxl_all, np.nan)
center = int(var_pxl_all.shape[1]/2)

# Calculate (horizontal) distances from center of corridor (in km)
for i in range(len(lat_pxl_all)):
    
    # Center point coordinates
    pc =  [lat_pxl_all[i, center], lon_pxl_all[i, center]]
    
    for j in range(lat_pxl_all.shape[1]):
        
        pj = [lat_pxl_all[i, j], lon_pxl_all[i, j]]
        
        distances[i, j] = gd.geodesic(pc, pj).km
        
        if j < center:
            
            distances[i, j] = - distances[i, j]
        
# Average distance along the west-east axis
dist_mean = np.mean(distances, axis = 0)
var_pxl_all_mean = np.nanmean(var_pxl_all, axis = 0)

# var_pxl_all_mean_sc = np.full_like(var_pxl_all_mean, np.nan)
# var_pxl_all_mean_sc[center - sc_width:center + sc_width] =\
#     var_pxl_all_mean[center - sc_width:center + sc_width]

# =============================================================================
# # Select N initial and final points in the array and interpolate the rest
# =============================================================================
var_pxl_all_mean_n = copy.deepcopy(var_pxl_all_mean)
N = 10
var_pxl_all_mean_n[N:-N] = np.nan

# Indices of non-nan elements
non_nan_indices = np.where(~np.isnan(var_pxl_all_mean_n))[0]

# Interpolate using non-nan elements
interpolation_points = non_nan_indices
var_pxl_all_mean_interp = np.interp(np.arange(len(var_pxl_all_mean_n)), 
                                    interpolation_points, 
                                    var_pxl_all_mean_n[non_nan_indices])


fig = plt.figure()
plt.plot(dist_mean, var_pxl_all_mean, label = 'Original values')
plt.plot(dist_mean, var_pxl_all_mean_interp, linestyle = ':', color = 'k', 
         label = 'Interpolated values')
plt.axvline(x = dist_mean[center], linestyle = ':', color='grey')
plt.ylabel('[' + cdict.varUnits[var] + ']')
plt.xlabel('Distance from corridor center, W to E [km]')
plt.title(var.upper() + ' across shipping corridor #' + sc)
plt.legend()
outfile = 'Figures/' + var.upper() + '_across_sc_' + str(sc)
fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

diff = var_pxl_all_mean - var_pxl_all_mean_interp

fig = plt.figure()
plt.plot(dist_mean, diff)
plt.axvline(x = dist_mean[center], linestyle = ':', color='grey')
plt.ylabel('[' + cdict.varUnits[var] + ']')
plt.xlabel('Distance from corridor center, W to E [km]')
plt.title(var.upper() + ' change due to shipping corridor #' + sc)
outfile = 'Figures/' + var.upper() + '_change_due_to_sc_' + str(sc)
fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

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