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


def find_shipping_corridor_center_coordinates(flag_sc):
    
    """
    This function returns the shipping corridor center latitude and longitude 
    per latitudinal line.
    
    Input: 2D binary array of shipping corridor flag
    Output: 1D arrays of SC center latitudes and longitudes
    """
    
    # Find first and last occurrences of ones (shipping corridor edges) in each 
    # line
    first_occurrences = np.argmax(flag_sc == 1, axis=1)

    # Find the last occurrence of value 1 in each row (by reversing the array 
    # along columns)
    last_occurrences = np.argmax((flag_sc == 1)[:, ::-1], axis=1)

    # Adjust the indices of the last occurrence to account for reversing
    last_occurrences = flag_sc.shape[1] - 1 - last_occurrences  

    # Define shipping corridor center lat and lon
    sc_centlat = np.empty(flag_sc.shape[0])
    sc_centlon = np.empty(flag_sc.shape[0])

    for i in range(flag_sc.shape[0]):
        
        index = int((last_occurrences[i] + first_occurrences[i])/2)
            
        sc_centlat[i] = lat_claas[i][index]
        sc_centlon[i] = lon_claas[i][index]
      
    del index  
    
    return sc_centlat, sc_centlon


def find_angle_bewteen_shipping_corrridor_and_north(sc_centlat, sc_centlon):
    
    """
    This function returns the angle between the shipping corridor (defined by 
    1D arrays of its center latitudes and longitudes) and the North direction.
    
    Input: 1D arrays of SC center latitudes and longitudes
    Output: angle (in radians)
    """
    
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
    
    return np.radians(angle_degrees)   


def find_line_perpendicular_to_corridor(c):
    
    """
    This function returns information on the grid cells that lie along the line
    perpendicular to the shipping corridor center (given by latitudinal line
    number c)
    
    Input: index of latitudinal line, based on SC flag 2D array.
    Output: 1D lists of distances (in km), lat indices and lon indices of the
            grid cells lying along the perpendicular line
    """
   
    # Corridor center pixel coordinates
    pc = [sc_centlat[c], sc_centlon[c]]
    
    # Line intercept
    b = sc_centlat[c] - angle_radians * sc_centlon[c]
    
    # Array indices of pixels falling within (half-pixel) the perpendicular 
    lat_indices = []
    lon_indices = []
    # Distances of the "line pixels" from the corridor center 
    distances = []
    
    for i in range(lat_claas.shape[0]):
        
        expected_lats = angle_radians * lon_claas[i, :] + b
                        
        lon_ind = np.argmin(abs(lat_claas[i, :] - expected_lats))
        
        if (lon_ind > 0) and (lon_ind < (lat_claas.shape[1] - 1)):
            
            lat_indices.append(i)
            lon_indices.append(lon_ind)
            
            pij = [lat_claas[i, lon_ind], lon_claas[i, lon_ind]]
            distances.append(gd.geodesic(pc, pij).km)
            
        else:
            
            continue

    return distances, lat_indices, lon_indices


def center_shipping_corridor_perpendicular_lines(all_lat_indices, 
                                                 all_lon_indices, 
                                                 all_distances):
    
    """
    This function takes lists of 1D arrays of lat/lon indices and distances 
    (in km) from shipping corridor centers and centers them so that they are 
    aligned to the shipping corridor center.
    
    Input: lists of 1D arrays of lat/lon indices and distances from shipping 
           corridor center.
    Output: 2D arrays of the same (input) variables aligned and centered along 
            the shipping corridor center. 
    """

    # Initialize lists to store centered data
    zero_indices = []
    zero_indices_reverse = []
    centered_dists = []
    centered_lat_inds = []
    centered_lon_inds = []

    # Center each list by adding NaN values
    for i in range(len(all_distances)):
        
        zero_indices.append(all_distances[i].index(0))
        zero_indices_reverse.append(len(all_distances[i]) -
                                    all_distances[i].index(0))
        
    max_zero_ind = max(zero_indices)
    max_zero_ind_rev = max(zero_indices_reverse)
        
    for i in range(len(all_distances)):
        
        zero_ind = all_distances[i].index(0)
        zero_ind_rev = len(all_distances[i]) - all_distances[i].index(0)
        
        pad_left = max_zero_ind - zero_ind
        pad_right = max_zero_ind_rev - zero_ind_rev
        
        centered_dist = np.concatenate([np.nan*np.ones(pad_left), all_distances[i], 
                                       np.nan*np.ones(pad_right)])
        centered_dists.append(centered_dist)
        
        centered_lat_ind = np.concatenate(
            [np.nan*np.ones(pad_left), all_lat_indices[i], 
             np.nan*np.ones(pad_right)])
        centered_lat_inds.append(centered_lat_ind)
        
        centered_lon_ind = np.concatenate(
            [np.nan*np.ones(pad_left), all_lon_indices[i], 
             np.nan*np.ones(pad_right)])
        centered_lon_inds.append(centered_lon_ind)
        
        
    return (np.vstack(centered_lat_inds), np.vstack(centered_lon_inds), 
            np.vstack(centered_dists))



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
        
    del args_list, pool 
        
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
# Load shipping corridor data and find angle with North
# =============================================================================    

flag_sc = np.load('flags_shipping_corridor_' + sc + '.npy')

sc_centlat, sc_centlon = find_shipping_corridor_center_coordinates(flag_sc)

angle_radians = find_angle_bewteen_shipping_corrridor_and_north(sc_centlat, 
                                                                sc_centlon)

# =============================================================================
# For each pixel at corridor center, find pixels along the line perpendicular 
# to the corridor 
# =============================================================================

num_processes = 10

if __name__ == "__main__":
    
    all_distances = []
    all_lat_indices = []
    all_lon_indices = []

    # Create a multiprocessing pool with the specified number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        
        results = pool.map(find_line_perpendicular_to_corridor, 
                           range(len(sc_centlat)))

    # Unpack and collect the results
    for c, (distances, lat_indices, lon_indices) in enumerate(results):
        
        all_distances.append(distances)
        all_lat_indices.append(lat_indices)
        all_lon_indices.append(lon_indices)        


# =============================================================================
# Center all perpendicular lines to the corridor center (zero distance) 
# =============================================================================

centered_lat_inds, centered_lon_inds, centered_dists = \
    center_shipping_corridor_perpendicular_lines(all_lat_indices, 
                                                 all_lon_indices, 
                                                 all_distances)
 
avg_distances = np.nanmean(centered_dists, axis=0)
    
# =============================================================================
# Analyze all pixels around center of shipping corridor
# =============================================================================

# List containg all values around center of corridor
# var_pxl_all = []; lat_pxl_all = []; lon_pxl_all = []

# halfwidth = 50

# for i in range(len(first_occurrences)):
    
#     if first_occurrences[i] != 0:
        
#         # Column at middle of shipping corridor
#         d_col = int((last_occurrences[i] + first_occurrences[i])/2)
        
#         # Save all points around the center of SC
#         if d_col - halfwidth > 0: 
            
#             var_pxl_all.append(var_data_mean[i][d_col-halfwidth : d_col+halfwidth])
#             lat_pxl_all.append(lat_claas[i][d_col-halfwidth : d_col+halfwidth])
#             lon_pxl_all.append(lon_claas[i][d_col-halfwidth : d_col+halfwidth])
        
# var_pxl_all = np.array(var_pxl_all)
# lat_pxl_all = np.array(lat_pxl_all)
# lon_pxl_all = np.array(lon_pxl_all)

# distances = np.full_like(lat_pxl_all, np.nan)
# center = int(var_pxl_all.shape[1]/2)

# # Calculate (horizontal) distances from center of corridor (in km)
# for i in range(len(lat_pxl_all)):
    
#     # Center point coordinates
#     pc =  [lat_pxl_all[i, center], lon_pxl_all[i, center]]
    
#     for j in range(lat_pxl_all.shape[1]):
        
#         pj = [lat_pxl_all[i, j], lon_pxl_all[i, j]]
        
#         distances[i, j] = gd.geodesic(pc, pj).km
        
#         if j < center:
            
#             distances[i, j] = - distances[i, j]
        
# # Average distance along the west-east axis
# dist_mean = np.mean(distances, axis = 0)
# var_pxl_all_mean = np.nanmean(var_pxl_all, axis = 0)

# var_pxl_all_mean_sc = np.full_like(var_pxl_all_mean, np.nan)
# var_pxl_all_mean_sc[center - sc_width:center + sc_width] =\
#     var_pxl_all_mean[center - sc_width:center + sc_width]

# =============================================================================
# # Select N initial and final points in the array and interpolate the rest
# =============================================================================
# var_pxl_all_mean_n = copy.deepcopy(var_pxl_all_mean)
# N = 10
# var_pxl_all_mean_n[N:-N] = np.nan

# # Indices of non-nan elements
# non_nan_indices = np.where(~np.isnan(var_pxl_all_mean_n))[0]

# # Interpolate using non-nan elements
# interpolation_points = non_nan_indices
# var_pxl_all_mean_interp = np.interp(np.arange(len(var_pxl_all_mean_n)), 
#                                     interpolation_points, 
#                                     var_pxl_all_mean_n[non_nan_indices])


# fig = plt.figure()
# plt.plot(dist_mean, var_pxl_all_mean, label = 'Original values')
# plt.plot(dist_mean, var_pxl_all_mean_interp, linestyle = ':', color = 'k', 
#          label = 'Interpolated values')
# plt.axvline(x = dist_mean[center], linestyle = ':', color='grey')
# plt.ylabel('[' + cdict.varUnits[var] + ']')
# plt.xlabel('Distance from corridor center, W to E [km]')
# plt.title(var.upper() + ' across shipping corridor #' + sc)
# plt.legend()
# outfile = 'Figures/' + var.upper() + '_across_sc_' + str(sc)
# fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

# diff = var_pxl_all_mean - var_pxl_all_mean_interp

# fig = plt.figure()
# plt.plot(dist_mean, diff)
# plt.axvline(x = dist_mean[center], linestyle = ':', color='grey')
# plt.ylabel('[' + cdict.varUnits[var] + ']')
# plt.xlabel('Distance from corridor center, W to E [km]')
# plt.title(var.upper() + ' change due to shipping corridor #' + sc)
# outfile = 'Figures/' + var.upper() + '_change_due_to_sc_' + str(sc)
# fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

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