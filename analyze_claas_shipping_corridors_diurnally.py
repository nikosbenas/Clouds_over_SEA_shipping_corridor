#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program analyzes the diurnal variation of CLAAS-3 data over the SE Atlantic shipping corridor. It is structured as follows:
1. Reads the CLAAS-3 data.
2. Reads the shipping corridor data.
3. Creates maps of time series averages per hour, and corresponding spatial averages.
4. Creates (average) profiles of data variation across the shipping corridor per hour. It also calculates the corresponding NoShip scenarios.
5. Calculates the corridor effect as a profile as above, and also as one simple average value.
6. Plots the results (optionally).
"""

from datetime import datetime
import numpy as np
import multiprocessing
import sys

from shipping_corridor_functions import block_average, calculate_NoShip_curve, calculate_across_corridor_average_and_std, calculate_area_weighted_average, center_shipping_corridor_perpendicular_lines, create_short_across_corridor_profiles, find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, make_map, plot_all_hourly_profiles, plot_diurnal, plot_profile_and_NoShip_line, read_lat_lon_arrays, read_monthly_time_series
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import FileNameStart


def process_index(c):

    '''
    Description:
        This function is a helper function used in a multiprocessing context to process each index c. It calls the find_line_perpendicular_to_corridor function with the necessary arguments for the given index c and returns the results.

    Inputs:
        - c: (int) Index representing a specific latitudinal line, based on the SC flag 2D array.

    Outputs:
        - distances: (list) A 1D list containing distances (in kilometers) of the grid cells lying along the perpendicular line from the corridor center.
        - lat_indices: (list) A 1D list containing latitude indices of the grid cells lying along the perpendicular line.
        - lon_indices: (list) A 1D list containing longitude indices of the grid cells lying along the perpendicular line.
    '''

    return find_line_perpendicular_to_corridor(c, sc_centlat, sc_centlon, angle_radians, lat_claas, lon_claas)
    
# =============================================================================
# Definitions
# =============================================================================

# Define variable to read and data folder
var = 'cfc'
data_folder = '/net/pc190604/nobackup/users/benas/CLAAS-3/Level_3/' + FileNameStart[var + '_mmdc']

# Uncertainty correlation coefficient for monthly averages
unc_coeff = 0.1

# Select the time period to analyze CLAAS-3 data
start_year = 2004
end_year = 2023

# Lats and lons at four corners of region
# north_lat = 10; south_lat = -35
# west_lon = -15; east_lon = 20 
north_lat = -10; south_lat = -20
west_lon = -10; east_lon = 10 

# Define bounding box: ul lon, lr lat,lr lon, ul lat
bounding_box = [west_lon, south_lat, east_lon, north_lat]

# For make_map function:
# [lon.min, lon.max, lat.min, lat.max]
plot_extent = [west_lon, east_lon, south_lat, north_lat] 
grid_extent = [west_lon, east_lon, south_lat, north_lat]

# Define a "time_series" dictionary, to include data and results from the time series analysis
time_series = {}
# Define a "centered" dictionary, to include data and results centered on the corridor center
centered = {}
# Define a "corridor_effect" dictionary, to include profiles, averages and uncertainties of monthly corridor effects.
corridor_effect = {}

# Create vector of dates for plotting
time_series['dates'] = [datetime.strptime(str(year) + str(month).zfill(2) + '01', '%Y%m%d')
         for year in range(start_year, end_year + 1) for month in range(1, 13)]

# Months dictionary 
month = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# =============================================================================
# Read CLAAS-3 data 
# =============================================================================

# Read CLAAS lat, lon once
claas3_aux_file = data_folder + '/' + FileNameStart[var + '_mmdc'] + '20040101000000419SVMSG01MA.nc'
lat_claas, lon_claas = read_lat_lon_arrays(claas3_aux_file)
del claas3_aux_file

# Find array indices of lat and lon at bounding box corners
istart, iend, jstart, jend = find_bounding_box_indices(bounding_box, lat_claas, lon_claas)

# Keep lat lon in bounding box only
lat_claas = lat_claas[istart:iend, jstart:jend]
lon_claas = lon_claas[istart:iend, jstart:jend]
    
# Loop over all years and months to read CLAAS-3 monthly diurnal data into 4D arrays
time_series['all_data'] = read_monthly_time_series(var + '_mmdc', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = True)


# =============================================================================
# Process Shipping Corridor data
# =============================================================================

# 1. Load shipping corridor data, find angle with North.
flag_sc = np.load('flags_shipping_corridor_2.npy')

# Match the reduced resolution of monthy diurnal data
flag_sc = block_average(flag_sc, 5)

sc_centlat, sc_centlon = find_shipping_corridor_center_coordinates(flag_sc, lat_claas, lon_claas)

angle_radians = find_angle_bewteen_shipping_corrridor_and_north(sc_centlat, 
                                                                sc_centlon)

# 2. For each pixel at corridor center, find pixels along the line perpendicular to the corridor.
num_processes = 10

if __name__ == "__main__":
    all_distances = []
    all_lat_indices = []
    all_lon_indices = []

    # Create a multiprocessing pool with the specified number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_index, range(len(sc_centlat)))

    # Unpack and collect the results
    for c, (distances, lat_indices, lon_indices) in enumerate(results):
        all_distances.append(distances)
        all_lat_indices.append(lat_indices)
        all_lon_indices.append(lon_indices)

# 3. Center all perpendicular lines to the corridor center (zero distance), find average distances from center. 
centered['latitude_indices'], centered['longitude_indices'], centered['distances'] = center_shipping_corridor_perpendicular_lines(all_lat_indices, all_lon_indices, all_distances)

del all_distances, all_lat_indices, all_lon_indices

# Calculate average distances along the corridor and set "western" part to negative
avg_distances = np.nanmean(centered['distances'], axis=0)
zero_index = np.where(avg_distances == 0)[0][0]
avg_distances[zero_index + 1:] = -avg_distances[zero_index + 1:]

# =============================================================================
# Analysis of time series
# =============================================================================

# 1. Maps of all-month averages per hour
time_series['mean_map_per_hour'] = np.nanmean(time_series['all_data'], axis = 3)
time_series['Nmonths_map_per_hour'] = np.nansum(time_series['all_data'], axis = 3) / time_series['mean_map_per_hour']
time_series['Nmonths_mean_per_hour'] = np.nanmean(time_series['Nmonths_map_per_hour'], axis = (1, 2))


# 2. Spatial time series averages per hour
mean_and_std  = [calculate_area_weighted_average(time_series['mean_map_per_hour'][i, :, :], lat_claas) for i in range(24)]

time_series['spatial_mean_per_hour'], time_series['spatial_std_per_hour'] = np.array(list(zip(*mean_and_std)))
time_series['spatial_Ncells_per_hour'] = np.array([np.count_nonzero(~np.isnan(time_series['mean_map_per_hour'][i, :, :])) for i in range(24)])

del mean_and_std

# Remove cases where map is not entirely covered
time_series['spatial_mean_per_hour'][time_series['spatial_Ncells_per_hour'] < lat_claas.size] = np.nan
# Remove cases where map is not entirely covered with all available months
time_series['spatial_mean_per_hour'][time_series['Nmonths_mean_per_hour'] < (np.nanmax(time_series['Nmonths_mean_per_hour'])-1)] = np.nan


# 3. Corridor-centered analysis

# Find data mean values centered along the shipping corridor
centered['mean_profile_per_hour'], centered['std_profile_per_hour'], centered['N_profile_per_hour'] = zip(*[calculate_across_corridor_average_and_std(centered['latitude_indices'], centered['longitude_indices'], time_series['mean_map_per_hour'][i, :, :]) for i in range(24)])    

centered['mean_profile_per_hour'] = np.array(centered['mean_profile_per_hour'])
centered['std_profile_per_hour'] = np.array(centered['std_profile_per_hour'])
centered['N_profile_per_hour'] = np.array(centered['N_profile_per_hour'])

# Create shorter profile plots
short_half_range = 350

avg_distances_short = create_short_across_corridor_profiles(short_half_range, avg_distances, avg_distances) 

centered['mean_profile_per_hour_short'] = np.array([create_short_across_corridor_profiles(short_half_range, avg_distances, profile) for profile in centered['mean_profile_per_hour']])

centered['std_profile_per_hour_short'] = np.array([create_short_across_corridor_profiles(short_half_range, avg_distances, profile) for profile in centered['std_profile_per_hour']])

centered['N_profile_per_hour_short'] = np.array([create_short_across_corridor_profiles(short_half_range, avg_distances, profile) for profile in centered['N_profile_per_hour']])


# Calculate curve to create NoShip profiles

corridor_half_range = 250 # Curve fitted based on th 250-400 km range from corridor center on either side. 
core_half_range = 75 # Average corridor effect based on the central 150 km-wide area.

centered['mean_NoShip_profile_per_hour'] = np.array([calculate_NoShip_curve(avg_distances, profile, corridor_half_range, 400, 3) for profile in centered['mean_profile_per_hour']])


# Create shorter NoShip profiles per month
centered['mean_NoShip_profile_per_hour_short'] = np.array([create_short_across_corridor_profiles(short_half_range, avg_distances, profile) for profile in centered['mean_NoShip_profile_per_hour']])



# Calculate profiles of differences (Ship - NoShip), ...
corridor_effect['mean_profile_per_hour'] = centered['mean_profile_per_hour_short'] - centered['mean_NoShip_profile_per_hour_short']

corridor_effect['std_profile_per_hour'] = centered['std_profile_per_hour_short']

# ... their averages and corresponding stds, per hour
corridor_effect['mean_per_hour'] = np.array([np.nanmean(corridor_effect['mean_profile_per_hour'][h, abs(avg_distances) < core_half_range]) for h in range(24)])

corridor_effect['std_per_hour'] = np.array([np.nanstd(corridor_effect['mean_profile_per_hour'][h, abs(avg_distances) < core_half_range]) for h in range(24)])

corridor_effect['N_per_hour'] = np.array([np.round(np.nansum(corridor_effect['mean_profile_per_hour'][h, abs(avg_distances) < core_half_range]) / corridor_effect['mean_per_hour'][h]) for h in range(24)])

# Remove cases where map is not entirely covered
corridor_effect['mean_per_hour'][time_series['spatial_Ncells_per_hour'] < lat_claas.size] = np.nan
corridor_effect['mean_profile_per_hour'][time_series['spatial_Ncells_per_hour'] < lat_claas.size] = np.nan
# Remove cases where map is not entirely covered with all available months
corridor_effect['mean_per_hour'][time_series['Nmonths_mean_per_hour'] < (np.nanmax(time_series['Nmonths_mean_per_hour'])-1)] = np.nan
corridor_effect['mean_profile_per_hour'][time_series['Nmonths_mean_per_hour'] < (np.nanmax(time_series['Nmonths_mean_per_hour'])-1)] = np.nan


# =============================================================================
# Plot results
# =============================================================================


# 1. Maps of all-month averages per hour
create_hourly_maps = False
if create_hourly_maps:

    for i in range(24):

        if not np.all(np.isnan(time_series['mean_map_per_hour'][i, :, :])):

            make_map(var, time_series['mean_map_per_hour'][i, :, :], var.upper() + ' ' + str(i).zfill(2) + ':30 UTC average', np.nanmin(time_series['mean_map_per_hour'][i, :, :]), np.nanmax(time_series['mean_map_per_hour'][i, :, :]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_map_average_hour_' + str(i).zfill(2) + '30UTC_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)

            make_map(var, time_series['Nmonths_map_per_hour'][i, :, :], var.upper() + ' ' + str(i).zfill(2) + ':30 UTC Nmonths', np.nanmin(time_series['Nmonths_map_per_hour'][i, :, :]), np.nanmax(time_series['Nmonths_map_per_hour'][i, :, :]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_map_Nmonths_hour_' + str(i).zfill(2) + '30UTC_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)


# 2. Spatial time series averages per hour
plot_diurnal_spatial_averages = True
if plot_diurnal_spatial_averages:

    plot_diurnal(var, time_series['spatial_mean_per_hour'], time_series['spatial_std_per_hour'], 'Diurnal variation', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_diurnal_mean.png', plot_zero_line = False, saveplot = True)


# 3. Corridor-centered plots
plot_24_profiles = False
if plot_24_profiles:

    for i in range(24):

        if not np.all(np.isnan(centered['mean_profile_per_hour'][i, :])):

            plot_profile_and_NoShip_line(var, centered['mean_profile_per_hour'][i, :], centered['std_profile_per_hour'][i, :], centered['mean_NoShip_profile_per_hour_short'][i, :], avg_distances, zero_index, var.upper() + ' across shipping corridor, ' + str(i).zfill(2) + ':30 UTC average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_long_profile_average_at_' + str(i).zfill(2) + '30UTC.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)

            plot_profile_and_NoShip_line(var, centered['mean_profile_per_hour_short'][i, :], centered['std_profile_per_hour_short'][i, :], centered['mean_NoShip_profile_per_hour_short'][i, :], avg_distances, zero_index, var.upper() + ' across shipping corridor, ' + str(i).zfill(2) + ':30 UTC average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_profile_average_at_' + str(i).zfill(2) + '30UTC.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)


# Plot diurnal corridor effect
plot_diurnal(var, corridor_effect['mean_per_hour'], corridor_effect['std_per_hour'], 'Diurnal corridor effect', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_diurnal_corridor_effect_and_std.png', plot_zero_line = True, saveplot = True)         

plot_all_hourly_profiles(var, corridor_effect['mean_profile_per_hour'], corridor_effect['std_profile_per_hour'], avg_distances, zero_index, avg_distances_short, ' ', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_all_hourly_diff_profiles_across_sc.png', plot_std_bands = False, plot_zero_line = True, saveplot = True)

print('check')
