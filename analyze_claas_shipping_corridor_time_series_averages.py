#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program analyzes time series average CLAAS-3 data over the SE Atlantic shipping corridor. It is structured as follows:
1. Reads the CLAAS-3 data.
2. Reads the shipping corridor data.
3. Calculates map of time series averages.
4. Creates (average) profile of data variation across the shipping corridor based on the time series average map. It also calculates the corresponding NoShip curve.
5. Calculates the corridor effect as a profile as above, and also as one simple average value.
"""

from datetime import datetime
import multiprocessing
import sys
import numpy as np
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import FileNameStart
from shipping_corridor_functions import calculate_NoShip_curve, calculate_across_corridor_average_and_std, calculate_running_mean, center_data_along_corridor, center_shipping_corridor_perpendicular_lines, create_short_across_corridor_profiles, find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, make_map, plot_change_and_zero_line, plot_time_series, read_lat_lon_arrays, read_monthly_time_series, plot_profile_and_NoShip_line


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

# Define variables to read and data folder
var = 'cre_liq'
data_folder = '/net/pc190604/nobackup/users/benas/CLAAS-3/Level_3/' + FileNameStart[var]

# Uncertainty correlation coefficient for monthly averages
unc_coeff = 0.1

# Select the time period to analyze CLAAS-3 data
start_year = 2004
end_year = 2023

# Lats and lons at four corners of region: ul, ur, lr, ll
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

# Create vector of dates for plotting
dates = [datetime.strptime(str(year) + str(month).zfill(2) + '01', '%Y%m%d')
         for year in range(start_year, end_year + 1) for month in range(1, 13)]

# Define a time series dictionary to include data, mean, std, uncertainty etc.
time_series = {}

# Define a "centered" dictionary, to include data and results centered on the corridor center
centered = {}

# =============================================================================
# Read CLAAS-3 data
# =============================================================================

# Read CLAAS lat, lon once
claas3_aux_file = '/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends/claas3_level3_aux_data_005deg.nc'
lat_claas, lon_claas = read_lat_lon_arrays(claas3_aux_file)

# Find array indices of lat and lon at bounding box corners
istart, iend, jstart, jend = find_bounding_box_indices(bounding_box, lat_claas, lon_claas)

# Keep lat lon in bounding box only
lat_claas = lat_claas[istart:iend, jstart:jend]
lon_claas = lon_claas[istart:iend, jstart:jend]

# Loop over all years and months to read CLAAS-3 data and their uncertainties into 3D arrays and include them in the time series dictionary
time_series['data'] = read_monthly_time_series(var, data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)
if var == 'cfc_day':
    time_series['unc'] = read_monthly_time_series('cfc_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)
else:
    time_series['unc'] = read_monthly_time_series(var + '_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)

# =============================================================================
# Process Shipping Corridor data
# =============================================================================

# 1. Load shipping corridor data, find angle with North.
flag_sc = np.load('flags_shipping_corridor_2.npy')

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
# Analysis of time series averages
# =============================================================================

# Calculate time series mean and number of months with data per grid cell
time_series['mean'] = np.nanmean(time_series['data'], axis = 2)
time_series['std'] = np.nanstd(time_series['data'], axis = 2)
time_series['Nmonths'] = np.round(np.nansum(time_series['data'], axis = 2) / time_series['mean']).astype(int)
time_series['unc_mean'] = np.sqrt(((1 / time_series['Nmonths']) * (time_series['std']**2)) + unc_coeff * (np.nanmean(time_series['unc'])**2))

# Find data mean values centered along the shipping corridor
centered['mean'], centered['std'], centered['N'] = calculate_across_corridor_average_and_std(centered['latitude_indices'], centered['longitude_indices'], time_series['mean'])  

# Calculate uncertainty of centered data
centered['unc'] = center_data_along_corridor(time_series['unc_mean'], centered['latitude_indices'], centered['longitude_indices'])
centered['unc_mean'] = np.sqrt(((1 / centered['N']) * (centered['std']**2)) + unc_coeff * (np.nanmean(centered['unc'], axis = 0)**2))

# Calculate curve to imitate absence of the shipping corridor

corridor_half_range = 250

centered['mean_NoShip'] = calculate_NoShip_curve(avg_distances, centered['mean'], corridor_half_range, 400, 3)

# Create shorter profile plots mean values and uncertainties, centered on the corridor
short_half_range = 350
avg_distances_short = create_short_across_corridor_profiles(short_half_range, avg_distances, avg_distances)
centered['mean_short'] = create_short_across_corridor_profiles(short_half_range, avg_distances, centered['mean'])
centered['mean_short_NoShip'] = create_short_across_corridor_profiles(short_half_range, avg_distances, centered['mean_NoShip'])
centered['unc_mean_short'] = create_short_across_corridor_profiles(short_half_range, avg_distances, centered['unc_mean'])

# And calculate corridor effect
corridor_effect = {}
corridor_effect['profile'] = centered['mean_short'] - centered['mean_short_NoShip']
corridor_effect['profile_unc'] = centered['unc_mean_short']
corridor_effect['mean'] = np.nanmean(corridor_effect['profile'][abs(avg_distances_short) < corridor_half_range])
corridor_effect['std'] = np.nanstd(corridor_effect['profile'][abs(avg_distances_short) < corridor_half_range])
corridor_effect['N_points'] = np.nansum(corridor_effect['profile'][abs(avg_distances_short) < corridor_half_range]) / corridor_effect['mean']
corridor_effect['unc_mean'] = np.sqrt(((1/corridor_effect['N_points']) * corridor_effect['std']**2) + (unc_coeff * np.nanmean(corridor_effect['profile_unc'][abs(avg_distances_short) < corridor_half_range])**2))


# Create maps of time series mean values and uncertainties
create_map = False
if create_map:

    # Of time series mean
    make_map(var, time_series['mean'], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' average', np.nanmin(time_series['mean']), np.nanmax(time_series['mean']), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/' + var.upper() + '_' + str(start_year) + '-' + str(end_year) + '_average.png', saveplot = True)

    # Of time series mean uncertainties
    make_map(var, time_series['unc_mean'], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' average uncertainty', np.nanmin(time_series['unc_mean']), np.nanmax(time_series['unc_mean']), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/' + var.upper() + '_' + str(start_year) + '-' + str(end_year) + '_average_uncertainty.png', saveplot = True)

create_profile_plots = False
if create_profile_plots:

    # Plot long profile of mean values
    plot_profile_and_NoShip_line(var, centered['mean'], centered['unc_mean'], centered['mean_NoShip'], avg_distances, zero_index, var.upper() + ' across shipping corridor, time series average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_across_sc_long.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)

    # Plot profile of mean values
    plot_profile_and_NoShip_line(var, centered['mean_short'], centered['unc_mean_short'], centered['mean_short_NoShip'], avg_distances_short, zero_index, var.upper() + ' across shipping corridor, time series average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_across_sc.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)


create_profile_difference_plots = False
if create_profile_difference_plots:

    # Plot profile of mean values
    plot_change_and_zero_line(var, corridor_effect['profile'], corridor_effect['profile_unc'], avg_distances_short, zero_index, corridor_effect['mean'], corridor_effect['unc_mean'], var.upper() + ' change due to shipping corridor', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_change_across_sc.png', plot_std_band = True, saveplot = True)

# =============================================================================
# Analysis of trends
# =============================================================================

# 1. Center all monthly time series data and uncertainties along the corridor 
centered['monthly_data'] = np.stack([center_data_along_corridor(time_series['data'][:, :, i], centered['latitude_indices'], centered['longitude_indices']) for i in range(time_series['data'].shape[2])], axis = 2)

centered['monthly_data_unc'] = np.stack([center_data_along_corridor(time_series['unc'][:, :, i], centered['latitude_indices'], centered['longitude_indices']) for i in range(time_series['unc'].shape[2])], axis = 2)

# 2 Calculate average profiles per month
centered['monthly_profiles'] = np.nanmean(centered['monthly_data'], axis = 0)
centered['monthly_profiles_std'] = np.nanstd(centered['monthly_data'], axis = 0)
centered['monthly_profiles_N'] = np.nansum(centered['monthly_data'], axis = 0) / centered['monthly_profiles']
centered['monthly_profiles_unc'] = np.sqrt((1 / centered['monthly_profiles_N']) * (centered['monthly_profiles_std']**2) + unc_coeff * (np.nanmean(centered['monthly_data_unc'], axis = 0)**2))

# 3. Fit "NoShip" curves per month
centered['monthly_profiles_NoShip'] = np.stack([calculate_NoShip_curve(avg_distances, centered['monthly_profiles'][:, i], corridor_half_range, 400, 3) for i in range(centered['monthly_profiles'].shape[1])], axis = 1)

# 4. Calculate corridor effect profile per month
corridor_effect['monthly_profiles'] = centered['monthly_profiles'] - centered['monthly_profiles_NoShip']
corridor_effect['monthly_profiles_unc'] = centered['monthly_profiles_unc']

corridor_effect['monthly_mean'] = np.stack([np.nanmean(corridor_effect['monthly_profiles'][abs(avg_distances_short) < corridor_half_range, i]) for i in range(corridor_effect['monthly_profiles'].shape[1])])

plot_time_series(dates, corridor_effect['monthly_mean'], var, 'time series of mean effects', 'test_time_series.png', saveplot = True)

smooth_effect = calculate_running_mean(corridor_effect['monthly_mean'], 12)

plot_time_series(dates, smooth_effect, var, 'time series of mean effects', 'test_time_series_smooth.png', saveplot = True)

# corridor_effect['std'] = np.nanstd(corridor_effect['profile'][abs(avg_distances_short) < corridor_half_range])
# corridor_effect['N_points'] = np.nansum(corridor_effect['profile'][abs(avg_distances_short) < corridor_half_range]) / corridor_effect['mean']
# corridor_effect['unc_mean'] = np.sqrt(((1/corridor_effect['N_points']) * corridor_effect['std']**2) + (unc_coeff * np.nanmean(corridor_effect['profile_unc'][abs(avg_distances_short) < corridor_half_range])**2))


# Print ALL MONTHLY PROFILES
plot_all_monthly_profiles = True
if plot_all_monthly_profiles:

    for i in range(centered['monthly_profiles'].shape[1]):

        plot_profile_and_NoShip_line(var, centered['monthly_profiles'][:, i], centered['monthly_profiles_unc'][:, i], centered['monthly_profiles_NoShip'][:, i], avg_distances, zero_index, var + ' profile, ' + dates[i].strftime("%Y-%m"), 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/All_monthly_profiles/' + var.upper() + '_long_profile_' + dates[i].strftime("%Y%m") + '.png', plot_NoShip_line=True, plot_std_band=True, saveplot=True)

print('check')

# for i in range(time_series['data'].shape[2]):

#     centered_monthly_data.append(center_data_along_corridor(time_series['data'][:, :, i], centered['latitude_indices'], centered['longitude_indices']))

# centered_monthly_data = np.array(centered_monthly_data)
