#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program analyzes the seasonal variation of CLAAS-3 data over the SE Atlantic shipping corridor. It is structured as follows:
1. Reads the CLAAS-3 data.
2. Reads the shipping corridor data.
3. Calculates maps of time series averages per month.
4. Creates (average) profile of data variation across the shipping corridor per month, based on corresponding time series average maps. It also calculates the corresponding NoShip curves.
5. Calculates the corridor effect as a profile per month as above, and also as  simple average values.
6. Plots results (optionally).
"""

from datetime import datetime
import multiprocessing
import sys
import numpy as np
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import FileNameStart, varSymbol
from shipping_corridor_functions import calculate_NoShip_curve, calculate_area_weighted_average, center_along_corridor_data_and_uncertainties_per_month,  center_shipping_corridor_perpendicular_lines, create_short_across_corridor_profiles, find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, make_map, plot_12_monthly_profiles, plot_intra_annual_variation, read_lat_lon_arrays, read_monthly_time_series, plot_profile_and_NoShip_line


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
var = 'cfc_day'
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
dates = {}
dates['months'] = []; dates['years'] = []
for year in range(start_year, end_year + 1):
    dates['years'].append(datetime(year, 1, 1))
    for month in range(1, 13):
        dates['months'].append(datetime.strptime(str(year) + str(month).zfill(2) + '01', '%Y%m%d'))

# Define a "centered" dictionary, to include data and results centered on the corridor center
centered = {}
# Define a "corridor_effect" dictionary, to include profiles, averages and uncertainties of monthly corridor effects.
corridor_effect = {}

# Months dictionary 
month_string = {
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
claas3_aux_file = '/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends/claas3_level3_aux_data_005deg.nc'
lat_claas, lon_claas = read_lat_lon_arrays(claas3_aux_file)
del claas3_aux_file

# Find array indices of lat and lon at bounding box corners
istart, iend, jstart, jend = find_bounding_box_indices(bounding_box, lat_claas, lon_claas)

# Keep lat lon in bounding box only
lat_claas = lat_claas[istart:iend, jstart:jend]
lon_claas = lon_claas[istart:iend, jstart:jend]

# Loop over all years and months to read CLAAS-3 data and their uncertainties into 3D arrays
data = read_monthly_time_series(var, data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)
if var == 'cfc_day':
    data_unc = read_monthly_time_series('cfc_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)
elif var == 'cot_liq_log':
    data_unc = read_monthly_time_series('cot_liq_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)    
else:
    data_unc = read_monthly_time_series(var + '_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal = False)

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
# Analysis of time series per individual month
# =============================================================================

# Reshape time series array to add a month dimension
shape_4d = (data.shape[0], data.shape[1], end_year + 1 - start_year, 12)
data_4d = data.reshape(shape_4d) 
data_unc_4d = data_unc.reshape(shape_4d) 

# Calculate time series mean over the region per individual month ...
mean_per_month = np.nanmean(data_4d, axis = 2)
std_per_month = np.nanstd(data_4d, axis = 2)
N_per_month = np.round(np.nansum(data_4d, axis = 2) / mean_per_month).astype(int)
# ... and propagate the uncertainty ...
unc_per_month = np.sqrt(((1 / N_per_month) * (std_per_month**2)) + unc_coeff * (np.nanmean(data_unc_4d, axis = 2)**2))
# ... and calculate area-weighted averages and uncertainties per month (12 values, intra-annual variation)
area_mean_per_month, area_std_per_month = calculate_area_weighted_average(mean_per_month, lat_claas)

area_unc_per_month = np.sqrt(((1 / mean_per_month[:,:,0].size) * (area_std_per_month**2)) + unc_coeff * (np.nanmean(unc_per_month, axis = (0, 1))**2))

# Center data and uncertainty per month along corridor
centered['mean_per_month'], centered['unc_per_month'] = center_along_corridor_data_and_uncertainties_per_month(centered, mean_per_month, unc_per_month)

# Find centered along the shipping corridor profile means per month
centered['monthly_profile_means'] = np.nanmean(centered['mean_per_month'], axis = 0)
centered['monthly_profile_std'] = np.nanstd(centered['mean_per_month'], axis = 0)
centered['monthly_profile_N'] = np.nansum(centered['mean_per_month'], axis = 0) / centered['monthly_profile_means']

centered['monthly_profile_unc'] = np.sqrt(((1 / centered['monthly_profile_N']) * (centered['monthly_profile_std']**2)) + unc_coeff * (np.nanmean(centered['unc_per_month'], axis = 0)**2))


# Create shorter profile plots
avg_distances_short = create_short_across_corridor_profiles(350, avg_distances, avg_distances)

centered['monthly_profile_means_short'] = np.swapaxes(np.array([create_short_across_corridor_profiles(350, avg_distances, centered['monthly_profile_means'][:, m]) for m in range(12)]), 0, 1)

centered['monthly_profile_unc_short'] = np.swapaxes(np.array([create_short_across_corridor_profiles(350, avg_distances, centered['monthly_profile_unc'][:, m]) for m in range(12)]), 0, 1)


# =============================================================================
# Calculate curve to create NoShip profiles 
# =============================================================================

corridor_half_range = 250 # Curve fitted based on th 250-400 km range from corridor center on either side. 
core_half_range = 75 # Average corridor effect based on the central 150 km-wide area.

centered['monthly_NoShip_profile_means_250'] = np.swapaxes(np.array([calculate_NoShip_curve(avg_distances, centered['monthly_profile_means'][:, m], corridor_half_range, 400, 3) for m in range(12)]), 0, 1)

# Calculate uncertainty of the no-ship curve (std of 4 fits)
mean_NoShip_fits = np.full((len(avg_distances), 12, 5), np.nan)

mean_NoShip_fits[:, :, 0] = np.swapaxes(np.array([calculate_NoShip_curve(avg_distances, centered['monthly_profile_means'][:, m], 150, 300, 3) for m in range(12)]), 0, 1)
mean_NoShip_fits[:, :, 1] = np.swapaxes(np.array([calculate_NoShip_curve(avg_distances, centered['monthly_profile_means'][:, m], 200, 350, 3) for m in range(12)]), 0, 1)
mean_NoShip_fits[:, :, 2] = centered['monthly_NoShip_profile_means_250']
mean_NoShip_fits[:, :, 3] = np.swapaxes(np.array([calculate_NoShip_curve(avg_distances, centered['monthly_profile_means'][:, m], 300, 450, 3) for m in range(12)]), 0, 1)
mean_NoShip_fits[:, :, 4] = np.swapaxes(np.array([calculate_NoShip_curve(avg_distances, centered['monthly_profile_means'][:, m], 350, 500, 3) for m in range(12)]), 0, 1)
centered['mean_NoShip_std'] = np.nanstd(mean_NoShip_fits, axis = 2)

# Create shorter NoShip profiles per month 
centered['monthly_NoShip_profile_means_short'] = np.swapaxes(np.array([create_short_across_corridor_profiles(350, avg_distances, centered['monthly_NoShip_profile_means_250'][:, m]) for m in range(12)]), 0, 1)

# Calculate profiles of differences (Ship - NoShip), their averages and corresponding uncertainties
corridor_effect['monthly_profiles'] = centered['monthly_profile_means_short'] - centered['monthly_NoShip_profile_means_short']

corridor_effect['monthly_profiles_unc'] = np.sqrt((centered['mean_NoShip_std'])**2 + (centered['monthly_profile_unc'])**2)

corridor_effect['monthly_mean'] = [np.nanmean(corridor_effect['monthly_profiles'][abs(avg_distances) < core_half_range, m]) for m in range(12)]

corridor_effect['monthly_std'] = np.array([np.nanstd(corridor_effect['monthly_profiles'][abs(avg_distances) < core_half_range, m]) for m in range(12)])

corridor_effect['monthly_N'] = np.array([np.round(np.nansum(corridor_effect['monthly_profiles'][abs(avg_distances) < core_half_range, m]) / corridor_effect['monthly_mean'][m]) for m in range(12)])

corridor_effect['monthly_unc'] = np.array([np.sqrt(((1 / corridor_effect['monthly_N'][m]) * (corridor_effect['monthly_std'][m]**2)) + unc_coeff * (np.nanmean(corridor_effect['monthly_profiles_unc'][abs(avg_distances) < core_half_range, m], axis = 0)**2)) for m in range(12)])

# Plot profiles: mean values and differences, separately (one plot/month) and all together. 
plot_monthly_profiles_separately = True
if plot_monthly_profiles_separately:

    for m in range(12):

        plot_profile_and_NoShip_line(var, centered['monthly_profile_means_short'][:, m], centered['monthly_profile_unc_short'][:, m], centered['monthly_NoShip_profile_means_short'][:, m], centered['mean_NoShip_std'][:, m], avg_distances, zero_index, varSymbol[var] + ' across shipping corridor, ' + month_string[m+1] + ' average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/'  + var.upper() + '_profile_average_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '.png', plot_NoShip_line = True, plot_data_unc = True, plot_NoShip_unc = True, saveplot = True)


plot_monthly_profiles_in_one_plot = True
if plot_monthly_profiles_in_one_plot:

    plot_12_monthly_profiles(var, month_string, varSymbol[var] + ' across shipping corridor', centered['monthly_profile_means_short'], centered['monthly_profile_unc_short'], avg_distances, zero_index, avg_distances_short, 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_12_monthly_mean_profiles_across_sc.png', plot_unc_bands = True, plot_zero_line = False, saveplot = True)


plot_monthly_difference_profiles_separately = True
if plot_monthly_difference_profiles_separately:

    for m in range(12):

        plot_profile_and_NoShip_line(var, corridor_effect['monthly_profiles'][:, m], corridor_effect['monthly_profiles_unc'][:, m], np.zeros_like(centered['monthly_profile_means_short'][:, m]), np.zeros_like(centered['monthly_profile_means_short'][:, m]), avg_distances_short, zero_index, varSymbol[var] + ' across shipping corridor, ' + month_string[m+1] + ' average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/'  + var.upper() + '_difference_profile_average_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '.png', plot_NoShip_line = True, plot_data_unc = True, plot_NoShip_unc = False, saveplot = True)


plot_monthly_diff_profiles_in_one_plot = True
if plot_monthly_diff_profiles_in_one_plot:

    plot_12_monthly_profiles(var, month_string, 'Corridor effect on ' + varSymbol[var], corridor_effect['monthly_profiles'], centered['monthly_profile_unc_short'], avg_distances, zero_index, avg_distances_short, 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_12_monthly_mean_diff_profiles_across_sc.png', plot_unc_bands = False, plot_zero_line = True, saveplot = True)        


# Create maps of mean values and uncertainties per month
create_monthly_maps = False
if create_monthly_maps:

    for m in range(12):

        # Of means per month
        make_map(var, mean_per_month[:, :, m], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' ' + month_string[m+1] + ' average', np.nanmin(mean_per_month[:, :, m]), np.nanmax(mean_per_month[:, :, m]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_map_average_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)

        # Of uncertainties per month
        make_map(var, unc_per_month[:, :, m], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' ' + month_string[m+1] + ' average uncertainty', np.nanmin(unc_per_month[:, :, m]), np.nanmax(unc_per_month[:, :, m]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_map_uncertainty_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)


# Plot seasonal variation of area averages and profile differences
plot_intra_annual= True
if plot_intra_annual:

    plot_intra_annual_variation(var, area_mean_per_month, area_unc_per_month, ' ', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_area_weighted_intra-annual_mean_and_uncertainty.png', plot_unc_band = True, plot_zero_line = False, saveplot = True)

    plot_intra_annual_variation(var, corridor_effect['monthly_mean'], corridor_effect['monthly_unc'], ' ', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Seasonal/' + var.upper() + '_intra-annual_corridor_effect_and_uncertainty.png', plot_unc_band = True, plot_zero_line = True, saveplot = True)


# Save seasonal averages and effects to combine variables in one plot (CDNC and CRE, LWP and CFC_DAY)
save_intra_annual = True
if save_intra_annual:

    # Intra-annual averages and uncertainties
    np.save('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_area_seasonal_mean_per_month_' + str(start_year) + '-' + str(end_year) + '.npy', area_mean_per_month.data)

    np.save('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_area_seasonal_uncertainty_per_month_' + str(start_year) + '-' + str(end_year) + '.npy', area_unc_per_month.data)

    # Intra-annual corridor effect and uncertainties
    np.save('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_corridor_effect_per_month_' + str(start_year) + '-' + str(end_year) + '.npy', corridor_effect['monthly_mean'])

    np.save('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_corridor_effect_unc_per_month_' + str(start_year) + '-' + str(end_year) + '.npy', corridor_effect['monthly_unc'])

print('check')