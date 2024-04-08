#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 07:16:13 2023

@author: benas
"""
from datetime import datetime
import numpy as np
import multiprocessing
import sys

from shipping_corridor_functions import calculate_across_corridor_average_and_std, calculate_area_weighted_average, center_shipping_corridor_perpendicular_lines, find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, make_map, plot_diurnal, plot_profile_and_NoShip_line, read_lat_lon_arrays, read_monthly_time_series
# sys.path.append('/usr/people/benas/Documents/CMSAF/python_modules/')
# from modis_python_functions import map_l3_var as create_map
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
import claas_trends_functions as ctf
from claas3_dictionaries import FileNameStart
import pvlib


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
var = 'lwp'
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
claas3_aux_file = data_folder + '/LWPmd20040101000000419SVMSG01MA.nc'
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
flag_sc = ctf.block_average(flag_sc, 5)

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

create_hourly_maps = False
if create_hourly_maps:

    for i in range(24):

        make_map(var, time_series['mean_map_per_hour'][i, :, :], var.upper() + ' ' + str(i).zfill(2) + ':30 UTC average', np.nanmin(time_series['mean_map_per_hour'][i, :, :]), np.nanmax(time_series['mean_map_per_hour'][i, :, :]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_map_average_hour_' + str(i).zfill(2) + '30UTC_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)

        make_map(var, time_series['Nmonths_map_per_hour'][i, :, :], var.upper() + ' ' + str(i).zfill(2) + ':30 UTC Nmonths', np.nanmin(time_series['Nmonths_map_per_hour'][i, :, :]), np.nanmax(time_series['Nmonths_map_per_hour'][i, :, :]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_map_Nmonths_hour_' + str(i).zfill(2) + '30UTC_' + str(start_year) + '-' + str(end_year) + '.png', saveplot = True)


# 2. Spatial time series averages per hour
mean_and_std  = [calculate_area_weighted_average(time_series['mean_map_per_hour'][i, :, :], lat_claas) for i in range(24)]

time_series['spatial_mean_per_hour'], time_series['spatial_std_per_hour'] = np.array(list(zip(*mean_and_std)))
time_series['spatial_Ncells_per_hour'] = np.array([np.count_nonzero(~np.isnan(time_series['mean_map_per_hour'][i, :, :])) for i in range(24)])

del mean_and_std

# Remove cases where map is not entirely covered
time_series['spatial_mean_per_hour'][time_series['spatial_Ncells_per_hour'] < lat_claas.size] = np.nan
# Remove cases where map is not entirely covered with all available months
time_series['spatial_mean_per_hour'][time_series['Nmonths_mean_per_hour'] < np.nanmax(time_series['Nmonths_mean_per_hour'])] = np.nan

plot_diurnal_spatial_averages = False
if plot_diurnal_spatial_averages:

    plot_diurnal(var, time_series['spatial_mean_per_hour'], time_series['spatial_std_per_hour'], var.upper() + ' diurnal variation', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_diurnal_mean.png', saveplot = True)


# 3. Corridor-centered analysis
# Find data mean values centered along the shipping corridor
centered['mean_per_hour'] = []; centered['std_per_hour'] = []; centered['N_per_hour'] = []
for i in range(24):

    mean, std, N = calculate_across_corridor_average_and_std(centered['latitude_indices'], centered['longitude_indices'], time_series['mean_map_per_hour'][i, :, :]) 
    centered['mean_per_hour'].append(mean)
    centered['std_per_hour'].append(std)
    centered['N_per_hour'].append(N)

# centered['mean_per_hour'], centered['std_per_hour'], centered['N_per_hour'] = zip(*[calculate_across_corridor_average_and_std(centered['latitude_indices'], centered['longitude_indices'], time_series['mean_map_per_hour'][i, :, :]) for i in range(24)])    

centered['mean_per_hour'] = np.array(centered['mean_per_hour'])
centered['std_per_hour'] = np.array(centered['std_per_hour'])
centered['N_per_hour'] = np.array(centered['N_per_hour'])

plot_24_profiles = True
if plot_24_profiles:

    for i in range(24):

        plot_profile_and_NoShip_line(var, centered['mean_per_hour'][i, :], centered['std_per_hour'][i, :], centered['mean_per_hour'][i, :], avg_distances, zero_index, var.upper() + ' across shipping corridor, ' + str(i).zfill(2) + ':30 UTC average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/Diurnal/' + var.upper() + '_profile_average_at_' + str(i).zfill(2) + '30UTC.png', plot_NoShip_line = False, plot_std_band = True, saveplot = True)

print('check')

'''
        


# =============================================================================
# OPTIONAL CODE: Calculate time series average per grid cell and plot results
# =============================================================================

if analyze_timeseries_average:
    
        
    # Find data mean values centered along the shipping corridor
    
    centered_data_ts_mean = np.full(((24,) + centered_lat_inds.shape), np.nan)
    for i in range(24):
        
        centered_data_ts_mean[i, :, :] = center_data_along_corridor(
            data_ts_mean[i, :, :], centered_lat_inds, centered_lon_inds)
    
    centered_data_ts_mean_avg = np.nanmean(centered_data_ts_mean, axis = 1)
    centered_data_ts_mean_N = (np.nansum(centered_data_ts_mean, axis = 1) /
                               centered_data_ts_mean_avg)
    
    # Calculate straight line to imitate absence of the shipping corridor ...
    centered_data_ts_mean_NoShip = np.full_like(centered_data_ts_mean_avg, 
                                                np.nan)
    # ... and define the corridor effect as their difference
    diff = np.full_like(centered_data_ts_mean_avg, np.nan)
    for i in range(24):
    
        centered_data_ts_mean_NoShip[i, :] = calculate_NoShip_line(
        avg_distances, centered_data_ts_mean_avg[i, :], 250)
        
        diff[i, :] = centered_data_ts_mean_avg[i, :] -\
            centered_data_ts_mean_NoShip[i, :]
    
    # OPTIONAL PLOTS
    if plot_diurnal_average_profiles:
        
        # Keep data falling at most 350 km from the corridor center
        avg_distances_350 = copy.deepcopy(avg_distances)
        avg_distances_350[abs(avg_distances) > 350] = np.nan
        
    ### Plot all time slot profiles in one plot
        saveplot = True    
        
        fig, ax = plt.subplots()
        
        # Define the tab20 colors
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Use the first 12 colors for the plot
        ax.set_prop_cycle(color=tab20_colors[:12])
        
        for i in range(24):
            
            if not np.isnan(data_ts_mean[i, :, :]).any():
            
                label = str(i).zfill(2) + ':00'
                ax.plot(avg_distances_350, diff[i, :], label = label)
            
        plt.axvline(x = avg_distances[zero_index], linestyle = ':', color='grey')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Distance from corridor center, W to E [km]')
        ax.set_ylabel('[' + cdict.varUnits[var] + ']')
        # ax.set_title(var.upper() + ' diurnal change across shipping corridor')
        
        if saveplot:
            outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
                str(end_year) + '/' + var.upper() + '_diurnal_change_profiles_across_sc.png'
            fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')
    
    ### 
    
        for i in range(24):
        
            if np.isnan(data_ts_mean[i, :, :]).all() == False:
            
                outfile = 'Figures/' + var.upper() + '/' + str(start_year) +\
                    '-' + str(end_year) + '/'  + var.upper() +\
                        '_time_series_mean_across_sc_long_at_' +\
                            str(i).zfill(2) + '00_UTC.png'
            
                plot_profile_and_NoShip_line(
                    centered_data_ts_mean_avg[i, :], 
                    centered_data_ts_mean_NoShip[i, :], var, avg_distances, 
                    zero_index, var.upper() + ' across shipping corridor, ' +\
                        'time series average at ' + str(i).zfill(2) + '00 UTC',
                        outfile, saveplot = False)
    
    # centered_data_ts_mean_avg[:, abs(avg_distances) > 350] = np.nan
            
            if np.isnan(data_ts_mean[i, :, :]).all() == False:
        
                outfile = 'Figures/' + var.upper() + '/' + str(start_year) +\
                    '-' + str(end_year) + '/' + var.upper() +\
                        '_time_series_mean_across_sc_at_' + str(i).zfill(2) +\
                            '00_UTC.png'
                            
                plot_profile_and_NoShip_line(
                    centered_data_ts_mean_avg[i, :], 
                    centered_data_ts_mean_NoShip[i, :], var, avg_distances_350,
                    zero_index, var.upper() + ' across shipping corridor, ' +\
                        'time series average at ' + str(i).zfill(2) + '00 UTC',
                        outfile, saveplot = False)
    
        # Plot difference profile due to SC
                outfile = 'Figures/' + var.upper() + '/' + str(start_year) +\
                    '-' + str(end_year) + '/' + var.upper() +\
                        '_time_series_mean_change_across_sc_at_' +\
                            str(i).zfill(2) + '00_UTC.png'
                    
                plot_profile_and_NoShip_difference(
                        avg_distances_350, diff[i, :], var, zero_index, var.upper() +
                        ' change across shipping corridor, time series ' +
                        'average at ' + str(i).zfill(2) + '00 UTC', outfile, 
                        saveplot = False)                    

    # CALCULATE WEIGHTED AVERAGES PER TIME SLOT
    avg_distances_250 = copy.deepcopy(avg_distances)
    diff_250 = copy.deepcopy(diff)
    diff_250[:, abs(avg_distances) > 250] = np.nan
    centered_data_ts_mean_avg[:, abs(avg_distances) > 250] = np.nan
    avg_distances_250[abs(avg_distances) > 250] = np.nan
    
    weights = 1 / (1 + abs(avg_distances_250))
    indices = np.where(np.logical_not(np.isnan(avg_distances_250)))[0]
    
    avg_diff = np.full(24, np.nan)
    diurnal_corridor_avg = np.full(24, np.nan)
    for i in range(24):
        
        avg_diff[i] = np.average(diff_250[i, indices], 
                                 weights = weights[indices])
        diurnal_corridor_avg[i] = np.average(
            centered_data_ts_mean_avg[i, indices], weights = weights[indices])

    
#### Plot average diurnal differences and spatial averages

for i in range(24):
    
    if np.isnan(data_ts_mean[i, :, :]).any():
        
        avg_diff[i] = np.nan
        diurnal_corridor_avg[i] = np.nan
        
        
# saveplot = True

# hours = np.arange(24)

# # Create a figure and left axis
# fig, ax1 = plt.subplots()

# # Plot average differences on the left axis
# ldiff, = ax1.plot(hours, diurnal_corridor_avg, color='k', label='Corridor average')
# ax1.set_ylim([0, 200])
# ax1.set_xlabel('Time (UTC)')
# ax1.set_ylabel('[' + cdict.varUnits[var] + ']', color='k')
# ax1.tick_params('y', colors='k')

# # Create a secondary y-axis on the right
# ax2 = ax1.twinx()

# # Plot spatial averages on the right axis
# lavg, = ax2.plot(hours, avg_diff, color='b', label='Corridor effect')
# ax2.set_ylim([0, 10])
# ax2.set_ylabel('[' + cdict.varUnits[var] + ']', color='b')
# ax2.tick_params('y', colors='b')

# lines = [ldiff, lavg]
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels)

# fig.tight_layout()

# plt.show()

# if saveplot:
#     outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
#         str(end_year) + '/' + var.upper() + '_diurnal_change_and_mean.png'
#     fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')
    
    
import matplotlib.gridspec as gridspec

saveplot = True

hours = np.arange(24)

# Create a 2x1 grid
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

# Top subplot
ax1 = plt.subplot(gs[0])

# Plot average differences on the left axis
ldiff, = ax1.plot(hours, diurnal_corridor_avg, color='k', label='Corridor average')
ax1.set_ylim([0, 100])
# ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('[' + cdict.varUnits[var] + ']', color='k')
ax1.tick_params('y', colors='k')

# Bottom subplot
ax2 = plt.subplot(gs[1], sharex=ax1)

# Plot spatial averages on the right axis
lavg, = ax2.plot(hours, avg_diff, color='b', label='Corridor effect')
ax2.set_ylim([-2, 0])
ax2.set_xlabel('Time (UTC)')
ax2.set_ylabel('[' + cdict.varUnits[var] + ']', color='b')
ax2.tick_params('y', colors='b')

lines = [ldiff, lavg]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels)

fig.tight_layout()

plt.show()

if saveplot:
    outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
        str(end_year) + '/' + var.upper() + '_diurnal_change_and_mean.png'
    fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

### Plot diurnal spatial averages and effects in same axis    
saveplot = True

fig = plt.figure()
plt.plot(hours, diurnal_corridor_avg, color='grey', label='Corridor average')
plt.plot(hours, avg_diff, color='b', label='Corridor effect')
plt.plot(hours[8:16], np.zeros(len(hours))[8:16], ':', color = 'grey')
plt.ylim([-3, 100])
plt.xlabel('Time (UTC)')
plt.ylabel('[' + cdict.varUnits[var] + ']', color='k')
plt.legend()
plt.show()

if saveplot:
    outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
        str(end_year) + '/' + var.upper() + '_diurnal_change_and_mean_1axis.png'
    fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')


    
### Plot diurnal spatial averages only    
# saveplot = False

# fig = plt.figure()
# plt.plot(hours, diurnal_corridor_avg, color='grey', label='Corridor average')
# plt.xlim([-1, 24])
# plt.ylim([3, 13])
# plt.xlabel('Time (UTC)')
# plt.ylabel('[' + cdict.varUnits[var] + ']', color='k')
# plt.legend()
# plt.show()

# if saveplot:
#     outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
#         str(end_year) + '/' + var.upper() + '_diurnal_mean.png'
#     fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')



# =============================================================================
# Create some test maps
# =============================================================================

# outfile = 'Figures/' + var.upper() + '/' + str(start_year) + '-' +\
#     str(end_year) + '/' + var.upper() + '_' + str(start_year) + '-' +\
#         str(end_year) + '_average.png'
# create_map(lat_claas, lon_claas, data_ts_mean, np.nanmin(data_ts_mean), 
#             np.nanmax(data_ts_mean), var.upper() + ' ' + str(start_year) +
#             '-' + str(end_year) + ' average', 'cm-3', 'viridis', 'neither', 
#             outfile, saveplot = False)

# for m in range(12):
    
#     outfile = 'test'
    
#     create_map(lat_claas, lon_claas, data_seas_mean[:, :, m], 
#                np.nanmin(data_seas_mean), np.nanmax(data_seas_mean), 
#                'CDNC average in month ' + str(m+1).zfill(2), 'cm-3', 'viridis', 
#                'neither', outfile, saveplot = False)


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
'''