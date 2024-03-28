from datetime import datetime
import multiprocessing
import sys
import numpy as np
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import FileNameStart
from shipping_corridor_functions import calculate_NoShip_curve, calculate_NoShip_line, calculate_across_corridor_average_and_std, center_data_along_corridor, center_shipping_corridor_perpendicular_lines, create_short_across_corridor_profiles, find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, make_map, read_lat_lon_arrays, read_monthly_time_series, plot_profile_and_NoShip_line


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
var = 'cdnc_liq'
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

# Find array indices of lat and lon at bounding box corners
istart, iend, jstart, jend = find_bounding_box_indices(bounding_box, lat_claas, lon_claas)

# Keep lat lon in bounding box only
lat_claas = lat_claas[istart:iend, jstart:jend]
lon_claas = lon_claas[istart:iend, jstart:jend]

# Loop over all years and months to read CLAAS-3 data and their uncertainties into 3D arrays
data = read_monthly_time_series(var, data_folder, start_year, end_year, istart, iend, jstart, jend)
data_unc = read_monthly_time_series(var + '_unc_mean', data_folder, start_year, end_year, istart, iend, jstart, jend)

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
centered_lat_indices, centered_lon_indices, centered_dists = center_shipping_corridor_perpendicular_lines(all_lat_indices, all_lon_indices, all_distances)

# Calculate average distances along the corridor and set "western" part to negative
avg_distances = np.nanmean(centered_dists, axis=0)
zero_index = np.where(avg_distances == 0)[0][0]
avg_distances[zero_index + 1:] = -avg_distances[zero_index + 1:]

# =============================================================================
# Analysis of time series per individual month
# =============================================================================

# Reshape time series array to add a month dimension
shape_4d = (data.shape[0], data.shape[1], end_year + 1 - start_year, 12)
data_4d = data.reshape(shape_4d) 
data_unc_4d = data_unc.reshape(shape_4d) 

# Calculate time series mean per individual month ...
mean_per_month = np.nanmean(data_4d, axis = 2)
std_per_month = np.nanstd(data_4d, axis = 2)
N_per_month = np.round(np.nansum(data_4d, axis = 2) / mean_per_month).astype(int)
# ... and propagate the uncertainty
unc_per_month = np.sqrt(((1 / N_per_month) * (std_per_month**2)) + unc_coeff * (np.nanmean(data_unc_4d, axis = 2)**2))

# Create maps of mean values and uncertainties per month
create_monthly_maps = True
if create_monthly_maps:

    for m in range(12):

        # Of means per month
        make_map(var, mean_per_month[:, :, m], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' ' + month_string[m+1] + ' average', np.nanmin(mean_per_month[:, :, m]), np.nanmax(mean_per_month[:, :, m]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/' + var.upper() + '_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '_average.png', saveplot = True)

        # Of uncertainties per month
        make_map(var, unc_per_month[:, :, m], var.upper() + ' ' + str(start_year) + '-' + str(end_year) + ' ' + month_string[m+1] + ' average uncertainty', np.nanmin(unc_per_month[:, :, m]), np.nanmax(unc_per_month[:, :, m]), grid_extent, plot_extent, 'viridis', 'neither', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/' + var.upper() + '_month_' + str(m+1).zfill(2) + '_' + str(start_year) + '-' + str(end_year) + '_uncertainty.png', saveplot = True)

# Find data mean values centered along the shipping corridor
centered_avg, centered_std, centered_N = calculate_across_corridor_average_and_std(centered_lat_indices, centered_lon_indices, time_series_mean)  

# Calculate uncertainty of centered data
centered_unc = center_data_along_corridor(time_series_unc_mean, centered_lat_indices, centered_lon_indices)
centered_avg_unc = np.sqrt(((1 / centered_N) * (centered_std**2)) + unc_coeff * (np.nanmean(centered_unc, axis = 0)**2))

# Calculate straight line to imitate absence of the shipping corridor (curve in the CFC case)

if var == 'cfc':
    centered_avg_NoShip = calculate_NoShip_curve(avg_distances, centered_avg, 250)
else:
    centered_avg_NoShip = calculate_NoShip_line(avg_distances, centered_avg, 250)

# Create shorter profile plots mean values and uncertainties, centered on the corridor
avg_distances_short = create_short_across_corridor_profiles(350, avg_distances, avg_distances)
centered_avg_short = create_short_across_corridor_profiles(350, avg_distances, centered_avg)
centered_avg_NoShip_short = create_short_across_corridor_profiles(350, avg_distances, centered_avg_NoShip)
centered_avg_unc_short = create_short_across_corridor_profiles(350, avg_distances, centered_avg_unc)


create_profile_plots = True
if create_profile_plots:

    # Plot long profile of mean values
    plot_profile_and_NoShip_line(var, centered_avg, centered_avg_unc, centered_avg_NoShip, avg_distances, zero_index, var.upper() + ' across shipping corridor, time series average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_across_sc_long.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)

    # Plot profile of mean values
    plot_profile_and_NoShip_line(var, centered_avg_short, centered_avg_unc_short, centered_avg_NoShip_short, avg_distances_short, zero_index, var.upper() + ' across shipping corridor, time series average', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_across_sc.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)


create_profile_difference_plots = True
if create_profile_difference_plots:

    # Plot profile of mean values
    plot_profile_and_NoShip_line(var, centered_avg_short - centered_avg_NoShip_short, centered_avg_unc_short, np.zeros_like(centered_avg_short), avg_distances_short, zero_index, var.upper() + ' change due to shipping corridor', 'Figures/' + var.upper() + '/' + str(start_year) + '-' + str(end_year) + '/'  + var.upper() + '_time_series_mean_change_across_sc.png', plot_NoShip_line = True, plot_std_band = True, saveplot = True)

print('check')