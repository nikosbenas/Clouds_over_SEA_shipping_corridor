

from datetime import datetime
import multiprocessing
import sys
import numpy as np
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import FileNameStart
from shipping_corridor_functions import find_angle_bewteen_shipping_corrridor_and_north, find_bounding_box_indices, find_line_perpendicular_to_corridor, find_shipping_corridor_center_coordinates, read_lat_lon_arrays, read_monthly_time_series


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
claas_data_folder = '/net/pc190604/nobackup/users/benas/CLAAS-3/Level_3/' + FileNameStart[var]

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

# =============================================================================
# Process CLAAS-3 data
# =============================================================================

# Read CLAAS lat, lon once
claas3_aux_file = '/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends/claas3_level3_aux_data_005deg.nc'
lat_claas, lon_claas = read_lat_lon_arrays(claas3_aux_file)

# Find array indices of lat and lon at bounding box corners
istart, iend, jstart, jend = find_bounding_box_indices(bounding_box, lat_claas, lon_claas)

# Keep lat lon in bounding box only
lat_claas = lat_claas[istart:iend, jstart:jend]
lon_claas = lon_claas[istart:iend, jstart:jend]

# Loop over all years and months to read MODIS data into a 3D array
claas_data = read_monthly_time_series(var, claas_data_folder, start_year, end_year, istart, iend, jstart, jend)

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


print('check')