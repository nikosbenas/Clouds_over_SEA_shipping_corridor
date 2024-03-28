import copy
import glob
import os
import geopy.distance as gd
import h5py
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np
import sys
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
from claas3_dictionaries import varUnits
import cartopy.crs as ccrs
import cartopy.feature as cf
from scipy.interpolate import interp1d


def find_bounding_box_indices(bounding_box, lat_array, lon_array):

    '''
    Description:
        Calculates indices of a bounding box within latitude and longitude arrays, ensuring istart < iend. 

    Inputs:
        - bounding_box: (list) [min_lat, max_lat, min_lon, max_lon]
        - lat_array: (2D NumPy array) Latitude values
        - lon_array: (2D NumPy array) Longitude values

    Outputs:
        - istart, iend, jstart, jend: (integers) Indices of the bounding box area
    '''

    istart_idx = np.argmin(np.abs(lat_array[:, 0] - bounding_box[1]))
    iend_idx = np.argmin(np.abs(lat_array[:, 0] - bounding_box[3]))

    # Ensure istart is less than iend
    istart = min(istart_idx, iend_idx) + 1
    iend = max(istart_idx, iend_idx) + 1

    jstart = np.argmin(np.abs(lon_array[0, :] - bounding_box[0])) + 1
    jend = np.argmin(np.abs(lon_array[0, :] - bounding_box[2])) + 1

    return istart, iend, jstart, jend


def read_lat_lon_arrays(modis_file):

    '''
    Description:
        Reads latitude and longitude data from a netCDF file and returns meshgrid arrays.

    Inputs:
        - modis_file: (string) Path to the MODIS file

    Outputs:
        - latData: (2D NumPy array) Latitude values in meshgrid format
        - lonData: (2D NumPy array) Longitude values in meshgrid format
    '''

    with h5py.File(modis_file, 'r') as file:

        latData = file['lat'][:]
        lonData = file['lon'][:]

    lonData, latData = np.meshgrid(lonData, latData)

    return latData, lonData


def read_monthly_time_series(var, data_folder, start_year, end_year, istart, iend, jstart, jend):

    '''
    Description:
        Reads monthly time series data for a specific variable from files in a given data folder, within a specified time period and spatial bounding box.

    Inputs:
        - var: (string) Variable name to read from the files
        - data_folder: (string) Path to the folder containing the data files
        - start_year: (int) Start year of the time series
        - end_year: (int) End year of the time series
        - istart: (int) Starting latitude index
        - iend: (int) Ending latitude index
        - jstart: (int) Starting longitude index
        - jend: (int) Ending longitude index

    Outputs:
        - var_data: (3D NumPy array) Time series data for the specified variable within the specified time period and spatial bounding box
    '''

    var_data = []

    for year in range(start_year, end_year + 1):

        for month in range(1, 13):

            print('Reading ' + var + ', ' + str(year) + '/' + str(month).zfill(2))

            # Find the file 
            file_pattern = str(year) + str(month).zfill(2)
            file_name = glob.glob(os.path.join(data_folder, f"*{file_pattern}*"))[0]

            with Dataset(file_name, 'r') as all_data:

                if all_data[var].ndim == 2: # e.g. MODIS data

                    var_data.append(all_data[var][istart:iend, jstart:jend])

                if all_data[var].ndim == 3: # e.g. CLAAS data

                    var_data.append(np.flipud(all_data[var][0, istart:iend, jstart:jend]))

                # Read fill value once
                if 'fill_value' not in locals():

                    fill_value = all_data[var]._FillValue

    # Convert list of 2D arrays into 3D array (lat, lon, time)
    var_data = np.dstack(var_data)

    # Replace fill value with nan
    var_data[var_data == fill_value] = np.nan

    # Adjust units of some variables
    if 'cre' in var:
        
        var_data = var_data * 1e+6 # convert m to micron
        
    if 'wp' in var: # LWP or IWP
    
        var_data = var_data * 1000 # convert kg/m2 to g/m2
        
    if 'cdnc' in var:
        
        var_data = var_data * 1e-6 # convert m-3 to cm-3

    return var_data


def calculate_area_weighted_average(array, lat_array):
    
    '''
    Description:
        Calculates the area-weighted average of a 2D or 3D array over latitude.
        ATTENTION: If the data array is masked, the latitude array should be similarly masked.

    Inputs:
        - array: (2D or 3D NumPy array) Data array to calculate the area-weighted average
        - lat_array: (2D NumPy array) Latitude array corresponding to the data array

    Outputs:
        - avg: (float or 1D NumPy array) Area-weighted average of the input array over latitude
    '''
    
    if array.ndim == 2:

        if np.all(np.isnan(array)):

            return np.nan
        
        else:

            return np.nansum(np.cos(np.radians(lat_array)) * array) / np.nansum(np.cos(np.radians(lat_array)))
        
    elif array.ndim == 3:

        if np.all(np.isnan(array)):

            return np.nan
        
        else:

            lat_array_tiled = np.tile(lat_array[:,:,np.newaxis], array.shape[2])

            return np.nansum(np.cos(np.radians(lat_array_tiled)) * array, axis=(0, 1)) / np.nansum(np.cos(np.radians(lat_array)), axis=(0, 1))
        
    else:

        raise ValueError("Input array should be either 2D or 3D.")


def plot_time_series(dates, array, var_name, title, saveplot, output_file):

    '''
    Description:
        Plots a time series of data with dates on the x-axis and the corresponding array values on the y-axis. Optionally, vertical dotted lines can be added at the beginning of each year. The plot includes labels for the x-axis, y-axis, and title. If specified, the plot can be saved to an output file.

    Inputs:
        - dates: (list or array) Dates corresponding to the data points in the time series
        - array: (list or array) Values of the time series data
        - var_name: (string) Name of the variable being plotted
        - title: (string) Title of the plot
        - saveplot: (bool) If True, the plot will be saved to an output file
        - output_file: (string) Path to the output file for saving the plot

    Outputs:
        - None
    '''

    fig = plt.figure()

    plt.plot(dates, array, color = 'b')

    # Add vertical dotted lines at the beginning of each year
    for year in range(min(dates).year, max(dates).year + 1):

        year_start = np.datetime64(str(year), 'Y')

        plt.axvline(x=year_start, color='grey', linestyle='--', linewidth=0.8)

    plt.xlabel('Month')

    plt.ylabel(var_name.upper() + ' [' + varUnits[var_name] + ']')

    plt.title(title)

    if saveplot:
        fig.savefig(output_file, dpi = 300, bbox_inches = 'tight')


def plot_two_time_series(dates, var_name_1, array_1, var_name_2, array_2, title, saveplot, output_file):

    '''
    Description:
        Plots two time series on the same plot with shared x-axis (dates) and dual y-axes. Each time series is represented by a line plot with specified colors. Vertical dotted lines are added at the beginning of each year for reference. The plot includes labels for the x-axis, y-axes, and title. If specified, the plot can be saved to an output file.

    Inputs:
        - dates: (list) Dates corresponding to the data points in the time series
        - var_name_1: (string) Name of the first variable being plotted
        - array_1: (1D NumPy array) Values of the first time series data
        - var_name_2: (string) Name of the second variable being plotted
        - array_2: (1D NumPy array) Values of the second time series data
        - title: (string) Title of the plot
        - saveplot: (bool) If True, the plot will be saved to an output file
        - output_file: (string) Path to the output file for saving the plot

    Outputs:
        - None
    '''

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Month')

    ax1.set_ylabel(varUnits[var_name_1])

    line1, = ax1.plot(dates, array_1, color = 'orange', label = var_name_1.upper())

    ax2 = ax1.twinx()

    ax2.set_ylabel(varUnits[var_name_2])

    line2, = ax2.plot(dates, array_2, color = 'blue', label = var_name_2.upper())

    # Combine handles and labels from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    # Create a single legend for both lines
    ax1.legend(lines, labels, loc='best')

    # Set y-axis tick labels color to match corresponding lines
    for line, ax in zip(lines, [ax1, ax2]):
        for tick_label in ax.get_yticklabels():
            tick_label.set_color(line.get_color())

    # Add vertical dotted lines at the beginning of each year
    for year in range(min(dates).year, max(dates).year + 1):

        year_start = np.datetime64(str(year), 'Y')

        plt.axvline(x=year_start, color='grey', linestyle='--', linewidth=0.8)

    plt.title(title)

    if saveplot:
        fig.savefig(output_file, dpi = 300, bbox_inches = 'tight')


def calculate_running_mean(array, window_size):

    '''
    Description:
        Calculates the running mean of an array using a specified window size. The running mean is computed by averaging values within a moving window centered at each element of the array. Values at the beginning and end of the array, where the window size extends beyond the array boundaries, are treated as missing values (NaN).

    Inputs:
        - array: (1D NumPy array) Input array for which the running mean will be calculated
        - window_size: (int or float) Size of the moving window for computing the mean

    Outputs:
        - result: (1D NumPy array) Array containing the running mean values, with NaN values at the edges where the window extends beyond the array boundaries
    '''
    
    result = np.full_like(array, np.nan)
    
    for i in range(len(result)):
        
        # First and last window_size/2 elements
        if (i <= int(window_size/2)) or (i >= int(len(result)-(window_size/2))):
            
            continue
        
        start = int(i - window_size/2+1)
        end = int(i + window_size/2+1)
        
        result[i] = np.nanmean(array[start:end])
    
    return result        


def find_shipping_corridor_center_coordinates(flag_sc, lat_array, lon_array):
    
    '''
    Description:
        This function finds the latitude and longitude coordinates of the center of a shipping corridor based on a binary flag array indicating the presence of the corridor. It identifies the first and last occurrences of ones (representing the edges of the corridor) in each row of the flag array and calculates the center coordinates based on these occurrences.

    Inputs:
        - flag_sc: (2D NumPy array) Binary flag array indicating the presence of the shipping corridor
        - lat_array: (2D NumPy array) Latitude coordinates corresponding to the flag array
        - lon_array: (2D NumPy array) Longitude coordinates corresponding to the flag array

    Outputs:
        - sc_centlat: (1D Numpy array) Latitude coordinates of the center of the shipping corridor
        - sc_centlon: (1D NumPy array) Longitude coordinates of the center of the shipping corridor
    '''
    
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
            
        sc_centlat[i] = lat_array[i][index]
        sc_centlon[i] = lon_array[i][index]
      
    del index  
    
    return sc_centlat, sc_centlon


def find_angle_bewteen_shipping_corrridor_and_north(sc_centlat, sc_centlon):
    
    '''
    Description:
        This function calculates the angle between a shipping corridor and the north direction. It performs linear regression on the latitude and longitude coordinates of the shipping corridor to determine its orientation. Then, it calculates the angle between the orientation line and the north direction.

    Inputs:
        - sc_centlat: (1D NumPy array) Latitude coordinates of the shipping corridor center
        - sc_centlon: (1D NumPy array) Longitude coordinates of the shipping corridor center

    Outputs:
         - angle_radians: (float) Angle between the shipping corridor and the north direction, in radians
    '''
    
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


def find_line_perpendicular_to_corridor(c, sc_centlat, sc_centlon, angle_radians, lat_claas, lon_claas):

    '''
    Description:
        This function calculates information about the grid cells lying along the line perpendicular to the shipping corridor center, based on the specified latitudinal line index (c).

    Inputs:
        - c: Index representing a specific latitudinal line, based on the SC flag 2D NumPy array.
        - sc_centlat: 1D NumPy array containing latitude coordinates of the shipping corridor center (par latitude line).
        - sc_centlon: 1D NumPy array containing longitude coordinates of the shipping corridor center (per latitude line).
        - angle_radians: (float) Angle between the shipping corridor and the north direction, expressed in radians.
        - lat_claas: 2D NumPy array containing latitude coordinates of the grid cells.
        - lon_claas: 2D NumPy array containing longitude coordinates of the grid cells.

    Outputs:
        - distances: A 1D list containing distances (in kilometers) of the grid cells lying along the perpendicular line from the corridor center.
        - lat_indices: A 1D list containing latitude indices of the grid cells lying along the perpendicular line.
        - lon_indices: A 1D list containing longitude indices of the grid cells lying along the perpendicular line.
    '''
    
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


def center_shipping_corridor_perpendicular_lines(perpendicular_lines_lat_indices, perpendicular_lines_lon_indices, perpendicular_lines_distances):

    '''
    Description:
        This function is designed to center shipping corridor perpendicular lines by padding them with NaN values. It takes lists of 1D arrays of latitude indices, longitude indices, and distances of grid cells lying along perpendicular lines from the corridor center as input. The function calculates the maximum index of the "zero" distance point in the lists and pads each list with NaN values to center them around this maximum index.

    Inputs:
        - perpendicular_lines_lat_indices: A list containing 1D arrays of latitude indices of grid cells lying along perpendicular lines from the corridor center (each 1D array corresponds to one line).
        - perpendicular_lines_lon_indices: A list containing 1D arrays of longitude indices of grid cells lying along perpendicular lines from the corridor center.
        - perpendicular_lines_distances: A list containing distances (in kilometers) of grid cells lying along perpendicular lines from the corridor center.

    Outputs:
        - centered_lat_indices: A 2D NumPy array containing centered latitude indices of grid cells, padded with NaN values.
        - centered_lon_indices: A 2D NumPy array containing centered longitude indices of grid cells, padded with NaN values.
        - centered_distances: A 2D NumPy array containing centered distances (in kilometers) of grid cells, padded with NaN values.
    '''

    # Initialize lists to store centered data
    zero_indices = []
    zero_indices_reverse = []
    centered_distances = []
    centered_lat_indices = []
    centered_lon_indices = []

    # Center each list by adding NaN values
    for i in range(len(perpendicular_lines_distances)):
        
        zero_indices.append(perpendicular_lines_distances[i].index(0))
        zero_indices_reverse.append(len(perpendicular_lines_distances[i]) -
                                    perpendicular_lines_distances[i].index(0))
        
    max_zero_ind = max(zero_indices)
    max_zero_ind_rev = max(zero_indices_reverse)
        
    for i in range(len(perpendicular_lines_distances)):
        
        zero_ind = perpendicular_lines_distances[i].index(0)
        zero_ind_rev = len(perpendicular_lines_distances[i]) - perpendicular_lines_distances[i].index(0)
        
        pad_left = max_zero_ind - zero_ind
        pad_right = max_zero_ind_rev - zero_ind_rev
        
        centered_dist = np.concatenate([np.nan*np.ones(pad_left), perpendicular_lines_distances[i], 
                                       np.nan*np.ones(pad_right)])
        centered_distances.append(centered_dist)
        
        centered_lat_ind = np.concatenate(
            [np.nan*np.ones(pad_left), perpendicular_lines_lat_indices[i], 
             np.nan*np.ones(pad_right)])
        centered_lat_indices.append(centered_lat_ind)
        
        centered_lon_ind = np.concatenate(
            [np.nan*np.ones(pad_left), perpendicular_lines_lon_indices[i], 
             np.nan*np.ones(pad_right)])
        centered_lon_indices.append(centered_lon_ind)
        
        
    return (np.vstack(centered_lat_indices), np.vstack(centered_lon_indices), 
            np.vstack(centered_distances))


def center_data_along_corridor(data_array, centered_lat_indices, centered_lon_indices):

    '''
    Description:
        This function centers a given 2D data array along a shipping corridor defined by latitude and longitude indices. It extracts the data values corresponding to the centered latitude and longitude indices from the provided data array, filling invalid indices with NaN values.

    Inputs:
        - data_array: A 2D NumPy array containing the original data.
        - centered_lat_indices: A 2D NumPy array containing centered latitude indices of grid cells along the shipping corridor.
        - centered_lon_indices: A 2D NumPy array containing centered longitude indices of grid cells along the shipping corridor.

    Outputs:
        - centered_data: A 2D NumPy array containing the data values centered along the shipping corridor. Invalid indices are filled with NaN values.
    '''
    
    valid_indices = ~np.isnan(centered_lat_indices) & ~np.isnan(centered_lon_indices)
    
    centered_data = np.full_like(centered_lat_indices, np.nan)
    
    centered_data[valid_indices] = data_array[
        centered_lat_indices[valid_indices].astype(int), 
        centered_lon_indices[valid_indices].astype(int)]

    return centered_data


def make_map(var, data_array, title, minval, maxval, grid_extent, plot_extent, cmap, ext, filename, saveplot):
    
    '''
    Description:
        This function generates a map plot using the provided array data, with customizable settings such as title, colorbar, and extent. It utilizes Matplotlib and Cartopy for plotting geographical data.

    Inputs:
        - var: A string with the name of the variable.
        - data_array: A 2D NumPy array containing the data to be plotted on the map.
        - title: A string representing the title of the plot.
        - minval: Minimum value of the color scale.
        - maxval: Maximum value of the color scale.
        - grid_extent: A list specifying the extent of the grid (format: [lon_min, lon_max, lat_min, lat_max]).
        - plot_extent: A list specifying the geographical extent of the plot (format: [lon_min, lon_max, lat_min, lat_max]).
        - cmap: A Matplotlib colormap object representing the colormap to be used for the plot.
        - ext: A string indicating the extension style for the colorbar (e.g., 'both', 'min', 'max', 'neither').
        - filename: A string specifying the filename for saving the plot.
        - saveplot: A boolean indicating whether to save the plot as an image file.

    Outputs:
        - None
    '''

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    ax.set_title(title)
    im = ax.imshow(data_array, vmin=minval, vmax=maxval, cmap=cmap,
                   extent=grid_extent, transform=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE)
    ax.set_extent(plot_extent)
    
    # Add a colorbar
    # cbar_ax = fig.add_axes([0.95, 0.35, 0.03, 0.3])
    cbar = fig.colorbar(im, shrink = 0.6, extend = ext)#, cax=cbar_ax)
    cbar.set_label('[' + varUnits[var] + ']')
    
    # Draw rectangle
    # rect = patches.Rectangle((grid_extent[0], grid_extent[2]),
    #                           grid_extent[1] - grid_extent[0],
    #                           grid_extent[3] - grid_extent[2],
    #                           linewidth=1, edgecolor='orange', facecolor='none')
    
    # rect = patches.Rectangle((-10, -20),
    #                           10 - -10,
    #                           -10 - -20,
    #                           linewidth=1, edgecolor='orange', facecolor='none')
    # ax.add_patch(rect)
    
    plt.tight_layout()
    
    if saveplot:
        print(f'Save figure: {filename}')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    # plt.show()
    plt.close()


def plot_profile_and_NoShip_line(var, profile_data, profile_data_std, profile_NoShip, distance, zero_index, title, outfile, plot_NoShip_line, plot_std_band, saveplot):

    '''
    Description:
        This function generates a plot displaying data profiles across the shipping corridor, with options to include a line representing data excluding the corridor ("SC excl.") and a shaded band indicating the standard deviation of the data ("SC incl.").

    Input:
        - var: A string of the variable name.
        - profile_data: A 1D NumPy array with the data profile across the corridor.
        - profile_data_std: A 1D NumPy array with the standard deviation of the data, used to plot the shaded band.
        - profile_NoShip: A 1D NumPy array with the data profile excluding the corridor.
        - distance: A 1D NumPy array with distance values from the corridor center, used as the x-axis.
        - zero_index: Index of the zero line on the x-axis.
        - title: A string with the title of the plot.
        - outfile: A string with the file path for saving the plot.
        - plot_NoShip_line: Boolean indicating whether to plot the "SC excl." line.
        - plot_std_band: Boolean indicating whether to plot the shaded band representing the standard deviation.
        - saveplot: Boolean indicating whether to save the plot.

    Output:
        - If saveplot is set to True, the plot is saved to the specified file path.
    '''
    
    fig = plt.figure()
    plt.plot(distance, profile_data, label = 'SC incl.')
    if plot_NoShip_line:
        plt.plot(distance, profile_NoShip, linestyle = ':', color = 'k', label = 'SC excl.')
        
    # plt.plot(distance, np.full_like(distance, 0), linestyle = ':', color='grey')
        
    plt.axvline(x = distance[zero_index], linestyle = ':', color='grey')

    # Plot standard deviation as a light blue band around the main data line
    if plot_std_band:
        plt.fill_between(distance, profile_data - profile_data_std, profile_data + profile_data_std, color='lightblue', alpha=0.5)

    plt.ylabel('[' + varUnits[var] + ']')
    plt.xlabel('Distance from corridor center, W to E [km]')
    plt.title(title)
    if plot_NoShip_line:
        plt.legend()
        
    if saveplot:

        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')


def calculate_NoShip_line(distance, profile_data, half_range):
    
    '''
    Description:
        This function calculates the NoShip line based on the shipping corridor center and a specified half range. It determines a line equation in the form y = ax + b, where 'y' represents the NoShip data, 'x' represents the distance from the center, and 'a' and 'b' are coefficients. the NoShip data are calculated based on linear interpolation. 

    Inputs:
        - distance: A 1D NumPy array representing the distance from the center for each data point.
        - profile_data: A 1D NumPy array containing the original profile data.
        - half_range: The half range in kilometers from the corridor center to define the two interpolation points.

    Outputs:
        - NoShip_line: A 1D NumPy array representing the linearly interpolated NoShip line values at each distance point.                            
    '''
    
    # Find indices of grid cells defining the range -half_range km to 
    # half_range km from center.
    iw = np.argmin(abs(-half_range - distance)) # index west
    ie = np.argmin(abs(half_range - distance)) # index east

    # Find a and b in line y = ax + b
    x1 = distance[iw]
    x2 = distance[ie]
    y1 = profile_data[iw]
    y2 = profile_data[ie]

    a = (y2 - y1) / (x2 - x1)
    b = y2 - a*x2

    # Find line values at distance points
    return a * distance + b


def calculate_NoShip_curve(distance, profile_data, half_range):
    
    '''
    Description:
        This function calculates the NoShip curve using a cubic polynomial fit. The cubic fit uses only data within a range of 350 km to half_range km from both sides of the corridor center.

    Inputs:
        - distance: A 1D NumPy array representing the distance from the corridor center for each data point.
        - profile_data: A 1D NumPy array containing the original profile data.
        - half_range: The half range in kilometers from the corridor center to define the corridor-affected range.

    Outputs:
        - y_interpolated: A 1D NumPy array representing the interpolated NoShip curve values at each distance point.                            
    '''

    # Find indices of grid cells defining the range -half_range km to 
    # half_range km from center.
    iw = np.argmin(abs(-half_range - distance)) # index west
    ie = np.argmin(abs(half_range - distance)) # index east

    # For the cubic fit, onsider only data between 350 km and 250 km from both sides of the corridor center
    iw_end = np.argmin(abs(-350 - distance)) # index west
    ie_start = np.argmin(abs(350 - distance)) # index east

    x_with_gap = np.concatenate((distance[ie_start:ie], distance[iw:iw_end]))
    y_with_gap = np.concatenate((profile_data[ie_start:ie], profile_data[iw:iw_end]))

    idx = np.isfinite(x_with_gap) & np.isfinite(y_with_gap)

    coefficients = np.polyfit(x_with_gap[idx], y_with_gap[idx], 3)
    a, b, c, d = coefficients

    y_interpolated = a*distance**3 + b*distance**2 + c*distance + d

    return y_interpolated


def calculate_across_corridor_average_and_std(centered_lat_indices, centered_lon_indices, data_array):

    '''
    Description:
        This function calculates the across-corridor average and standard deviation from centered data along the shipping corridor.

    Inputs:
        - centered_lat_indices: A 2D NumPy array containing centered latitude indices along the shipping corridor.
        - centered_lon_indices: A 2D NumPy array containing centered longitude indices along the shipping corridor.
        - data_array: A 2D NumPy array containing "map" data values.

    Outputs:
        - centered_data_avg: A 1D NumPy array representing the across-corridor average of the data.
        - centered_data_std: A 1D NumPy array representing the across-corridor standard deviation of the data.
        - centered_data_N: A 1D NumPy array representing the across-corridor number of grid cells used in the average.
    '''
    
    centered_data = center_data_along_corridor(data_array, centered_lat_indices, centered_lon_indices)

    centered_data_avg = np.nanmean(centered_data, axis = 0)
    centered_data_std = np.nanstd(centered_data, axis = 0)

    centered_data_N = (np.nansum(centered_data, axis = 0) / centered_data_avg)
    return centered_data_avg, centered_data_std, centered_data_N


def create_short_across_corridor_profiles(limit, avg_distances, data_long):

    '''
    Description:
        This function creates a short across-corridor profile by limiting the distance range based on a specified threshold. It selectively removes data points beyond the limit from the input profile.

    Inputs:
        - limit: A float representing the distance threshold beyond which data points are removed.
        - avg_distances: A 1D NumPy array representing the across-corridor average distances of the data.
        - data_long: A 1D NumPy array representing the across-corridor average of the mean data.
        
    Outputs:
        - data_short: A 1D NumPy array representing the shortened across-corridor average of the mean data.
    '''

    data_short = copy.deepcopy(data_long)

    data_short[abs(avg_distances) > limit] = np.nan

    return data_short


def center_along_corridor_data_and_uncertainties_per_month(centered, mean_per_month, unc_per_month):

    '''
    Description:
        - This function centers monthly mean data and uncertainties along the shipping corridor. The data are in 3D arrays, with the third dimension denoting the month (1-12).

    Inputs:
        - centered (dict): A dictionary containing centered latitude and longitude indices.
        - mean_per_month: A 3D NumPy array containing time series mean data per month.
        - unc_per_month: A 3D NumPy array containing time series mean uncertainty data per month.

    Outputs:
        - centered['mean_per_month']: A 3D NumPy array representing the centered along the shipping corridor mean data per month.
        - centered['unc_per_month']: A 3D NumPy array representing the centered monthly along the shipping corridor uncertainty data per month.
    '''

    centered_mean_per_month_list = []
    centered_unc_per_month_list = []

    for m in range(12):

        centered_mean_per_month_list.append(center_data_along_corridor(mean_per_month[: ,:, m], centered['latitude_indices'], centered['longitude_indices']))

        centered_unc_per_month_list.append(center_data_along_corridor(unc_per_month[: ,:, m], centered['latitude_indices'], centered['longitude_indices']))

    centered['mean_per_month'] = np.stack(centered_mean_per_month_list, axis = 2)
    centered['unc_per_month'] = np.stack(centered_unc_per_month_list, axis = 2)

    return centered['mean_per_month'], centered['unc_per_month']



