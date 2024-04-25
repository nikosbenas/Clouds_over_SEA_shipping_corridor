import copy
import glob
import os
import geopy.distance as gd
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
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


def block_average(array, block_size):

    '''
    Description:
        This function downsamples a 2D numpy array by averaging blocks of a specified size.

    Inputs:
        - array: The input 2D numpy array to be downsampled.
        - block_size (int or float): The size of the blocks (both rows and columns) over which to average the array.

    Output:
        - The downsampled 2D array where each element is the average of a block of size (block_size, block_size) from the input array. The dimensions of the output array are (nrows_blocks, ncols_blocks).
    '''
    
    # Number of rows and columns in the array
    nrows, ncols = array.shape
    
    # Calculate number of blocks in rows and columns
    nrows_blocks = nrows // block_size
    ncols_blocks = ncols // block_size

    # Reshape the array into blocks of size (block_size, block_size)
    arr_reshaped =\
        (array[:nrows_blocks*block_size, :ncols_blocks*block_size].
         reshape(nrows_blocks, block_size, ncols_blocks, block_size))
    
    # Average the blocks along the last two axes to get downsampled array
    return arr_reshaped.mean(axis=(1, 3))


def read_lat_lon_arrays(input_file):

    '''
    Description:
        Reads latitude and longitude data from a netCDF file and returns meshgrid arrays.

    Inputs:
        - input_file: (string) Path to the MODIS file

    Outputs:
        - latData: (2D NumPy array) Latitude values in meshgrid format
        - lonData: (2D NumPy array) Longitude values in meshgrid format
    '''

    with h5py.File(input_file, 'r') as file:

        latData = file['lat'][:]
        lonData = file['lon'][:]

    lonData, latData = np.meshgrid(lonData, latData)

    return latData, lonData


def read_monthly_time_series(var, data_folder, start_year, end_year, istart, iend, jstart, jend, read_diurnal):

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
        - read_diurnal: (boolean) Flag for reading monthly mean diurnal data

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

                    if read_diurnal:

                        file_data = all_data[var][:, istart:iend, jstart:jend]

                        for i in range(24):

                            file_data[i, :, :] = np.flipud(file_data[i, :, :])

                        var_data.append(file_data)

                    else:

                        var_data.append(np.flipud(all_data[var][0, istart:iend, jstart:jend]))

                # Read fill value once
                if 'fill_value' not in locals():

                    fill_value = all_data[var]._FillValue

    # Convert list of 2D arrays into 3D array (lat, lon, time)
    if read_diurnal:
        var_data = np.stack(var_data, axis = 3)
    else:
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
        - std_dev: (float or 1D NumPy array) Area-weighted standard deviation of the input array over latitude
    '''
    
    weights = np.cos(np.radians(lat_array))

    if array.ndim == 2:

        if np.all(np.isnan(array)):

            return np.nan, np.nan
        
        else:

            avg = np.nansum(weights * array) / np.nansum(weights)

            std_dev = np.sqrt(np.nansum(weights * (array - avg)**2) / np.nansum(weights))

            return avg, std_dev
        
    elif array.ndim == 3:

        if np.all(np.isnan(array)):

            return np.nan, np.nan
        
        else:

            lat_array_tiled = np.tile(lat_array[:,:,np.newaxis], array.shape[2])

            weights_3D = np.cos(np.radians(lat_array_tiled))

            avg = np.nansum(weights_3D * array, axis=(0, 1)) / np.nansum(weights, axis=(0, 1))

            std_dev = np.zeros(array.shape[2])
            for m in range(array.shape[2]):

                std_dev[m] = np.sqrt(np.nansum(weights * (array[:,:,m] - avg[m])**2) / np.nansum(weights))
        
            return avg, std_dev
        
    else:

        raise ValueError("Input array should be either 2D or 3D.")


def plot_time_series(dates, array, array_unc, var_name, title, output_file, plot_unc_band, plot_zero_line, saveplot):

    '''
    Description:
        Plots a time series of data with dates on the x-axis and the corresponding array values on the y-axis. Optionally, vertical dotted lines can be added at the beginning of each year. The plot includes labels for the x-axis, y-axis, and title. If specified, the plot can be saved to an output file.

    Inputs:
        - dates: (list or array) Dates corresponding to the data points in the time series.
        - array: (list or array) Values of the time series data.
        - array_unc: (list or array) Values of the time series data uncertainties.
        - var_name: (string) Name of the variable being plotted.
        - title: (string) Title of the plot.
        - output_file: (string) Path to the output file for saving the plot.
        - plot_unc_band: Boolean indicating whether to plot uncertainty bands around the main data line.
        - plot_zero_line: Boolean indicating whether to plot a dotted grey zero line.
        - saveplot: Boolean indicating wether to save th plot to an output file.

    Outputs:
        - None
    '''

    fig = plt.figure()

    plt.plot(dates, array, color = 'b')

    # Add vertical dotted lines at the beginning of each year
    for year in range(min(dates).year, max(dates).year + 1):

        year_start = np.datetime64(str(year), 'Y')

        plt.axvline(x=year_start, color='grey', linestyle='--', linewidth=0.8)

    # Plot uncertainty as a light blue band around the main data line
    if plot_unc_band:

        plt.fill_between(dates, array - array_unc, array + array_unc, color = 'lightblue', alpha = 0.5, linewidth = 0)

    if plot_zero_line:

        plt.axhline(y = 0, linestyle = ':', color = 'k')

    plt.xlabel('Month')

    plt.ylabel(var_name.upper() + ' [' + varUnits[var_name] + ']')

    plt.title(title)

    if saveplot:
        fig.savefig(output_file, dpi = 300, bbox_inches = 'tight')

    plt.close()


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

    plt.close()


def plot_intra_annual_variation(var, array, unc, title, outfile, plot_std_band, plot_zero_line, saveplot):

    '''
    Description:
        This function generates a plot of the intra-annual (seasonal) variation of a variable throughout the year. It displays values of the variable for each month, optionally accompanied by a band representing propagated uncertainties. The plot can be saved to a file if specified.

    Input:
        - var: Variable name (e.g., 'cdnc_liq', 'cre_liq').
        - array: 1D NumPy array containing the values of the variable for each month.
        - unc: 1D NumPy array containing the propagated uncertainties for each month.
        - title: a string containing the title of the plot.
        - outfile: Filepath to save the plot.
        - plot_std_band: Boolean indicating whether to plot uncertainty bands around the main data line.
        - plot_zero_line: Boolean indicating whether to plot a dotted line at y = 0.
        - saveplot: Boolean indicating whether to save the plot to outfile.
    '''

    fig = plt.figure()

    plt.plot(np.arange(1,13), array, color = 'k')

    # Plot standard deviation as a light grey band around the main data line
    if plot_std_band:

        plt.fill_between(np.arange(1,13), array - unc, array + unc, color = 'lightgrey', alpha = 0.5, linewidth = 0)

    if plot_zero_line:

        plt.axhline(y = 0, linestyle = ':', color = 'k')


    plt.ylabel('[' + varUnits[var] + ']')
    plt.xlabel('Month')
    plt.title(title)
    
    if saveplot:
        
        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

    plt.close()


def plot_intra_annual_variation_for_two(var1, array1, unc1, var2, array2, unc2, title, outfile, plot_std_band, plot_zero_line, saveplot):
    '''
    Description:
        This function generates a plot of the intra-annual (seasonal) variation of two variables throughout the year. It displays values of the variables for each month, optionally accompanied by bands representing propagated uncertainties. The plot can be saved to a file if specified.

    Input:
        - var1: Variable name for the first y-axis (e.g., 'cdnc_liq', 'cre_liq').
        - array1: 1D NumPy array containing the values of the first variable for each month.
        - unc1: 1D NumPy array containing the propagated uncertainties for the first variable for each month.
        - var2: Variable name for the second y-axis.
        - array2: 1D NumPy array containing the values of the second variable for each month.
        - unc2: 1D NumPy array containing the propagated uncertainties for the second variable for each month.
        - title: a string containing the title of the plot.
        - outfile: Filepath to save the plot.
        - plot_std_band: Boolean indicating whether to plot uncertainty bands around the main data lines.
        - plot_zero_line: Boolean indicating whether to plot dotted lines at y = 0 in both y axes.
        - saveplot: Boolean indicating whether to save the plot to outfile.
    '''

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel(var1.upper() + ' [' + varUnits[var1] + ']', color=color1)
    ax1.plot(np.arange(1,13), array1, color=color1)
    if plot_std_band:
        ax1.fill_between(np.arange(1,13), array1 - unc1, array1 + unc1, color='lightblue', alpha=0.5, linewidth = 0)
    if plot_zero_line:
        ax1.axhline(y = 0, linestyle = ':', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # second y-axis that shares the same x-axis

    color2 = 'tab:orange'
    ax2.set_ylabel(var2.upper() + ' [' + varUnits[var2] + ']', color=color2)
    ax2.plot(np.arange(1,13), array2, color=color2)
    if plot_std_band:
        ax2.fill_between(np.arange(1,13), array2 - unc2, array2 + unc2, color='navajowhite', alpha=0.5, linewidth = 0)
    if plot_zero_line:
        ax2.axhline(y = 0, linestyle = ':', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title(title)

    if saveplot:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')

    plt.close()



def calculate_running_mean(array, array_unc, unc_coef, window_size):

    '''
    Description:
        Calculates the running mean of an array using a specified window size. The running mean is computed by averaging values within a moving window centered at each element of the array. Values at the beginning and end of the array, where the window size extends beyond the array boundaries, are treated as missing values (NaN).

    Inputs:
        - array: (1D NumPy array) Input array for which the running mean will be calculated.
        - array_unc: (1D NumPy array) Uncertainty values of input array elements, needed for the calculation of running mean uncertainties.
        - unc_coef: (float) Uncertainty coefficient [0, 1] value, characterizing the correlation between uncertainty values. It is needed for the calculation of running mean uncertainties.
        - window_size: (int or float) Size of the moving window for computing the mean.

    Outputs:
        - result: (1D NumPy array) Array containing the running mean values, with NaN values at the edges where the window extends beyond the array boundaries.
        - unc: (1D NumPy array) Array containing the propagated uncertainties of the running mean values, with NaN values at the edges where the window extends beyond the array boundaries.
    '''
    
    result = np.full_like(array, np.nan)
    unc = np.full_like(array, np.nan)
    
    for i in range(len(result)):
        
        # First and last window_size/2 elements
        if (i <= int(window_size/2)) or (i >= int(len(result)-(window_size/2))):
            
            continue
        
        start = int(i - window_size/2+1)
        end = int(i + window_size/2+1)
        
        result[i] = np.nanmean(array[start:end])

        unc[i] = np.sqrt((1/window_size) * (np.nanstd(array[start:end])**2) + unc_coef * (np.nanmean(array_unc)**2))
    
    return result, unc        


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
        - cmap: A Matplotlib colormap object representing the colormap to be used for the plot. Can be 'viridis' for continuous values, 'RdBu_r' for diverging.
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
    cbar = fig.colorbar(im, shrink = 0.6, extend = ext)#, cax=cbar_ax)
    cbar.set_label('[' + varUnits[var] + ']')

    plt.tight_layout()
    
    if saveplot:

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
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
        plt.fill_between(distance, profile_data - profile_data_std, profile_data + profile_data_std, color='lightblue', alpha=0.5, linewidth = 0)

    plt.ylabel('[' + varUnits[var] + ']')
    plt.xlabel('Distance from corridor center, W to E [km]')
    plt.title(title)
    if plot_NoShip_line:
        plt.legend()
        
    if saveplot:

        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

    plt.close()


def plot_change_and_zero_line(var, profile_data, profile_data_std, distance, zero_index, mean_val, unc_val, title, outfile, plot_std_band, saveplot):

    '''
    Description:
        This function generates a plot displaying profiles of changes across the shipping corridor, the zero line, and profile mean and uncertainty values, with options to include a a shaded band indicating the propagated uncertainty of the profile data.

    Input:
        - var: A string of the variable name.
        - profile_data: A 1D NumPy array with the data profile across the corridor.
        - profile_data_std: A 1D NumPy array with the standard deviation of the data, used to plot the shaded band.
        - distance: A 1D NumPy array with distance values from the corridor center, used as the x-axis.
        - zero_index: Index of the zero line on the x-axis.
        - mean_val: Mean value to display in a text box.
        - unc_val: Uncertainty value to display in a text box.
        - title: A string with the title of the plot.
        - outfile: A string with the file path for saving the plot.
        - plot_std_band: Boolean indicating whether to plot the shaded band representing the standard deviation.
        - saveplot: Boolean indicating whether to save the plot.

    Output:
        - If saveplot is set to True, the plot is saved to the specified file path.
    '''
    
    fig = plt.figure()
    plt.plot(distance, profile_data, label = 'SC incl.')
        
    plt.axvline(x = distance[zero_index], linestyle = ':', color='grey')

    plt.axhline(y = 0, linestyle = ':', color='grey')

    # Plot standard deviation as a light blue band around the main data line
    if plot_std_band:
        plt.fill_between(distance, profile_data - profile_data_std, profile_data + profile_data_std, color='lightblue', alpha=0.5, linewidth = 0)

    plt.ylabel('[' + varUnits[var] + ']')
    plt.xlabel('Distance from corridor center, W to E [km]')
    plt.title(title)
        
    # Add text box with mean and uncertainty values
    mean_str = f"{mean_val:.2f}"
    unc_str = f"{unc_val:.2f}"
    plt.text(0.95, 0.05, f"Mean = {mean_str} $\pm$ {unc_str} {varUnits[var]}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=10)


    if saveplot:

        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

    plt.close()


def calculate_NoShip_curve(distance, profile_data, corridor_half_range, outer_half_range, fit_order):
    
    '''
    Description:
        This function calculates the NoShip curve using a polynomial fit. The  fit uses only data within a range of 400 km to corridor_half_range km from both sides of the corridor center.

    Inputs:
        - distance: A 1D NumPy array representing the distance from the corridor center for each data point.
        - profile_data: A 1D NumPy array containing the original profile data.
        - corridor_half_range: The half range in kilometers from the corridor center to define the corridor-affected range.
        - outer_half_range: the outer limit of the unaffected area to be used for fitiing.
        - fit_order: order of the polynomial to be fitted.

    Outputs:
        - y_interpolated: A 1D NumPy array representing the interpolated NoShip curve values at each distance point.                            
    '''

    # Find indices of grid cells defining the range -corridor_half_range km to 
    # corridor_half_range km from center.
    iw = np.argmin(abs(-corridor_half_range - distance)) # index west
    ie = np.argmin(abs(corridor_half_range - distance)) # index east

    # For the cubic fit, onsider only data between 400 km and 250 km from both sides of the corridor center
    iw_end = np.argmin(abs(-outer_half_range - distance)) # index west
    ie_start = np.argmin(abs(outer_half_range - distance)) # index east

    x_with_gap = np.concatenate((distance[ie_start:ie], distance[iw:iw_end]))
    y_with_gap = np.concatenate((profile_data[ie_start:ie], profile_data[iw:iw_end]))

    idx = np.isfinite(x_with_gap) & np.isfinite(y_with_gap)

    if not any(idx): # All idx elements are false

        return np.full_like(distance, np.nan)

    coefficients = np.polyfit(x_with_gap[idx], y_with_gap[idx], fit_order)

    if fit_order == 1:

        a, b = coefficients

        y_interpolated = a*distance + b

    if fit_order == 3:

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


def plot_12_monthly_profiles(var, month_string, title, profiles, unc_profiles, avg_distances, zero_index, avg_distances_short, outfile, plot_unc_bands, plot_zero_line, saveplot):

    '''
    Description:
        This function generates a plot displaying 12 monthly profiles of a variable across the shipping corridor, with optional uncertainty bands and the zero line (when differences are plotted). The plot can be saved to a file if specified.

    Input:
        - var: Variable name (e.g., 'cdnc_liq', 'cre_liq').
        - month_string: Dictionary mapping month numbers to their names.
        - title: Title of the plot.
        - profiles: 2D NumPy array (n, 12) containing the monthly profiles to be plotted.
        - unc_profiles: 2D NumPy array (n, 12) containing the uncertainties of the monthly profiles.
        - avg_distances: 1D NumPy array (n) representing distances from the corridor center.
        - zero_index: Index of the zero value on the avg_distances array.
        - avg_distances_short: 1D NumPy array (n) containing NaN values for distances larger than a threshold.
        - outfile: Filepath to save the plot.
        - plot_unc_bands: Boolean indicating whether to plot uncertainty bands.
        - plot_zero_line: Boolean indicating whether to plot a zero line.
        - saveplot: Boolean indicating whether to save the plot to outfile.

    Output:
        - None.
    '''
    
    fig, ax = plt.subplots()

    # Define the tab20 colors
    tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Use the first 12 colors for the plot
    colors = tab20_colors[:12]

    for i in range(12):

        label = month_string[i + 1]

        mean = profiles[:, i]

        ax.plot(avg_distances_short, mean, label = label, color = colors[i])

        if plot_unc_bands:

            unc = unc_profiles[:, i]

            ax.fill_between(avg_distances_short, mean - unc, mean + unc, color = colors[i], alpha = 0.3, linewidth = 0)

    if plot_zero_line:

        ax.plot(avg_distances_short, mean - mean, color = 'grey', linestyle = ':')

    plt.axvline(x = avg_distances[zero_index], linestyle = ':', color='grey')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Distance from corridor center, W to E [km]')
    ax.set_ylabel('[' + varUnits[var] + ']')
    ax.set_title(title)

    if saveplot:

        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

    plt.close()


def plot_all_hourly_profiles(var, profiles, std_profiles, avg_distances, zero_index, avg_distances_short, title, outfile, plot_std_bands, plot_zero_line, saveplot):

    '''
    Description:
        This function generates a plot displaying all available hourly profiles of a variable across the shipping corridor, with optional standard deviation bands and the zero line (when differences are plotted). The plot can be saved to a file if specified.

    Input:
        - var: Variable name (e.g., 'cdnc_liq', 'cre_liq').
        - profiles: 2D NumPy array (24, n) containing the monthly profiles to be plotted.
        - std_profiles: 2D NumPy array (24, n) containing the standard deviations of the monthly profiles.
        - avg_distances: 1D NumPy array (n) representing distances from the corridor center.
        - zero_index: Index of the zero value on the avg_distances array.
        - avg_distances_short: 1D NumPy array (n) containing NaN values for distances larger than a threshold.
        - title: Title of the plot.
        - outfile: Filepath to save the plot.
        - plot_std_bands: Boolean indicating whether to plot standard deviation bands.
        - plot_zero_line: Boolean indicating whether to plot a zero line.
        - saveplot: Boolean indicating whether to save the plot to outfile.

    Output:
        - None.
    '''
    
    fig, ax = plt.subplots()

    # Define the tab20 colors
    if ('cfc' in var) or ('cth' in var):
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 24))
    else:
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))

    colors = tab20_colors[:]
    color_index = 0

    for i in range(24):

        mean = profiles[i, :]

        if not np.all(np.isnan(mean)):

            label = str(i).zfill(2) + ':30'

            ax.plot(avg_distances_short, mean, label = label, color = colors[color_index])

            if plot_std_bands:

                unc = std_profiles[:, i]

                ax.fill_between(avg_distances_short, mean - unc, mean + unc, color = colors[color_index], alpha = 0.3, linewidth = 0)

            color_index += 1

    if plot_zero_line:

        ax.plot(avg_distances_short, np.zeros_like(avg_distances_short), color = 'grey', linestyle = ':')

    plt.axvline(x = avg_distances[zero_index], linestyle = ':', color='grey')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Distance from corridor center, W to E [km]')
    ax.set_ylabel('[' + varUnits[var] + ']')
    ax.set_title(title)

    if saveplot:

        fig.savefig(outfile, dpi = 300, bbox_inches = 'tight')

    plt.close()


def plot_diurnal(var, y_diurnal, y_std, title, output_file, plot_zero_line, saveplot):

    '''
    Description:
        This function generates a diurnal plot showing the variation of a variable over a 24-hour period. The plot also displays shaded regions indicating the standard deviation. The plot can be optionally saved to an output file.

    Input:
        - var: A string representing the variable name.
        - y_diurnal: A 1D NumPy array containing the values of the variable over a 24-hour period.
        - y_std: A 1D NumPy array containing the standard deviation of the variable over a 24-hour period.
        - title: A string with the title of the plot.
        - output_file: A string with the file path for saving the plot.
        - plot_zero_line: Boolean indicating whether to plot a dotted line at y = 0.
        - saveplot: A boolean indicating whether to save the plot to the output file.

    Output:
        - If saveplot is set to True, the plot is saved to the specified output file.
    '''

    hours = np.arange(0.5, 24, 1)

    fig, ax = plt.subplots()

    # Set x-axis tick labels to display time in the format 'H:MM'
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{int(hour)}:{30 if hour % 1 == 0.5 else '00'}" for hour in hours])

    ax.plot(hours, y_diurnal, color='k')
    ax.fill_between(hours, y_diurnal - y_std, y_diurnal + y_std, color = 'k', alpha = 0.3, linewidth = 0)

    if plot_zero_line:

        plt.axhline(y = 0, linestyle = ':', color = 'k')

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('[' + varUnits[var] + ']', color='k')
    ax.set_title(title)

    if saveplot:

        fig.savefig(output_file, dpi = 300, bbox_inches = 'tight')


def calculate_average_profiles_per_month(unc_coeff, centered):

    '''
    This function calculates statistical properties of monthly data in a time series, centered along the shipping corridor. It takes in an uncertainty coefficient and a dictionary with monthly data and associated uncertainties centered along the corridor, then calculates the mean, standard deviation, and propagated uncertainty for each monthly profile, updating the dictionary with these calculated values.

    Inputs:
        - unc_coeff: A float representing the uncertainty coefficient. This value is used in the uncertainty calculation.
        - centered: A dictionary containing the following keys:
            'monthly_data': A 3D numpy array (centered lat, centered lon, time) containing centered monthly data.
            'monthly_data_unc': A 3D numpy array (centered lat, centered lon, time) containing the uncertainties associated with the monthly data.

    Outputs:
        The function updates the centered dictionary with the following keys:
            - 'monthly_profiles': A 2D numpy array (mean longitudinal distance, time) representing the mean of the monthly data along the corridor center.
            - 'monthly_profiles_std': A 2D numpy array representing the standard deviation of the monthly data along the corridor center.
            - 'monthly_profiles_N': A 2D numpy array representing the number of non-NaN values used in the mean calculation.
            - 'monthly_profiles_unc': A 2D numpy array representing the combined uncertainty of the monthly data, calculated as the square root of the sum of squared standard deviation and the squared mean of the uncertainty data multiplied by the given uncertainty coefficient.
    '''

    centered['monthly_profiles'] = np.nanmean(centered['monthly_data'], axis = 0)

    centered['monthly_profiles_std'] = np.nanstd(centered['monthly_data'], axis = 0)

    centered['monthly_profiles_N'] = np.nansum(centered['monthly_data'], axis = 0) / centered['monthly_profiles']

    centered['monthly_profiles_unc'] = np.sqrt((1 / centered['monthly_profiles_N']) * (centered['monthly_profiles_std']**2) + unc_coeff * (np.nanmean(centered['monthly_data_unc'], axis = 0)**2))



def calculate_average_corridor_value_per_month(unc_coeff, centered, core_half_range, avg_distances_short):

    '''
    This function calculates monthly averages and relevant statistical properties specifically for data within the shipping corridor (based on monthly averaged profiles), defined by a core half-range, based on average distances. It updates a dictionary with mean, standard deviation, sample size, and uncertainty values for the monthly data within the corridor.

    Inputs:
        - unc_coeff: A float representing the uncertainty coefficient. This value is used in the uncertainty calculation.
        - centered: A dictionary containing the following keys:
            'monthly_profiles': A 2D numpy array containing monthly profiles of data, centered along the corridor.
            'monthly_profiles_unc': A 2D numpy array containing uncertainties associated with the monthly profiles.
            core_half_range: A float representing the half-range of the corridor core.
            avg_distances_short: A 1D numpy array of average distances from the corridor center.

    Outputs:
        The function updates the centered dictionary with the following keys:
            - 'monthly_corridor_mean': A 1D numpy array representing the mean of the monthly profiles for data within the corridor core defined by the given half-range.
            - 'monthly_corridor_std': A 1D numpy array representing the standard deviation of the monthly profiles for data within the corridor core.
            - 'monthly_corridor_N': A 1D numpy array representing the number of non-NaN values used in each mean calculation within the corridor core.
            - 'monthly_corridor_unc': A 1D numpy array representing the combined uncertainty of the monthly profiles within the core corridor, calculated as the square root of the sum of squared standard deviation and the squared mean of the uncertainty data multiplied by the given uncertainty coefficient.
    '''

    centered['monthly_corridor_mean'] = np.stack([np.nanmean(centered['monthly_profiles'][abs(avg_distances_short) < core_half_range, i]) for i in range(centered['monthly_profiles'].shape[1])])

    centered['monthly_corridor_std'] = np.stack([np.nanstd(centered['monthly_profiles'][abs(avg_distances_short) < core_half_range, i]) for i in range(centered['monthly_profiles'].shape[1])])

    centered['monthly_corridor_N'] = np.stack([np.sum(np.isfinite(centered['monthly_profiles'][abs(avg_distances_short) < core_half_range, i])) for i in range(centered['monthly_profiles'].shape[1])])

    centered['monthly_corridor_unc'] = np.sqrt((1 / centered['monthly_corridor_N']) * (centered['monthly_corridor_std']**2) + unc_coeff * (np.nanmean(centered['monthly_profiles_unc'], axis = 0)**2))


def calculate_temporal_mean_corridor_effect(temporal, unc_coeff, centered, core_half_range, avg_distances_short, corridor_effect):

    '''
    This function calculates monthly averages and relevant statistical properties for the corridor effect within a specified corridor core half-range. It updates a dictionary with mean, standard deviation, sample size, and uncertainty values for the monthly average corridor effect.

    Inputs:
        - temporal: a string denoting the temporal resolution of averaging. It can be either 'monthly' or 'annual'.
        - unc_coeff: A float representing the uncertainty coefficient. This value is used in the uncertainty calculation.
        - centered: A dictionary containing a key temporal + '_profiles_unc' which is a 2D numpy array of uncertainties associated with the temporal profiles.
        - core_half_range: A float representing the half-range of the corridor core.
        - avg_distances_short: A 1D numpy array of average distances west and east from the corridor center.
        - corridor_effect: A dictionary containing the following key:
            temporal + '_profiles': A 2D numpy array containing temporal mean profiles of corridor effect.

    Outputs:
        The function updates the corridor_effect dictionary with the following keys:
            - temporal + '_mean': A 1D numpy array representing the mean of the temporal corridor effect profiles for data within the corridor core, defined by the given half-range.
            - temporal + '_std': A 1D numpy array representing the standard deviation of the temporal corridor effect profiles within the core corridor.
            - temporal + '_N': A 1D numpy array representing the number of non-NaN values used in each mean calculation within the corridor core.
            - temporal + '_mean_unc': A 1D numpy array representing the combined uncertainty of the temporal mean corridor effect, calculated as the square root of the sum of squared standard deviation and the squared mean of the uncertainty data multiplied by the given uncertainty coefficient.
    '''

    corridor_effect[temporal + '_mean'] = np.stack([np.nanmean(corridor_effect[temporal + '_profiles'][abs(avg_distances_short) < core_half_range, i]) for i in range(corridor_effect[temporal + '_profiles'].shape[1])])

    corridor_effect[temporal + '_std'] = np.stack([np.nanstd(corridor_effect[temporal + '_profiles'][abs(avg_distances_short) < core_half_range, i]) for i in range(corridor_effect[temporal + '_profiles'].shape[1])])

    corridor_effect[temporal + '_N'] = np.stack([np.sum(np.isfinite(corridor_effect[temporal + '_profiles'][abs(avg_distances_short) < core_half_range, i])) for i in range(corridor_effect[temporal + '_profiles'].shape[1])])

    corridor_effect[temporal + '_mean_unc'] = np.sqrt((1 / corridor_effect[temporal + '_N']) * (corridor_effect[temporal + '_std']**2) + unc_coeff * (np.nanmean(centered[temporal + '_profiles_unc'], axis = 0)**2))


def calculate_annual_average_profiles(unc_coeff, centered):

    '''
    This function calculates the annual average profiles of data across the corridor based on max. 12 monthly averages, as well as the associated standard deviation, number of observations, and propagated uncertainty for each year.

    Inputs:
        - unc_coeff: The uncertainty coefficient, used in the uncertainty calculation.
        - centered: A dictionary containing the following key-value pairs:
            - 'monthly_profiles': A 2D NumPy array of monthly profiles, where each row represents a profile and each column represents a month.
            - 'monthly_profiles_unc': A 2D NumPy array of monthly profile uncertainties, with the same dimensions as centered['monthly_profiles'].

    Outputs:
        - The function calculates and updates the following key-value pairs in the centered dictionary:
            - 'annual_profiles_mean': A 2D NumPy array containing the mean of the monthly profiles for each year. The array is calculated by reshaping the monthly profiles to group them into years (each group contains 12 months), and then computing the mean across the third dimension (months).
            - 'annual_profiles_std': A 2D NumPy array containing the standard deviation of the monthly profiles for each year. The array is calculated in a similar way as annual_profiles_mean.
            - 'annual_profiles_N': A 2D NumPy array containing the number of  non-NaN values used in the annual mean calculations.
            - 'annual_profiles_unc': A 2D NumPy array containing the propagated uncertainty of the annual profiles for each year. 
    '''

    centered['annual_profiles_mean'] = np.nanmean(centered['monthly_profiles'].reshape((centered['monthly_profiles'].shape[0], int(centered['monthly_profiles'].shape[1] / 12), 12)), axis = 2)

    centered['annual_profiles_std'] = np.nanstd(centered['monthly_profiles'].reshape((centered['monthly_profiles'].shape[0], int(centered['monthly_profiles'].shape[1] / 12), 12)), axis = 2)

    centered['annual_profiles_N'] = np.nansum(centered['monthly_profiles'].reshape((centered['monthly_profiles'].shape[0], int(centered['monthly_profiles'].shape[1] / 12), 12)), axis = 2) / centered['annual_profiles_mean']

    centered['annual_profiles_unc'] = np.sqrt((1/centered['annual_profiles_N'])*(centered['annual_profiles_std']**2) + unc_coeff*(np.nanmean(centered['monthly_profiles_unc'].reshape((centered['monthly_profiles_unc'].shape[0], int(centered['monthly_profiles_unc'].shape[1] / 12), 12)), axis = 2)**2))


def calculate_temporal_average_profiles_from_specific_months(centered, months_initials, unc_coeff):

    '''
    This function calculates the annual or specific-months average profiles of data across the corridor, based on max. 12 monthly averages (for the "annual" case), as well as the associated standard deviation, number of observations, and propagated uncertainty for each year.

    Inputs:
        - centered: A dictionary containing the following key-value pairs:
            - 'monthly_profiles': A 2D NumPy array of monthly profiles, where each row represents a profile and each column represents a month.
            - 'monthly_profiles_unc': A 2D NumPy array of monthly profile uncertainties, with the same dimensions as centered['monthly_profiles'].
        - months_initials: A string containing either 'annual' or month initials, e.g. 'SON', 'ASOND', etc.
        - unc_coeff: The uncertainty coefficient, used in the uncertainty calculation.

    Outputs:
        - The function calculates and updates the following key-value pairs in the centered dictionary:
            - months_initials + '_profiles_mean': A 2D NumPy array containing the mean of the profiles for each year. The array is calculated by reshaping the monthly profiles to group them into years (each group contains 12 months), and then computing the mean across (part of) the third dimension (specified months).
            - months_initials + '_profiles_std': A 2D NumPy array containing the standard deviation of the monthly profiles for each year. The array is calculated in a similar way as the months_initials + _profiles_mean array.
            - months_initials + '_profiles_N': A 2D NumPy array containing the number of  non-NaN values used in the yearly mean calculations.
            - months_initials + '_profiles_unc': A 2D NumPy array containing the propagated uncertainty of the yearly profiles for each year. 
    '''


    # Define a dictionary mapping month initials to month numbers
    month_dict = {
        'J': 1, 'F': 2, 'M': 3, 'A': 4, 'M': 5, 'J': 6, 'J': 7, 'A': 8,
        'S': 9, 'O': 10, 'N': 11, 'D': 12
    }
    
    # Check if months_initials is 'annual'
    if months_initials.lower() == 'annual':
        # Average all 12 months if the input is 'annual'
        months_list = list(range(12))
    else:
        # Convert month initials to month numbers
        months_list = [month_dict[month] - 1 for month in months_initials.upper()]
    
    # Get the shape of the monthly data
    num_years = centered['monthly_profiles'].shape[1] // 12
    
    # Reshape the monthly data into (num_months, num_years, 12)
    reshaped_data = centered['monthly_profiles'].reshape(centered['monthly_profiles'].shape[0], num_years, 12)
    reshaped_data_unc = centered['monthly_profiles_unc'].reshape(centered['monthly_profiles_unc'].shape[0], num_years, 12)
    
    # Select the specified months for each year
    selected_months_data = reshaped_data[:, :, months_list]
    selected_months_data_unc = reshaped_data_unc[:, :, months_list]
    
    # Calculate the mean along the selected months axis 
    centered[months_initials + '_profiles_mean'] = np.nanmean(selected_months_data, axis = 2)

    centered[months_initials + '_profiles_std'] = np.nanstd(selected_months_data, axis = 2)
    
    centered[months_initials + '_profiles_N'] = np.nansum(selected_months_data, axis = 2) / centered[months_initials + '_profiles_mean']

    centered[months_initials + '_profiles_unc'] = np.sqrt((1/centered[months_initials + '_profiles_N'])*(centered[months_initials + '_profiles_std']**2) + unc_coeff * (np.nanmean(selected_months_data_unc, axis = 2)**2))


def analyze_annual_trends_from_specific_months(temp_res, unc_coeff, centered, avg_distances, corridor_half_range, core_half_range, avg_distances_short, corridor_effect):

    '''
    Analyzes yearly trends from specific months in the time series data, including fitting curves without the effect of ship emissions, calculating profiles of corridor effects, and determining yearly mean corridor effects.

    Functionality:
        - Calls the function calculate_temporal_average_profiles_from_specific_months to calculate temporal averages from specific months of the year in the time series data. The function modifies the centered dictionary.
        - Fits the "NoShip" curves (curves without the effect of ship emissions) per year for the given data. This is done using the calculate_NoShip_curve function for each year of the time series. The fitted curves are stored in the centered dictionary under a new key, combining the specified temporal resolution with _profiles_NoShip.
        - Calculates yearly profiles of corridor effects by subtracting the "NoShip" profiles from the original profiles. The resulting profiles of corridor effects and the corresponding uncertainties are stored in the corridor_effect dictionary.
        - Calls the function calculate_temporal_mean_corridor_effect to calculate the yearly mean corridor effects, which is stored in the corridor_effect dictionary.

    Input:
        - temp_res (str): Specifies the temporal resolution ('annual' or specific month initials, e.g. 'SON'), to analyze the data.
        - unc_coeff (float): Uncertainty coefficient used for calculations of propagated uncertainty.
        - centered (dict): A dictionary containing data about centered monthly or annual profiles, profiles without the effect of ships, and related uncertainties.
        - avg_distances (ndarray): Array containing average distances from the shipping corridor center.
        - corridor_half_range (float): The half range of the corridor used for the calculations.
        - core_half_range (float): The half range of the core corridor used for mean corridor effect calculations.
        - avg_distances_short (ndarray): Shorter average distances in the data set.
        - corridor_effect (dict): A dictionary to store data about corridor effects and related uncertainties.

    Output:
        - The function modifies the centered and corridor_effect dictionaries by adding and updating data related to the analysis of annual or specific-month trends. Specifically, it adds data on the "NoShip" curves, profiles of corridor effects, and annual mean corridor effects.
    '''

    calculate_temporal_average_profiles_from_specific_months(centered, temp_res, unc_coeff)

    # Fit "NoShip" curves per year
    centered[temp_res + '_profiles_NoShip'] = np.stack([calculate_NoShip_curve(avg_distances, centered[temp_res + '_profiles_mean'][:, i], corridor_half_range, 400, 3) for i in range(centered[temp_res + '_profiles_mean'].shape[1])], axis = 1)

    # Calculate annual profiles of corridor effects
    corridor_effect[temp_res + '_profiles'] = centered[temp_res + '_profiles_mean'] - centered[temp_res + '_profiles_NoShip']
    corridor_effect[temp_res + '_profiles_unc'] = centered[temp_res + '_profiles_unc']

    # Calculate annual mean corridor effects
    calculate_temporal_mean_corridor_effect(temp_res, unc_coeff, centered, core_half_range, avg_distances_short, corridor_effect)


def calculate_map_of_time_series_means(unc_coeff, data, data_unc, iyear, imonth, eyear, emonth, first_year):

    '''
    Description:
        This function calculates the mean, standard deviation, number of values and mean uncertainty from a given 3D array (data) of time series data, as well as an associated array of uncertainties (data_unc). The function returns the calculated values.

    Input:
        - unc_coeff: A numerical coefficient used in the uncertainty calculation.
        - data: A 3D NumPy array representing time series data, with the third dimension representing time.
        - data_unc: A 3D NumPy array representing uncertainties associated with the time series data.
        - iyear, imonth: start year and month for calculations.
        - eyear, emonth: end year and month for calculations.
        - first_year: first year of the entire time series.

    Output:
        - mean: A 2D array representing the mean values of the input data along the time dimension.
        - std: A 2D array representing the standard deviation of the input data along the time dimension.
        - N: A 2D array representing the count of non-NaN values used in the mean calculation for each element in the input data along the time dimension.
        - unc_mean: A 2D array representing the propagated uncertainty of the mean, given unc_coeff, std, and data_unc.
    '''

    # Find start and end indices for averaging
    si = (iyear - first_year) * 12 + (imonth - 1)
    ei = (eyear - first_year) * 12 + emonth

    mean = np.nanmean(data[:, :, si:ei], axis = 2)
    std = np.nanstd(data[:, :, si:ei], axis = 2)
    N = np.round(np.nansum(data[:, :, si:ei], axis = 2) / mean).astype(int)
    unc_mean = np.sqrt(((1 / N) * (std**2)) + unc_coeff * (np.nanmean(data_unc[:, :, si:ei])**2))

    return mean, std, N, unc_mean


def find_overlap(array1, unc1, array2, unc2):

    # Calculate the lower and upper bounds for each cell in both arrays
    lower1 = array1 - unc1
    upper1 = array1 + unc1
    
    lower2 = array2 - unc2
    upper2 = array2 + unc2
    
    # Check for overlap at each grid cell
    # Overlap occurs if max of the lower bounds <= min of the upper bounds
    overlap = (np.maximum(lower1, lower2) <= np.minimum(upper1, upper2))
    
    return overlap


def check_significance_of_difference(array1, unc1, array2, unc2):

    # Calculate the absolute difference between the two arrays
    abs_diff = np.abs(array2 - array1)
    
    # Calculate the combined uncertainty
    unc_comb = np.sqrt(unc1**2 + unc2**2)
    
    significant = abs_diff >= unc_comb

    return significant

def plot_data_with_mask(var, data_array, bool_array, title, minval, maxval, grid_extent, plot_extent, cmap_color, cmap_grey, ext, filename, saveplot):

    # Create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the title
    ax.set_title(title)

    # Masked arrays for True and False cases
    masked_data_false = np.ma.masked_where(bool_array, data_array)
    masked_data_true = np.ma.masked_where(~bool_array, data_array)

    # Plot False cases using grayscale colormap
    im_grey = ax.imshow(masked_data_false, cmap=cmap_grey, vmin=minval, vmax=maxval, extent=grid_extent, transform=ccrs.PlateCarree())

    # Plot True cases using color colormap
    im_color = ax.imshow(masked_data_true, cmap=cmap_color, vmin=minval, vmax=maxval, extent=grid_extent, transform=ccrs.PlateCarree())

    # Add coastlines
    ax.add_feature(cf.COASTLINE)

    # Set the plot extent
    ax.set_extent(plot_extent)

    # Add a single colorbar using the color colormap
    cbar = fig.colorbar(im_color, shrink=0.6, extend=ext)
    cbar.set_label(f'[{varUnits[var]}]')

    # Add colorbars for both greyscale and color portions
    # cbar_grey = fig.colorbar(im_grey, shrink=0.6, extend=ext, orientation='vertical', pad=0.05)
    

    # Adjust the layout
    plt.tight_layout()

    # Save the plot if required
    if saveplot:
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    # Close the plot
    plt.close()