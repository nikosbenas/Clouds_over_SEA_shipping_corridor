#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program creates a binary mask of shipping corridors based on ship density
data from the World Bank:
datacatalog.worldbank.org/search/dataset/0037580/Global-Shipping-Traffic-Density

The data (originally at 0.005 deg. resolution) are first aggregated to a 
resolution coarser than CLAAS in order to avoid noisy pixels in the resulting
flag. Then the resolution is increased to the CLAAS level 3 0.05 degrees.

Created on Fri Jun 16 10:08:25 2023

@author: benas
"""
import rasterio
import numpy as np
import sys
sys.path.append('/usr/people/benas/Documents/CMSAF/python_modules/')
from modis_python_functions import map_l3_var as create_map
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')


def get_rowcol_from_latlon(lat, lon, filename):
    
    """
    Function to return row and column numbers of array from a geotiff file
    given latitude, longitude and file name.
    
    Input:
        
        lat, lon: latitude and longitude at pont of interest (floats)
        filename: name of geotiff file (string)
        
    Output:
        
        row, col: row and column of point of interest (ints)
        
    """
    
    # Open the GeoTIFF file
    with rasterio.open(filename) as src:
        
        # Convert latitude and longitude to pixel coordinates
        row, col = src.index(lon, lat)
        
        # Round row and col to nearest integers
        row = int(np.round(row))
        col = int(np.round(col))
        
    return row, col


def get_lat_lon_arrays(file_name, rs, cs, stride):
    
    """
    Function to return latitude and longitude arrays from a geotiff file based
    on the dataset and the affine transformation matrix.
    
    Input:
        
        file_name: name of the geotiff file (string)
        rs, cs: rows and columns at the four corners of the area of interest 
               (lists of four integers)
        stride: the step to be used when creating the arrays
        
    Output:
        
        lat, lon: arrays of latitude and longitude
        
    """    
    
    with rasterio.open(file_name) as src: 
                                                        
        cols, rows = np.meshgrid(np.arange(cs[0], cs[1], stride), 
                                 np.arange(rs[0], rs[3], stride))
            
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            
        lons= np.array(xs)
            
        lats = np.array(ys)
                
    return lats, lons


def sum_grid_cells(array, block_size):
    
    """
    
    Given array and block_size, reduce the resolution of the array by summing
    the values in each block.
    
    Output is the reduced-resolution array.
    
    """
    
    # Calculate the number of blocks in each dimension
    num_blocks_row = array.shape[0] // block_size
    num_blocks_col = array.shape[1] // block_size
   
    # Reshape the array into blocks of the specified size
    blocks = array[:num_blocks_row * block_size, :num_blocks_col * block_size]
    blocks = blocks.reshape(num_blocks_row, block_size, num_blocks_col, 
                            block_size)
   
    # Calculate the sum of values within each block
    block_sums = np.sum(blocks, axis=(1, 3))
   
    return block_sums


def average_grid_cells(array, block_size):
    
    """
    
    Given array and block_size, reduce the resolution of the array by averaging
    the values in each block.
    
    Output is the reduced-resolution array.
    
    """
    
    # Calculate the number of blocks in each dimension
    num_blocks_row = array.shape[0] // block_size
    num_blocks_col = array.shape[1] // block_size
   
    # Reshape the array into blocks of the specified size
    blocks = array[:num_blocks_row * block_size, :num_blocks_col * block_size]
    blocks = blocks.reshape(num_blocks_row, block_size, num_blocks_col, 
                            block_size)
   
    # Calculate the sum of values within each block
    block_averages = np.mean(blocks, axis=(1, 3))
    
    return block_averages


# =============================================================================
# Read lat, lon and ship density at region of interest
# =============================================================================

# Open the GeoTIFF file
geotiff_filename = '/nobackup/users/benas/Ship_Density/shipdensity_global.tif'

# Lats and lons at four corners of region: ul, ur, lr, ll
# toplat = -10; bottomlat = -35
# leftlon = -10; rightlon = 20 
toplat = -10; bottomlat = -20
leftlon = -10; rightlon = 10 

lats = [toplat, toplat, bottomlat, bottomlat]
lons = [leftlon, rightlon, rightlon, leftlon]
del toplat, bottomlat, leftlon, rightlon

rows = []; cols = []
for i in range(len(lats)):
    
    r, c = get_rowcol_from_latlon(lats[i], lons[i], geotiff_filename)
    
    rows.append(r)
    cols.append(c)

stride = 1

lat, lon = get_lat_lon_arrays(geotiff_filename, rows, cols, stride)

src = rasterio.open(geotiff_filename)

data = src.read(1, window = ((rows[0], rows[3]), (cols[0], cols[1])))

src.close()

del rows, cols, geotiff_filename, src, stride, c, i, r

# =============================================================================
# Reduce lat, lon and ship data resolution to match the CLAAS-3 resolution 
# =============================================================================

block_size_for_flag = 50
# Sum data and average lat and lon
data_for_flag = sum_grid_cells(data, block_size_for_flag)
lat_for_flag = average_grid_cells(lat, block_size_for_flag)
lon_for_flag = average_grid_cells(lon, block_size_for_flag)

block_size = 10
# Sum data and average lat and lon
data_reduced = sum_grid_cells(data, block_size)
lat_reduced = average_grid_cells(lat, block_size)
lon_reduced = average_grid_cells(lon, block_size)

del lat, lon, data, block_size, block_size_for_flag


# =============================================================================
# Create binary flag to identify shipping corridors
# =============================================================================

flag_from_mean_plus_1s = np.where(data_for_flag > 
                                  (np.mean(data_for_flag) + 
                                   np.std(data_for_flag)), 1, 0)


# Increase resolution of flag to match the CLAAS-3 resolution
# Define the desired higher resolution
new_shape = data_reduced.shape

# Compute the repeat factor along each axis
repeat_factor = (new_shape[0] // flag_from_mean_plus_1s.shape[0], 
                 new_shape[1] // flag_from_mean_plus_1s.shape[1])

# Increase the resolution using the Kronecker product
flag_reduced = np.kron(flag_from_mean_plus_1s, np.ones(repeat_factor))

del new_shape, repeat_factor, data_for_flag

# =============================================================================
# Create some test maps
# =============================================================================

outfile = 'Figures/Flags_shipping_corridor.png'
create_map(lat_reduced, lon_reduced, flag_reduced, 0, 1, 
            'Ship density flag from mean + 1 std', '-', 'Reds', 'neither', 
            outfile, saveplot = False)

# =============================================================================
# Save flag array to npy file
# =============================================================================

outfile = 'flags_shipping_corridor_2.npy'
np.save(outfile, flag_reduced)
