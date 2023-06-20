#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:08:25 2023

@author: benas
"""
import rasterio
import numpy as np
import sys
sys.path.append('/usr/people/benas/Documents/CMSAF/python_modules/')
from modis_python_functions import map_l3_var as create_map
sys.path.append('/data/windows/m/benas/Documents/CMSAF/CLAAS-3/CLAAS-3_trends')
import claas_trends_functions as ctf
import claas3_dictionaries as cdict
import h5py
from datetime import datetime 
from netCDF4 import Dataset


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
    blocks = blocks.reshape(num_blocks_row, block_size, num_blocks_col, block_size)
   
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
    blocks = blocks.reshape(num_blocks_row, block_size, num_blocks_col, block_size)
   
    # Calculate the sum of values within each block
    block_averages = np.mean(blocks, axis=(1, 3))
    
    return block_averages


# =============================================================================
# Read lat, lon and ship density at region of interest
# =============================================================================

# Open the GeoTIFF file
geotiff_filename = '/nobackup/users/benas/Ship_Density/shipdensity_global.tif'

# Lats and lons at four corners of region: ul, ur, lr, ll
lats = [-10, -10, -35, -35]
lons = [-10, 20, 20, -10]

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

block_size = 10

# Sum data and average lat and lon
data_reduced = sum_grid_cells(data, block_size)
lat_reduced = average_grid_cells(lat, block_size)
lon_reduced = average_grid_cells(lon, block_size)

# =============================================================================
# Read CLAAS-3 data
# =============================================================================

# Define bounding box: ul lon, lr lat,lr lon, ul lat
bounding_box = [lons[0], lats[2], lons[2], lats[0]]
stride = 1

# Define variables to read
var_list = ['cdnc_liq']

# Select the time period to analyze CLAAS-3 data
start_year = 2004
end_year = 2022

# Create vector of dates for plotting
dates = [datetime.strptime(str(year) + str(month).zfill(2) + '01', '%Y%m%d')
         for year in range(start_year, end_year + 1) for month in range(1, 13)]

# Read full CLAAS-3 lat and lon arrays, needed to find b.box indices
claas3_l3_aux_data_file = '/data/windows/m/benas/Documents/CMSAF/CLAAS-3/' +\
    'CLAAS-3_trends/claas3_level3_aux_data_005deg.nc'
f = h5py.File(claas3_l3_aux_data_file, 'r')
latData = f['lat'][:]
lonData = f['lon'][:]
f.close()
lonData, latData = np.meshgrid(lonData, latData)

istart, iend, jstart, jend = ctf.find_bbox_indices(bounding_box, latData,
                                                   lonData)

del latData, lonData 

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
        
        del claas3_file, claas3_data

outfile = './testmap_SE_Atlantic_reduced_resolution.png'

create_map(lat_reduced, lon_reduced, data_reduced, np.min(data_reduced), 
           np.max(data_reduced), 'test', '-', 'Reds', 'neither', outfile, 
           saveplot = True)

# Access the dataset attributes
# print(src.crs)        # Coordinate reference system
# print(src.bounds)     # Bounding box coordinates
# print(src.width)      # Width of the raster in pixels
# print(src.height)     # Height of the raster in pixels
# print(src.count)      # Number of bands in the raster
# print(src.dtypes)     # Data types of the bands
# print(src.transform)  # Affine transformation matrix


# # Read the raster data as a NumPy array
# data = src.read(1)

# # Perform further operations on the data or the dataset
# # For example, you can visualize the raster using Matplotlib or other libraries

# # Close the dataset
# src.close()
