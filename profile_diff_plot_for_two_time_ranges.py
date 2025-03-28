import numpy as np
from shipping_corridor_functions import plot_change_and_zero_line_for_two

var = 'cal'


distance = np.load('npy_arrays_for_plots/' + var.upper() + '/avg_distances_350.npy')

# Plot across-corridor profiles of change
array1 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_change_across_sc_2004-2019.npy')
array2 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_change_across_sc_2020-2023.npy')
unc1 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_change_unc_across_sc_2004-2019.npy')
unc2 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_change_unc_across_sc_2020-2023.npy')
label1 = '2004-2019'
label2 = '2020-2023'
title = ''
outfile = 'Figures/' + var.upper() + '/' + var.upper() + '_time_series_mean_change_across_sc_before_and_after_2020.png'
plot_std_band = True
plot_zero_line = True
saveplot = True

plot_change_and_zero_line_for_two(var, array1, unc1, label1, array2, unc2, label2, distance, 103, title, outfile, plot_std_band, saveplot)


# Plot across-corridor profiles of means
array1 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_across_sc_2004-2019.npy')
array2 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_across_sc_2020-2023.npy')
unc1 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_unc_across_sc_2004-2019.npy')
unc2 = np.load('npy_arrays_for_plots/' + var.upper() + '/' + var.upper() + '_time_series_mean_unc_across_sc_2020-2023.npy')
label1 = '2004-2019'
label2 = '2020-2023'
title = ''
outfile = 'Figures/' + var.upper() + '/' + var.upper() + '_time_series_mean_across_sc_before_and_after_2020.png'
plot_std_band = True
plot_zero_line = False
saveplot = True

plot_change_and_zero_line_for_two(var, array1, unc1, label1, array2, unc2, label2, distance, 103, title, outfile, plot_std_band, saveplot)