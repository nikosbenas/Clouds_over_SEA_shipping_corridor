import numpy as np

var = 'cfc_day'
core_half_range = 75

distance = np.load('npy_arrays_for_plots/avg_distances.npy')

array19 = np.load('npy_arrays_for_plots/' + var.upper() + '_time_series_mean_change_across_sc_2004-2019.npy')
array23 = np.load('npy_arrays_for_plots/' + var.upper() + '_time_series_mean_change_across_sc_2020-2023.npy')

core19 = array19[np.abs(distance) < core_half_range]
core23 = array23[np.abs(distance) < core_half_range]

diff23_19 = core23-core19

diff_mean = np.mean(diff23_19)
diff_2sigma = 2 * np.std(diff23_19)

print('check')
