import pickle
import numpy as np
import pandas as pd
import os
import process_data
import gmm
import figure_functions
import kmeans_utils
import write_results


curr_dir = os.getcwd()
data_path = os.path.join(curr_dir, '..', 'data')
belltown_path = os.path.join(data_path, 'Belltown_Hour')
fig_path = os.path.join(curr_dir, '..', 'nature_figs')
results_path = os.path.join(curr_dir, '..', 'nature_results')
path = belltown_path
pickle.dump('belltown', open(os.path.join(data_path, 'background_img_name.p'), 'wb'))

time1 = 51
time2 = 58
num_comps = 4

k_values = [3, 5, 10]
p_value = .01

month_year_start = (3, 2016)
month_year_end = (7, 2016)


params = process_data.load_data(data_path=data_path, load_paths=[path], 
                                month_year_start=month_year_start, month_year_end=month_year_end, 
                                verbose=False)
element_keys, loads, gps_loc, park_data, idx_to_day_hour, day_hour_to_idx = params


figure_functions.contour_plot(loads, gps_loc, time1, 
                              fig_path, filename='contours1.png')
figure_functions.contour_plot(loads, gps_loc, time2, 
                              fig_path, filename='contours2.png')
figure_functions.mixture_plot(loads, gps_loc, time1, 
                              fig_path, filename='mixture_plot1.png')
figure_functions.mixture_plot(loads, gps_loc, time2, 
                              fig_path, filename='mixture_plot2.png')

area_map = pickle.load(open(os.path.join(data_path, 'belltown_subareas.p'), 'rb'))

results = gmm.locational_demand_analysis(park_data, gps_loc, num_comps, 
                                         k_values, area_map, verbose=False)

days = [result[0] for result in results]
hours = [result[1] for result in results]

time_avg_consistency = [result[2] for result in results]

write_results.write_gmm_results(time_avg_consistency, results_path)

morans_mixture = [result[3] for result in results] 
morans_dist_mixture = [result[4] for result in results] 

morans_area = [result[5] for result in results]
morans_dist_area = [result[6] for result in results]

morans_dist = [result[7] for result in results]

morans_neighbor = [result[8] for result in results]
morans_3 = [neighbor[3] for neighbor in morans_neighbor]
morans_5 = [neighbor[5] for neighbor in morans_neighbor]
morans_10 = [neighbor[10] for neighbor in morans_neighbor]

gmm_var = [result[9] for result in results]
np.savetxt(os.path.join(results_path, 'gmm_var.csv'), np.array(gmm_var), delimiter=',')

sdot_var = [result[10] for result in results]
np.savetxt(os.path.join(results_path, 'sdot_var.csv'), np.array(sdot_var), delimiter=',')

centers = [result[11] for result in results]

distances, centroids = kmeans_utils.get_distances(centers=centers, num_comps=num_comps)
best_time = distances.mean(axis=1).argmin()

write_results.write_centroid_distance_results(days=days, hours=hours,
                                              distances=distances,
                                              results_path=results_path)

figure_functions.centroid_plots(centers=centers, gps_loc=gps_loc, 
                                times=best_time, fig_path=fig_path, 
                                num_comps=num_comps)

all_morans = [morans_mixture, morans_dist_mixture, morans_area, morans_dist_area, 
              morans_dist, morans_3, morans_5, morans_10]

auto_names = ['mixture', 'mixture_dist', 'area', 'area_dist', 'dist', 'k_3', 'k_5', 'k_10']

all_I = []
all_p_one = []
all_p_two = []

for j in xrange(len(auto_names)):

    results_path_curr = os.path.join(results_path, auto_names[j])

    I_avg, p_one_side, p_two_side = write_results.write_moran_results(days, hours, 
                                                                     all_morans[j], 
                                                                     p_value, results_path_curr)
    all_I.append(I_avg)
    all_p_one.append(p_one_side)
    all_p_two.append(p_two_side)

avg_moran = np.vstack((all_I, all_p_one, all_p_two))
index = ['Moran I Over All Days and Times', 
         'Significant One Sided P Value Percentage Average Over All Days And Times', 
         'Significant Two Sided P Value Percentage Average Over All Days And Times']
avg_df = pd.DataFrame(avg_moran, index=index, columns=auto_names)
avg_df.to_csv(os.path.join(results_path, 'Moran_Results.csv'), sep=',')