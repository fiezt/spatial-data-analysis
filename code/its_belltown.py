import pickle
import numpy as np
import pandas as pd
import os
import gmm
import datetime
import figure_functions
import process_data
import kmeans_utils
import write_results


curr_dir = os.getcwd()
data_path = os.path.join(curr_dir, '..', 'data')
belltown_path = os.path.join(data_path, 'Belltown_Hour')
path = belltown_path

fig_path = os.path.join(curr_dir, '..',  'ITS_figs', 'belltown_figs')
fig_paths_seasonal = []
fig_paths_seasonal.append(os.path.join(fig_path, 'summer_2016'))
fig_paths_seasonal.append(os.path.join(fig_path, 'fall_2016'))
fig_paths_seasonal.append(os.path.join(fig_path, 'winter_2017'))
fig_paths_seasonal.append(os.path.join(fig_path, 'spring_2017'))
fig_paths_seasonal.append(os.path.join(fig_path, 'summer_2017'))

fig_paths_price = []
fig_paths_price.append(os.path.join(fig_path, 'price_before'))
fig_paths_price.append(os.path.join(fig_path, 'price_after'))

results_path = os.path.join(curr_dir, '..', 'ITS_results', 'belltown_results')
results_paths_seasonal = []
results_paths_seasonal.append(os.path.join(results_path, 'summer_2016'))
results_paths_seasonal.append(os.path.join(results_path, 'fall_2016'))
results_paths_seasonal.append(os.path.join(results_path, 'winter_2017'))
results_paths_seasonal.append(os.path.join(results_path, 'spring_2017'))
results_paths_seasonal.append(os.path.join(results_path, 'summer_2017'))

results_paths_price = []
results_paths_price.append(os.path.join(results_path, 'price_before'))
results_paths_price.append(os.path.join(results_path, 'price_after'))

area_map = pickle.load(open(os.path.join(data_path, 'belltown_subareas.p'), 'rb'))
pickle.dump('belltown', open(os.path.join(data_path, 'background_img_name.p'), 'wb'))

time1 = 59
time2 = 63
num_comps = 4

k_values = [3, 5, 10]
p_value = .01


# Loading the data.
all_loads = []
all_element_keys = []
all_gps = []
all_park_data = []
starts = [(6, 2016), (9, 2016), (12, 2016), (3, 2017), (6, 2017)]
ends = [(8, 2016), (11, 2016), (2, 2017), (5, 2017), (8, 2017)]

for pair in zip(starts, ends):
    
    month_year_start = pair[0]
    month_year_end = pair[1]
        
    params = process_data.load_data(data_path=data_path, load_paths=[path], 
                                    month_year_start=month_year_start, month_year_end=month_year_end, 
                                    verbose=False)
    element_keys, loads, gps_loc, park_data, idx_to_day_hour, day_hour_to_idx = params
    
    all_element_keys.append(element_keys)
    all_loads.append(loads)
    all_gps.append(gps_loc)
    all_park_data.append(park_data)    
    
all_keys_seasonal = all_element_keys 
all_gps_seasonal = all_gps
all_loads_seasonal = all_loads
all_park_data_seasonal = all_park_data

all_loads = []
all_gps = []
all_element_keys = []
all_park_data = []
starts = [(6, 2016), (6, 2017)]
ends = [(6, 2016), (6, 2017)]

for pair in zip(starts, ends):
    
    month_year_start = pair[0]
    month_year_end = pair[1]
        
    params = process_data.load_data(data_path=data_path, load_paths=[path], 
                                    month_year_start=month_year_start, month_year_end=month_year_end, 
                                    verbose=False)
    element_keys, loads, gps_loc, park_data, idx_to_day_hour, day_hour_to_idx = params
    
    all_loads.append(loads)
    all_gps.append(gps_loc)
    all_element_keys.append(element_keys)
    all_park_data.append(park_data)    
  
    
all_keys_price = all_element_keys 
all_gps_price = all_gps
all_loads_price = all_loads
all_park_data_price = all_park_data


# Getting contour and mixture plot.
for i in xrange(len(fig_paths_seasonal)):
    figure_functions.temporal_day_plots(all_loads_seasonal[i], fig_paths_seasonal[i], 
                                        filename='temporal_day_plots.png')
    figure_functions.contour_plot(all_loads_seasonal[i], all_gps_seasonal[i], time1, 
                                  fig_paths_seasonal[i], filename='contours1.png')
    figure_functions.contour_plot(all_loads_seasonal[i], all_gps_seasonal[i], time2, 
                                  fig_paths_seasonal[i], filename='contours2.png')
    figure_functions.mixture_plot(all_loads_seasonal[i], all_gps_seasonal[i], time1, 
                                  fig_paths_seasonal[i], filename='mixture_plot1.png')
    figure_functions.mixture_plot(all_loads_seasonal[i], all_gps_seasonal[i], time2, 
                                  fig_paths_seasonal[i], filename='mixture_plot2.png')
    

# Getting temporal plots.
for i in xrange(len(fig_paths_seasonal)-1):
    figure_functions.temporal_change_plot(all_loads_seasonal[i], all_loads_seasonal[i+1], 
                                          all_keys_seasonal[i], all_keys_seasonal[i+1], 
                                          color_option=i+1, fig_path=fig_paths_seasonal[i+1],
                                          filename='posneg.png') 

    figure_functions.temporal_mean_diff_plot(all_loads_seasonal[i], all_loads_seasonal[i+1], 
                                             all_keys_seasonal[i], all_keys_seasonal[i+1], 
                                             color_option=i+1, fig_path=fig_paths_seasonal[i+1],
                                             filename='diff.png') 
    
# Difference in each season.
for i in xrange(len(fig_paths_price)):
    figure_functions.temporal_day_plots(all_loads_price[i], fig_paths_price[i], 
                                        filename='temporal_day_plots.png')
    figure_functions.contour_plot(all_loads_price[i], all_gps_price[i], time1, 
                                  fig_paths_price[i], filename='contours1.png')
    figure_functions.contour_plot(all_loads_price[i], all_gps_price[i], time2, 
                                  fig_paths_price[i], filename='contours2.png')
    figure_functions.mixture_plot(all_loads_price[i], all_gps_price[i], time1, 
                                  fig_paths_price[i], filename='mixture_plot1.png')
    figure_functions.mixture_plot(all_loads_price[i], all_gps_price[i], time2, 
                                  fig_paths_price[i], filename='mixture_plot2.png')

# Difference before and after price change.
for i in xrange(len(fig_paths_price)-1):
    figure_functions.temporal_change_plot(all_loads_price[i], all_loads_price[i+1], 
                                          all_keys_price[i], all_keys_price[i+1], 
                                          color_option=5, fig_path=fig_paths_price[i+1],
                                          filename='posneg.png') 

    figure_functions.temporal_mean_diff_plot(all_loads_price[i], all_loads_price[i+1], 
                                             all_keys_price[i], all_keys_price[i+1], 
                                             color_option=5, fig_path=fig_paths_price[i+1],
                                             filename='diff.png') 

# Creating the animations.
for i in xrange(len(fig_paths_seasonal)):
    figure_functions.create_animation(all_loads_seasonal[i], all_gps_seasonal[i],
                                      fig_paths_seasonal[i],
                                      animation_path=fig_paths_seasonal[i],
                                      num_comps=num_comps)
    
for i in xrange(len(fig_paths_price)):
    figure_functions.create_animation(all_loads_price[i], all_gps_price[i],
                                      fig_paths_price[i],
                                      animation_path=fig_paths_price[i],
                                      num_comps=num_comps)

    
# Showing how we learn similar models at these times.
times1 = [2, 14, 26, 38, 50]
times2 = [26, 27, 28, 29, 30]

for i in xrange(len(fig_paths_seasonal)):
    figure_functions.mixture_plot(all_loads_seasonal[i], all_gps_seasonal[i],
                                  times1, fig_paths_seasonal[i], shape=(1,5), 
                                  filename='mixture_daily.png')
    figure_functions.mixture_plot(all_loads_seasonal[i], all_gps_seasonal[i],
                                  times2, fig_paths_seasonal[i], shape=(1,5),
                                  filename='mixture_hourly.png')
    
for i in xrange(len(fig_paths_price)):
    figure_functions.mixture_plot(all_loads_price[i], all_gps_price[i],
                                  times1, fig_paths_price[i], shape=(1,5), 
                                  filename='mixture_daily.png')
    
    figure_functions.mixture_plot(all_loads_price[i], all_gps_price[i],
                                  times2, fig_paths_price[i], shape=(1,5), 
                                  filename='mixture_hourly.png')

    
# Getting the results for each season.
for i in xrange(len(results_paths_seasonal)):
    results = gmm.locational_demand_analysis(all_park_data_seasonal[i], 
                                             all_gps_seasonal[i],
                                             num_comps, k_values, area_map, verbose=False)
        
    days = [result[0] for result in results]
    hours = [result[1] for result in results]
    
    time_avg_consistency = [result[2] for result in results]
    
    write_results.write_gmm_results(time_avg_consistency, results_paths_seasonal[i])
    
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
    np.savetxt(os.path.join(results_paths_seasonal[i], 'gmm_var.csv'), np.array(gmm_var), delimiter=',')
    
    sdot_var = [result[10] for result in results]
    np.savetxt(os.path.join(results_paths_seasonal[i], 'sdot_var.csv'), np.array(sdot_var), delimiter=',')
    
    centers = [result[11] for result in results]
    
    distances, centroids = kmeans_utils.get_distances(centers=centers, num_comps=num_comps)
    best_time = distances.mean(axis=1).argmin()
    
    write_results.write_centroid_distance_results(days=days, hours=hours,
                                                  distances=distances,
                                                  results_path=results_paths_seasonal[i])
    
    figure_functions.centroid_plots(centers=centers, gps_loc=all_gps_seasonal[i], 
                                    times=best_time, fig_path=fig_paths_seasonal[i], 
                                    num_comps=num_comps)
    
    all_morans = [morans_mixture, morans_dist_mixture, morans_area, morans_dist_area, 
                  morans_dist, morans_3, morans_5, morans_10]
    
    auto_names = ['mixture', 'mixture_dist', 'area', 'area_dist', 'dist', 'k_3', 'k_5', 'k_10']
    
    all_I = []
    all_p_one = []
    all_p_two = []
    
    for j in xrange(len(auto_names)):
        
        results_path = os.path.join(results_paths_seasonal[i], auto_names[j])

        I_avg, p_one_side, p_two_side = write_results.write_moran_results(days, hours, 
                                                                         all_morans[j], 
                                                                         p_value, results_path)
        all_I.append(I_avg)
        all_p_one.append(p_one_side)
        all_p_two.append(p_two_side)
    
    avg_moran = np.vstack((all_I, all_p_one, all_p_two))
    index = ['Moran I Over All Days and Times', 
             'Significant One Sided P Value Percentage Average Over All Days And Times', 
             'Significant Two Sided P Value Percentage Average Over All Days And Times']
    avg_df = pd.DataFrame(avg_moran, index=index, columns=auto_names)
    avg_df.to_csv(os.path.join(results_paths_seasonal[i], 'Moran_Results.csv'), sep=',')
    

# Getting the results before and after the price change.
for i in xrange(len(results_paths_price)):
    results = gmm.locational_demand_analysis(all_park_data_price[i], 
                                             all_gps_price[i],
                                             num_comps, k_values, data_path, verbose=False)
    
    days = [result[0] for result in results]
    hours = [result[1] for result in results]
    
    time_avg_consistency = [result[2] for result in results]
    
    write_results.write_gmm_results(time_avg_consistency, results_paths_price[i])
    
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
    np.savetxt(os.path.join(results_paths_price[i], 'gmm_var.csv'), np.array(gmm_var), delimiter=',')
    
    sdot_var = [result[10] for result in results]
    np.savetxt(os.path.join(results_paths_price[i], 'sdot_var.csv'), np.array(sdot_var), delimiter=',')
    
    centers = [result[11] for result in results]
    
    distances, centroids = kmeans_utils.get_distances(centers=centers, num_comps=num_comps)
    best_time = distances.mean(axis=1).argmin()
    
    write_results.write_centroid_distance_results(days=days, hours=hours,
                                                  distances=distances,
                                                  results_path=results_paths_price[i])
    
    figure_functions.centroid_plots(centers=centers, gps_loc=all_gps_price[i], 
                                    times=best_time, fig_path=fig_paths_price[i], 
                                    num_comps=num_comps)
    
    all_morans = [morans_mixture, morans_dist_mixture, morans_area, morans_dist_area, 
                  morans_dist, morans_3, morans_5, morans_10]
    
    auto_names = ['mixture', 'mixture_dist', 'area', 'area_dist', 'dist', 'k_3', 'k_5', 'k_10']
    
    all_I = []
    all_p_one = []
    all_p_two = []
    
    for j in xrange(len(auto_names)):
        
        results_path = os.path.join(results_paths_price[i], auto_names[j])

        I_avg, p_one_side, p_two_side = write_results.write_moran_results(days, hours, 
                                                                         all_morans[j], 
                                                                         p_value, results_path)
        all_I.append(I_avg)
        all_p_one.append(p_one_side)
        all_p_two.append(p_two_side)
    
    avg_moran = np.vstack((all_I, all_p_one, all_p_two))
    index = ['Moran I Over All Days and Times', 
             'Significant One Sided P Value Percentage Average Over All Days And Times', 
             'Significant Two Sided P Value Percentage Average Over All Days And Times']
    avg_df = pd.DataFrame(avg_moran, index=index, columns=auto_names)
    avg_df.to_csv(os.path.join(results_paths_price[i], 'Moran_Results.csv'), sep=',')

    
# Plotting the differences between and after the price change.
for area in ['North', 'South']:
    idx1 = []
    keys1 = []
    for key in all_keys_price[0]:
        if area_map[key] == area:
            idx1.append(all_keys_price[0].index(key))
            keys1.append(key)

    gps1 = all_gps_price[0][idx1]
    loads1 = all_loads_price[0][idx1]

    idx2 = []
    keys2 = []
    for key in all_keys_price[1]:
        if area_map[key] == area:
            idx2.append(all_keys_price[1].index(key))
            keys2.append(key)

    gps2 = all_gps_price[1][idx2]
    loads2 = all_loads_price[1][idx2]

    figure_functions.temporal_change_plot(loads1, loads2, keys1, keys2, 
                                          color_option=5, fig_path=fig_path,
                                          filename='posneg_' + area + '.png') 

    figure_functions.temporal_mean_diff_plot(loads1, loads2, keys1, keys2, 
                                             color_option=5, fig_path=fig_path,
                                             filename='diff_' + area + '.png') 

