import numpy as np
import pandas as pd
import pickle
import os
import process_data
import gmm
import figure_functions
import kmeans_utils
import write_results


curr_dir = os.getcwd()
data_path = os.path.join(curr_dir, '..', 'data')
belltown_path = os.path.join(data_path, 'Belltown_Hour')
commcore_path = os.path.join(data_path, 'CommercialCore_Hour')
path = [belltown_path, commcore_path]

fig_path = os.path.join(curr_dir, '..',  'ITS_figs', 'belltown_commcore_figs')
fig_paths_seasonal = []
fig_paths_seasonal.append(os.path.join(fig_path, 'summer_2017'))

results_path = os.path.join(curr_dir, '..', 'ITS_results', 'belltown_commcore_results')
results_paths_seasonal = []
results_paths_seasonal.append(os.path.join(results_path, 'summer_2016'))
results_paths_seasonal.append(os.path.join(results_path, 'fall_2016'))
results_paths_seasonal.append(os.path.join(results_path, 'winter_2017'))
results_paths_seasonal.append(os.path.join(results_path, 'spring_2017'))
results_paths_seasonal.append(os.path.join(results_path, 'summer_2017'))

pickle.dump('belltown_commcore', open(os.path.join(data_path, 'background_img_name.p'), 'wb'))
area_map = pickle.load(open(os.path.join(data_path, 'belltown_commcore_subareas.p'), 'rb'))

time1 = 15
num_comps = 6

k_values = [3, 5, 10]
p_value = .01

# Load the data.
all_loads = []
all_element_keys = []
all_gps = []
all_park_data = []
starts = [(6, 2016), (9, 2016), (12, 2016), (3, 2017), (6, 2017)]
ends = [(8, 2016), (11, 2016), (2, 2017), (5, 2017), (8, 2017)]

for pair in zip(starts, ends):
    
    month_year_start = pair[0]
    month_year_end = pair[1]
        
    params = process_data.load_data(data_path=data_path, load_paths=path, 
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


# Mixture plot.
figure_functions.mixture_plot(all_loads_seasonal[-1], all_gps_seasonal[-1], time1, 
                              fig_paths_seasonal[-1], filename='mixture1.png',
                              num_comps=num_comps)

# Getting the results.
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

