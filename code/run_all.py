import os
import sys
import numpy as np
import pandas as pd
import utils
import gmm
import figure_functions
import kmeans_utils


def run_figures(element_keys, loads, gps_loc, num_comps, data_path, fig_path,
                animation_path, time, show_fig):
    """Create the visualizations of the spatial characteristics of the parking data.

    :param element_keys: List containing the keys of block-faces to draw on map.
    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param num_comps: Integer number of mixture components for the model.
    :param data_path: File path to the paystation_info.csv and blockface_locs.p.
    :param fig_path: Path to retrieve the background image from.
    :param animation_path: Path to save the animation video to.
    :param time: Integer column index to get the load data from.
    :param show_fig: Bool indicating whether to show the images.
    """

    figure_functions.plot_neighborhoods(element_keys=element_keys, data_path=data_path,
                                        fig_path=fig_path)

    figure_functions.plot_paid_areas(element_keys=element_keys, data_path=data_path,
                                     fig_path=fig_path)

    figure_functions.surface_plot(loads=loads, gps_loc=gps_loc, time=time,
                                  fig_path=fig_path, show_fig=show_fig)

    figure_functions.interpolation(loads=loads, gps_loc=gps_loc, time=time,
                                   fig_path=fig_path, show_fig=show_fig)

    figure_functions.triangular_grid(loads=loads, gps_loc=gps_loc, time=time,
                                     fig_path=fig_path, show_fig=show_fig)

    figure_functions.contour_plot(loads=loads, gps_loc=gps_loc, time=time,
                                  fig_path=fig_path, show_fig=show_fig)

    figure_functions.voronoi(gps_loc=gps_loc, fig_path=fig_path, show_fig=show_fig)

    figure_functions.spatial_heterogeneity(loads=loads, time=time,
                                           fig_path=fig_path, show_fig=show_fig)

    figure_functions.temporal_heterogeneity(loads=loads, fig_path=fig_path,
                                            show_fig=show_fig)

    figure_functions.temporal_day_plots(loads=loads, fig_path=fig_path,
                                        show_fig=show_fig)

    figure_functions.temporal_hour_plots(loads=loads, fig_path=fig_path,
                                         show_fig=show_fig)

    figure_functions.model_selection(loads=loads, gps_loc=gps_loc,
                                     fig_path=fig_path, show_fig=show_fig)

    figure_functions.mixture_plot(loads=loads, gps_loc=gps_loc, times=time,
                                  fig_path=fig_path, num_comps=num_comps,
                                  show_fig=show_fig)

    figure_functions.create_animation(loads=loads, gps_loc=gps_loc,
                                      fig_path=fig_path,
                                      animation_path=animation_path,
                                      num_comps=num_comps)


def gmm_simulations(park_data, gps_loc, times, k, p_value, fig_path, results_path):
    """Running several GMM tests including prediction and spatial correlation.
    
    :return park_data: Multi-index DataFrame with data sorted by date and block key.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param fig_path: File path to save figures to.
    :param results_path: File path to save results files to.
    """

    results = gmm.locational_demand_analysis(park_data, gps_loc, k)

    days = [result[0] for result in results]
    hours = [result[1] for result in results]
    errors = [result[2] for result in results]
    morans_mixture = [result[3] for result in results]
    morans_area = [result[4] for result in results]
    morans_adjacent = [result[5] for result in results]
    means = [result[6] for result in results]

    write_gmm_results(errors, results_path)
    write_moran_results(days, hours, morans_mixture, morans_area,
                        morans_adjacent, p_value, results_path)

    distances, centroids = kmeans_utils.get_distances(means)
    write_centroid_distance_results(days, hours, means, distances, results_path)

    all_time_points = kmeans_utils.get_centroid_circle_paths(distances, centroids)

    fig, ax = figure_functions.centroid_radius(centroids, all_time_points, gps_loc,
                                               times=times, fig_path=fig_path, shape=(1,len(times)))

    fig, ax = figure_functions.centroid_plots(means, gps_loc, times=times, 
                                              fig_path=fig_path, shape=(1,len(times)))


def write_gmm_results(errors, results_path):
    """Write gmm prediction accuracy results to a text file for easy loading into latex.

    :param errors: List of error percentages corresponding to the day and hour index.
    :param results_path: File path to save results files to.
    """

    with open(os.path.join(results_path, 'gmm_pred_results.txt'), 'wb') as f: 
        errors = np.array(errors).reshape((6, len(errors)/6))
        
        day_avg_errs = errors.mean(axis=1)
        hour_avg_errs = errors.mean(axis=0)
        
        day_count = 0
        for day in errors:
            day_errors = [str(err) + '% & ' for err in day]
            day_errors.append(str(round(day_avg_errs[day_count], 2)) + '% \\\ \n')
            day_errors = ''.join(day_errors)
            
            f.write(day_errors)
            
            day_count += 1
        
        hour_errors = [str(round(hour_avg_errs[i], 2)) + '% & ' 
                       for i in xrange(len(hour_avg_errs)-1)]

        hour_errors.append(str(round(hour_avg_errs[-1], 2)) + '% \\\ \n')
        hour_errors = ''.join(hour_errors)
        
        f.write(hour_errors)

    index = []
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday'}

    
def write_moran_results(days, hours, morans_mix, morans_area, morans_adj, p_value, results_path):
    """Writing all the Moran autocorrelation results to files.

    :param days: List of days indexes.
    :param hours: List of hour indexes which correspond with the days indexes.
    :param morans_mix: List of tuples with each tuple containing the Moran I, 
    expectation, variance, z score, one sided p, and two sided p values using 
    the connections of the mixture components as the weight matrix.
    :param morans_adj: List of tuples with each tuple containing the Moran I, 
    expectation, variance, z score, one sided p, and two sided p values using 
    the adjacent connections as the weight matrix.
    :param results_path: File path to save results files to.
    """

    I_mix = [[morans_mix[j][i][0] for i in xrange(len(morans_mix[j]))] for j in xrange(len(morans_mix))]
    one_sided_mix = [[morans_mix[j][i][4] for i in xrange(len(morans_mix[j]))] for j in xrange(len(morans_mix))]
    two_sided_mix = [[morans_mix[j][i][5] for i in xrange(len(morans_mix[j]))] for j in xrange(len(morans_mix))]

    I_area = [[morans_area[j][i][0] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]
    one_sided_area = [[morans_area[j][i][4] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]
    two_sided_area = [[morans_area[j][i][5] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]

    I_adj = [[morans_adj[j][i][0] for i in xrange(len(morans_adj[j]))] for j in xrange(len(morans_adj))]
    one_sided_adj = [[morans_adj[j][i][4] for i in xrange(len(morans_adj[j]))] for j in xrange(len(morans_adj))]
    two_sided_adj = [[morans_adj[j][i][5] for i in xrange(len(morans_adj[j]))] for j in xrange(len(morans_adj))]

    index = []
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday'}

    for d, h in zip(days, hours):
        day = day_map[d]

        if h < 12:
            hour = str(h) + ':00 AM'
        elif hour == 12:
            hour = str(h) + ':00 PM'
        else:
            hour = str(h-12) + ':00 PM'
            
        index.append(day + ' ' + hour)
        

    ###################### Writing moran mixture results #######################
    I_mix_df = pd.DataFrame(data=I_mix, index=index)
    I_mix_df = I_mix_df.fillna('')
    I_mix_df.to_csv(os.path.join(results_path, 'moran_mix_I_results.csv'), 
                    sep=',', header=False)

    one_sided_mix_df = pd.DataFrame(data=one_sided_mix, index=index)
    one_sided_mix_df = one_sided_mix_df.fillna('')
    one_sided_mix_df.to_csv(os.path.join(results_path, 'p_one_sided_mix_results.csv'), 
                            sep=',', header=False)

    two_sided_mix_df = pd.DataFrame(data=two_sided_mix, index=index)
    two_sided_mix_df = two_sided_mix_df.fillna('')
    two_sided_mix_df.to_csv(os.path.join(results_path, 'p_two_sided_mix_results.csv'), 
                            sep=',', header=False)

    one_sided_mix_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in one_sided_mix]

    one_sided_mix_sig_df = pd.DataFrame(data=one_sided_mix_sig, index=index)
    one_sided_mix_sig_df = one_sided_mix_sig_df.fillna('')
    one_sided_mix_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_mix_significance.csv'), 
                            sep=',', header=False)

    two_sided_mix_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in two_sided_mix]

    two_sided_mix_sig_df = pd.DataFrame(data=two_sided_mix_sig, index=index)
    two_sided_mix_sig_df = two_sided_mix_sig_df.fillna('')
    two_sided_mix_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_mix_significance.csv'), 
                                sep=',', header=False)
    ############################################################################


    ###################### Writing moran area results ##########################
    I_area_df = pd.DataFrame(data=I_area, index=index)
    I_area_df = I_area_df.fillna('')
    I_area_df.to_csv(os.path.join(results_path, 'moran_area_I_results.csv'), 
                    sep=',', header=False)

    one_sided_area_df = pd.DataFrame(data=one_sided_area, index=index)
    one_sided_area_df = one_sided_area_df.fillna('')
    one_sided_area_df.to_csv(os.path.join(results_path, 'p_one_sided_area_results.csv'), 
                            sep=',', header=False)

    two_sided_area_df = pd.DataFrame(data=two_sided_area, index=index)
    two_sided_area_df = two_sided_area_df.fillna('')
    two_sided_area_df.to_csv(os.path.join(results_path, 'p_two_sided_area_results.csv'), 
                            sep=',', header=False)

    one_sided_area_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in one_sided_area]

    one_sided_area_sig_df = pd.DataFrame(data=one_sided_area_sig, index=index)
    one_sided_area_sig_df = one_sided_area_sig_df.fillna('')
    one_sided_area_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_area_significance.csv'), 
                            sep=',', header=False)

    two_sided_area_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in two_sided_area]

    two_sided_area_sig_df = pd.DataFrame(data=two_sided_area_sig, index=index)
    two_sided_area_sig_df = two_sided_area_sig_df.fillna('')
    two_sided_area_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_area_significance.csv'), 
                                sep=',', header=False)
    ############################################################################


    ###################### Writing moran adjacent results ######################
    I_adj_df = pd.DataFrame(data=I_adj, index=index)
    I_adj_df = I_adj_df.fillna('')
    I_adj_df.to_csv(os.path.join(results_path, 'moran_adj_I_results.csv'), sep=',', header=False)

    one_sided_adj_df = pd.DataFrame(data=one_sided_adj, index=index)
    one_sided_adj_df = one_sided_adj_df.fillna('')
    one_sided_adj_df.to_csv(os.path.join(results_path, 'p_one_sided_adj_results.csv'), sep=',', header=False)

    two_sided_adj_df = pd.DataFrame(data=two_sided_adj, index=index)
    two_sided_adj_df = two_sided_adj_df.fillna('')
    two_sided_adj_df.to_csv(os.path.join(results_path, 'p_two_sided_adj_results.csv'), sep=',', header=False)

    one_sided_adj_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in one_sided_adj]

    one_sided_adj_sig_df = pd.DataFrame(data=one_sided_adj_sig, index=index)
    one_sided_adj_sig_df = one_sided_adj_sig_df.fillna('')
    one_sided_adj_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_adj_significance.csv'), 
                                sep=',', header=False)

    two_sided_adj_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                         for row in two_sided_adj]

    two_sided_adj_sig_df = pd.DataFrame(data=two_sided_adj_sig, index=index)
    two_sided_adj_sig_df = two_sided_adj_sig_df.fillna('')
    two_sided_adj_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_adj_significance.csv'), 
                                sep=',', header=False)
    ############################################################################


    ###################### Writing moran average results #######################
    I_mix_avg = np.array([item for sublist in I_mix for item in sublist]).mean()
    p_one_mix_sig_avg = np.array(one_sided_mix_sig).mean()
    p_two_mix_sig_avg = np.array(two_sided_mix_sig).mean()

    I_area_avg = np.array([item for sublist in I_area for item in sublist]).mean()
    p_one_area_sig_avg = np.array(one_sided_area_sig).mean()
    p_two_area_sig_avg = np.array(two_sided_area_sig).mean()

    I_adj_avg = np.array([item for sublist in I_adj for item in sublist]).mean()
    p_one_adj_sig_avg = np.array(one_sided_adj_sig).mean()
    p_two_adj_sig_avg = np.array(two_sided_adj_sig).mean()

    avgs = np.array([[I_mix_avg, I_area_avg, I_adj_avg], 
                     [p_one_mix_sig_avg, p_one_area_sig_avg, p_one_adj_sig_avg], 
                     [p_two_mix_sig_avg, p_two_area_sig_avg, p_two_adj_sig_avg]])

    index = ['Moran I Over All Days and Times', 
             'Significant One Sided P Value Percentage Average Over All Days And Times', 
             'Significant Two Sided P Value Percentage Average Over All Days And Times']

    cols = ['Using Mixture Connections as Weight Matrix', 
            'Using Paid Area Connections as Weight Matrix',
            'Using Adjacent Connections as Weight Matrix']

    avg_df = pd.DataFrame(avgs, index=index, columns=cols)
    avg_df.to_csv(os.path.join(results_path, 'moran_averages.csv'), sep=',')
    ############################################################################


def write_centroid_distance_results(days, hours, means, distances, results_path):
    """Writing the average distance to centroid of cluster centroids from each point.

    :param days: List of days indexes.
    :param hours: List of hour indexes which correspond with the days indexes.
    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains a 2-d array that contains each of the centroids
    for a GMM fit of a particular date for the weekday and hour.
    :param distances: Numpy array of 2 dimensions with the mean as the crow flies distance
    from all points in a cluster to the clusters centroid for each centroid in each row.
    :param results_path: File path to save results files to.
    """

    distances = distances.mean(axis=1)

    index = []
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday'}

    for d, h in zip(days, hours):
        day = day_map[d]

        if h < 12:
            hour = str(h) + ':00 AM'
        elif hour == 12:
            hour = str(h) + ':00 PM'
        else:
            hour = str(h-12) + ':00 PM'
            
        index.append(day + ' ' + hour)

    col = ['Average Distance in Meters']

    distances_df = pd.DataFrame(data=distances, index=index, columns=col)
    distances_df.to_csv(os.path.join(results_path, 'average_distances.csv'), 
                        sep=',', header=False)


def main():
    curr_dir = os.getcwd()
    data_path = curr_dir + '/../data/'
    fig_path = curr_dir + '/../figs/'
    results_path = curr_dir + '/../results/'
    animation_path = curr_dir + '/../animation/'

    if len(sys.argv) == 1:
        # Default times.
        time = 58
        time1 = 50
        time2 = 58
    elif len(sys.argv) >= 3 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        time = int(sys.argv[1])
        time1 = int(sys.argv[1])
        time2 = int(sys.argv[2])
    else:
        print('Bad Input: Exiting Now')
        return

    run_figs = True
    run_sims = True

    if 'figs' in sys.argv:
        run_figs = False

    if 'sims' in sys.argv:
        run_sims = False

    if run_figs and run_sims:
        params = utils.load_data(data_path)
        gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx = params
        run_figures(avg_loads, gps_loc, N, P, fig_path, animation_path, time, time1, time2)
        
        park_data = utils.load_daily_data(park_data)
        gmm_simulations(park_data, gps_loc, N, fig_path, results_path)
    elif run_figs:
        params = utils.load_data(data_path)
        gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx = params
        run_figures(avg_loads, gps_loc, N, P, fig_path, animation_path, time, time1, time2)
    elif run_sims:
        park_data, gps_loc, N = utils.load_daily_data_standalone(data_path)
        gmm_simulations(park_data, gps_loc, N, fig_path, results_path)
    

if __name__ == '__main__':
    main()