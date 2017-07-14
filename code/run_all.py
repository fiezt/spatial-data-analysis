import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import utils
import gmm
import figure_functions
import kmeans_utils


def run_figures(loads, gps_loc, N, P, fig_path, animation_path, time, time1, time2):
    """Create the visualizations of the spatial characteristics of the parking data.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param P: Integer number of total hours in the week with loads.
    :param fig_path: File path to save figures to.
    :param animation_path: File path to save mixture animation to.
    :param time: Integer index of column to use from loads for majority of plots.
    :param time1: Integer index of column index to use for mixture and contour plots.
    :param time2: Integer index of column index to use for mixture and contour plots.
    """

    figure_functions.model_selection(loads, gps_loc, P, fig_path)
    figure_functions.create_animation(loads, gps_loc, N, P, fig_path, animation_path)
    
    fig, ax = figure_functions.mixture_plot(loads=loads, gps_loc=gps_loc, 
                                            times=[time1], N=N, fig_path=fig_path, 
                                            shape=(1,1), filename='mixture1.png',
                                            title='')
    fig, ax = figure_functions.mixture_plot(loads=loads, gps_loc=gps_loc, 
                                            times=[time2], N=N, fig_path=fig_path, 
                                            shape=(1,1), filename='mixture2.png',
                                            title='')

    fig, ax = figure_functions.surface_plot(loads=loads, gps_loc=gps_loc, time=time, 
                                            fig_path=fig_path)

    fig, ax = figure_functions.interpolation(loads=loads, gps_loc=gps_loc, time=time,
                                             N=N, fig_path=fig_path)

    fig, ax = figure_functions.triangular_grid(loads=loads, gps_loc=gps_loc, time=time,
                                               N=N, fig_path=fig_path)

    fig, ax = figure_functions.contour_plot(loads=loads, gps_loc=gps_loc, time=time1,
                                            title='', 
                                            N=N, filename='contour1.png', fig_path=fig_path, 
                                            contours=10)

    fig, ax = figure_functions.contour_plot(loads=loads, gps_loc=gps_loc, time=time2,
                                            title='', 
                                            N=N, filename='contour2.png', fig_path=fig_path, 
                                            contours=10)

    fig, ax = figure_functions.voronoi(gps_loc=gps_loc, N=N, fig_path=fig_path)
    
    fig, ax = figure_functions.spatial_heterogeneity(loads=loads, time=time, 
                                                     N=N, fig_path=fig_path)

    fig, ax = figure_functions.temporal_heterogeneity(loads=loads, time=time, 
                                                      P=P, fig_path=fig_path)

    fig, ax = figure_functions.temporal_day_plots(loads=loads, P=P, fig_path=fig_path)

    fig, ax = figure_functions.temporal_hour_plots(loads=loads, fig_path=fig_path)


def gmm_simulations(park_data, gps_loc, N, fig_path, results_path):
    """Running several GMM tests including prediction and spatial correlation.
    
    :return park_data: Multi-index DataFrame with data sorted by date and block key.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param fig_path: File path to save figures to.
    :param results_path: File path to save results files to.
    """

    results = gmm.locational_demand_analysis(park_data, gps_loc, N)

    days = [result[0] for result in results]
    hours = [result[1] for result in results]
    errors = [result[2] for result in results]
    morans_mixture = [result[3] for result in results]
    morans_adjacent = [result[4] for result in results]
    means = [result[5] for result in results]

    write_gmm_results(errors, results_path)
    write_moran_results(days, hours, morans_mixture, morans_adjacent, results_path)
    write_centroid_distance_results(days, hours, means, results_path)

    fig, ax = figure_functions.centroid_plots(means, gps_loc, N, times=range(6), 
                                              fig_path=fig_path, shape=(2,3))


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

    
def write_moran_results(days, hours, morans_mix, morans_adj, results_path):
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

    one_sided_mix_sig = [round(len(np.where(np.array(row) < .05)[0])*100/float(len(row)), 2) 
                         for row in one_sided_mix]

    one_sided_mix_sig_df = pd.DataFrame(data=one_sided_mix_sig, index=index)
    one_sided_mix_sig_df = one_sided_mix_sig_df.fillna('')
    one_sided_mix_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_mix_significance.csv'), 
                            sep=',', header=False)

    two_sided_mix_sig = [round(len(np.where(np.array(row) < .05)[0])*100/float(len(row)), 2) 
                         for row in two_sided_mix]

    two_sided_mix_sig_df = pd.DataFrame(data=two_sided_mix_sig, index=index)
    two_sided_mix_sig_df = two_sided_mix_sig_df.fillna('')
    two_sided_mix_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_mix_significance.csv'), 
                                sep=',', header=False)

    I_adj_df = pd.DataFrame(data=I_adj, index=index)
    I_adj_df = I_adj_df.fillna('')
    I_adj_df.to_csv(os.path.join(results_path, 'moran_adj_I_results.csv'), sep=',', header=False)

    one_sided_adj_df = pd.DataFrame(data=one_sided_adj, index=index)
    one_sided_adj_df = one_sided_adj_df.fillna('')
    one_sided_adj_df.to_csv(os.path.join(results_path, 'p_one_sided_adj_results.csv'), sep=',', header=False)

    two_sided_adj_df = pd.DataFrame(data=two_sided_adj, index=index)
    two_sided_adj_df = two_sided_adj_df.fillna('')
    two_sided_adj_df.to_csv(os.path.join(results_path, 'p_two_sided_adj_results.csv'), sep=',', header=False)

    one_sided_adj_sig = [round(len(np.where(np.array(row) < .05)[0])*100/float(len(row)), 2) 
                         for row in one_sided_adj]

    one_sided_adj_sig_df = pd.DataFrame(data=one_sided_adj_sig, index=index)
    one_sided_adj_sig_df = one_sided_adj_sig_df.fillna('')
    one_sided_adj_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_adj_significance.csv'), 
                                sep=',', header=False)

    two_sided_adj_sig = [round(len(np.where(np.array(row) < .05)[0])*100/float(len(row)), 2) 
                         for row in two_sided_adj]

    two_sided_adj_sig_df = pd.DataFrame(data=two_sided_adj_sig, index=index)
    two_sided_adj_sig_df = two_sided_adj_sig_df.fillna('')
    two_sided_adj_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_adj_significance.csv'), 
                                sep=',', header=False)

    I_mix_avg = np.array([item for sublist in I_mix for item in sublist]).mean()
    p_one_mix_sig_avg = np.array(one_sided_mix_sig).mean()
    p_two_mix_sig_avg = np.array(two_sided_mix_sig).mean()

    I_adj_avg = np.array([item for sublist in I_adj for item in sublist]).mean()
    p_one_adj_sig_avg = np.array(one_sided_adj_sig).mean()
    p_two_adj_sig_avg = np.array(two_sided_adj_sig).mean()

    avgs = np.array([[I_mix_avg, I_adj_avg], [p_one_mix_sig_avg, p_one_adj_sig_avg], 
                     [p_two_mix_sig_avg, p_two_adj_sig_avg]])
    index = ['Moran I Over All Days and Times', 
             'Significant One Sided P Value Percentage Average Over All Days And Times', 
             'Significant Two Sided P Value Percentage Average Over All Days And Times']

    cols = ['Using Mixture Connections as Weights Matrix', 'Using Adjacent Connections as Weight Matrix']
    avg_df = pd.DataFrame(avgs, index=index, columns=cols)
    avg_df.to_csv(os.path.join(results_path, 'moran_averages.csv'), sep=',')


def write_centroid_distance_results(days, hours, means, results_path):
    """Writing the average distance to centroid of cluster centroids from each point.

    :param days: List of days indexes.
    :param hours: List of hour indexes which correspond with the days indexes.
    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains a 2-d array that contains each of the centroids
    for a GMM fit of a particular date for the weekday and hour.
    :param results_path: File path to save results files to.
    """

    distances = kmeans_utils.get_distances(means)
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