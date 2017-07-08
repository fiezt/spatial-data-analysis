import os
import numpy as np
import pandas as pd
from collections import defaultdict
import utils
import gmm
import figure_functions
import kmeans_utils


def run_figures(avg_loads, gps_loc, park_data, N, P, idx_to_day_hour, 
                day_hour_to_idx, fig_path, animation_path):
    """

    """

    figure_functions.model_selection(avg_loads, gps_loc, P, fig_path)
    figure_functions.create_animation(avg_loads, gps_loc, N, P, fig_path, animation_path)

    time = 58
    time1 = 50
    time2 = 58

    
    fig, ax = figure_functions.mixture_plot(loads=avg_loads, gps_loc=gps_loc, 
                                            times=[time1], N=N, fig_path=fig_path, 
                                            shape=(1,1), filename='friday_10am_gmm.png',
                                            title='')
    fig, ax = figure_functions.mixture_plot(loads=avg_loads, gps_loc=gps_loc, 
                                            times=[time2], N=N, fig_path=fig_path, 
                                            shape=(1,1), filename='friday_6pm_gmm.png',
                                            title='')
    

    fig, ax = figure_functions.surface_plot(loads=avg_loads, gps_loc=gps_loc, time=time, 
                                            fig_path=fig_path)

    fig, ax = figure_functions.interpolation(loads=avg_loads, gps_loc=gps_loc, time=time,
                                             N=N, fig_path=fig_path)

    fig, ax = figure_functions.triangular_grid(loads=avg_loads, gps_loc=gps_loc, time=time,
                                               N=N, fig_path=fig_path)

    fig, ax = figure_functions.contour_plot(loads=avg_loads, gps_loc=gps_loc, time=time1,
                                            title='Friday 10:00 AM Average Load Contours', 
                                            N=N, filename='friday_10am.png', fig_path=fig_path, 
                                            contours=10)

    fig, ax = figure_functions.contour_plot(loads=avg_loads, gps_loc=gps_loc, time=time2,
                                            title='Friday 6:00 PM Average Load Contours', 
                                            N=N, filename='friday_6pm.png', fig_path=fig_path, 
                                            contours=10)

    fig, ax = figure_functions.voronoi(gps_loc=gps_loc, N=N, fig_path=fig_path)
    
    fig, ax = figure_functions.spatial_heterogeneity(loads=avg_loads, time=time, 
                                                     N=N, fig_path=fig_path)

    fig, ax = figure_functions.temporal_heterogeneity(loads=avg_loads, time=time, 
                                                      P=P, fig_path=fig_path)

    fig, ax = figure_functions.temporal_day_plots(loads=avg_loads, P=P, fig_path=fig_path)

    fig, ax = figure_functions.temporal_hour_plots(loads=avg_loads, fig_path=fig_path)


def gmm_simulations(park_data, gps_loc, N, fig_path, results_path):
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

    create_centroid_figure(means, gps_loc, N, fig_path)


def write_gmm_results(errors, results_path):
    """

    """

    with open(os.path.join(results_path, 'gmm_pred_results.txt'), 'wb') as f: 
        errors = np.array(errors).reshape((6,12))
        
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


def write_moran_results(days, hours, morans_mix, morans_adj, results_path):
    """

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
    """

    """

    distances = kmeans_utils.get_time_scores(means)
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

    col = 'Average Distance in Meters'

    distances_df = pd.DataFrame(data=distances, index=index, columns=col)
    distances_df.to_csv(os.path.join(results_path, 'average_distances.csv'), 
                        sep=',', header=False)


def create_centroid_figure(means, gps_loc, N, fig_path):
    """

    """
    
    fig, ax = figure_functions.centroid_plots(means, gps_loc, N, times=range(6), 
                                              fig_path=fig_path, shape=(2,3))


def main():
    curr_dir = os.getcwd()
    data_path = curr_dir + '/../data/'
    fig_path = curr_dir + '/../figs/'
    results_path = curr_dir + '/../results/'
    animation_path = curr_dir + '/../animation/'

    
    params = utils.load_data(data_path)
    gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx = params

    run_figures(avg_loads, gps_loc, park_data, N, P, idx_to_day_hour, 
                day_hour_to_idx, fig_path, animation_path)
    
    park_data = utils.load_daily_data(park_data)
    gmm_simulations(park_data, gps_loc, N, fig_path, results_path)
    

if __name__ == '__main__':
    main()