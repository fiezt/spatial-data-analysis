import numpy as np
import pandas as pd
import os


def write_gmm_results(consistencies, results_path):
    """Write gmm consistency results to a text file for easy loading into latex.

    :param consistencies: List of consistency measures corresponding to the day and hour indexes.
    :param results_path: Path to save results files to.
    """

    with open(os.path.join(results_path, 'gmm_consistency_results.txt'), 'wb') as f:
        consistencies = np.array(consistencies).reshape((6, len(consistencies)/6))
        
        day_avg = consistencies.mean(axis=1)
        hour_avg = consistencies.mean(axis=0)
        
        day_count = 0
        for day in consistencies:
            day_consistencies = [str(measure) + '\% & ' for measure in day]
            day_consistencies.append(str(round(day_avg[day_count], 2)) + '\% \\\ \n')
            day_consistencies = ''.join(day_consistencies)
            
            f.write(day_consistencies)
            
            day_count += 1
        
        hour_consistencies = [str(round(hour_avg[i], 2)) + '\% & ' for i in xrange(len(hour_avg)-1)]
        hour_consistencies.append(str(round(hour_avg[-1], 2)) + '\% \\\ \n')
        hour_consistencies = ''.join(hour_consistencies)
        
        f.write(hour_consistencies)

    
def write_moran_results(days, hours, morans_mixture, morans_area, morans_neighbor, p_value, results_path):
    """Writing all the Moran autocorrelation results to files.

    :param days: List of days indexes.
    :param hours: List of hour indexes which correspond with the days indexes.
    :param morans_mixture: List of tuples for each date in the training set
    with each tuple containing the Moran I value, Moran expectation value, Moran
    variance, Moran z score, Moran one sided p value, and Moran two sided p
    value using the connections of the mixture components as the weight matrix.
    :param morans_area: List of tuples for each date in the training set
    with each tuple containing the Moran I value, Moran expectation value, Moran
    variance, Moran z score, Moran one sided p value, and Moran two sided p
    value using the paid area connections from sdot as the weight matrix.
    :param morans_neighbor: List of tuples for each date in the training set
    with each tuple containing the Moran I value, Moran expectation value, Moran
    variance, Moran z score, Moran one sided p value, and Moran two sided p
    value using the neighbor connections as the weight matrix.
    :param p_value: Float p value to use to measure significance.
    :param results_path: File path to save results files to.
    """

    I_mix = [[morans_mixture[j][i][0] for i in xrange(len(morans_mixture[j]))] for j in xrange(len(morans_mixture))]
    one_sided_mix = [[morans_mixture[j][i][4] for i in xrange(len(morans_mixture[j]))] for j in xrange(len(morans_mixture))]
    two_sided_mix = [[morans_mixture[j][i][5] for i in xrange(len(morans_mixture[j]))] for j in xrange(len(morans_mixture))]

    I_area = [[morans_area[j][i][0] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]
    one_sided_area = [[morans_area[j][i][4] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]
    two_sided_area = [[morans_area[j][i][5] for i in xrange(len(morans_area[j]))] for j in xrange(len(morans_area))]

    I_neighbor = [[morans_neighbor[j][i][0] for i in xrange(len(morans_neighbor[j]))] for j in xrange(len(morans_neighbor))]
    one_sided_neighbor = [[morans_neighbor[j][i][4] for i in xrange(len(morans_neighbor[j]))] for j in xrange(len(morans_neighbor))]
    two_sided_neighbor = [[morans_neighbor[j][i][5] for i in xrange(len(morans_neighbor[j]))] for j in xrange(len(morans_neighbor))]

    index = []
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday'}

    for d, h in zip(days, hours):
        day = day_map[d]

        if h < 12:
            hour = str(h) + ':00 AM'
        elif h == 12:
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


    ###################### Writing moran neighbor results ######################
    I_neighbor_df = pd.DataFrame(data=I_neighbor, index=index)
    I_neighbor_df = I_neighbor_df.fillna('')
    I_neighbor_df.to_csv(os.path.join(results_path, 'moran_neighbor_I_results.csv'), sep=',', header=False)

    one_sided_neighbor_df = pd.DataFrame(data=one_sided_neighbor, index=index)
    one_sided_neighbor_df = one_sided_neighbor_df.fillna('')
    one_sided_neighbor_df.to_csv(os.path.join(results_path, 'p_one_sided_neighbor_results.csv'), sep=',', header=False)

    two_sided_neighbor_df = pd.DataFrame(data=two_sided_neighbor, index=index)
    two_sided_neighbor_df = two_sided_neighbor_df.fillna('')
    two_sided_neighbor_df.to_csv(os.path.join(results_path, 'p_two_sided_neighbor_results.csv'), sep=',', header=False)

    one_sided_neighbor_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2)
                         for row in one_sided_neighbor]

    one_sided_neighbor_sig_df = pd.DataFrame(data=one_sided_neighbor_sig, index=index)
    one_sided_neighbor_sig_df = one_sided_neighbor_sig_df.fillna('')
    one_sided_neighbor_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_neighbor_significance.csv'),
                                sep=',', header=False)

    two_sided_neighbor_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2)
                         for row in two_sided_neighbor]

    two_sided_neighbor_sig_df = pd.DataFrame(data=two_sided_neighbor_sig, index=index)
    two_sided_neighbor_sig_df = two_sided_neighbor_sig_df.fillna('')
    two_sided_neighbor_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_neighbor_significance.csv'),
                                sep=',', header=False)
    ############################################################################


    ###################### Writing moran average results #######################
    I_mix_avg = np.array([item for sublist in I_mix for item in sublist]).mean()
    p_one_mix_sig_avg = np.array(one_sided_mix_sig).mean()
    p_two_mix_sig_avg = np.array(two_sided_mix_sig).mean()

    I_area_avg = np.array([item for sublist in I_area for item in sublist]).mean()
    p_one_area_sig_avg = np.array(one_sided_area_sig).mean()
    p_two_area_sig_avg = np.array(two_sided_area_sig).mean()

    I_neighbor_avg = np.array([item for sublist in I_neighbor for item in sublist]).mean()
    p_one_neighbor_sig_avg = np.array(one_sided_neighbor_sig).mean()
    p_two_neighbor_sig_avg = np.array(two_sided_neighbor_sig).mean()

    avgs = np.array([[I_mix_avg, I_area_avg, I_neighbor_avg],
                     [p_one_mix_sig_avg, p_one_area_sig_avg, p_one_neighbor_sig_avg],
                     [p_two_mix_sig_avg, p_two_area_sig_avg, p_two_neighbor_sig_avg]])

    index = ['Moran I Over All Days and Times', 
             'Significant One Sided P Value Percentage Average Over All Days And Times', 
             'Significant Two Sided P Value Percentage Average Over All Days And Times']

    cols = ['Using Mixture Connections as Weight Matrix', 
            'Using Paid Area Connections as Weight Matrix',
            'Using Nearest Neighbor Connections as Weight Matrix']

    avg_df = pd.DataFrame(avgs, index=index, columns=cols)
    avg_df.to_csv(os.path.join(results_path, 'moran_averages.csv'), sep=',')
    ############################################################################


def write_centroid_distance_results(days, hours, distances, results_path):
    """Writing the average distance to centroid of cluster centroids from each point.

    :param days: List of days indexes.
    :param hours: List of hour indexes which correspond with the days indexes.
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
        elif h == 12:
            hour = str(h) + ':00 PM'
        else:
            hour = str(h-12) + ':00 PM'
            
        index.append(day + ' ' + hour)

    col = ['Average Distance in Meters']

    distances_df = pd.DataFrame(data=distances, index=index, columns=col)
    distances_df.to_csv(os.path.join(results_path, 'average_distances.csv'), 
                        sep=',', header=False)