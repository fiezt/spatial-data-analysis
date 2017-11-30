import numpy as np
import pandas as pd
import os


def write_gmm_results(consistencies, results_path, filename='consistencies.csv'):
    """Writing the consistency results to a file.

    :param consistencies: List of consistency values to write to file.
    :param results_path: File path to save the file to.
    :param filename: File name to save the file as.
    """

    consistency_results = np.array(consistencies).reshape((6, -1))
    d = consistency_results.shape[1]

    # Hourly average.
    hourly = np.round(consistency_results.mean(axis=0), 1)

    # Daily average.
    daily = np.concatenate((np.round(consistency_results.mean(axis=1), 1), [np.nan]))

    consistency_results = np.vstack((consistency_results, hourly))
    consistency_results = np.hstack((consistency_results, daily.reshape((-1, 1))))

    columns = range(8, 8 + d) + ['Average Daily']
    index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Average Hourly']

    consistency_results = pd.DataFrame(consistency_results, index=index, columns=columns)
    consistency_results.to_csv(os.path.join(results_path, filename))


def write_moran_results(days, hours, morans, p_value, results_path):
    """Writing the Moran autocorrelation results to files.

    :param days: List of days indexes.
    :param hours: List of hour indexes.
    :param morans: List of tuples for each date in the training set with
    each tuple containing the Moran I value, Moran expectation value, Moran
    variance, Moran z score, Moran one sided p value, and Moran two sided p.
    :param p_value: Float p value to use to measure significance.
    :param results_path: File path to save results files to.

    :return I_avg: Float average Moran I value.
    :return p_one_side_sig_avg: Float percentage of significant one sided p value instances.
    :return p_two_side_sig_avg: Float percentage of significant two sided p value instances.
    """

    I = [[morans[j][i][0] for i in xrange(len(morans[j]))] for j in xrange(len(morans))]
    one_sided = [[morans[j][i][4] for i in xrange(len(morans[j]))] for j in xrange(len(morans))]
    two_sided = [[morans[j][i][5] for i in xrange(len(morans[j]))] for j in xrange(len(morans))]

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
        

    ###################### Writing moran results #######################
    I_df = pd.DataFrame(data=I, index=index)
    I_df = I_df.fillna('')
    I_df.to_csv(os.path.join(results_path, 'I_results.csv'), 
                sep=',', header=False)

    one_sided_df = pd.DataFrame(data=one_sided, index=index)
    one_sided_df = one_sided_df.fillna('')
    one_sided_df.to_csv(os.path.join(results_path, 'p_one_sided_results.csv'), 
                        sep=',', header=False)

    two_sided_df = pd.DataFrame(data=two_sided, index=index)
    two_sided_df = two_sided_df.fillna('')
    two_sided_df.to_csv(os.path.join(results_path, 'p_two_sided_results.csv'), 
                        sep=',', header=False)

    one_sided_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                     for row in one_sided]

    one_sided_sig_df = pd.DataFrame(data=one_sided_sig, index=index)
    one_sided_sig_df = one_sided_sig_df.fillna('')
    one_sided_sig_df.to_csv(os.path.join(results_path, 'p_one_sided_significance.csv'), 
                            sep=',', header=False)

    two_sided_sig = [round(len(np.where(np.array(row) < p_value)[0])*100/float(len(row)), 2) 
                     for row in two_sided]

    two_sided_sig_df = pd.DataFrame(data=two_sided_sig, index=index)
    two_sided_sig_df = two_sided_sig_df.fillna('')
    two_sided_sig_df.to_csv(os.path.join(results_path, 'p_two_sided_significance.csv'), 
                            sep=',', header=False)
    ############################################################################


    ###################### Computing Moran average results #######################
    I_avg = np.array([item for sublist in I for item in sublist]).mean()
    p_one_sig_avg = np.array(one_sided_sig).mean()
    p_two_sig_avg = np.array(two_sided_sig).mean()

    return I_avg, p_one_sig_avg, p_two_sig_avg


def write_centroid_distance_results(days, hours, distances, results_path):
    """Writing the average distance to centroid of cluster centroids from each point.

    :param days: List of days indexes.
    :param hours: List of hour indexes.
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