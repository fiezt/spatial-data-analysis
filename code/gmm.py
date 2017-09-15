import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
import moran_auto
import itertools
import multiprocessing
import functools
import warnings
warnings.filterwarnings('ignore')


def locational_demand_analysis(park_data, gps_loc, num_comps, k, verbose=True):
    """Find GMM consistency and spatial autocorrelation at each day of the week and time of day.

    This function finds the consistency of the GMM fit over time and also
    finds the spatial autocorrelation characteristics using Moran's I with
    3 types of weight matrices and gets the centroids at each fit of the
    mixture model for each day of the week and time of the day.
    
    :param park_data: Multi-index DataFrame containing datetimes in the first
    level index and block-face keys in the second level index. Values include
    the corresponding loads.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param num_comps: Integer number of mixture components for the model.
    :param k: Integer number of neighbors to use for the Moran weighting matrix.
    :param verbose: Bool indicator of whether to print progress.

    :return results: List of tuples with each tuple containing the day of week 
    as an integer, integer hour of the day, consistency measure, Moran parameters
    using weighting matrix of the mixture components, Moran parameters using
    weighting matrix of the paid area, Moran parameters using the weighting matrix
    of the k nearest neighbor block-faces, and list of numpy arrays of the GPS
    coordinates of the centroids of the cluster for each fit.
    """

    days = sorted(park_data['Day'].unique())
    hours = sorted(park_data['Hour'].unique())

    times = list(itertools.product(days, hours))

    pool = multiprocessing.Pool()

    func = functools.partial(locational_demand_one_time, park_data, gps_loc, times, num_comps, k, verbose)
    results = pool.map(func, range(len(times)))

    pool.close()
    pool.join()

    return results


def locational_demand_one_time(park_data, gps_loc, times, num_comps, k, verbose, iteration):
    """Find GMM consistency and spatial autocorrelation at one day of the week and time of day.

    This function finds the consistency of the GMM fit over time and also
    finds the spatial autocorrelation characteristics using Moran's I with
    3 types of weight matrices and gets the centroids at each fit of the
    mixture model for a single day of the week and time of the day.
    
    :param park_data: Multi-index DataFrame containing datetimes in the first
    level index and block-face keys in the second level index. Values include
    the corresponding loads.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param times: List of tuples, with each tuple containing day and hour pair.
    :param num_comps: Integer number of mixture components for the model.
    :param k: Integer number of neighbors to use for the Moran weighting matrix.
    :param iteration: Integer iteration number of the multiprocessing.

    :return day: Integer day of week.
    :return hour: Integer hour of the day.
    :return time_avg_consistency: Float of the consistency percentage of the mixture
    model assignments (a test point is consistent if it stays in the same component),
    averaged over fitting a model on a date, testing on all others, averaging 
    over the testing consistencies and then averaging over all these averages for
    each time a model is fit on a date.
    :return morans_mixture: List of tuples for each date in the training set 
    with each tuple containing the Moran I value, Moran expectation value, Moran 
    variance, Moran z score, Moran one sided p value, and Moran two sided p 
    value using the connections of the mixture components as the weight matrix.
    :return morans_area: List of tuples for each date in the training set
    with each tuple containing the Moran I value, Moran expectation value, Moran
    variance, Moran z score, Moran one sided p value, and Moran two sided p
    value using the paid area connections from sdot as the weight matrix.
    :return morans_neighbor: List of tuples for each date in the training set
    with each tuple containing the Moran I value, Moran expectation value, Moran 
    variance, Moran z score, Moran one sided p value, and Moran two sided p 
    value using the neighbor connections as the weight matrix.
    :return centers: List of numpy arrays of the centroids of each fit.
    """

    # Bool indicating whether to drop blockfaces where the load is negligible.
    mask_small = False

    time = times[iteration]
    day = time[0]
    hour = time[1]

    if verbose:
        print('Starting day %d and hour %d' % (day, hour))

    # Getting the data for the current day and hour combination.
    data_df = park_data.loc[(park_data['Day'] == day) & (park_data['Hour'] == hour)]
    block_keys = sorted(data_df.index.get_level_values(1).unique().tolist())

    average_consistencies = []
    centers = [] 
    morans_mixture = [] 
    morans_area = []
    morans_neighbor = []

    for train_time in data_df.index.get_level_values(0).unique().tolist():
        train_data = data_df.xs(train_time, level=0)

        # If indicator set, make negligible loads nan so they will be dropped.
        if mask_small:
            train_data.loc[(train_data['Load'] <= .05), 'Load'] = np.nan

        # Dropping block-faces which were closed or had no supply.
        train_data = train_data.dropna()

        N = len(train_data)

        train_active_index = train_data.index.tolist()
        train_mask = [block_keys.index(train_active_index[i]) for i in xrange(len(train_active_index))]  
        train_to_label_map = {train_active_index[i]: i for i in xrange(len(train_active_index))}

        # Getting the data and normalizing the features.
        train_loads = train_data['Load'].values.reshape((-1, 1))
        train = np.hstack((train_loads, gps_loc[train_mask]))
        scaler = MinMaxScaler().fit(train)
        train = scaler.transform(train)

        # Fitting the mixture model.
        gmm = mixture.GaussianMixture(n_init=10, n_components=num_comps,
                                      covariance_type='diag').fit(train)

        # Scaling the mean back to GPS coordinates and saving the centroids.
        means = np.vstack(([(mean[1:] - scaler.min_[1:])/(scaler.scale_[1:]) for mean in gmm.means_]))
        centers.append(means)

        # Getting the labels by choosing the component which maximizes the posterior probability.
        train_labels = gmm.predict(train)

        # Getting spatial correlation statistics for Moran's I using mixture component connections.
        weights = moran_auto.get_mixture_weights(train_labels, N)        
        I = moran_auto.moran_mixture(train_loads[:, 0], train_labels, N)
        expectation = moran_auto.moran_expectation(N)
        variance = moran_auto.moran_variance(train_loads[:, 0], weights, N)
        z_score = moran_auto.z_score(I, expectation, variance)
        p_one_sided, p_two_sided = moran_auto.p_value(z_score)

        morans_mixture.append([I, expectation, variance, z_score, p_one_sided, p_two_sided])

        # Getting spatial correlation statistics for Moran's I using mixture paid area connections.
        weights = moran_auto.get_area_weights(train_active_index, N)        
        I = moran_auto.moran_area(train_loads[:, 0], train_active_index, N)
        expectation = moran_auto.moran_expectation(N)
        variance = moran_auto.moran_variance(train_loads[:, 0], weights, N)
        z_score = moran_auto.z_score(I, expectation, variance)
        p_one_sided, p_two_sided = moran_auto.p_value(z_score)

        morans_area.append([I, expectation, variance, z_score, p_one_sided, p_two_sided])

        # Getting spatial correlation statistics for Moran's I using nearest neighbor connections.
        weights = moran_auto.get_neighbor_weights(gps_loc[train_mask], N, k)
        I = moran_auto.moran_neighbor(train_loads[:, 0], gps_loc[train_mask], N, k)
        expectation = moran_auto.moran_expectation(N)
        variance = moran_auto.moran_variance(train_loads[:, 0], weights, N)
        z_score = moran_auto.z_score(I, expectation, variance)
        p_one_sided, p_two_sided = moran_auto.p_value(z_score)

        morans_neighbor.append([I, expectation, variance, z_score, p_one_sided, p_two_sided])

        consistencies = []

        # For each other day of data, get labels of new data using model that was fit.
        for test_time in data_df.index.get_level_values(0).unique().tolist():

            # Skipping predicting on the time that was trained on.
            if test_time == train_time:
                continue

            test_data = data_df.xs(test_time, level=0)

            # If indicator set, make negligible loads nan so they will be dropped.
            if mask_small:
                test_data.loc[test_data['Load'] <= .05, 'Load'] = np.nan

            # Dropping block-faces which were closed or had no supply.
            test_data = test_data.dropna()

            # Keeping the block-faces which were used in fitting the model.
            test_data = test_data.loc[test_data.index.isin(train_active_index)]
            test_active_index = test_data.index.tolist()
            test_mask = [block_keys.index(test_active_index[i]) for i in xrange(len(test_active_index))]

            # Getting the data and normalizing the features.
            test_loads = test_data['Load'].values.reshape((-1, 1))
            test = np.hstack((test_loads, gps_loc[test_mask]))
            test = scaler.transform(test)

            # Getting the labels by choosing the component which maximizes the posterior probability.
            test_labels = gmm.predict(test)

            # Next block gets the fraction of block-faces assigned to the same component.
            same = 0
            changed = 0

            for i in xrange(len(test_labels)):
                key = test_active_index[i]
                idx = train_to_label_map[key]

                if test_labels[i] == train_labels[idx]:
                    same += 1
                else:
                    changed += 1

            consistency = same/float(same + changed)
            consistencies.append(consistency)

        # Getting average consistency over all test sets.
        average_consistencies.append(np.array(consistencies).mean())

    # Average consistency for the particular day and hour combination.
    time_avg_consistency = round(np.array(average_consistencies).mean() * 100, 2)

    if verbose:
        print('Finished day %d and hour %d' % (day, hour))

    return day, hour, time_avg_consistency, morans_mixture, morans_area, morans_neighbor, centers