import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from moran_auto import moran_mixture
import itertools
import multiprocessing
import functools


def locational_demand_analysis(park_data, gps_loc, N):
    """Find GMM prediction error at each day of the week and time of day.
    
    :param park_data: Multi-index DataFrame containing dates and blockface key
    indexes and the corresponding loads, hour, and day.

    :param gps_loc: Numpy array with each row containing the GPS coordinates of 
    each blockface ordered the same as in park_data.

    :param N: Integer count of the number of blockfaces.

    :return results: List of tuples with each tuple containing the day of week 
    as an integer, integer hour of the day, prediction accuracy, and list of 
    numpy arrays of the GPS coordinates of the centroids of the cluster for each
    fit.
    """

    days = sorted(park_data['Day'].unique())
    hours = sorted(park_data['Hour'].unique())

    times = list(itertools.product(days, hours))

    pool = multiprocessing.Pool()

    func = functools.partial(GMM, park_data, gps_loc, times, N)
    results = pool.map(func, range(len(times)))

    pool.close()
    pool.join()

    return results


def GMM(park_data, gps_loc, times, N, iter):
    """Finding the GMM prediction error for a day and hour combination.
    
    :param park_data: Multi-index DataFrame containing dates and blockface key
    indexes and the corresponding loads, hour, and day.
    :param gps_loc: Numpy array with each row containing the GPS coordinates of 
    each blockface ordered the same as in park_data.
    :param times: List of tuples, with each tuple containing day and hour pair.
    :param N: Integer count of the number of blockfaces.
    :param iter: Integer iteration number of the multiprocessing.

    :return result: Tuple containing the integer day of week, integer hour of
    the day, float time_avg_accuracy of the accuracy percentage, float 
    time_avg_moran of the average Moran I autocorrelation, and list of 
    numpy arrays of the centroids of each fit.
    """

    time = times[iter]
    day = time[0]
    hour = time[1]

    data_df = park_data.loc[(park_data['Day'] == day) & (park_data['Hour'] == hour)]

    # Each row is an element key, and each column is a date.
    data = data_df['Load'].values.reshape((-1, N)).T

    P = data.shape[1]

    average_accuracies = []

    centers = []

    morans = []

    # Fitting the model for each date for the given day and hour combination.
    for train_time in xrange(P):

        train = np.hstack((data[:, train_time, None], gps_loc))

        # Saving the scaling so it can be applied to the test set as well.
        unscaled_loads = train[:,0]
        scaler = MinMaxScaler().fit(train)
        train = scaler.transform(train)

        gmm = mixture.GaussianMixture(n_init=200, n_components=4, 
                                      covariance_type='diag').fit(train)

        # Scaling the mean and covariances back to GPS coordinates.
        means = np.vstack(([(mean[1:] - scaler.min_[1:])/(scaler.scale_[1:]) for mean in gmm.means_]))
        covs = np.dstack(([np.diag((cov[1:])/(scaler.scale_[1:]**2)) for cov in gmm.covariances_])).T

        centers.append(means)

        train_labels = gmm.predict(train)

        morans.append(moran_mixture(unscaled_loads, train_labels, N))

        accuracies = []

        # For each other day of data, predict using model that was fit.
        for test_time in xrange(P):

            if test_time == train_time:
                continue

            test = np.hstack((data[:, test_time, None], gps_loc))

            test = scaler.transform(test)

            test_labels = gmm.predict(test)

            correct_idx = [i for i in range(N) if train_labels[i] == test_labels[i]]
            accuracy = len(correct_idx)/float(N)

            accuracies.append(accuracy)

        # Getting average prediction accuracy over all test sets.
        average_accuracies.append(np.array(accuracies).mean())

    # Average error for the particular day and hour combination.
    time_avg_accuracy = round(100.0 - np.array(average_accuracies).mean() * 100, 2)

    # Average Moran autocorrelation for the particular day and hour combo.
    time_avg_moran = np.array(morans).mean()

    result = (day, hour, time_avg_accuracy, time_avg_moran, centers)
    
    return result




