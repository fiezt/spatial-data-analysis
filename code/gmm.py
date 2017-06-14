import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
import itertools
import multiprocessing
import functools


def locational_demand_analysis(park_data, gps_loc, N):
    """

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
    """

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

    for train_time in xrange(P):

        train = np.hstack((data[:, train_time, None], gps_loc))

        # Saving the scaling so it can be applied to the test set as well.
        scaler = MinMaxScaler().fit(train)
        train = scaler.transform(train)

        gmm = mixture.GaussianMixture(n_init=200, n_components=4, 
                                      covariance_type='diag').fit(train)

        # Scaling the mean and covariances back to gps coordinates.
        means = np.vstack(([(mean[1:] - scaler.min_[1:])/(scaler.scale_[1:]) for mean in gmm.means_]))
        covs = np.dstack(([np.diag((cov[1:])/(scaler.scale_[1:]**2)) for cov in gmm.covariances_])).T

        centers.append(means)

        train_labels = gmm.predict(train)

        accuracies = []

        # For each other day of data, predict using model determine accuracy of model.
        for test_time in xrange(P):

            if test_time == train_time:
                continue

            test = np.hstack((data[:, test_time, None], gps_loc))

            test = scaler.transform(test)

            test_labels = gmm.predict(test)

            correct_idx = [i for i in range(N) if train_labels[i] == test_labels[i]]
            accuracy = len(correct_idx)/float(N)

            accuracies.append(accuracy)

        average_accuracies.append(np.array(accuracies).mean())

    # Average accuracy for the particular day and hour combination.
    time_avg_accuracy = np.array(average_accuracies).mean()
    
    return (day, hour, time_avg_accuracy, centers)
