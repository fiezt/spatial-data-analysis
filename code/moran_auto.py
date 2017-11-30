import numpy as np
import scipy.stats as st


def get_neighbor_weights(gps_loc, N, k):
    """Get the weight matrix for Moran I by using k nearest neighbor connections.

    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param N: Integer number of samples (locations).
    :param k: Integer number of neighbors to use for the weighting matrix.

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))

    for i in xrange(N):
        # Finding the k-nearest neighbors.
        neighbors = np.vstack(sorted([(j, np.linalg.norm(gps_loc[i] - gps_loc[j])) for j in xrange(N)],
                                     key=lambda x: x[1])[1:k+1])[:, 0].astype('int')
        weights[i, neighbors] = 1

    return weights


def get_dist_weights(gps_loc, N):
    """Get the weight matrix for Moran I by using distance based metric.

    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param N: Integer number of samples (locations).

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))

    for i in xrange(N):
        # Computing distance to each sample normalized between 0 and 1.
        dist = np.linalg.norm(gps_loc[i] - gps_loc, axis=1)**2
        dist /= dist.max()
        dist = -1*(1 - dist)
        weights[i] = dist

    di = np.diag_indices(N)
    weights[di] = 0

    return weights


def get_area_weights(train_active_index, N, area_map, subarea_to_key):
    """Get the weight matrix for Moran I by using the paid area connections.

    :param train_active_index: Numpy array of the active block-face keys.
    :param N: Integer number of samples (locations).
    :param area_map: Dictionary from key to subarea.
    :param subarea_to_key: Dictionary from subarea to key.

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))
    for i in xrange(N):
        # Finding blocks in the same subarea.
        weights[i, subarea_to_key[area_map[train_active_index[i]]]] = 1
        
    di = np.diag_indices(N)
    weights[di] = 0

    return weights


def get_dist_area_weights(train_active_index, gps_loc, N, area_map, subarea_to_key):
    """Get the weight matrix for Moran I by using the paid area connections by distance.

    :param train_active_index: Numpy array of the active block-face keys.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param N: Integer number of samples (locations).
    :param area_map: Dictionary from key to subarea.
    :param subarea_to_key: Dictionary from subarea to key.

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))
    all_idx = range(N)

    for i in xrange(N):
        # Finding blocks in the same and different areas.
        same_area = subarea_to_key[area_map[train_active_index[i]]]
        diff_area = list(set(all_idx) - set(same_area))

        # Computing distance to each block in same subarea normalized between 0 and 1.
        dist = np.linalg.norm(gps_loc[i] - gps_loc, axis=1)**2
        dist /= dist[same_area].max()
        dist[diff_area] = 1
        dist = -1*(1 - dist)

        weights[i] = dist
        
    di = np.diag_indices(N)
    weights[di] = 0

    return weights


def get_mixture_weights(train_labels, N):
    """Calculate the Moran I weight matrix using the mixture connections.

    :param train_labels: Numpy array containing label for each data point.
    :param N: Integer number of samples (locations).

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))

    for i in xrange(N):
        label = train_labels[i]
        matching = np.where(train_labels == label)[0].tolist()
        weights[i, matching] = 1

    di = np.diag_indices(N)
    weights[di] = 0

    return weights


def get_dist_mixture_weights(train_labels, gps_loc, N):
    """Calculate the Moran I weight matrix using the mixture connections.
    
    :param train_labels: Numpy array containing label for each data point.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param N: Integer number of samples (locations).

    :return weights: Numpy array of the weight matrix.
    """

    weights = np.zeros((N, N))

    for i in xrange(N):
        label = train_labels[i]

        # Finding blocks with the same and different label.
        not_matching = np.where(train_labels != label)[0].tolist()
        matching = np.where(train_labels == label)[0].tolist()

        # Computing distance to each block in same component normalized between 0 and 1.
        dist = np.linalg.norm(gps_loc[i] - gps_loc, axis=1)**2
        dist /= dist[matching].max()
        dist[not_matching] = 1
        dist = -1*(1 - dist)
        weights[i] = dist

    di = np.diag_indices(N)
    weights[di] = 0

    return weights


def moran_I(x, N, weights):
    """Calculating the Moran I.

    :param x: Numpy array of the loads.
    :param N: Integer number of samples.
    :param weights: Numpy array of the weight matrix.

    :return I: Float of Moran I.
    """

    W = weights.sum()
    z = x - x.mean()

    top = sum(weights[i,j]*z[i]*z[j] for i in xrange(N) for j in xrange(N)) 
    bottom = np.dot(z.T, z)

    I = (N/W) * top/bottom

    return I


def moran_expectation(N):
    """Calculate the expected value of the Moran I.

    :param N: Integer number of samples (locations).

    :return expectation: Float of expectation of Moran I.
    """

    expectation = -1./(N - 1.)

    return expectation


def moran_variance(x, w, N):
    """Calculating the variance of the Moran I.
    
    :param x: Numpy array of the loads.
    :param w: Numpy array of the weight matrix.
    :param N: Integer number of samples (locations).

    :return var: Float of variance of Moran I.
    """

    W = w.sum()

    z = x - x.mean()

    s_1 = .5 * sum((w[i,j] + w[j,i])**2 for i in xrange(N) for j in xrange(N))

    s_2 = sum((sum(w[i,j] for j in xrange(N)) + sum(w[j,i] for j in xrange(N)))**2 for i in xrange(N))

    s_3 = (N**(-1) * (z**4).sum())/(N**(-1) * (z**2).sum())**2

    s_4 = (N**2 - 3.*N + 3.)*s_1 - N*s_2 + 3.*W**2

    s_5 = (N**2 - N)*s_1 - 2.*N*s_2 + 6.*W**2

    var = ((N*s_4 - s_3*s_5)/((N - 1.) * (N - 2.) * (N - 3.) * W**2)) - moran_expectation(N)**2

    return var


def z_score(I, expectation, variance):
    """Calculate the z-score for the Moran I.
    
    :param I: Float of Moran I.
    :param expectation: Float of expectation of Moran I.
    :param variance: Float of variance of Moran I.

    :return z: Float of z-score for the Moran I.
    """

    z = (I - expectation)/np.sqrt(variance)

    return z


def p_value(z):
    """Calculating the one and two sided p-value for the Moran z-score.
    
    :param z: Float of z-score for the Moran I.

    :return p_one_sided, p_two_sided: Float of one sided and two sided p values.
    """

    p_one_sided = st.norm.sf(abs(z)) 
    p_two_sided = st.norm.sf(abs(z))*2 

    return p_one_sided, p_two_sided