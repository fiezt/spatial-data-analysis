import numpy as np
import pickle
import os
import scipy.stats as st


def get_mixture_weights(train_labels, N):
    """Calculate the Moran I weight matrix using the mixture connections.
    
    :param train_labels: Numpy array like containing class label for each data 
    point in x.
    :param N: Integer number of samples.

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


def get_adjacent_weights(block_keys, N, path=os.getcwd() + '/../data'):
    """Get the weight matrix for Moran I by using adjacent connections.
    
    :param block_keys: List of block key identifiers.
    :param N: Integer number of spatial units.
    :param path: File path to the data about neighboring blocks.

    :return weights: Numpy array weight matrix.
    """

    weights = np.genfromtxt(os.path.join(path, "belltown-adjacency1.txt"), delimiter=" ")

    return weights


def moran_adjacent(x, block_keys, N, path=os.getcwd() + '/../data', weight_func=get_adjacent_weights):
    """Find the Moran I using the adjacent weight matrix.

    :param x: Numpy array of the variable of interest.
    :param block_keys: List of ordered block key identifiers.
    :param N: Integer number of samples.
    :param path: The path to the location of the adjacency information.
    :param weight_func: Function to get the weights for Moran's I.

    :return I: Moran I.
    """

    weights = weight_func(block_keys, N, path)

    W = weights.sum()
    z = x - x.mean()

    top = sum(weights[i,j]*z[i]*z[j] for i in xrange(N) for j in xrange(N)) 
    bottom = np.dot(z.T, z)

    I = (N/W) * top/bottom

    return I


def moran_mixture(x, train_labels, N, weight_func=get_mixture_weights):
    """Calculating the Moran I autocorrelation for the mixture model weights.

    The weight matrix is used by giving weight 1 at the column index if the
    column has the same label as the row index does. All other weights are 
    set to 0, and the diagonal is set to 0. The variable of interest is then x.

    :param x: Numpy array of the variable of interest.
    :param train_labels: Numpy array like containing class label for each data 
    point in x.
    :param N: Integer number of samples.
    :param weight_func: Function to get the weights for Moran's I.

    :return I: float of Moran I autocorrelation.
    """

    weights = weight_func(train_labels, N)

    W = weights.sum()
    z = x - x.mean()

    top = sum(weights[i,j]*z[i]*z[j] for i in xrange(N) for j in xrange(N)) 
    bottom = np.dot(z.T, z)

    try:
        I = (N/W) * top/bottom
    except:
        I = None

    return I


def moran_expectation(N):
    """Calculate the expected value of Moran's I.

    :param N: Integer of the number of spatial units

    :return expectation: Float of the expectation of Moran's I.
    """

    expectation = -1./(N - 1.)

    return expectation


def moran_variance(x, w, N):
    """Calculating the variance of the Moran I.
    
    :param x: Numpy array of the variable of interest.
    :param w: Numpy array of the weight matrix.
    :param N: Integer number of samples.

    :return var: Variance of the Moran I.
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
    
    :param I: Moran I.
    :param expectation: Expectation of Moran I.
    :param variance: Variance of Moran I.

    :return z: z-score for the Moran I.
    """

    z = (I - expectation)/np.sqrt(variance)

    return z


def p_value(z):
    """Calculating the one and two sided p-value for the Moran z-score.
    
    :param z: Float z score of the Moran I.

    :return p_one_sided, p_two_sided: one sided and two sided p values.
    """

    p_one_sided = st.norm.sf(abs(z)) 

    p_two_sided = st.norm.sf(abs(z))*2 

    return p_one_sided, p_two_sided