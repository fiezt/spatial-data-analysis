import numpy as np
from sklearn.cluster import KMeans


def get_time_scores(means):
    """Getting the Kmeans score for each time of day and day of week provided.

    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains 2-d arrays that contain each of the centroids
    for a GMM fit of a particular date for the weekday and hour.

    :return scores: List of tuples with the first item in each tuple being the 
    index of the means provided, and the second item the kmeans score. The 
    tuples are sorted by the kmeans score in descending order.
    :return times: Sorted list of times, sorted by the kmeans score for the time.
    """

    scores = []

    P = len(means)

    for time in xrange(P):

        data = np.vstack((means[time]))

        kmeans = KMeans(n_clusters=4).fit(data)
        labels = kmeans.labels_.tolist()

        scores.append((time, kmeans.score(data)))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
    times = [score[0] for score in scores]

    return scores, times


def get_distances(means):
    """
    
    """
    
    num_comps = 4
    all_time_dist = []
    
    for time in xrange(len(means)):
        data = np.vstack((means[time]))
        
        kmeans = KMeans(n_clusters=num_comps, n_init=50).fit(data)
        labels = kmeans.labels_.tolist()  
        centroids = kmeans.cluster_centers_
        
        current_time_dist = []
        for i in xrange(num_comps):
            curr_points = data[np.where(np.array(labels) == i)[0].tolist()]

            dist = np.array([measure(curr_points[j], centroids[i]) for j in xrange(len(curr_points))]).mean()  

            current_time_dist.append(dist)
        
        all_time_dist.append(current_time_dist)
    
    distances = np.vstack((all_time_dist))
    
    return distances


def as_the_crow_flies_distance(point1, point2):
    """

    """

    lat1 = point1[0]
    lon1 = point1[1]
    lat2 = point2[0]
    lon2 = point2[1]

    radius = 6378.137

    dlat = (lat2 * np.pi)/180. - (lat1 * np.pi)/180.
    dlon = (lon2 * np.pi)/180. - (lon1 * np.pi)/180.

        
    a = (np.sin(dlat/2.) * np.sin(dlat/2.)) \
        + (np.cos((lat1 * np.pi)/180.) * np.cos((lat2 * np.pi)/180.) * np.sin(dlon/2.) * np.sin(dlon/2.))
            
    c = 2. * np.arctan(np.sqrt(a)/np.sqrt(1-a))
        
    d = radius * c
        
    return d * 1000