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

    P = len(times)

    for time in xrange(P):

        data = np.vstack((means[time]))

        kmeans = KMeans(n_clusters=4).fit(data)
        labels = kmeans.labels_.tolist()

        scores.append((time, kmeans.score(data)))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
    times = [score[0] for score in scores]

    return scores, times


def same_size_kmeans():
    pass











