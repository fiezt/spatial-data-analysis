import numpy as np
from sklearn.cluster import KMeans


def get_time_scores(means):
    """Getting the k-means score for each time of day and day of week provided.

    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains 2-d arrays that contain each of the centroids
    for a GMM fit of a particular date for the weekday and hour.

    :return scores: List of tuples with the first item in each tuple being the 
    index of the means provided, and the second item the k-means score. The 
    tuples are sorted by the k-means score in descending order.
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


def get_distances(means, num_comps=4):
    """Find average distance between points in a cluster and the centroid.

    This function takes in a parameter, means, which is a list of list of numpy 
    arrays in which each numpy array in the inner list contains a set of 
    centroids from the Gaussian mixture modeling procedure for a particular date
    and the inner list represents a particular day of the week and hour of the
    day combo. For each of the times indexes indicated by the times parameter,
    the centroids that were fit at a particular day of the week and hour of the
    day combo are clustered. The average distance between the points in a 
    cluster and the centroid of a cluster for each cluster is returned for each time.

    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains 2-d arrays that contain each of the centroids
    for a GMM fit of a particular date for the weekday and hour.
    :param num_comps: Integer number of cluster components for the model.

    :return distances: Numpy array of 2 dimensions with the mean as the crow flies distance
    from all points in a cluster to the clusters centroid for each centroid in each row.
    :return centroids: Numpy array of 3 dimensions with the first dimension
    the time, the second the centroids for that time, and the last the GPS coords
    of the centroid for the time and centroid.
    """
    
    all_time_dist = []
    all_time_centroids = []
    
    for time in xrange(len(means)):
        data = np.vstack((means[time]))
        
        kmeans = KMeans(n_clusters=num_comps, n_init=50).fit(data)
        labels = kmeans.labels_.tolist()  
        centroids = kmeans.cluster_centers_
        
        current_time_dist = []
        current_time_centroids = []

        for i in xrange(num_comps):
            curr_points = data[np.where(np.array(labels) == i)[0].tolist()]

            dist = np.array([as_the_crow_flies_distance(curr_points[j], centroids[i]) 
                            for j in xrange(len(curr_points))]).mean()  

            current_time_dist.append(dist)
            current_time_centroids.append(centroids[i])
        
        all_time_dist.append(current_time_dist)
        all_time_centroids.append(current_time_centroids)

    distances = np.vstack((all_time_dist))
    centroids = np.array((all_time_centroids))
    
    return distances, centroids


def as_the_crow_flies_distance(point1, point2):
    """Calculate the as the crow flies distance between two GPS locations.

    :param point1: Tuple or list containing a latitude and longitude.
    :param point2: Tuple or list containing a latitude and longitude.

    :return distance: As the crow flies distance in meters between the two points.
    """

    lat1 = np.deg2rad(point1[0])
    lon1 = np.deg2rad(point1[1])
    lat2 = np.deg2rad(point2[0])
    lon2 = np.deg2rad(point2[1])

    radius = 6366.566

    dlat = lat2 - lat1 
    dlon = lon2 - lon1 

    a = (np.sin(dlat/2.) * np.sin(dlat/2.)) + (np.cos(lat1) * np.cos(lat2) 
                                               * np.sin(dlon/2.) * np.sin(dlon/2.))
            
    c = 2. * np.arctan(np.sqrt(a)/np.sqrt(1-a))
        
    distance = radius * c
    distance *= 1000
        
    return distance


def get_centroid_circle_paths(distances, centroids):
    """Find the path for a circle of radius of the distance around each centroid.

    :param distances: Numpy array of 2 dimensions with the mean as the crow flies distance
    from all points in a cluster to the clusters centroid for each centroid in each row.
    :param centroids: Numpy array of 3 dimensions with the first dimension
    the time, the second the centroids for that time, and the last the GPS coords
    of the centroid for the time and centroid.
    
    :return all_time_points: Numpy array of 4 dimensions with the first dimension 
    the number of times the centroids were found for, the second dimension the 
    number of centroids at the time, the third dimension the points for the 
    circle with the last dimension each point in GPS coords.
    """

    all_time_points = []

    for i in xrange(centroids.shape[0]):
        
        curr_time_points = []
        
        for j in xrange(centroids.shape[1]):
            lat = centroids[i, j, 0]
            lon = centroids[i, j, 1]

            r_earth = 6366.566
            km_dist = distances[i, j]/1000.

            delta = km_dist/r_earth

            points = []

            # Finding the destination point given distance and bearing from start point.
            for k in np.arange(0,360, 4):
                theta = np.deg2rad(k)

                lat_r = np.deg2rad(lat)
                lon_r = np.deg2rad(lon)

                new_lat_r = np.arcsin(np.sin(lat_r)*np.cos(delta) + np.cos(lat_r)*np.sin(delta)*np.cos(theta))
                new_lon_r = lon_r + np.arctan2(np.sin(theta)*np.sin(delta)*np.cos(lat_r), np.cos(delta) - np.sin(lat_r)*np.sin(new_lat_r))

                new_lat = np.rad2deg(new_lat_r)
                new_lon = np.rad2deg(new_lon_r)

                points.append([new_lat, new_lon])
            
            curr_time_points.append(points)

        all_time_points.append(curr_time_points)

    all_time_points = np.array(all_time_points)

    return all_time_points