import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Ellipse
import os
import numpy as np
import figure_functions
from map_overlay import MapOverlay


def init_animation(gps_loc, num_comps, fig_path):
    """Initializing figure for animation.
    
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param num_comps: Integer number of mixture components for the model.
    :param fig_path: Path to retrieve the background image from.
    
    :return fig: figure object containing the loaded background image.
    :return ax: ax object.
    :return scatter: Matplotlib path collections object for plotting the block
    location center points.
    :return scatter_centroid: Matplotlib path collections object for plotting 
    mixture centroids.
    :return patches: Matplotlib patches objects for the mixture ellipses.
    :return ellipses: Ellipse patch objects for the mixture ellipses.
    :return mp: MapOverlay object for plotting over the map.
    :return center: Tuple of the center of the image with respect to gps coords.
    :return pix_center: List of the x and y pixel positions of the center of the image.
    """

    up_left, bottom_right, img_size, fig_name = figure_functions.setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in range(len(gps_loc))])

    # Setting center of image.
    center = ((up_left[0] - bottom_right[0])/2., (up_left[1] - bottom_right[1])/2.)
    pix_center = mp.to_image_pixel_position(list(center))

    fig = plt.figure(figsize=(18, 16))
    ax = plt.axes(xlim=(min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100),
                  ylim=(min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))

    im = imread(os.path.join(fig_path, fig_name))
    ax.imshow(im)
    ax.invert_yaxis()

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # Adding in the midpoints of the block faces to the map as points.
    scatter = ax.scatter([], [], s=175, color='red', edgecolor='black')
    
    ax.xaxis.label.set_fontsize(25)
    ax.set_title('Gaussian Mixture Model on Occupancy Data and Location', fontsize=25)

    # Setting up the centroids for the mixture components.
    scatter_centroid = ax.scatter([], [], s=500, color='red', edgecolor='black')

    # Setting up the ellipses for the mixture components.
    patches = [Ellipse(xy=(0, 0), width=0, height=0, angle=0, edgecolor='black', 
               facecolor='none', lw='4') for comp in range(2*num_comps)]
    ellipses = [ax.add_patch(patches[comp]) for comp in range(2*num_comps)]

    return fig, ax, scatter, scatter_centroid, patches, ellipses, mp, center, pix_center


def animate(frame, times, ax, scatter, scatter_centroid, patches, ellipses, 
            mp, default_means, center, pix_center, loads, gps_loc, num_comps):
    """Animating the mixture model for a set of times in a day.
    
    :param frame: The iteration number of the animation frame.
    :param times: List of time indexes to create the animation for.
    :param ax: ax object for plotting and labels.
    :param scatter: Matplotlib path collections object for plotting the block
    location center points.
    :param scatter_centroid: Matplotlib path collections object for plotting 
    mixture centroids.
    :param patches: Matplotlib patches objects for the mixture ellipses.
    :param ellipses: Ellipse patch objects for the mixture ellipses.
    :param mp: MapOverlay object for plotting over the map.
    :param default_means: Numpy array of the default locations for the 
    centroids, to attempt to keep the colors for the clusters the same.
    :param center: Tuple of the center of the image with respect to gps coords.
    :param pix_center: List of the x and y pixel positions of the center of the image.
    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param num_comps: Integer number of mixture components for the model.
    
    :return: Updated ax, scatter, scatter_centroid, scatter_centroid, patches,
    and ellipses objects.
    """    
    
    time = times[frame]

    num_times = loads.shape[1]

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in range(len(gps_loc))])

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    pix_pos = pix_pos[mask]

    scatter.set_offsets(pix_pos)

    # Getting the data and normalizing the features.
    cluster_data = np.hstack((loads[mask][:, time, None], gps_loc[mask]))
    scaler = MinMaxScaler().fit(cluster_data)
    cluster_data = scaler.transform(cluster_data)

    # Fitting the mixture model.
    gmm = mixture.GaussianMixture(n_init=200, n_components=num_comps, 
                                  covariance_type='diag').fit(cluster_data)
    
    # Scaling the mean and covariances back to GPS coordinates.
    means = np.vstack(([(mean[1:] - scaler.min_[1:])/(scaler.scale_[1:]) for mean in gmm.means_]))
    covs = np.dstack(([np.diag((cov[1:])/(scaler.scale_[1:]**2)) for cov in gmm.covariances_])).T

    # Getting the labels by choosing the component which maximizes the posterior probability.
    labels = gmm.predict(cluster_data)    

    if num_comps == 4:

        colors = ['blue', 'deeppink', 'aqua', 'lawngreen']
        color_codes = {}

        for i in range(num_comps):
            # Finding the default centroid closest to the current centroid.
            dists = [(j, np.linalg.norm(means[i] - default_means[j])) for j in range(num_comps)]
            best_colors = sorted(dists, key=lambda item:item[1])

            # Finding the color that is unused that is closest to the current centroid.
            unused_colors = [color[0] for color in best_colors if color[0] 
                             not in color_codes.values()]

            # Choosing the closest centroid that is not already used.
            choice = unused_colors[0]
            color_codes[i] = choice
    else:
        colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, num_comps)]
        color_codes = {i:i for i in range(num_comps)}

    # Setting the cluster colors based off of the labels.
    scatter.set_color([colors[color_codes[labels[i]]] for i in range(len(labels))]) 
    scatter.set_edgecolor(['black' for i in range(len(labels))])
    
    ellipse_num = 0
    
    # Updating the ellipses for each of the components.
    for i in range(num_comps):
        lambda_, v = np.linalg.eig(covs[i])
        lambda_ = np.sqrt(lambda_)
        
        # Getting the ellipses for the 1st and 2nd standard deviations.
        for j in [1, 2]:
            
            # Converting mean in gps coords to pixel positions.
            xy = mp.to_image_pixel_position(list(means[i, :]))
            
            # Width and height of the ellipses in gps coords.
            width = lambda_[0]*j*2
            height = lambda_[1]*j*2 
            
            # Center of the ellipse in pixel positions.
            new_center = (center[0]+width, center[1]+height)
            new_center = mp.to_image_pixel_position(list(new_center))
            
            # New width and height of the ellipses in pixel positions.
            width = abs(new_center[0] - pix_center[0])
            height = abs(new_center[1] - pix_center[1])
            
            # Updating the ellipses for the animation.
            patches[ellipse_num].center = xy
            patches[ellipse_num].width = width
            patches[ellipse_num].height = height
            patches[ellipse_num].edgecolor = colors[color_codes[i]]
            
            ellipse_num += 1
            
    # Converting the centroids to pixel positions from gps coords.
    pix_means = np.array([mp.to_image_pixel_position(list(means[i, :])) for i in range(len(means))])
    
    # Updating the centroids for the animations.
    scatter_centroid.set_offsets(pix_means)

    hour = 8 + (time % (num_times/6))
    if hour < 12:
        hour = str(hour) + ':00 AM'
    elif hour == 12:
        hour = str(hour) + ':00 PM'
    else:
        hour = str(hour - 12) + ':00 PM'

    day = time/(num_times/6)

    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday'}
    ax.set_xlabel(days[day] + ' ' + hour)
    
    return ax, scatter, scatter_centroid, scatter_centroid, patches, ellipses