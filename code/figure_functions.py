import numpy as np
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import matplotlib
import mixture_animation
from matplotlib import cm
from matplotlib import animation 
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread
import seaborn as sns
sns.reset_orig()
from scipy.spatial import voronoi_plot_2d
from scipy.spatial import Voronoi
from map_overlay import MapOverlay


def setup_image():
    """Specifying parameters of image to overlay plots on.

    :return upleft, bttmright, imgsize: gps coords of upper left corner of 
    image, gps coords of bottom right corner of image, and the pixel size 
    of the image respectively.
    """

    # GPS coordinates at corners of figure that will be used as the background.
    upleft = [47.6197793,-122.3592749]
    bttmright = [47.607274, -122.334786]

    # Figure size of background that will be used.
    imgsize = [1135,864]

    return upleft, bttmright, imgsize


def surface_plot(loads, gps_loc, time, fig_path, filename='surface.png'):
    """Create 3D surface plot of load data.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    x = gps_loc[:, 1, None]
    y = gps_loc[:, 0, None]
    z = loads[:, time, None]

    x = MinMaxScaler().fit_transform(x)
    y = MinMaxScaler().fit_transform(y)

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    X, Y = np.meshgrid(xi, yi)
    Z = ml.griddata(x[:, 0], y[:, 0], z[:, 0], xi, yi)

    light = LightSource(0, 0)
    illuminated_surface = light.shade(Z, cmap=cm.hot_r)
     
    fig = plt.figure()

    ax = plt.gca(projection='3d')

    ax.view_init(azim=89.9999999999, elev=70)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=illuminated_surface, 
                    linewidth=1, antialiased=True)

    ax.set_xticks([]) 
    ax.set_yticks([]) 

    ax.set_xlabel('Latitude' )
    ax.set_ylabel('Longitude')
    ax.set_title('Geospatial Load')

    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def interpolation(loads, gps_loc, time, N, fig_path, filename='interpolation.png'):
    """Create interpolation of the load data on the map.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param time: Integer column index to get the load data from.
    :param N: Integer number of samples (locations).
    :param fig_path: Path to save file plot to and read background image from.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Translating GPS coordinates to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    x = pixpos[:, 0, None]
    y = pixpos[:, 1, None]
    z = loads[:, time, None]

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    X, Y = np.meshgrid(xi, yi)
    Z = ml.griddata(x[:, 0], y[:, 0], z[:, 0], xi, yi)

    fig = plt.figure(figsize=(18,16))
    ax = plt.axes(xlim=(min(pixpos[:,0])-100, max(pixpos[:,0])+100), 
                  ylim=(min(pixpos[:,1])-100, max(pixpos[:,1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, 'belltown.png'))
    ax.imshow(im)
    ax.invert_yaxis()

    # Interpolating between the block locations to create continuous heat map.
    plt.pcolormesh(xi, yi, Z, vmin=z.min(), vmax=z.max(), cmap='jet')
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax, cmap='jet')
    cbar.ax.tick_params(labelsize=24) 

    ax.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def triangular_grid(loads, gps_loc, time, N, fig_path, filename='triangle.png'):
    """Create triangular grid of the load data on the map.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param time: Integer column index to get the load data from.
    :param N: Integer number of samples (locations).
    :param fig_path: Path to save file plot to and read background image from.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Translating GPS coordinates to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    x = pixpos[:, 0, None]
    y = pixpos[:, 1, None]
    z = loads[:, time, None]

    fig = plt.figure(figsize=(18,16))
    ax = plt.axes(xlim=(min(pixpos[:,0])-100, max(pixpos[:,0])+100), 
                  ylim=(min(pixpos[:,1])-100, max(pixpos[:,1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, 'belltown.png'))
    ax.imshow(im)

    ax.invert_yaxis()

    # Creating unstructured triangle graph.
    ax.tripcolor(x[:, 0], y[:, 0], z[:, 0], edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet') 
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize=24) 

    ax.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax


def contour_plot(loads, gps_loc, time, N, fig_path, title,
                 contours=10, filename='contours.png'):
    """Create contour plot of the load data on the map.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param time: Integer column index to get the load data from.
    :param N: Integer number of samples (locations).
    :param fig_path: Path to save file plot to and read background image from.
    :param title: Figure title for the plot.
    :param contours: Integer number of contour levels to use.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Translating GPS coordinates to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    x = pixpos[:, 0, None]
    y = pixpos[:, 1, None]
    z = loads[:, time, None]

    fig = plt.figure(figsize=(18,16))
    ax = plt.axes(xlim=(min(pixpos[:,0])-100, max(pixpos[:,0])+100), 
                  ylim=(min(pixpos[:,1])-100, max(pixpos[:,1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, 'belltown.png'))
    ax.imshow(im)
    ax.set_title(title, fontsize=40)

    ax.invert_yaxis()

    # Creating contour map with the last argument the level of contours.
    ax.tricontourf(x[:, 0], y[:, 0], z[:, 0], contours, vmin=z.min(), vmax=z.max(), cmap='jet') 
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax, label='Load')
    cbar.ax.tick_params(labelsize=30) 

    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=40)
    text.set_font_properties(font)

    ax.axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def voronoi(gps_loc, N, fig_path, filename='voronoi.png'):
    """Create voronoi diagram of the block locations on the map.

    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Translating GPS coordinates to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    fig = plt.figure(figsize=(18,16))
    ax = plt.axes(xlim=(min(pixpos[:,0]), max(pixpos[:,0])), 
                  ylim=(min(pixpos[:,1]), max(pixpos[:,1])))

    # Computing the Voronoi Tesselation.
    vor = Voronoi(pixpos)

    # Plotting the Voronoi Diagram.
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Colorizing the Voronoi Diagram.
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)

    plt.plot(pixpos[:,0], pixpos[:,1], 'ko')

    ax.invert_yaxis()
    plt.axis('off')

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    :param vor: Voronoi input diagram
    :param radius: Float distance to 'points at infinity'.

    :param regions: List of tuples indexes of vertices in each revised Voronoi regions.
    :param vertices: List of tuples coordinates for revised Voronoi vertices. 
    Same as coordinates of input vertices, with 'points at infinity' appended 
    to the end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point.
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions.
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region.
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region.
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region.
                continue

            # Compute the missing endpoint of an infinite ridge.
            t = vor.points[p2] - vor.points[p1]  # Tangent.
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal.

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def spatial_heterogeneity(loads, time, N, fig_path, filename='spatial_heterogeneity.png'):
    """Plot histogram of the loads at a given time to demonstrate spatial heterogeneity.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param time: Integer column index to get the load data from.
    :param N: Integer number of samples (locations).
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    bins = range(N)

    # Getting the load data for a specific time to plot for each block key.
    counts = loads[:, time]

    sns.set()
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    ax.set_xticks(bins)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=22)
    ax.yaxis.set_ticks_position('left')

    plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')
    plt.xlim([0, N])

    plt.title('Spatial Heterogeneity', fontsize=22)
    plt.xlabel('Blockface Key', fontsize=22)
    plt.ylabel('Load', fontsize=22)

    plt.tight_layout()

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax


def temporal_heterogeneity(loads, time, P, fig_path, filename='temporal_heterogeneity.png'):
    """Plot average load across belltown at each time and hour combination
    to demonstrate temporal heterogeneity. 

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param time: Integer column index to get the load data from.
    :param P: Integer number of total hours in the week with loads.
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    bins = range(P)

    # Getting the mean load over all locations at each time for the plot.
    counts = np.mean(loads, axis=0)

    sns.set()
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    ax.set_xticks(bins)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=22)
    ax.yaxis.set_ticks_position('left')

    plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')
    plt.xlim([-.3, P+.2])

    # 10 hour days.
    if P == 60:
        # Adding lines to separate days more clearly.
        ax.axvline(x=0, color='black')
        ax.axvline(x=10, color='black')
        ax.axvline(x=20, color='black')
        ax.axvline(x=30, color='black')
        ax.axvline(x=40, color='black')
        ax.axvline(x=50, color='black')
        ax.axvline(x=60, color='black')

        plt.title('Temporal Heterogeneity', fontsize=22)
        plt.ylabel('Load', fontsize=22)

        # Labels of the day of the week for each portion of the plot.
        ax.annotate('Monday',xy=(1.7,-.05),xytext=(1.7,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Tuesday',xy=(11.7,-.05),xytext=(11.7,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Wednesday',xy=(20.35,-.05),xytext=(20.35,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Thursday',xy=(31.6,-.05),xytext=(31.6,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Friday',xy=(42.8,-.05),xytext=(42.8,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Saturday',xy=(51.6,-.05),xytext=(51.6,-.05), annotation_clip=False, fontsize=16)

    # 12 hour days.
    elif P == 72:
        # Adding lines to separate days more clearly.
        ax.axvline(x=0, color='black')
        ax.axvline(x=12, color='black')
        ax.axvline(x=24, color='black')
        ax.axvline(x=36, color='black')
        ax.axvline(x=48, color='black')
        ax.axvline(x=60, color='black')
        ax.axvline(x=72, color='black')

        plt.title('Temporal Heterogeneity', fontsize=22)
        plt.ylabel('Load', fontsize=22)

        # Labels of the day of the week for each portion of the plot.
        ax.annotate('Monday', xy=(2,-.05), xytext=(2,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Tuesday', xy=(14,-.05), xytext=(14,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Wednesday', xy=(24.5,-.05), xytext=(24.5,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Thursday', xy=(38,-.05), xytext=(38,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Friday', xy=(51,-.05), xytext=(51,-.05), annotation_clip=False, fontsize=16)
        ax.annotate('Saturday', xy=(62,-.05), xytext=(62,-.05), annotation_clip=False, fontsize=16)

    plt.tight_layout()

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax


def temporal_day_plots(loads, P, fig_path, filename='temporal_day_plots.png'):
    """Create subplots containing bar plots of the load for each day of the week.


    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param P: Integer number of total hours in the week with loads.
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    sns.set()
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(60,10))

    day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                5: 'Friday', 6: 'Saturday'}

    # Getting the indexes that correspond to each of the days in list of list.
    days = [[i for i in range(j, j+(P/6))] for j in range(0, P, (P/6))]

    i = 1

    for day in days:
        bins = range(8, (P/6) + 8)

        # Getting the mean loads for the indexes corresponding to the hours of the given day.
        counts = loads.mean(axis=0)[day]

        ax1 = plt.subplot(1,6,i)
        
        ax1.set_xticks(np.arange(min(bins), max(bins)+1, 1))
        x_labels = [str(b) + 'AM' if b < 12 else str(b-12) + 'PM' 
                    if b != 12 else str(b) + 'PM' for b in bins]
        ax1.set_xticklabels(x_labels)
        
        plt.title(day_dict[i], fontsize=22)
        
        plt.setp(ax1.get_xticklabels(), fontsize=22, rotation=60)
        plt.setp(ax1.get_yticklabels(), fontsize=22)
        
        for tick in ax1.xaxis.get_majorticklabels():
            tick.set_horizontalalignment('left')
        
        plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')
        plt.xlim([min(bins), max(bins)+1])
        
        i += 1

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def temporal_hour_plots(loads, fig_path, filename='temporal_hour_plots.png'):
    """Creating subplots of the average load for each hour of the day by day of week.

    This function creates a subplot for each hour of the day and in each subplot
    is a bar plot with the average load at each day of the week for that hour.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    sns.set()
    sns.set_style("whitegrid")

    nrows = 2
    ncols = (loads.shape[1]/6)/2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 21))

    P = loads.shape[1]

    # Getting the indexes that correspond to each of the hours in list of list.
    hours = [[j + i*(P/6) for i in range(6)] for j in range((P/6))]

    i = 1
    for hour in hours:
        bins = range(6)

        # Getting the mean loads for the indexes corresponding to the days of the given hour.
        counts = loads.mean(axis=0)[hour]
        
        ax1 = plt.subplot(nrows, ncols, i)
        ax1.set_xticks(np.arange(min(bins), max(bins)+1, 1))
        ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
        plt.setp(ax1.get_xticklabels(), fontsize=22, rotation=60)
        plt.setp(ax1.get_yticklabels(), fontsize=22)
        
        if i + 7 < 12:
            title = str(7+i) + ':00 AM'    
        elif i + 7 == 12:
            title = str(7+i) + ':00 PM'
        elif i + 7 > 12:
            title = str(7+i-12) + ':00 PM'
            
        plt.title(title, fontsize=22)
        
        plt.bar(bins, counts, color='red', align='center')
        
        i += 1
        
    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    return fig, ax


def mixture_plot(loads, gps_loc, times, N, fig_path, 
                 default_means=np.array([[47.61337195, -122.34394369], [47.6144188, -122.34992362],
                                         [47.61707117, -122.34971354], [47.61076144, -122.34305349]]),
                 num_comps=4, shape=None, filename='mixture_plot.png', 
                 title='Gaussian Mixture Model on Average Load Distribution and Location'):
    """Plotting the mixture model results at a time or times of day.

    This function first creates a mixture model of the load data and the spatial
    characteristics, e.g. the GPS coordinates of the block locations. Thus the 
    mixture model has features for each block of the load at the time, and the
    latitude and longitude of the center of the block. The centroids and the 
    curves of the first two standard deviations of spatial components are drawn
    on the map. Each block is indicated by a scatter point and colored according
    to the probabilistic assignment the mixture model designates the block.
    
    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param times: List of column indexes to get the load data from.
    :param N: Integer number of samples (locations).
    :param num_comps: Integer number of mixture components for the model.
    :param fig_path: Path to save file plot to and read background image from.
    :param default_means: Numpy array, each row containing an array of the 
    lat and long to use as the default mean so colors will stay the same
    when making several plots.
    :param shape: Tuple of the row and col dimension of subplots.
    :param filename: Name to save the file as.
    :param title: Title for the figure.
    
    :return fig, ax: Matplotlib figure and axes objects.
    :return means: Numpy array where each row contains an array of the lat and
    long of a centroid of the GMM.
    """
    
    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Converting the gps locations to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    # Setting center of image.
    center = ((upleft[0] - bttmright[0])/2., (upleft[1] - bttmright[1])/2.)
    pix_center = mp.to_image_pixel_position(list(center))

    P = loads.shape[1]
    
    if isinstance(times, list):
        num_figs = len(times)
    else:
        num_figs = 1
    
    if shape == None:
        fig = plt.figure(figsize=(18*num_figs, 16))
        fs = 35
        fs_x = 35
    else:
        fig = plt.figure(figsize=(18*shape[1], 16*shape[0]))
        fs = 35
        fs_x = 35
    
    for fig_count in range(1, num_figs+1):
        
        if shape == None:
            ax = fig.add_subplot(1, num_figs, fig_count)
        else:
            ax = fig.add_subplot(shape[0], shape[1], fig_count)
            
        ax.set_xlim((min(pixpos[:,0])-100, max(pixpos[:,0])+100))
        ax.set_ylim((min(pixpos[:,1])-100, max(pixpos[:,1])+100))
        
        if isinstance(times, list):
            time = times[fig_count-1]
        else:
            time = times
        
        ax.invert_yaxis()

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        im = imread(os.path.join(fig_path, "belltown.png"))
        ax.imshow(im)

        # Adding in the midpoints of the block faces to the map as points.
        scatter = ax.scatter(pixpos[:, 0], pixpos[:, 1], s=175, color='red', edgecolor='black')

        ax.xaxis.label.set_fontsize(fs_x)

        scatter_centroid = ax.scatter([], [], s=500, color='red', edgecolor='black')

        patches = [Ellipse(xy=(0, 0), width=0, height=0, angle=0, edgecolor='black', 
                   facecolor='none', lw='4') for comp in range(2*num_comps)]

        ellipses = [ax.add_patch(patches[comp]) for comp in range(2*num_comps)]

        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}

        if num_comps == 4:
            colors = ['blue', 'deeppink', 'aqua', 'lawngreen']
        else:
            colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0,1,num_comps)]

        cluster_data = np.hstack((loads[:, time, None], gps_loc))

        scaler = MinMaxScaler().fit(cluster_data)
        cluster_data = scaler.transform(cluster_data)

        gmm = mixture.GaussianMixture(n_init=200, n_components=num_comps, 
                                      covariance_type='diag').fit(cluster_data)

        # Scaling the mean and covariances back to gps coordinates.
        means = np.vstack(([(mean[1:] - scaler.min_[1:])/(scaler.scale_[1:]) for mean in gmm.means_]))
        covs = np.dstack(([np.diag((cov[1:])/(scaler.scale_[1:]**2)) for cov in gmm.covariances_])).T

        labels = gmm.predict(cluster_data)    

        if num_comps == 4:
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
            color_codes = {i:i for i in range(num_comps)}

        # Setting the cluster colors to keep the colors the same each iteration.
        scatter.set_color([colors[color_codes[labels[i]]] for i in range(len(labels))]) 
        scatter.set_edgecolor(['black' for i in range(len(labels))])

        num = 0

        # Updating the ellipses for each of the components.
        for i in range(num_comps):
            lambda_, v = np.linalg.eig(covs[i])
            lambda_ = np.sqrt(lambda_)

            # Getting the ellipses for the 1st and 2nd standard deviations.
            for j in [1, 2]:

                # Converting mean in gps coords to pixel positions.
                xy = mp.to_image_pixel_position(list(means[i,:]))

                # Width and height of the ellipses in gps coords.
                width = lambda_[0]*j*2
                height = lambda_[1]*j*2 

                # Center of the ellipse in pixel positions.
                new_center = (center[0]+width, center[1]+height)
                new_center = mp.to_image_pixel_position(list(new_center))

                # New width and height of the ellipses in pixel positions.
                width = abs(new_center[0] - pix_center[0])
                height = abs(new_center[1] - pix_center[1])

                # Updating the ellipses.
                patches[num].center = xy
                patches[num].width = width
                patches[num].height = height
                patches[num].edgecolor = colors[color_codes[i]]

                num += 1

        # Converting the centroids to pixel positions from gps coords.
        pix_means = np.array([mp.to_image_pixel_position(list(means[i,:])) for i in range(len(means))])

        # Updating the centroids for the animations.
        scatter_centroid.set_offsets(pix_means)

        hour = 8 + (time % (P/6))
        if hour < 12:
            hour = str(hour) + ':00 AM'
        else:
            hour = str(hour - 12) + ':00 PM'

        day = time/(P/6)

        ax.set_xlabel(days[day] + ' ' + hour)

    fig.tight_layout()
    fig.suptitle(title, fontsize=fs)

    if shape[0] > 1 and shape[1] == 1:
        plt.subplots_adjust(top=0.975)
    else:
        plt.subplots_adjust(top=0.98)

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax

 
def centroid_plots(means, gps_loc, N, times, fig_path, num_comps=4, 
                   shape=None, filename='centroid_plot.png', 
                   title='Centroids'):
    
    """Plotting the centroids from different dates at the same weekday and hour.

    This function takes in a parameter, means, which is a list of list of numpy 
    arrays in which each numpy array in the inner list contains a set of 
    centroids from the Gaussian mixture modeling procedure for a particular date
    and the inner list represents a particular day of the week and hour of the
    day combo. For each of the times indexes indicated by the times parameter,
    the centroids that were fit at a particular day of the week and hour of the
    day combo are clustered. These clusters are then plotted as scatter points
    at the locations they are. Each cluster is colored differently to indicate
    the cluster difference.
    
    :param means: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains a 2-d array that contains each of the centroids
    for a GMM fit of a particular date for the weekday and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param times: List of indexes to get the load data from.
    :param fig_path: Path to save file plot to and read background image from.
    :param num_comps: Integer number of mixture components for the model.
    :param shape: Tuple of the row and col dimension of subplots.
    :param filename: Name to save the file as.
    :param title: Title for the figure.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    upleft, bttmright, imgsize = setup_image()

    mp = MapOverlay(upleft, bttmright, imgsize)

    # Converting the gps locations to pixel positions.
    pixpos = np.array([mp.to_image_pixel_position(list(gps_loc[i,:])) for i in range(N)])

    # Setting center of image.
    center = ((upleft[0] - bttmright[0])/2., (upleft[1] - bttmright[1])/2.)
    pix_center = mp.to_image_pixel_position(list(center))

    P = len(means)
    
    if isinstance(times, list):
        num_figs = len(times)
    else:
        num_figs = 1
    
    if shape == None:
        fig = plt.figure(figsize=(18*num_figs, 16))
        fs = 35
        fs_x = 35
    else:
        fig = plt.figure(figsize=(18*shape[1], 16*shape[0]))
        fs = 35
        fs_x = 35
    
    for fig_count in range(1, num_figs+1):
        
        if shape == None:
            ax = fig.add_subplot(1, num_figs, fig_count)
        else:
            ax = fig.add_subplot(shape[0], shape[1], fig_count)
            
        ax.set_xlim((min(pixpos[:,0])-100, max(pixpos[:,0])+100))
        ax.set_ylim((min(pixpos[:,1])-100, max(pixpos[:,1])+100))
        
        if isinstance(times, list):
            time = times[fig_count-1]
        else:
            time = times
        
        ax.invert_yaxis()

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        im = imread(os.path.join(fig_path, "belltown.png"))
        ax.imshow(im)

        # Clustering the centroids.
        data = np.vstack((means[time]))
        kmeans = KMeans(n_clusters=num_comps, n_init=50).fit(data)
        labels = kmeans.labels_.tolist()

        # Converting the centroid locations to pixel positions.
        data_pixpos = np.array([mp.to_image_pixel_position(list(data[i,:])) for i in range(len(data))])

        # Adding in the centroids to the map as points.
        scatter = ax.scatter(data_pixpos[:, 0], data_pixpos[:, 1], s=500, color='red', edgecolor='black')

        ax.xaxis.label.set_fontsize(fs_x)

        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}

        if num_comps == 4:
            colors = ['blue', 'deeppink', 'aqua', 'lawngreen']
        else:
            colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0,1,num_comps)]

        # Setting the centroid colors to be the same within a cluster.
        scatter.set_color([colors[labels[i]] for i in range(len(labels))]) 
        scatter.set_edgecolor(['black' for i in range(len(labels))])

        hour = time % (P/6)
        day = time/(P/6)

        ax.set_xlabel(days[day] + ' ' + str(8+hour) + ':00')

    fig.tight_layout()
    fig.suptitle(title, fontsize=fs)

    if shape[0] > 1 and shape[1] == 1:
        plt.subplots_adjust(top=0.975)
    else:
        plt.subplots_adjust(top=0.98)

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax


def model_selection(loads, gps_loc, P, fig_path):
    """Using various criterion to evaluate the number of components to use in GMM.

    This function creates a Gaussian mixture model with a varying range of 
    components for each of the times in the parameter loads. The average
    akaike information criterion, bayesian information criterion, and likelihood
    are then averaged over all times and plotted in order to find the correct
    number of components to use in the model and these graphs are plotted.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param P: Integer number of total hours in the week with loads.
    :param fig_path: Path to save file plot to.
    """

    model_selection = {"likelihood": [], "bic": [], "aic": []} 
    min_comps = 1
    max_comps = 21

    # Fitting mixture models at each possible time the loads are available.
    for time in range(P):    
        likelihoods = []
        bics = []
        aics = []

        # Varying the number of components and getting AIC, BIC, and likelihood.
        for num_comps in range(min_comps, max_comps):
            cluster_data = np.hstack((loads[:, time, None], gps_loc))

            scaler = MinMaxScaler().fit(cluster_data)
            cluster_data = scaler.transform(cluster_data)

            gmm = mixture.GaussianMixture(n_init=10, n_components=num_comps, 
                                          covariance_type='diag').fit(cluster_data)

            likelihoods.append(gmm.lower_bound_)
            bics.append(gmm.bic(cluster_data))
            aics.append(gmm.aic(cluster_data))

        model_selection['likelihood'].append(likelihoods)
        model_selection['bic'].append(bics)
        model_selection['aic'].append(aics)  

    sns.set()
    sns.set_style("whitegrid")
    
    # Likelihood model selection plot.
    plt.figure()

    # Getting the mean likelihood over all times for varying number of components.
    mean_likelihood = np.mean(np.vstack((model_selection['likelihood'])), axis=0)

    plt.plot(range(min_comps, max_comps), mean_likelihood, 'o-', color='red')
    plt.xlim([-0.1, 20.1])

    # Plotting vertical lines to make the separation of number of components clear.
    plt.axvline(x=0, color='black')
    plt.axvline(x=5, color='black')
    plt.axvline(x=10, color='black')
    plt.axvline(x=15, color='black')
    plt.axvline(x=20, color='black')
    plt.xlabel('Number of Components')
    plt.ylabel('Likelihood')
    plt.title('Likelihood Model Selection')
    plt.savefig(os.path.join(fig_path, 'likelihood_model.png'))

    # BIC model selection plot.
    plt.figure()

    # Getting the mean BIC over all times for varying number of components.
    mean_bic = np.mean(np.vstack((model_selection['bic'])), axis=0)

    plt.plot(range(min_comps, max_comps), mean_bic, 'o-', color='red')
    plt.xlim([-0.1, 20.1])

    # Plotting vertical lines to make the separation of number of components clear.
    plt.axvline(x=0, color='black')
    plt.axvline(x=5, color='black')
    plt.axvline(x=10, color='black')
    plt.axvline(x=15, color='black')
    plt.axvline(x=20, color='black')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.title('BIC Model Selection')
    plt.savefig(os.path.join(fig_path, 'bic_model.png'))

    # AIC model selection plot.
    plt.figure()

    # Getting the mean AIC over all times for varying number of components.
    mean_aic = np.mean(np.vstack((model_selection['aic'])), axis=0)

    # Plotting vertical lines to make the separation of number of components clear.
    plt.plot(range(min_comps, max_comps), mean_aic, 'o-', color='red')
    plt.xlim([-0.1, 20.1])

    plt.axvline(x=0, color='black')
    plt.axvline(x=5, color='black')
    plt.axvline(x=10, color='black')
    plt.axvline(x=15, color='black')
    plt.axvline(x=20, color='black')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    plt.title('AIC Model Selection')
    plt.savefig(os.path.join(fig_path, 'aic_model.png'))  

    sns.reset_orig()


def create_animation(loads, gps_loc, N, P, fig_path, animation_path, num_comps=4):
    """Create an animation of the GMM model using figures of each hour of load data.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    :param P: Integer number of total hours in the week with loads.
    :param fig_path: Path to save file plot to.
    :param animation_path: Path to save the animation video to.
    :param num_comps: Integer number of mixture components for the model.
    """

    params = mixture_animation.init_animation(gps_loc, num_comps, N, fig_path)
    fig, ax, scatter, scatter_centroid, patches, ellipses, mp, center, pix_center = params

    times = range(P)

    # Default means in attempt to keep the colors the same at each time index.
    default_means = np.array([[47.61348888, -122.34343007],[47.61179196, -122.34500616],
                              [47.61597088, -122.35054099],[47.61706817, -122.34617185]])

    ani = animation.FuncAnimation(fig=fig, func=mixture_animation.animate, frames=P, 
                                  fargs=(times, ax, scatter, scatter_centroid, patches, 
                                         ellipses, mp, default_means, center, 
                                         pix_center, loads, gps_loc, num_comps, ), 
                                  interval=200)


    FFwriter = animation.FFMpegWriter(fps=1)
    ani.save(os.path.join(animation_path, 'mixture.mp4'), writer=FFwriter)