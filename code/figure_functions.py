import numpy as np
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import LightSource
import os
from sklearn.preprocessing import MinMaxScaler
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
    :param time: Column index to get the load data from.
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
    :param time: Column index to get the load data from.
    :param N: Number of samples (locations).
    :param fig_path: Path to save file plot to.
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
    plt.pcolormesh(xi, yi, Z)
    plt.scatter(x, y, c=z, s=200)

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax)
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
    :param time: Column index to get the load data from.
    :param N: Number of samples (locations).
    :param fig_path: Path to save file plot to.
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
    ax.tripcolor(x[:, 0], y[:, 0], z[:, 0]) 
    plt.scatter(x,y,c=z,s=200)

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
    :param time: Column index to get the load data from.
    :param N: Number of samples (locations).
    :param fig_path: Path to save file plot to.
    :param title: Figure title for the plot.
    :param contours: Number of contour levels to use.
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
    ax.tricontourf(x[:, 0], y[:, 0], z[:, 0], contours) 
    plt.scatter(x,y,c=z,s=200)

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
    :param N: Number of samples (locations).
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

    # Computing the Voronoi Tesselation
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

    :param regions : List of tuples indexes of vertices in each revised Voronoi regions.
    :param vertices : List of tuples coordinates for revised Voronoi vertices. 
    Same as coordinates of input vertices, with 'points at infinity' appended to the
    end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def spatial_heterogeneity(loads, time, N, fig_path, filename='spatial_heterogeneity.png'):
    """Plot histogram of the loads at a given time to demonstrate spatial heterogeneity.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param time: Column index to get the load data from.
    :param N: Number of samples (locations).
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    bins = range(N)
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

    plt.bar(bins, counts, 1, color='red', align='center')

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
    :param time: Column index to get the load data from.
    :param P: Number of times.
    :param fig_path: Path to save file plot to.
    :param filename: Name of the file to save.

    :return fig, ax: Matplotlib figure and axes objects.
    """

    bins = range(P)
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

    plt.bar(bins, counts, color='red')

    ax.axvline(x=0.2, color='black')
    ax.axvline(x=10, color='black')
    ax.axvline(x=20, color='black')
    ax.axvline(x=30, color='black')
    ax.axvline(x=40, color='black')
    ax.axvline(x=50, color='black')
    ax.axvline(x=59.7, color='black')

    plt.title('Temporal Heterogeneity', fontsize=22)
    plt.ylabel('Load', fontsize=22)

    ax.annotate('Monday',xy=(2,-.05),xytext=(2,-.05), annotation_clip=False, fontsize=16)
    ax.annotate('Tuesday',xy=(12,-.05),xytext=(12,-.05), annotation_clip=False, fontsize=16)
    ax.annotate('Wednesday',xy=(20.65,-.05),xytext=(20.65,-.05), annotation_clip=False, fontsize=16)
    ax.annotate('Thursday',xy=(31.90,-.05),xytext=(31.90,-.05), annotation_clip=False, fontsize=16)
    ax.annotate('Friday',xy=(43.1,-.05),xytext=(43.1,-.05), annotation_clip=False, fontsize=16)
    ax.annotate('Saturday',xy=(51.9,-.05),xytext=(51.9,-.05), annotation_clip=False, fontsize=16)

    plt.tight_layout()

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    return fig, ax


def plot_all(loads, gps_loc, time, N, P, fig_path):
    """Produce all plots including surface plot, interpolation, triangular 
    grid, contour plot, voronoi, spatial_heterogeneity, temporal_heterogeneity.

    :param loads: Numpy array with each row containing the load for a day of 
    week and time, where each column is a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param time: Column index to get the load data from.
    :param N: Number of samples (locations).
    :param P: Number of times.
    :param fig_path: Path to save file plot to.

    :return: Nothing is returned but the plots are plotted and saved.
    """

    fig, ax = surface_plot(loads=loads, gps_loc=gps_loc, time=4, 
                           fig_path=fig_path)
    plt.show()

    fig, ax = interpolation(loads=loads, gps_loc=gps_loc, time=time,
                            N=N, fig_path=fig_path)
    plt.show()

    fig, ax = triangular_grid(loads=loads, gps_loc=gps_loc, time=time,
                              N=N, fig_path=fig_path)
    plt.show()

    fig, ax = contour_plot(loads=loads, gps_loc=gps_loc, time=time,
                           N=N, fig_path=fig_path)
    plt.show()

    fig, ax = voronoi(gps_loc=gps_loc, N=N, fig_path=fig_path)
    plt.show()

    fig, ax = spatial_heterogeneity(loads=loads, time=time, 
                                    N=N, fig_path=fig_path)
    plt.show()

    fig, ax = temporal_heterogeneity(loads=loads, time=time, 
                                     P=P, fig_path=fig_path)
    plt.show()

