import numpy as np
import pandas as pd
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import matplotlib
import mixture_animation
from matplotlib import cm
from matplotlib import animation 
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import gmplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.misc import imread
from scipy.spatial import Voronoi
from map_overlay import MapOverlay
import seaborn as sns
sns.reset_orig()

"""
# GPS coordinates of upper left corner of the image.
background_up_left = [47.6197793, -122.3592749]

# GPS coordinates of bottom right corner of the image.
background_bottom_right = [47.607274, -122.334786]

# Pixel dimensions of the image.
background_img_size = [1135, 864]

# Background figure to overlay on. Must be in figure directory.
background_fig_name = 'belltown.png'
"""

"""
background_up_left = [47.619062, -122.358414]
background_bottom_right = [47.608167, -122.328308]
background_img_size = [1403, 745]
background_fig_name = 'seattle_medium.png'
"""


background_up_left = [47.618721, -122.357463]
background_bottom_right = [47.601608, -122.312430]
background_img_size = [1769, 996]
background_fig_name = 'seattle_big.png'



def setup_image():
    """Specifying parameters of image to overlay plots on.

    :return up_left: List of GPS coordinates of upper left corner of the image.
    :return bottom_right: List of GPS coordinates of bottom right corner of the image.
    :return img_size: List of pixel dimensions of the image.
    :return fig_name: Name of figure to use as the background image.
    """

    up_left = background_up_left
    bottom_right = background_bottom_right

    img_size = background_img_size

    fig_name = background_fig_name

    return up_left, bottom_right, img_size, fig_name


def plot_neighborhoods(element_keys, data_path, fig_path, filename='neighborhood_map.html'):
    """Plotting on google maps the paid parking block-faces given colored by neighborhood.
    
    :param element_keys: List containing the keys of block-faces to draw on map.
    :param data_path: File path to the paystation_info.csv and blockface_locs.p.
    :param fig_path: Path to save the html file of blockfaces drawn on the map.
    :param filename: Name to save file of the map, must end in .html.
    """
    
    with open(os.path.join(data_path, 'blockface_locs.p'), 'rb') as f:
        locations = pickle.load(f)

    gmap = gmplot.GoogleMapPlotter(47.612676, -122.345028, 15)

    area_info = pd.read_csv(os.path.join(data_path, 'paystation_info.csv'))
    area_info = area_info[['ELMNTKEY', 'PAIDAREA', 'SUBAREA']]

    all_keys = area_info['ELMNTKEY'].unique().tolist()

    colors = iter(['b', 'g', 'r', 'm', 'k', 'orange', 'deepskyblue', 
                   'purple', 'firebrick', 'darkcyan', 'gold', 'limegreen', 'gray'])

    area_dict = {}

    for key in element_keys:
        curr_block = locations[key]

        lat1, lat2 = curr_block[1], curr_block[-2]
        lon1, lon2 = curr_block[0], curr_block[-3]

        if key in all_keys:
            area = area_info.loc[area_info['ELMNTKEY'] == key]['PAIDAREA'].unique().tolist()[0]

            if area not in area_dict:
                area_dict[area] = next(colors)
        else:
            continue

        gmap.plot([lat1, lat2], [lon1, lon2], color=area_dict[area], edge_width=4)

    gmap.draw(os.path.join(fig_path, filename))


def plot_paid_areas(element_keys, data_path, fig_path, filename='paidarea_map.html'):
    """Plotting on google maps the paid parking blockfaces given colored by each paid area.
    
    :param element_keys: List containing the keys of block-faces to draw on map.
    :param data_path: File path to the paystation_info.csv and blockface_locs.p.
    :param fig_path: Path to save the html file of blockfaces drawn on the map.
    :param filename: Name to save file of the map, must end in .html.
    """
    
    with open(os.path.join(data_path, 'blockface_locs.p'), 'rb') as f:
        locations = pickle.load(f)

    gmap = gmplot.GoogleMapPlotter(47.612676, -122.345028, 15)
    
    area_info = pd.read_csv(os.path.join(data_path, 'paystation_info.csv'))
    area_info = area_info[['ELMNTKEY', 'PAIDAREA', 'SUBAREA']]
    all_keys = area_info['ELMNTKEY'].unique().tolist()

    colors = iter(['b', 'g', 'r', 'm', 'k', 'orange', 'deepskyblue', 
                   'purple', 'firebrick', 'darkcyan', 'gold', 'limegreen', 'gray'])

    area_dict = {}

    for key in element_keys:
        curr_block = locations[key]

        lat1, lat2 = curr_block[1], curr_block[-2]
        lon1, lon2 = curr_block[0], curr_block[-3]

        if key in all_keys:
            neighborhood = area_info.loc[area_info['ELMNTKEY'] == key]['PAIDAREA'].unique().tolist()[0]
            subarea = area_info.loc[area_info['ELMNTKEY'] == key]['SUBAREA'].unique().tolist()[0]
            area = (neighborhood, subarea)

            if area not in area_dict:
                area_dict[area] = next(colors)
        else:
            continue

        gmap.plot([lat1, lat2], [lon1, lon2], color=area_dict[area], edge_width=4)

                
    gmap.draw(os.path.join(fig_path, filename))


def surface_plot(loads, gps_loc, time, fig_path, filename='surface.png', show_fig=False):
    """Create 3D surface plot of load data.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block-face.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    gps_loc = gps_loc[mask]
    loads = loads[mask]

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
     
    plt.figure()

    ax = plt.gca(projection='3d')
    ax.view_init(azim=89.9999999999, elev=70)

    # Creating 3D surface plot of the block-face loads.
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=illuminated_surface, 
                    linewidth=1, antialiased=True)

    ax.set_xticks([]) 
    ax.set_yticks([]) 

    ax.set_xlabel('Latitude' )
    ax.set_ylabel('Longitude')

    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def interpolation(loads, gps_loc, time, fig_path, filename='interpolation.png', show_fig=False):
    """Create interpolation of the load data on the map.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    plt.figure(figsize=(18, 16))
    ax = plt.axes(xlim=(min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100),
                  ylim=(min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, fig_name))
    ax.imshow(im)
    ax.invert_yaxis()

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    pix_pos = pix_pos[mask]
    loads = loads[mask]

    x = pix_pos[:, 0, None]
    y = pix_pos[:, 1, None]
    z = loads[:, time, None]

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    Z = ml.griddata(x[:, 0], y[:, 0], z[:, 0], xi, yi)

    # Interpolating between the block locations to create continuous heat map.
    plt.pcolormesh(xi, yi, Z, vmin=z.min(), vmax=z.max(), cmap='jet')

    # Plotting each of the block-faces on the map colored by the interpolation.
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax, cmap='jet', label='Parking Occupancy')

    # Converting labels for color bar to be occupancy percentage.
    old_labels = cbar.ax.get_yticklabels()
    new_labels = map(lambda label: str(int(float(label.get_text())*100)) + '%', old_labels)
    cbar.ax.set_yticklabels(new_labels)

    # Resizing text properties.
    cbar.ax.tick_params(labelsize=30) 
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=40)
    text.set_font_properties(font)

    # Getting label for figure next block of code.
    days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
    num_times = loads.shape[1]

    hour = 8 + (time % (num_times/6))
    if hour < 12:
        hour = str(hour) + ':00 AM'
    elif hour == 12:
        hour = str(hour) + ':00 PM'
    else:
        hour = str(hour - 12) + ':00 PM'

    day = time/(num_times/6)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    ax.set_xlabel(days[day] + ' ' + hour)
    ax.xaxis.label.set_fontsize(35)

    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def triangular_grid(loads, gps_loc, time, fig_path, filename='triangle.png', show_fig=False):
    """Create triangular grid of the load data on the map.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    plt.figure(figsize=(18, 16))
    ax = plt.axes(xlim=(min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100),
                  ylim=(min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, fig_name))
    ax.imshow(im)
    ax.invert_yaxis()

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    pix_pos = pix_pos[mask]
    loads = loads[mask]

    x = pix_pos[:, 0, None]
    y = pix_pos[:, 1, None]
    z = loads[:, time, None]

    # Creating unstructured triangle graph.
    ax.tripcolor(x[:, 0], y[:, 0], z[:, 0], edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Plotting each of the block-faces on the map colored by the triangle.
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax, cmap='jet', label='Parking Occupancy')

    # Converting labels for color bar to be occupancy percentage.
    old_labels = cbar.ax.get_yticklabels()
    new_labels = map(lambda label: str(int(float(label.get_text())*100)) + '%', old_labels)
    cbar.ax.set_yticklabels(new_labels)

    # Resizing text properties.
    cbar.ax.tick_params(labelsize=30) 
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=40)
    text.set_font_properties(font)

    # Getting label for figure next block of code.
    days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
    num_times = loads.shape[1]

    hour = 8 + (time % (num_times/6))
    if hour < 12:
        hour = str(hour) + ':00 AM'
    elif hour == 12:
        hour = str(hour) + ':00 PM'
    else:
        hour = str(hour - 12) + ':00 PM'

    day = time/(num_times/6)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    ax.set_xlabel(days[day] + ' ' + hour)
    ax.xaxis.label.set_fontsize(35)

    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def contour_plot(loads, gps_loc, time, fig_path, contours=10, filename='contours.png', show_fig=False):
    """Create contour plot of the load data on the map.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param contours: Integer number of contour levels to use for contour plot.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    plt.figure(figsize=(18, 16))
    ax = plt.axes(xlim=(min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100),
                  ylim=(min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))

    # Plotting background image of the map.
    im = imread(os.path.join(fig_path, fig_name))
    ax.imshow(im)
    ax.invert_yaxis()

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    pix_pos = pix_pos[mask]
    loads = loads[mask]

    x = pix_pos[:, 0, None]
    y = pix_pos[:, 1, None]
    z = loads[:, time, None] 

    # Creating contour map of the occupancy on the image.
    ax.tricontourf(x[:, 0], y[:, 0], z[:, 0], contours, vmin=z.min(), vmax=z.max(), cmap='jet')

    # Plotting each of the block-faces on the map colored by the contour.
    plt.scatter(x, y, c=z, s=200, edgecolor='black', vmin=z.min(), vmax=z.max(), cmap='jet')

    # Resizing the color bar to be size of image and adding it to the figure.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(cax=cax, label='Parking Occupancy')

    # Converting labels for color bar to be occupancy percentage.
    old_labels = cbar.ax.get_yticklabels()
    new_labels = map(lambda label: str(int(float(label.get_text())*100)) + '%', old_labels)
    cbar.ax.set_yticklabels(new_labels)

    # Resizing text properties.
    cbar.ax.tick_params(labelsize=30) 
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=40)
    text.set_font_properties(font)

    # Getting label for figure next block of code.
    days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
    num_times = loads.shape[1]

    hour = 8 + (time % (num_times/6))
    if hour < 12:
        hour = str(hour) + ':00 AM'
    elif hour == 12:
        hour = str(hour) + ':00 PM'
    else:
        hour = str(hour - 12) + ':00 PM'

    day = time/(num_times/6)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    ax.set_xlabel(days[day] + ' ' + hour)
    ax.xaxis.label.set_fontsize(35)

    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def voronoi(gps_loc, fig_path, filename='voronoi.png', show_fig=False):
    """Create voronoi diagram of the block locations on the map.

    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param fig_path: Path to save the image to.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    plt.figure(figsize=(18, 16))
    ax = plt.axes(xlim=(min(pix_pos[:, 0]), max(pix_pos[:, 0])),
                  ylim=(min(pix_pos[:, 1]), max(pix_pos[:, 1])))

    # Computing the Voronoi Tesselation.
    vor = Voronoi(pix_pos)

    # Plotting the Voronoi Diagram.
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Colorizing the Voronoi Diagram.
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)

    plt.plot(pix_pos[:, 0], pix_pos[:, 1], 'ko')

    ax.invert_yaxis()
    plt.axis('off')

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    :param vor: Voronoi input diagram
    :param radius: Float distance to 'points at infinity'.
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
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def spatial_heterogeneity(loads, time, fig_path, filename='spatial_heterogeneity.png', show_fig=False):
    """Plot histogram of the loads at a given time to demonstrate spatial heterogeneity.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param time: Integer column index to get the load data from.
    :param fig_path: Path to save the image to.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    # Dropping block-faces with nan (closed) or negligible load.
    mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
    loads = loads[mask]

    N = len(loads)
    bins = range(N)

    # Getting the load data for a specific time and scaling.
    counts = loads[:, time] * 100

    sns.set()
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.set_xticks(bins)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=22)
    ax.yaxis.set_ticks_position('left')

    plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')
    plt.xlim([0, N])

    plt.title('Block-Face Occupancies', fontsize=22)
    plt.xlabel('Block-Faces', fontsize=22)
    plt.ylabel(r'Occupancy $\%$', fontsize=22)

    plt.tight_layout()

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    if show_fig:
        plt.show()

    plt.close()


def temporal_heterogeneity(loads, fig_path, filename='temporal_heterogeneity.png', show_fig=False):
    """Plot average occupancy percentage across at each time and hour combination.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param fig_path: Path to save the image to.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    num_times = loads.shape[1]
    bins = range(num_times)

    # Setting block-faces with negligible load to nan so they are ignored when getting mean.
    with np.errstate(invalid='ignore'):
        loads[loads <= .05] = np.nan

    # Getting the mean load over all locations at each time for the plot.
    counts = np.nanmean(loads, axis=0) * 100

    sns.set()
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.set_xticks(bins)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticklabels([])
    plt.setp(ax.get_yticklabels(), fontsize=22)
    ax.yaxis.set_ticks_position('left')

    plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')
    plt.xlim([-.3, num_times + .2])

    # 10 hour days.
    if num_times == 60:
        # Adding lines to separate days more clearly.
        ax.axvline(x=0, color='black')
        ax.axvline(x=10, color='black')
        ax.axvline(x=20, color='black')
        ax.axvline(x=30, color='black')
        ax.axvline(x=40, color='black')
        ax.axvline(x=50, color='black')
        ax.axvline(x=60, color='black')

        plt.title('Daily Occupancy Profiles', fontsize=22)
        plt.ylabel(r'Occupancy $\%$', fontsize=22)

        # Labels of the day of the week for each portion of the plot.
        ax.annotate('Monday',xy=(1.7,-5),xytext=(1.7,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Tuesday',xy=(11.7,-5),xytext=(11.7,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Wednesday',xy=(20.35,-5),xytext=(20.35,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Thursday',xy=(31.6,-5),xytext=(31.6,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Friday',xy=(42.8,-5),xytext=(42.8,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Saturday',xy=(51.6,-5),xytext=(51.6,-5), annotation_clip=False, fontsize=16)

    # 12 hour days.
    elif num_times == 72:
        # Adding lines to separate days more clearly.
        ax.axvline(x=0, color='black')
        ax.axvline(x=12, color='black')
        ax.axvline(x=24, color='black')
        ax.axvline(x=36, color='black')
        ax.axvline(x=48, color='black')
        ax.axvline(x=60, color='black')
        ax.axvline(x=72, color='black')

        plt.title('Daily Occupancy Profiles', fontsize=22)
        plt.ylabel(r'Occupancy $\%$', fontsize=22)

        # Labels of the day of the week for each portion of the plot.
        ax.annotate('Monday', xy=(2,-5), xytext=(2,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Tuesday', xy=(14,-5), xytext=(14,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Wednesday', xy=(24.5,-5), xytext=(24.5,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Thursday', xy=(38,-5), xytext=(38,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Friday', xy=(51,-5), xytext=(51,-5), annotation_clip=False, fontsize=16)
        ax.annotate('Saturday', xy=(62,-5), xytext=(62,-5), annotation_clip=False, fontsize=16)

    plt.tight_layout()

    sns.reset_orig()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    if show_fig:
        plt.show()

    plt.close()


def temporal_day_plots(loads, fig_path, filename='temporal_day_plots.png', show_fig=False):
    """Create subplots containing bar plots of the load for each day of the week.

    This function creates a subplot for each day of week and in each subplot
    is a bar plot with the average load at each hour of the day.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param fig_path: Path to save the image to.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    sns.set()
    sns.set_style("whitegrid")

    plt.subplots(nrows=1, ncols=6, figsize=(60, 10))

    # Setting block-faces with negligible load to nan so they are ignored when getting mean.
    with np.errstate(invalid='ignore'):
        loads[loads <= .05] = np.nan

    num_times = loads.shape[1]

    # List of lists, where each inner list contains the indexes for a day.
    days = [[i for i in xrange(j, j+(num_times/6))] for j in xrange(0, num_times, (num_times/6))]

    day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',
                5: 'Friday', 6: 'Saturday'}

    day_count = 1

    for day in days:
        bins = range(8, (num_times/6) + 8)

        # Getting the mean loads for the hours of the given day and scaling for plot.
        counts = np.nanmean(loads, axis=0)[day] * 100

        ax = plt.subplot(1, 6, day_count)
        ax.set_xticks(np.arange(min(bins), max(bins)+1, 1))
        x_labels = [str(b) + 'AM' if b < 12 else str(b-12) + 'PM' 
                    if b != 12 else str(b) + 'PM' for b in bins]
        ax.set_xticklabels(x_labels)
        
        plt.title(day_dict[day_count], fontsize=36)
        plt.ylabel(r'Occupancy $\%$', fontsize=32)
        
        plt.setp(ax.get_xticklabels(), fontsize=28, rotation=60)
        plt.setp(ax.get_yticklabels(), fontsize=28)
        
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment('left')
        
        plt.bar(bins, counts, width=1, color='red', edgecolor='black', align='edge')

        plt.xlim([min(bins), max(bins)+1])
        plt.ylim([0, 80])
        
        day_count += 1

    sns.reset_orig()

    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def temporal_hour_plots(loads, fig_path, filename='temporal_hour_plots.png', show_fig=False):
    """Creating subplots of the average load for each hour of the day.

    This function creates a subplot for each hour of the day and in each subplot
    is a bar plot with the average load at each day of the week for that hour.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param fig_path: Path to save the image to.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """

    sns.set()
    sns.set_style("whitegrid")

    num_times = loads.shape[1]

    num_rows = 2
    num_cols = (num_times/6)/2
    plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(7*num_cols, 21))

    # Setting block-faces with negligible load to nan so they are ignored when getting mean.
    with np.errstate(invalid='ignore'):
        loads[loads <= .05] = np.nan

    # List of lists, where each inner list contains the indexes for an hour of the day.
    hours = [[j + i*(num_times/6) for i in xrange(6)] for j in xrange((num_times/6))]

    hour_count = 1

    for hour in hours:
        bins = range(6)

        # Getting the mean loads for the days of the given hour and scaling for plot.
        counts = np.nanmean(loads, axis=0)[hour] * 100
        
        ax1 = plt.subplot(num_rows, num_cols, hour_count)
        ax1.set_xticks(np.arange(min(bins), max(bins)+1, 1))
        ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'])
        plt.setp(ax1.get_xticklabels(), fontsize=22, rotation=60)
        plt.setp(ax1.get_yticklabels(), fontsize=22)
        
        if hour_count + 7 < 12:
            title = str(7 + hour_count) + ':00 AM'
        elif hour_count + 7 == 12:
            title = str(7 + hour_count) + ':00 PM'
        elif hour_count + 7 > 12:
            title = str(7 + hour_count - 12) + ':00 PM'
            
        plt.ylabel(r'Occupancy $\%$', fontsize=24)
        plt.title(title, fontsize=24)
        
        plt.bar(bins, counts, color='red', align='center')

        plt.ylim([0, 80])
        
        hour_count += 1
        
    sns.reset_orig()

    plt.tight_layout()

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def mixture_plot(loads, gps_loc, times, fig_path, num_comps=4,
                 default_means=np.array([[47.61337195, -122.34394369], [47.6144188, -122.34992362],
                                         [47.61707117, -122.34971354], [47.61076144, -122.34305349]]),
                 shape=None, filename='mixture_plot.png', show_fig=False):
    """Plotting the mixture model results at a time or times of day.

    This function first creates a mixture model of the load data and the spatial
    characteristics, e.g. the GPS coordinates of the block locations. Thus the 
    mixture model has features for each block of the load at the time, and the
    latitude and longitude of the center of the block. The centroids and the 
    curves of the first two standard deviations of spatial components are drawn
    on the map. Each block is indicated by a scatter point and colored according
    to the probabilistic assignment the mixture model designates the block.
    
    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param times: List or integer of column index(es) to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param num_comps: Integer number of mixture components for the model.
    :param default_means: Numpy array, each row containing an array of the
    lat and long to use as the default mean so colors will stay the same
    when making several plots.
    :param shape: Tuple of the row and col dimension of subplots.
    :param filename: Name to save the image as.
    :param show_fig: Bool indicating whether to show the image.
    """
    
    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    # Setting center of image.
    center = ((up_left[0] - bottom_right[0])/2., (up_left[1] - bottom_right[1])/2.)
    pix_center = mp.to_image_pixel_position(list(center))

    num_times = loads.shape[1]
    
    if isinstance(times, list):
        num_figs = len(times)
    else:
        num_figs = 1
    
    if shape is None:
        fig = plt.figure(figsize=(18*num_figs, 16))
        fs = 35
    else:
        fig = plt.figure(figsize=(18*shape[1], 16*shape[0]))
        fs = 35
    
    for fig_count in xrange(1, num_figs+1):
        
        if shape is None:
            ax = fig.add_subplot(1, num_figs, fig_count)
        else:
            ax = fig.add_subplot(shape[0], shape[1], fig_count)
            
        ax.set_xlim((min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100))
        ax.set_ylim((min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))
        
        if isinstance(times, list):
            time = times[fig_count-1]
        else:
            time = times

        im = imread(os.path.join(fig_path, fig_name))
        ax.imshow(im)
        ax.invert_yaxis()

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax.xaxis.label.set_fontsize(fs)

        # Dropping block-faces with nan (closed) or negligible load.
        mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))
        pix_pos_time = pix_pos[mask]

        # Adding in the midpoints of the block faces to the map as points.
        scatter = ax.scatter(pix_pos_time[:, 0], pix_pos_time[:, 1], s=175, color='red', edgecolor='black')

        # Setting up the centroids for the mixture components.
        scatter_centroid = ax.scatter([], [], s=500, color='red', edgecolor='black')

        # Setting up the ellipses for the mixture components.
        patches = [Ellipse(xy=(0, 0), width=0, height=0, angle=0, edgecolor='black', 
                   facecolor='none', lw='4') for comp in xrange(2*num_comps)]
        ellipses = [ax.add_patch(patches[comp]) for comp in range(2*num_comps)]

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

            for i in xrange(num_comps):
                # Finding the default centroid closest to the current centroid.
                dists = [(j, np.linalg.norm(means[i] - default_means[j])) for j in xrange(num_comps)]
                best_colors = sorted(dists, key=lambda item:item[1])

                # Finding the color that is unused that is closest to the current centroid.
                unused_colors = [color[0] for color in best_colors if color[0] 
                                 not in color_codes.values()]

                # Choosing the closest centroid that is not already used.
                choice = unused_colors[0]
                color_codes[i] = choice
        else:
            colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, num_comps)]
            color_codes = {i:i for i in xrange(num_comps)}

        # Setting the cluster colors based off of the labels.
        scatter.set_color([colors[color_codes[labels[i]]] for i in xrange(len(labels))])
        scatter.set_edgecolor(['black' for i in xrange(len(labels))])

        ellipse_num = 0

        # Updating the ellipses for each of the components.
        for i in xrange(num_comps):
            lambda_, v = np.linalg.eig(covs[i])
            lambda_ = np.sqrt(lambda_)

            # Getting the ellipses for the 1st and 2nd standard deviations.
            for j in [1, 2]:

                # Converting mean in GPS coords to pixel positions.
                xy = mp.to_image_pixel_position(list(means[i, :]))

                # Width and height of the ellipses in GPS coords.
                width = lambda_[0]*j*2
                height = lambda_[1]*j*2 

                # Center of the ellipse in pixel positions.
                new_center = (center[0] + width, center[1] + height)
                new_center = mp.to_image_pixel_position(list(new_center))

                # New width and height of the ellipses in pixel positions.
                width = abs(new_center[0] - pix_center[0])
                height = abs(new_center[1] - pix_center[1])

                # Updating the ellipses.
                patches[ellipse_num].center = xy
                patches[ellipse_num].width = width
                patches[ellipse_num].height = height
                patches[ellipse_num].edgecolor = colors[color_codes[i]]

                ellipse_num += 1

        # Converting the centroids to pixel positions from GPS coords.
        pix_means = np.array([mp.to_image_pixel_position(list(means[i, :])) for i in xrange(len(means))])

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

        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
        ax.set_xlabel(days[day] + ' ' + hour)

    fig.tight_layout()

    if shape is not None and shape[0] > 1 and shape[1] == 1:
        plt.subplots_adjust(top=0.975)
    else:
        plt.subplots_adjust(top=0.98)

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')
    
    if show_fig:
        plt.show()

    plt.close()

 
def centroid_plots(centers, gps_loc, times, fig_path, num_comps,
                   shape=None, filename='centroid_plot.png', show_fig=False):
    """Plotting the centroids from different dates at the same weekday and hour.

    This function takes in a parameter, centers, which is a list of list of numpy
    arrays in which each numpy array in the inner list contains a set of 
    centroids from the Gaussian mixture modeling procedure for a particular date
    and the inner list represents a particular day of the week and hour of the
    day combo. For each of the times indexes indicated by the times parameter,
    the centroids that were fit at a particular day of the week and hour of the
    day combo are clustered. These clusters are then plotted as scatter points
    at the locations they are. Each cluster is colored differently to indicate
    the cluster difference.
    
    :param centers: List of list of numpy arrays, with each outer list containing
    and inner list which contains a list of numpy arrays where each numpy array
    in the inner lists contains a 2-d array that contains each of the centroids
    for a GMM fit of a particular date for the weekday and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param times: List or integer of column index(es) to get the load data from.
    :param fig_path: Path to save image to and retrieve background image from.
    :param num_comps: Integer number of mixture components for the model.
    :param shape: Tuple of the row and col dimension of subplots.
    :param filename: Name to save the file as.
    :param show_fig: Bool indicating whether to show the image.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    num_times = len(centers)
    
    if isinstance(times, list):
        num_figs = len(times)
    else:
        num_figs = 1
    
    if shape is None:
        fig = plt.figure(figsize=(18*num_figs, 16))
        fs = 35
    else:
        fig = plt.figure(figsize=(18*shape[1], 16*shape[0]))
        fs = 35

    for fig_count in xrange(1, num_figs+1):
        
        if shape is None:
            ax = fig.add_subplot(1, num_figs, fig_count)
        else:
            ax = fig.add_subplot(shape[0], shape[1], fig_count)
            
        ax.set_xlim((min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100))
        ax.set_ylim((min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))
        
        if isinstance(times, list):
            time = times[fig_count-1]
        else:
            time = times

        im = imread(os.path.join(fig_path, fig_name))
        ax.imshow(im)
        ax.invert_yaxis()

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax.xaxis.label.set_fontsize(fs)

        # Getting the data, clustering the centroids, and getting class labels.
        data = np.vstack((centers[time]))
        kmeans = KMeans(n_clusters=num_comps, n_init=50).fit(data)
        labels = kmeans.labels_.tolist()

        # Translating the centroid GPS coordinates to pixel positions.
        data_pix_pos = np.array([mp.to_image_pixel_position(list(data[i, :])) for i in xrange(len(data))])

        # Adding in the centroids to the map as points.
        scatter = ax.scatter(data_pix_pos[:, 0], data_pix_pos[:, 1], s=500, color='red', edgecolor='black')

        if num_comps == 4:
            colors = ['deeppink', 'lawngreen', 'blue', 'aqua']
        else:
            colors = [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, num_comps)]

        # Setting the centroid colors to be the same within a cluster.
        scatter.set_color([colors[labels[i]] for i in xrange(len(labels))])
        scatter.set_edgecolor(['black' for i in xrange(len(labels))])

        hour = 8 + (time % (num_times/6))
        day = time/(num_times/6)

        if hour < 12:
            hour = str(hour) + ':00 AM'
        elif hour == 12:
            hour = str(hour) + ':00 PM'
        else:
            hour = str(hour - 12) + ':00 PM'

        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
        ax.set_xlabel(days[day] + ' ' + hour)

    fig.tight_layout()

    if shape is not None and shape[0] > 1 and shape[1] == 1:
        plt.subplots_adjust(top=0.975)
    else:
        plt.subplots_adjust(top=0.98)

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def centroid_radius(centroids, all_time_points, gps_loc, times, fig_path, 
                    shape=None, filename='centroid_radius.png', show_fig=False):
    """Plotting centroids at a time and date and circle around each.

    The input of the centroids are the centroids of all the centers that
    were found by the locational demand analysis function in figure_functions.py.
    The mean distance to these centroids are then used as the radius of the 
    circle that is plotted around each centroid. 
    
    :param centroids: Numpy array of 3 dimensions with the first dimension the 
    number of times the centroids were found for, the second dimension the 
    number of centroids at the time, and the third dimension the GPS coords of each centroid.
    :param all_time_points: Numpy array of 4 dimensions with the first dimension 
    the number of times the centroids were found for, the second dimension the 
    number of centroids at the time, the third dimension the points for the 
    circle with the last dimension each point in GPS coords.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param times: List or integer of column index(es) to get the load data from.
    :param fig_path: Path to save the image to and retrieve background image from.
    :param shape: Tuple of the row and col dimension of subplots.
    :param filename: Name to save the image as.
    """

    up_left, bottom_right, img_size, fig_name = setup_image()

    mp = MapOverlay(up_left, bottom_right, img_size)

    num_times = centroids.shape[0]

    # Translating GPS coordinates to pixel positions.
    pix_pos = np.array([mp.to_image_pixel_position(list(gps_loc[i, :])) for i in xrange(len(gps_loc))])

    if isinstance(times, list):
        num_figs = len(times)
    else:
        num_figs = 1

    if shape is None:
        fig = plt.figure(figsize=(18*num_figs, 16))
        fs = 35
    else:
        fig = plt.figure(figsize=(18*shape[1], 16*shape[0]))
        fs = 35

    for fig_count in xrange(1, num_figs+1):

        if shape is None:
            ax = fig.add_subplot(1, num_figs, fig_count)
        else:
            ax = fig.add_subplot(shape[0], shape[1], fig_count)

        ax.set_xlim((min(pix_pos[:, 0])-100, max(pix_pos[:, 0])+100))
        ax.set_ylim((min(pix_pos[:, 1])-100, max(pix_pos[:, 1])+100))

        if isinstance(times, list):
            time = times[fig_count-1]
        else:
            time = times

        im = imread(os.path.join(fig_path, fig_name))
        ax.imshow(im)
        ax.invert_yaxis()

        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax.xaxis.label.set_fontsize(fs)

        # Adding the centroids to the map.
        scatter_centroid = ax.scatter(centroids[0][:, 0], centroids[0][:, 1], s=500, 
                                      color='red', edgecolor='black')

        # Adding in the circle around each centroid having radius of average distance to centroid of all points.
        for comp in xrange(centroids.shape[1]):
            path = np.array([list(mp.to_image_pixel_position(all_time_points[time, comp, i])) 
                             for i in xrange(all_time_points.shape[2])])
            
            poly = plt.Polygon(path, fill=None, edgecolor='black', lw=4)
            
            ax.add_patch(poly)

        # Converting the centroids to pixel positions from GPS coords.
        pix_means = np.array([mp.to_image_pixel_position(list(centroids[time, i])) for i in xrange(centroids.shape[1])])

        # Updating the centroid locations.
        scatter_centroid.set_offsets(pix_means)

        hour = 8 + (time % (num_times/6))
        if hour < 12:
            hour = str(hour) + ':00 AM'
        elif hour == 12:
            hour = str(hour) + ':00 PM'
        else:
            hour = str(hour - 12) + ':00 PM'

        day = time/(num_times/6)

        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday'}
        ax.set_xlabel(days[day] + ' ' + hour)

    fig.tight_layout()

    if shape is not None and shape[0] > 1 and shape[1] == 1:
        plt.subplots_adjust(top=0.975)
    else:
        plt.subplots_adjust(top=0.98)

    plt.savefig(os.path.join(fig_path, filename), bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.close()


def model_selection(loads, gps_loc, fig_path, show_fig=False):
    """Using various criterion to evaluate the number of components to use in GMM.

    This function creates a Gaussian mixture model with a varying range of 
    components for each of the times in the parameter loads. The average
    akaike information criterion, bayesian information criterion, and likelihood
    are then averaged over all times and plotted in order to find the correct
    number of components to use in the model and these graphs are plotted.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param fig_path: Path to save the image to.
    :param show_fig: Bool indicating whether to show the image.
    """

    model_selection = {"likelihood": [], "bic": [], "aic": []} 
    min_comps = 1
    max_comps = 21

    num_times = loads.shape[1]

    # Fitting mixture models at each possible time the loads are available.
    for time in xrange(num_times):

        likelihoods = []
        bics = []
        aics = []

        # Varying the number of components and getting AIC, BIC, and likelihood.
        for num_comps in xrange(min_comps, max_comps):

            # Dropping block-faces with nan (closed) or negligible load.
            mask = ((~np.isnan(loads[:, time])) & (~(loads[:, time] <= 0.05)))

            # Getting the data and normalizing the features.
            cluster_data = np.hstack((loads[mask][:, time, None], gps_loc[mask]))
            scaler = MinMaxScaler().fit(cluster_data)
            cluster_data = scaler.transform(cluster_data)

            # Fitting the mixture model.
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

    if show_fig:
        plt.show()

    plt.close()

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

    if show_fig:
        plt.show()

    plt.close()

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

    if show_fig:
        plt.show()

    plt.close()

    sns.reset_orig()


def create_animation(loads, gps_loc, fig_path, animation_path, num_comps=4,
                     filename='mixture.mp4'):
    """Create an animation of the GMM model using figures of each hour of load data.

    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param fig_path: Path to retrieve the background image from.
    :param animation_path: Path to save the animation video to.
    :param num_comps: Integer number of mixture components for the model.
    :param filename: Name to save the animation as, needs to be mp4 extension.
    """

    params = mixture_animation.init_animation(gps_loc, num_comps, fig_path)
    fig, ax, scatter, scatter_centroid, patches, ellipses, mp, center, pix_center = params

    num_times = loads.shape[1]
    times = range(num_times)

    # Default means in attempt to keep the colors the same at each time index if num_comps is 4.
    default_means = np.array([[47.61348888, -122.34343007],[47.61179196, -122.34500616],
                              [47.61597088, -122.35054099],[47.61706817, -122.34617185]])

    ani = animation.FuncAnimation(fig=fig, func=mixture_animation.animate, frames=num_times,
                                  fargs=(times, ax, scatter, scatter_centroid, patches, 
                                         ellipses, mp, default_means, center, 
                                         pix_center, loads, gps_loc, num_comps, ), 
                                  interval=200)

    FFwriter = animation.FFMpegWriter(fps=1)
    ani.save(os.path.join(animation_path, filename), writer=FFwriter)