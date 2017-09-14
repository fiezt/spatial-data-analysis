import os
import sys
import process_data
import gmm
import figure_functions
import kmeans_utils
import write_results


def run_figures(element_keys, loads, gps_loc, num_comps, data_path, fig_path,
                animation_path, time, show_fig):
    """Create the visualizations of the spatial characteristics of the parking data.

    :param element_keys: List containing the keys of block-faces to draw on map.
    :param loads: Numpy array where each row is the load data for a block-face
    and each column corresponds to a day of week and hour.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param num_comps: Integer number of mixture components for the model.
    :param data_path: File path to the paystation_info.csv and blockface_locs.p.
    :param fig_path: Path to retrieve the background image from and save images to.
    :param animation_path: Path to save the animation video to.
    :param time: Integer column index to get the load data from.
    :param show_fig: Bool indicating whether to show the images.
    """

    figure_functions.plot_neighborhoods(element_keys=element_keys, data_path=data_path,
                                        fig_path=fig_path)

    figure_functions.plot_paid_areas(element_keys=element_keys, data_path=data_path,
                                     fig_path=fig_path)

    figure_functions.surface_plot(loads=loads, gps_loc=gps_loc, time=time,
                                  fig_path=fig_path, show_fig=show_fig)

    figure_functions.interpolation(loads=loads, gps_loc=gps_loc, time=time,
                                   fig_path=fig_path, show_fig=show_fig)

    figure_functions.triangular_grid(loads=loads, gps_loc=gps_loc, time=time,
                                     fig_path=fig_path, show_fig=show_fig)

    figure_functions.contour_plot(loads=loads, gps_loc=gps_loc, time=time,
                                  fig_path=fig_path, show_fig=show_fig)

    figure_functions.voronoi(gps_loc=gps_loc, fig_path=fig_path, show_fig=show_fig)

    figure_functions.spatial_heterogeneity(loads=loads, time=time,
                                           fig_path=fig_path, show_fig=show_fig)

    figure_functions.temporal_heterogeneity(loads=loads, fig_path=fig_path,
                                            show_fig=show_fig)

    figure_functions.temporal_day_plots(loads=loads, fig_path=fig_path,
                                        show_fig=show_fig)

    figure_functions.temporal_hour_plots(loads=loads, fig_path=fig_path,
                                         show_fig=show_fig)

    figure_functions.model_selection(loads=loads, gps_loc=gps_loc,
                                     fig_path=fig_path, show_fig=show_fig)

    figure_functions.mixture_plot(loads=loads, gps_loc=gps_loc, times=time,
                                  fig_path=fig_path, num_comps=num_comps,
                                  show_fig=show_fig)

    figure_functions.create_animation(loads=loads, gps_loc=gps_loc,
                                      fig_path=fig_path,
                                      animation_path=animation_path,
                                      num_comps=num_comps)


def run_gmm_simulations(park_data, gps_loc, time, num_comps, k, p_value, fig_path, results_path):
    """Running several GMM tests including consistency measures and spatial correlation.
    
    :param park_data: Multi-index DataFrame containing datetimes in the first
    level index and block-face keys in the second level index. Values include
    the corresponding loads.
    :param gps_loc: Numpy array with each row containing the lat, long pair
    midpoints for a block-face.
    :param time: Integer column index to get the load data from.
    :param num_comps: Integer number of mixture components for the model.
    :param k: Integer number of neighbors to use for the Moran weighting matrix.
    :param p_value: Float p value to use to measure significance.
    :param fig_path: Path to save the images to.
    :param results_path: Path to save results files to.
    """

    results = gmm.locational_demand_analysis(park_data=park_data, gps_loc=gps_loc,
                                             num_comps=num_comps, k=k)

    days = [result[0] for result in results]
    hours = [result[1] for result in results]
    consistencies = [result[2] for result in results]
    morans_mixture = [result[3] for result in results]
    morans_area = [result[4] for result in results]
    morans_neighbor = [result[5] for result in results]
    centers = [result[6] for result in results]

    write_results.write_gmm_results(consistencies=consistencies, results_path=results_path)

    write_results.write_moran_results(days=days, hours=hours, morans_mixture=morans_mixture,
                                      morans_area=morans_area, morans_neighbor=morans_neighbor,
                                      p_value=p_value, results_path=results_path)

    distances, centroids = kmeans_utils.get_distances(centers=centers, num_comps=num_comps)

    write_results.write_centroid_distance_results(days=days, hours=hours,
                                                  distances=distances,
                                                  results_path=results_path)

    all_time_points = kmeans_utils.get_centroid_circle_paths(distances=distances,
                                                             centroids=centroids)

    figure_functions.centroid_plots(centers=centers, gps_loc=gps_loc, times=time,
                                    fig_path=fig_path, num_comps=num_comps)

    figure_functions.centroid_radius(centroids=centroids, all_time_points=all_time_points,
                                     gps_loc=gps_loc, times=time, fig_path=fig_path)


def main():
    curr_dir = os.getcwd()
    data_path = curr_dir + '/../data/'
    fig_path = curr_dir + '/../figs/'
    results_path = curr_dir + '/../results/'
    animation_path = curr_dir + '/../animation/'
    belltown_path = data_path + '/Belltown_Hour'

    time = int(sys.argv[1])
    num_comps = int(sys.argv[2])
    k = int(sys.argv[3])
    p_value = float(sys.argv[4])
    option = sys.argv[5]

    params = process_data.load_data(data_path=data_path, load_paths=belltown_path)
    element_keys, loads, gps_loc, park_data, idx_to_day_hour, day_hour_to_idx = params

    if option == 'all':
        run_figures(element_keys=element_keys, loads=loads, gps_loc=gps_loc,
                    num_comps=num_comps, data_path=data_path, fig_path=fig_path,
                    animation_path=animation_path, time=time, show_fig=False)

        run_gmm_simulations(park_data=park_data, gps_loc=gps_loc, time=time,
                            num_comps=num_comps, k=k, p_value=p_value, fig_path=fig_path,
                            results_path=results_path)
    elif option == 'figs':
        run_figures(element_keys=element_keys, loads=loads, gps_loc=gps_loc,
                    num_comps=num_comps, data_path=data_path, fig_path=fig_path,
                    animation_path=animation_path, time=time, show_fig=False)
    elif option == 'sims':
        run_gmm_simulations(park_data=park_data, gps_loc=gps_loc, time=time,
                            num_comps=num_comps, k=k, p_value=p_value, fig_path=fig_path,
                            results_path=results_path)


if __name__ == '__main__':
    main()