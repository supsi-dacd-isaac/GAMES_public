import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from seaborn._statistics import KDE
import warnings
from distance_util import bhattacharyya_distance
import osmnx as ox
from mclp import *
from models.macroscopic_traffic_model import Macroscopic_traffic_model

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.cluster.vq import kmeans, vq


def plot_data_points(data, ax=None, alpha=0.02):
    x, y = data[:, 0], data[:, 1]
    if ax is None:
        plt.scatter(x, y, c='b', alpha=alpha)
    else:
        ax.scatter(x, y, c='b', alpha=alpha)


def plot_kmeans_clustering(centers, ax=None, alpha=0.8):
    if ax is None:
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', alpha=alpha)
    else:
        ax.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', alpha=alpha)

def quantile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels


def estimate_bivariate_density(data_x, data_y, levels=20, thresh=0.05):
    estimate_kws = {'bw_method': 'scott', 'bw_adjust': 1, 'gridsize': 200, 'cut': 3, 'clip': None, 'cumulative': False}
    estimator = KDE(**estimate_kws)

    observations = pd.DataFrame()
    observations['x'], observations['y'] = data_x, data_y

    min_variance = observations.var().fillna(0).min()
    observations = observations["x"], observations["y"]
    # Estimate the density of observations at this level
    singular = math.isclose(min_variance, 0)
    # Input the weights
    weights = None
    try:
        if not singular:
            density, support = estimator(*observations, weights=weights)
    except np.linalg.LinAlgError:
        # Testing for 0 variance doesn't catch all cases where scipy raises,
        # but we can also get a ValueError, so we need this convoluted approach
        singular = True

    if singular:
        msg = (
            "KDE cannot be estimated (0 variance or perfect covariance). "
            "Pass `warn_singular=False` to disable this warning."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)

    densities = {}
    densities[()] = density

    levels = np.linspace(thresh, 1, levels)
    draw_levels = {
        k: quantile_to_level(d, levels)
        for k, d in densities.items()
    }
    return estimator, density, support, draw_levels


def coverage_objective(x_ev , x_d_rad, acceptance_radius=500):
    # Objective function to minimize coverage given a distribution and EV locations

    lat_targets = x_d_rad[:, 1]
    lon_targets = x_d_rad[:, 0]
    x_ev_rad = np.radians(x_ev)

    lat_targets = lat_targets[:, np.newaxis]  # Convert to column vector
    lon_targets = lon_targets[:, np.newaxis]  # Convert to column vector

    lat_diff = lat_targets - x_ev_rad[:, 1]  # Compute latitude differences
    lon_diff = lon_targets - x_ev_rad[:, 0]  # Compute longitude differences

    # Compute the matrix of pairwise distances using the Haversine formula
    AVG_EARTH_RADIUS = 6_371_000
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_targets) * np.cos(x_ev_rad[:, 1]) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = AVG_EARTH_RADIUS * c
    is_covered = distances<=acceptance_radius

    mean_coverage = np.mean(np.any(is_covered, axis=1))
    # mean_distance_km = distance/(len(x_ev_rad)*1_000)
    if mean_coverage >= 0.5:
        print('great coverage', mean_coverage)
        plt.scatter(lon_targets, lat_targets, alpha=0.05, c='b', label='demand points ~ f_dem(x)')
        plt.scatter(lon_targets[np.any(is_covered, axis=1)], lat_targets[np.any(is_covered, axis=1)], alpha=0.15, c='r',
                    label='covered')
        plt.scatter(x_ev_rad[:, 0], x_ev_rad[:, 1], c='k',
                    label=' x_guess')
        plt.grid()
        plt.legend()
        plt.xlabel('LON')
        plt.ylabel('LAT')
        plt.show()

    return -mean_coverage

def fit_kde(trip_data, bandwidth=0.01):
    """
    Fit a Kernel Density Estimator (KDE) to trip departure data.

    Parameters:
    - trip_data: DataFrame containing trip departure coordinates as 'LATs' and 'LONs' columns.
    - bandwidth: Bandwidth parameter for the KDE.

    Returns:
    - kde: Fitted KDE model.
    """
    # Extract latitude and longitude data
    data = trip_data[['LATs', 'LONs']].values
    # Initialize and fit the KDE model
    kde = KernelDensity(bandwidth=bandwidth, metric="haversine", kernel="gaussian", algorithm="ball_tree")
    kde.fit(data)
    return kde


def visualize_kde(kde, grid_points, figsize=(10, 8)):
    # Compute the log density values for the grid points
    log_density = kde.score_samples(grid_points)

    # estimate log density on the grid points
    density = np.exp(log_density)

    # Create a contour plot of the KDE
    plt.figure(figsize=figsize)
    n_grid_points = np.sqrt(len(grid_points)).astype(int)
    lat_grid, lon_grid = grid_points[:, 0].reshape((n_grid_points, n_grid_points)), grid_points[:, 1].reshape(
        (n_grid_points, n_grid_points))
    plt.contourf(lon_grid, lat_grid, density.reshape((n_grid_points, n_grid_points)), cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Latitude (radians)')
    plt.ylabel('Longitude (radians)')
    plt.title('KDE Fitted to Trip Departure Data')
    plt.show()


def optimize_coverage(df, control_area, n_ev, radius):
    # Optimize EV locations to minimize coverage given the KDE distribution
    lat_lon_data = df[['LATs', 'LONs']]
    lat_lon_data['LATs_rad'] = np.radians(lat_lon_data['LATs'])
    lat_lon_data['LONs_rad'] = np.radians(lat_lon_data['LONs'])
    data_points = np.array(lat_lon_data[['LATs_rad', 'LONs_rad']])

    # Create and fit the KDE model
    kde = KernelDensity(bandwidth=radius, metric='haversine', kernel='gaussian')
    kde.fit(data_points)
    # Generate initial EV locations (you can initialize them differently)
    initial_x_ev = np.random.rand(n_ev, 2) * 2 * np.pi - np.pi  # Random initial locations within the control area
    # Define bounds for EV locations (within the control area)
    bounds = [(min(control_area[:, 0]), max(control_area[:, 0])), (min(control_area[:, 1]), max(control_area[:, 1]))]
    # Minimize the coverage objective function using scipy's minimize
    result = minimize(coverage_objective, initial_x_ev,
                      args=(np.exp(kde.score_samples(data_points)), control_area, radius),
                      method='trust-constr', bounds=bounds)

    optimized_x_ev = result.x.reshape(-1, 2)
    return optimized_x_ev


def initial_guess_xevs(data_points, n_ev=50):
    # Generate initial EV locations  # Random initial locations within the control area
    p_levels = np.linspace(.25, .75, 30)
    q_lon = np.quantile(data_points[:, 0], p_levels)
    q_lat = np.quantile(data_points[:, 1], p_levels)
    rand_lat_rad = np.random.choice(q_lat, n_ev)
    rand_lon_rad = np.random.choice(q_lon, n_ev)
    initial_x_ev = np.vstack([rand_lon_rad, rand_lat_rad]).T
    guess_x_ev_flat = initial_x_ev.flatten()  # Flatten values for the optimizer

    return guess_x_ev_flat


def optimize_distance_fleet(df, n_ev, f_opt_x):
    # lat_min, lon_min, lat_max, lon_max
    # Optimize EV locations to minimize coverage given the KDE distribution
    lat_lon_data = df[['LATs', 'LONs']]
    data_points = np.array(lat_lon_data[['LONs', 'LATs']])

    guess_x_ev_flat = initial_guess_xevs(data_points, n_ev=n_ev)

    # Use the genetic algorithm for optimization
    bounds = [(lon_min, lon_max), (lat_min, lat_max)] * n_ev
    result = differential_evolution(func=f_opt_x, bounds = bounds,
                                    args=(n_ev,),
                                    disp=True, maxiter=500,
                                    popsize=10, tol=0.01) #, workers=4

    optimized_x_ev = result.x.reshape(n_ev, 2)
    distance = f_opt_x(result.x)

    # visualize
    plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.05, c='b', label='demand points ~ f_dem(x)')
    plt.scatter(guess_x_ev_flat.reshape(n_ev, 2)[:, 0], guess_x_ev_flat.reshape(n_ev, 2)[:, 1], c='k',
                label='initial x_guess')
    plt.scatter(optimized_x_ev[:, 0], optimized_x_ev[:, 1], c='r', label='optimized x_ev')
    plt.grid()
    plt.legend()
    plt.xlabel('LON')
    plt.ylabel('LAT')
    plt.show()
    return optimized_x_ev, distance


def generate_grid_points(lat_min, lat_max, lon_min, lon_max, grid_resolution):
    """
    Generate a grid of points within specified latitude and longitude ranges.

    Parameters:
    - lat_min, lat_max: Minimum and maximum latitude values.
    - lon_min, lon_max: Minimum and maximum longitude values.
    - grid_resolution: Number of points along each dimension.

    Returns:
    - grid_points: Array of grid points.
    """

    # Create a grid of latitude and longitude values
    latitudes = np.linspace(lat_min, lat_max, grid_resolution)
    longitudes = np.linspace(lon_min, lon_max, grid_resolution)

    # Generate all possible combinations of latitude and longitude
    grid_points = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)

    return grid_points


def kde_coverage(df_trips, lat_min, lon_min, lat_max, lon_max, bandwidth, visualize=True):
    # Calculate coverage using KDE distribution and EV locations
    kde = fit_kde(df_trips, bandwidth=bandwidth)

    if visualize:
        grid_points = generate_grid_points(lat_min, lat_max, lon_min, lon_max, 7)
        visualize_kde(kde, grid_points)

    # Calculate coverage using KDE and EV locations
    coverage = 0.0
    for i in range(len(control_area)):
        coverage += np.exp(kde.score_samples(np.array([[control_area[i][0], control_area[i][1]]])))

    return coverage


# Example usage:
# df_trip_departures = pd.read_csv('trip_departures.csv')  # Load your trip departure data
# control_area = np.array([[lat1, lon1], [lat2, lon2], ...])  # Define your control area boundary
# n_ev = 10  # Number of EVs in the fleet
# radius = 0.01  # Bandwidth for KDE estimation (adjust as needed)

# Optimized EV locations
# optimized_locations = optimize_coverage(df_trip_departures, control_area, n_ev, radius)
# print("Optimized EV Locations:", optimized_locations)

# Coverage using KDE
# kde_cov = kde_coverage(df_trip_departures, control_area, radius)
# print("KDE Coverage:", kde_cov)

# Example usage:
# kde = fit_kde(df_trip_departures, bandwidth=0.01)  # Fit KDE to your data
# grid_points = generate_grid_points()  # Define your grid points
# visualize_kde(kde, grid_points)

if __name__ == '__main__':
    df = pd.read_pickle(
        r'C:\Users\roberto.rocchetta.in\OneDrive - SUPSI\Desktop\GAMES project\Codes\gym_relocation\data\autotel\df_TelAviv_for_gym_episodes.pkl')
    lon_min, lon_max = 34.72, 34.87
    lat_min, lat_max = 32.01, 32.15
    control_area = np.array([[lat_min, lon_min], [lat_max, lon_max]])  # Define your control area boundary

    Gtr = Macroscopic_traffic_model(geographical_area='Tel Aviv, Israel')
    data = np.array(df[['LONs','LATs']].values)

    ## K-means clustering
    n_st = 50
    data = data[:650_000]
    centers = kmeans(data, n_st)[0]
    cluster = vq(data, centers)[0]

    vor = Voronoi(centers)

    # Plotting
    fig, ax = ox.plot_graph(Gtr.G_drive, node_alpha=0.1, bgcolor="#cccccc",
                            node_color='k', edge_color='k', show=False,
                            edge_alpha=0.1)
    plot_data_points(data, ax=ax)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
    plot_kmeans_clustering(centers, ax=ax, alpha=0.99)
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)
    plt.show()
    """ for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

    center = centers.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            # Getting distance between points
            t = centers[pointidx[1]] - centers[pointidx[0]]

            # Normalizing to a unit vector. To do that, just get the norm
            # (by the Pythagorean theorem), and divide the vector by the norm
            t = t / np.linalg.norm(t)

            n = np.array([-t[1], t[0]])  # normal
            midpoint = centers[pointidx].mean(axis=0)
            far_point = (vor.vertices[i] +
                         np.sign(np.dot(midpoint - center, n)) * n * 50)
            plt.plot([vor.vertices[i, 0], far_point[0]],
                     [vor.vertices[i, 1], far_point[1]], 'k--')




    points = data[np.random.randint(0, len(data), (1, 1_300))]
    # Number of sites to select
    n_st, M = 80, 500
    # Service radius of each site
    radius = 0.004
    # Candidate site size (random sites generated)
    # Run mclp opt_sites is the location of optimal sites and f is the points covered
    opt_sites, f = mclp(points, n_st, radius, M)

    fig, ax = ox.plot_graph(Gtr.G_drive, node_alpha=0.1, bgcolor="#808080", node_color='k', edge_color='k', show=False,
                            edge_alpha=0.3)
    ax.scatter(points[:, 0], points[:, 1], 20, marker='x', c='b', alpha=0.9)
    (centers, ax=None, alpha=0.8)
    plt.scatter(opt_sites[:, 0], opt_sites[:, 1], 10, c='r', marker='+')
    for site in opt_sites:
        circle = plt.Circle(site, radius, color='r', fill=True, lw=2, alpha=0.3)
        ax.add_artist(circle)
    ax.axis('equal')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)
    plt.show()

    ev = pd.DataFrame()
    ev['LAT'] = np.random.uniform(lat_min, lat_max, 100)
    ev['LON'] = np.random.uniform(lon_min, lon_max, 100)

    set1 = df[['LONs', 'LATs']].values
    set2 = ev[['LON', 'LAT']].values
    db = bhattacharyya_distance(set1, set2)
    # kl = kl_divergence_between_sets(set1,set2)
    n_ev = 30  # Number of EVs in the fleet

    X_h, BD_h = [], []
    for h in range(24):
        df_h_temp = df[df['hour'] == h]

        # Optimize EV locations to minimize coverage given the KDE distribution
        lat_lon_data = df_h_temp[['LATs', 'LONs']]
        data_points = np.array(lat_lon_data[['LONs', 'LATs']])
        data_points_rad = np.radians(data_points)
        # Define the Bhattacharyya function for the genetic algorithm
        def bhattacharyya_fun_x(x, n_ev):
            return bhattacharyya_distance(x.reshape(n_ev, 2), data_points) * 1e8

        def coverage_fun_x(x, n_ev):
            return coverage_objective(x.reshape(n_ev, 2), data_points_rad[:10_000])


        x_h, coverage_h = optimize_distance_fleet(df=df_h_temp, n_ev=n_ev, f_opt_x=coverage_fun_x)
        x_h, bd_h = optimize_distance_fleet(df=df_h_temp, n_ev=n_ev, f_opt_x=bhattacharyya_fun_x)
        X_h.append(x_h)
        BD_h.append(bd_h)

    x_all, bd_all = optimize_distance_fleet(df=df, n_ev=n_ev)

    bandwidth = 0.0001  # Bandwidth for KDE estimation (adjust as needed)
    df_1 = df[df['hour'] == 3]
    kde_estimator, density_xy, support_xy, draw_levels = estimate_bivariate_density(data_x=df_1['LONs'].values,
                                                                                    data_y=df_1['LATs'].values,
                                                                                    levels=20,
                                                                                    thresh=0.05)

    density, support = kde_estimator._eval_bivariate(ev['LON'], ev['LAT'])
    # sb.kdeplot(y=df[df.hour == 1].LATs, x=df[df.hour == 1].LONs)

    # Generate grid points for visualization
    grid_resolution = 5  # Adjust the resolution as needed
    grid_points = generate_grid_points(lat_min, lat_max, lon_min, lon_max, grid_resolution)

    kde = fit_kde(trip_data=df, bandwidth=0.01)
    visualize_kde(kde, grid_points)

    # Create and fit a Support Vector Data Description (SVDD)
    nu = 0.05  # Adjust the nu parameter as needed
    svdd = OneClassSVM(kernel='rbf', nu=nu)
    svdd.fit(df[['LATs', 'LONs']])

    # Score the grid points using the SVDD
    svdd_scores = -svdd.decision_function(grid_points)

    # Create and fit a Gaussian Mixture Model (GMM)
    n_components = 3  # Adjust the number of components as needed
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(df[['LATs', 'LONs']])
    # Score the grid points using the GMM
    gmm_scores = -gmm.score_samples(grid_points)

    # Coverage using KDE
    kde_cov = kde_coverage(df, control_area, bandwidth, visualize=True)
    print("KDE Coverage:", kde_cov)

    # Optimized EV locations
    optimized_locations = optimize_coverage(df, control_area, n_ev, bandwidth)
    print("Optimized EV Locations:", optimized_locations)
"""