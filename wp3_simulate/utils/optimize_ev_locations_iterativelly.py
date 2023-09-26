import numpy as np
import pandas as pd
import seaborn as sb
import pulp as pl
import matplotlib.pyplot as plt
from mclp import *



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

def show_covered_area(x_ev_rad, x_d_rad, acceptance_radius=300):
    # Objective function to minimize coverage given a distribution and EV locations

    lat_targets = x_d_rad[:, 1]
    lon_targets = x_d_rad[:, 0]

    lat_targets = lat_targets[:, np.newaxis]  # Convert to column vector
    lon_targets = lon_targets[:, np.newaxis]  # Convert to column vector

    lat_diff = lat_targets - x_ev_rad[:, 1]  # Compute latitude differences
    lon_diff = lon_targets - x_ev_rad[:, 0]  # Compute longitude differences

    # Compute the matrix of pairwise distances using the Haversine formula
    AVG_EARTH_RADIUS = 6_371_000
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_targets) * np.cos(x_ev_rad[:, 1]) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = AVG_EARTH_RADIUS * c
    is_covered = distances <= acceptance_radius

    mean_coverage = np.mean(np.any(is_covered, axis=1))

    print('great coverage', mean_coverage)
    plt.scatter(lon_targets, lat_targets, alpha=0.05, c='b', label='demand points ~ f_dem(x)')
    plt.scatter(lon_targets[np.any(is_covered, axis=1)], lat_targets[np.any(is_covered, axis=1)], alpha=0.15, c='r',
                label='covered')
    plt.scatter(x_ev_rad[:, 0], x_ev_rad[:, 1], c='k',
                label=' x_optimized')
    plt.grid()
    plt.legend()
    plt.xlabel('LON')
    plt.ylabel('LAT')
    plt.show()

    return mean_coverage

# Function to perform Linear optimization (coverage) for a given number of EVs
def optimize_pulp_min_zeta(x_d_rad, n_ev, half_manhattan_max, lon_min, lat_min, lon_max, lat_max):
    # try rescaling for numerical stability
    x_d = x_d_rad
    # Convert latitude and longitude bounds to radians
    lon_min_rad, lat_min_rad = np.radians(lon_min), np.radians(lat_min)
    lon_max_rad, lat_max_rad = np.radians(lon_max), np.radians(lat_max)
    # Get the number of demand points
    n_d = len(x_d)
    # Create a PuLP problem
    prob = pl.LpProblem("EV_Location_Optimization", pl.LpMinimize)
    # Create decision variables for centers, zeta, and min_zeta with bounds
    center_x = {}
    center_y = {}
    zeta = {}
    min_zeta = {}
    for i in range(n_ev):
        center_x[i] = pl.LpVariable(f"center_x_{i}", lowBound=0.9 * lon_min_rad, upBound=1.1 * lon_max_rad,
                                    cat='Continuous')
        center_y[i] = pl.LpVariable(f"center_y_{i}", lowBound=0.9 * lat_min_rad, upBound=1.1 * lat_max_rad,
                                    cat='Continuous')
        min_zeta[i] = pl.LpVariable(f"min_zeta_{i}", lowBound=0, cat='Continuous')
        for j in range(n_d):
            zeta[(i, j)] = pl.LpVariable(f"zeta_{i}_{j}", lowBound=0, cat='Continuous')

    # Define the objective function to minimize the sum of min_zeta
    prob += pl.lpSum(min_zeta[i] for i in range(n_ev))
    # Define the constraints using L1 norm and set min_zeta values
    for i in range(n_ev):
        for j in range(n_d):
            dist_expr_x = center_x[i] - x_d[j][0]
            dist_expr_y = center_y[i] - x_d[j][1]
            prob += dist_expr_x <= half_manhattan_max + zeta[(i, j)]
            prob += dist_expr_x >= -half_manhattan_max - zeta[(i, j)]
            prob += dist_expr_y <= half_manhattan_max + zeta[(i, j)]
            prob += dist_expr_y >= -half_manhattan_max - zeta[(i, j)]

    for j in range(n_d):
        prob += min_zeta[j] <= pl.lpSum(zeta[(i, j)] for i in range(n_ev))
    # Solve the PuLP problem
    prob.solve()
    # Extract the optimal values
    optimal_centers = np.zeros((n_ev, 2))
    optimal_zeta = np.zeros((n_ev, n_d))
    for i in range(n_ev):
        for j in range(n_d):
            optimal_zeta[i][j] = zeta[i, j].varValue
        optimal_centers[i][0] = center_x[i].varValue
        optimal_centers[i][1] = center_y[i].varValue

    return optimal_centers, optimal_zeta



if __name__ == '__main__':
    df = pd.read_pickle(
        r'C:\Users\roberto.rocchetta.in\OneDrive - SUPSI\Desktop\GAMES project\Codes\gym_relocation\data\autotel\df_TelAviv_for_gym_episodes.pkl')

    lat_lon_data = df[['LATs', 'LONs']]
    data = np.array(lat_lon_data[['LONs', 'LATs']])
    data_points = np.radians(data)

    lon_min, lon_max = 34.72, 34.87
    lat_min, lat_max = 32.01, 32.15
    lon_min_rad, lon_max_rad, lat_min_rad, lat_max_rad = map(np.radians, [lon_min, lat_min, lon_max, lat_max])
    ev = pd.DataFrame()
    ev['LAT'] = np.random.uniform(lat_min, lat_max, 100)
    ev['LON'] = np.random.uniform(lon_min, lon_max, 100)

    # Number of sites to select
    K = 30
    # Service radius of each site
    manhattan_max = 0.0001
    # Candidate site size (random sites generated)
    M = 500
    # Run mclp opt_sites is the location of optimal sites and f is the points covered
    opt_sites, f = mclp(data_points[:1500], K, manhattan_max, M)
    plot_result(data_points[:1500], opt_sites, manhattan_max)
    plt.show()
    """ ----------------- """
    """  try with binary linear programming """
    """ ----------------- """
    radiant_scaler=1.0
    n_ev = 5  # Number of geometric centers
    n_d = 100  # Number of demand points
    manhattan_max = 0.0001
    nd_rge = range(n_d)
    x_demand = data_points[:n_d,:]
    # Create a linear programming problem
    model = pl.LpProblem("Rectangle_Covering_Maximization", pl.LpMinimize)

    # Decision variables
    # Coordinates of the EVs
    # Create decision variables for centers and zeta with bounds
    center_coords_var_x = {}
    center_coords_var_y = {}
    zeta = {}
    for i in range(n_ev):
        center_coords_var_x[i] = pl.LpVariable(f"center_x_{i}", lowBound=lon_min_rad * radiant_scaler * 0.8,
                                    upBound=lon_max_rad * radiant_scaler * 1.2, cat='Continuous')
        center_coords_var_y[i] = pl.LpVariable(f"center_y_{i}", lowBound=lat_min_rad * radiant_scaler * 0.8,
                                    upBound=lat_max_rad * radiant_scaler * 1.2, cat='Continuous')

    # Binary variable indicating whether a demand point i is covered by the rectangle centered on the EVs j
    ev_covers_demand_var = [[pl.LpVariable(f"ev_{i}_covers_demand_{j}", cat=pl.LpBinary) for j in nd_rge] for i in range(n_ev)]
    # Binary variable indicating whether a demand point i is covered by any vehicle
    demand_cover_vars = [pl.LpVariable(f"demand_{i}_is_covered", cat=pl.LpBinary) for i in nd_rge]


    # Objective function: Maximize the number of covered demand points
    model += pl.lpSum(demand_cover_vars[i] for i in nd_rge), "Objective maximize covered demand"

    for i in range(n_ev):
        for j in nd_rge:
            # Constraints to ensure that the binary variable indicates whether a center covers a demand point
            model += ev_covers_demand_var[i][j] >= demand_cover_vars[j], f"min_coverage_{i}_{j}"


    for i in range(n_ev):
        for j in range(n_d):
            diff_x = center_coords_var_x[i] - x_demand[j][0]
            diff_y = center_coords_var_y[i] - x_demand[j][1]

            model += diff_x <= manhattan_max + 1e3 * (1 - ev_covers_demand_var[i][j])
            model += diff_y <= manhattan_max + 1e3 * (1 - ev_covers_demand_var[i][j])
            model += -diff_x <= manhattan_max + 1e3 * (1 - ev_covers_demand_var[i][j])
            model += -diff_y <= manhattan_max + 1e3 * (1 - ev_covers_demand_var[i][j])

    # Solve the linear programming problem
    model.solve()

    centers = np.zeros((n_ev, 2))
    cover_ij = np.zeros((n_ev, n_d))
    for i in range(n_ev):
        centers[i, 0] = center_coords_var_x[i].varValue
        centers[i, 1] = center_coords_var_y[i].varValue
        for j in range(n_d):
            cover_ij[i][j] = ev_covers_demand_var[i][j].varValue


    plt.figure(figsize=(8, 6))
    is_covered =np.min(cover_ij, axis=0)==1
    plt.scatter(x_demand[:, 0], x_demand[:, 1], label='Demand Points', marker='x', color='blue', alpha=0.1)
    plt.scatter(x_demand[is_covered, 0], x_demand[is_covered, 1], label='Demand Points', marker='d', color='red', alpha=0.3)
    plt.scatter(centers[:, 0], centers[:, 1], 20, label='Centers', marker='o', color='red')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Centers and Demand Points')
    plt.legend()
    plt.grid(True)
    plt.show()


    """ ----------------- """
    """  Create a PuLP optimal placement problem for a fleet with 3 EVs """
    """ ----------------- """

    n_ev = 3
    radiant_scaler = 1
    manhattan_max = 0.0001
    n_d = 20
    x_demand = data_points[:n_d] * radiant_scaler
    prob = pl.LpProblem("EV_Location_Optimization", pl.LpMinimize)
    # Create decision variables for centers, zeta, and min_zeta with bounds

    # Create decision variables for centers and zeta with bounds
    center_x = {}
    center_y = {}
    zeta = {}

    for i in range(n_ev):
        center_x[i] = pl.LpVariable(f"center_x_{i}", lowBound=lon_min_rad * radiant_scaler * 0.8,
                                    upBound=lon_max_rad * radiant_scaler * 1.2, cat='Continuous')
        center_y[i] = pl.LpVariable(f"center_y_{i}", lowBound=lat_min_rad * radiant_scaler * 0.8,
                                    upBound=lat_max_rad * radiant_scaler * 1.2, cat='Continuous')

        for j in range(n_d):
            zeta[(i, j)] = pl.LpVariable(f"zeta_{i}_{j}", lowBound=0,  cat='Continuous')
            zeta[(i, j)].varValue = 1.0  # Set the initial value to 1.0

    # prob += pl.lpSum(min_zeta[j] for j in range(n_d))
    prob += pl.lpSum(zeta[(i, j)] for i in range(n_ev) for j in range(n_d))
    # Define the constraints using L1 norm and set min_zeta values

    for j in range(n_d):
        demand_covered = pl.lpSum(zeta[i, j] for i in range(n_ev))
        prob += demand_covered >= 1  # At least one EV covers the demand point

    for i in range(n_ev):
        for j in range(n_d):
            dist_expr_x_ij = center_x[i] - x_demand[j][0]
            dist_expr_y_ij = center_y[i] - x_demand[j][1]
            prob += dist_expr_x_ij <= 0.5 * manhattan_max + zeta[i, j]
            prob += dist_expr_x_ij >= -0.5 * manhattan_max - zeta[i, j]
            prob += dist_expr_y_ij <= 0.5 * manhattan_max + zeta[i, j]
            prob += dist_expr_y_ij >= -0.5 * manhattan_max - zeta[i, j]


    # Solve the PuLP problem
    prob.solve()

    # Extract the optimal values

    optimal_centers = np.zeros((n_ev, 2))
    optimal_zeta = np.zeros((n_ev, n_d))

    c1 = []
    manhattan_distances = []
    for i in range(n_ev):
        for j in range(n_d):
            optimal_zeta[i][j] = zeta[i, j].varValue
        optimal_centers[i][0] = center_x[i].varValue
        optimal_centers[i][1] = center_y[i].varValue

    c1 = []
    manhattan_distances = []
    for i in range(n_ev):
        for j in range(n_d):
            dist_expr_lon = optimal_centers[i][0] - x_demand[j][0]
            dist_expr_lat = optimal_centers[i][1] - x_demand[j][1]
            Manhattan_distance = abs(dist_expr_lon) + abs(dist_expr_lat)
            manhattan_distances.append(Manhattan_distance)
            # print(Manhattan_distance)
            c1.append(Manhattan_distance <= manhattan_max + optimal_zeta[i, j])

    plt.scatter(x_demand[:, 0], x_demand[:, 1], c='b', alpha=0.1, label='demand points')
    plt.scatter(optimal_centers[0, :], optimal_centers[1, :], c='r', label='optimal EV coord')
    plt.legend()
    plt.grid
    plt.show()

    """ try multple sizes """

    num_evs_range = [2, 3, 4, 5, 30, 100]  # Adjust as needed
    # Iterate over different numbers of EVs
    for num_evs in num_evs_range:
        """optimal_centers, optimal_zeta = optimize_for_num_evs_pulp(x_d=x_demand,
                                                                  n_ev=num_evs, r_dist=radius_fixed,
                                                                  lon_min=lon_min, lon_max=lon_max,
                                                                  lat_min=lat_min, lat_max=lat_max)"""
        manhattan_max = 0.001
        x_d = np.array(lat_lon_data[['LONs', 'LATs']])[:, n_d]
        optimal_centers, optimal_zeta = optimize_for_num_evs_pulp(x_d=x_d, n_ev=num_evs,
                                                                  half_manhattan_max=manhattan_max, lon_min=lon_min,
                                                                  lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)

        # Print the results for each number of EVs
        print(f"Optimal Centers for {num_evs} EVs:")
        print(optimal_centers)
        print("Optimal Zeta:")
        print(optimal_zeta)
        # visualize
        show_covered_area(optimal_centers, x_demand, acceptance_radius=300)
