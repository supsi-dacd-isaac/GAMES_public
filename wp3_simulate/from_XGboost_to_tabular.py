from multiprocessing import Pool
import matplotlib.pyplot as plt
import csv
from trip_sampler import *
from scipy.sparse import csc_matrix
from wp3_simulate.sim.simulate.car_sharing_patterns import (assign_CarSharing_mode_from_table,
                                                            assign_FAST_CarSharing_mode_from_table,
                                                            assign_mode)
# load data
acts_path = 'data/users/trips_features_from_Zurich_with_distances_and_decision_times.csv'  # user trips with decision times
acts_gdf_mode = sim.simulate.car_sharing_patterns.load_trips(acts_path)
acts_gdf_mode = acts_gdf_mode.sort_values("mode_decision_time")  # sort by mode decision time, not by person

stations_path = "data/station_scenario/Zurich_231.csv"
stations_gdf = sim.simulate.car_sharing_patterns.load_station_scenario(stations_path)

# load the tabular choice_model and xgboost choice model
xgboost_model_path = "sim/trained_models/mode_choice/xgb_model.p"
with open(xgboost_model_path, "rb") as infile:  # load the trained model for user choice of transportation mode:
    mode_choice_model = pickle.load(infile)

mode_choice_table_path = "sim/trained_models/mode_choice/sparse_mode_choice_table.csv"
with open(mode_choice_table_path, "rb") as infile:  # load the trained model for user choice of transportation mode:
    tabular_mode_choice = pickle.load(infile)

X_baseline = stations_gdf['n_cars']
# Define functions for the simulation
def assign_distribution(X, n_stations: int = 231, fleet_size: int = 495):  # max_ev_per_station: int =20
    """ assign X=(n1,n2,....,n_stations) cars to each one of the ns stations"""
    ev_list = [i for i in range(fleet_size)]
    ev_distribution_list = [[] for _ in range(n_stations)]
    X = X.astype(int)
    for id, x in enumerate(X):
        ev_distribution_list[id] = ev_list[:int(x)]
        ev_list = ev_list[int(x):]  # remove assigned vehicles
    return X, ev_distribution_list


def save_mode_choice_table(n_dist_discrete: int = 251, max_dist=25_000, min_dist=0):
    """ -------------- prepare mode_choice table -------------- """
    DISTANCES = np.linspace(min_dist, max_dist, n_dist_discrete)
    sparse_mode_choice_table = csc_matrix((len(acts_gdf_mode), n_dist_discrete), dtype=np.int8).toarray()
    count = 0
    for idx, row in acts_gdf_mode.iterrows():
        mode_row = []
        for d in DISTANCES:
            row["feat_distance_to_station_origin"] = d
            if mode_choice_model(row) == "Mode::CarsharingMobility":
                mode_row.append(1)
            else:
                mode_row.append(0)
        sparse_mode_choice_table[count, :] = mode_row
        count += 1
        print('\033[94m trips done: ' + str(count) + '/' + str(len(acts_gdf_mode)) + '\033[0m')
    # Create the DataFrame with row indices and column headers
    df_mode_choices = pd.DataFrame(sparse_mode_choice_table, index=acts_gdf_mode.index, columns=DISTANCES)
    df_mode_choices.to_pickle('sparse_mode_choice_table.csv')
    return df_mode_choices


def simulate_carsharing_reservations_table(x):
    st = time.time()
    stations_gdf['n_cars'], stations_gdf['vehicle_list'] = assign_distribution(x)
    acts_gdf_simulated = assign_CarSharing_mode_from_table(acts_gdf_mode, stations_gdf, tabular_mode_choice)
    sim_reservations = sim.simulate.car_sharing_patterns.derive_reservations(acts_gdf_simulated)
    elapsed_time = time.time() - st
    print('ELAPSED TIME: ', elapsed_time, '   SECONDS')
    Reward_activation, Reward_booking_duration, Reward_drive_rate = 25, 3, 0.65  # CHF/event# CHF/hour# CHF/km
    return -(np.shape(sim_reservations)[0] * Reward_activation +
             sim_reservations['drive_km'].sum() * Reward_drive_rate +
             sim_reservations['duration'].sum() * Reward_booking_duration)


def simulate_FAST_carsharing_reservations_from_table(x):
    st = time.time()
    stations_gdf['n_cars'], stations_gdf['vehicle_list'] = assign_distribution(x)
    acts_gdf_simulated = assign_FAST_CarSharing_mode_from_table(acts_gdf_mode, stations_gdf, tabular_mode_choice)
    sim_reservations = sim.simulate.car_sharing_patterns.derive_reservations(acts_gdf_simulated)
    elapsed_time = time.time() - st
    print('ELAPSED TIME: ', elapsed_time, '   SECONDS')
    Reward_activation, Reward_booking_duration, Reward_drive_rate = 25, 3, 0.65  # CHF/event# CHF/hour# CHF/km
    return -(np.shape(sim_reservations)[0] * Reward_activation +
             sim_reservations['drive_km'].sum() * Reward_drive_rate +
             sim_reservations['duration'].sum() * Reward_booking_duration)


def simulate_carsharing_reservations_xgboost(x):
    st = time.time()
    stations_gdf['n_cars'], stations_gdf['vehicle_list'] = assign_distribution(x)
    acts_gdf_simulated = assign_mode(acts_gdf_mode, stations_gdf, mode_choice_model)
    sim_reservations = sim.simulate.car_sharing_patterns.derive_reservations(acts_gdf_simulated)
    elapsed_time = time.time() - st
    print('ELAPSED TIME: ', elapsed_time, '   SECONDS')
    Reward_activation, Reward_booking_duration, Reward_drive_rate = 25, 3, 0.65  # CHF/event# CHF/hour# CHF/km
    negative_revenue = -(np.shape(sim_reservations)[0] * Reward_activation +
                         sim_reservations['drive_km'].sum() * Reward_drive_rate +
                         sim_reservations['duration'].sum() * Reward_booking_duration)
    return negative_revenue


def simulate_random_relocations_tabular_vs_xgboost(i):
    n_source_stations = 20
    from_st, to_st, n_relocated_cars = np.random.randint(0, n_source_stations), \
                                       np.random.randint(n_source_stations, len(X_baseline)), \
                                       np.random.randint(0, min(X_baseline[:n_source_stations]))
    X_relocated = X_baseline.copy()
    X_relocated.iloc[from_st] -= n_relocated_cars
    X_relocated.iloc[to_st] += n_relocated_cars
    apprx_reward = -simulate_FAST_carsharing_reservations_from_table(X_relocated)
    xgb_reward = -simulate_carsharing_reservations_xgboost(X_relocated)
    return apprx_reward, xgb_reward


if __name__ == '__main__':
    # save_mode_choice_table
    save_dir = "wp3_simulate/data/analyze_emulator_errors"

    """
    num_processes = 4  # Number of parallel processes
    num_iterations = 2

    with Pool(num_processes) as pool:
            results = pool.map(simulate_random_relocations_tabular_vs_xgboost, range(num_iterations))
    apprx_rewards, xgb_rewards = zip(*results)
    """

    # Initialize empty lists to store the loaded data
    apprx_rewards ,xgb_rewards = [], []

    filename = 'data/analyze_emulator_errors/lists_tabular_vs_xgboost.csv'
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip the header row
            if row[0].startswith('List'):
                continue
            # Extract the data from each row
            list_name = row[0]
            values = list(map(float, row[1:]))
            # Assign the values to the appropriate list based on the list name
            if list_name == 'List 1':
                apprx_rewards = values
            elif list_name == 'List 2':
                xgb_rewards = values

    for i in range(25):
        approx_r, xgb_r = simulate_random_relocations_tabular_vs_xgboost(i)
        apprx_rewards.append(approx_r)
        xgb_rewards.append(xgb_r)
    normalized_abs_errors = [abs(j - i) / j for i, j in zip(apprx_rewards, xgb_rewards)]

    plt.subplots(figsize=(10, 10))
    plt.scatter(apprx_rewards, xgb_rewards)
    plt.xlabel('revenue booking - tabular approximation')
    plt.ylabel('revenue booking - xgboost model')
    plt.grid()
    plt.show()
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['List 1'] + apprx_rewards)
        writer.writerow(['List 2'] + xgb_rewards)
        writer.writerow(['List 3'] + normalized_abs_errors)

    """
    apprx_rewards, xgb_rewards = [], []
    for i in range(20):
        # Shuffle the list to randomize the order
        from_st, to_st, n_relocated_cars = np.random.randint(0, 10), np.random.randint(20, 100), np.random.randint(1, 5)
        X_relocated = X_baseline.copy()
        X_relocated.iloc[from_st] = X_relocated.iloc[from_st] - n_relocated_cars
        X_relocated.iloc[to_st] = X_relocated.iloc[to_st] + n_relocated_cars

        apprx_rewards.append(-simulate_FAST_carsharing_reservations_from_table(X_relocated))
        xgb_rewards.append(-simulate_carsharing_reservations_xgboost(X_relocated))"""

