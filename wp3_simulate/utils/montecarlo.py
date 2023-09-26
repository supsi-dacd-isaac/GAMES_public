import pandas as pd
import numpy as np
import time
from colorama import Fore, Style  # Import the colorama library to simplify ANSI escape code usage
import multiprocessing

orange_color, blue_color, cyan_color = Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.CYAN  # Define the color codes

def MonteCarlo_fleet_sizes(RelocationGridCityEnv,
               config_file,
               n_steps=5,
               n_episodes=10,
               fleet_sizes=range(5, 10)):

    """"""
    fleet_conf = config_file['Fleet']
    stations_conf = config_file['Stations']
    mean_booking_scores = []
    df_mc_samples = pd.DataFrame([])

    for nev in fleet_sizes:  # repeat for increasing fleet size
        config_file['Fleet']['sizes'] = [nev, nev, nev, nev, nev]
        random_fleet_scenario(fleet_conf, stations_conf,
                              name_save='df_fleet_scenario_telaviv.csv',
                              save=True)
        n_trips, tot_km, tot_min, step_id, episode_no = [], [], [], [], []
        mean_is_idle, mean_time_to_return_min, mean_idle_time_min, mean_SOC = [], [], [], []
        print(f'{blue_color} EV size 5x{nev}{Style.RESET_ALL}')  # Reset color to default
        for e in range(n_episodes):
            print(f'{orange_color}MC simulation, episode : {e}{Style.RESET_ALL}')  # Reset color to default
            env = RelocationGridCityEnv(config_file=config_file) # re-start the environment
            tic = time.time()
            for i in range(n_steps):
                df_trip_demand_samples = env.sample_trip_demand_scenario()
                df_reservations = env.simulate_reservations(df_trip_demand_samples)
                n_trips.append(len(df_reservations))
                tot_km.append(df_reservations['dis_km'].sum())
                tot_min.append(df_reservations['dur_min'].sum())

                mean_is_idle.append(env.df_fleet.ev_is_idle_state.mean())
                mean_time_to_return_min.append(env.df_fleet.time_to_return_min.mean())
                mean_idle_time_min.append(env.df_fleet.idle_time_min.mean())
                mean_SOC.append(env.df_fleet.SOC.mean())

                step_id.append(i)
                episode_no.append(e)

            toc = time.time()
            print(f'{cyan_color} episode time : {round(toc - tic, 2)}{Style.RESET_ALL} seconds')  # Reset color to default

        df_mc_samples['n' + str(nev)] = n_trips
        df_mc_samples['km' + str(nev)] = tot_km
        df_mc_samples['min' + str(nev)] = tot_min
        df_mc_samples['step_id' + str(nev)] = step_id
        df_mc_samples['episode_no' + str(nev)] = episode_no
        df_mc_samples['fleet_size' + str(nev)] = sum([nev, nev, nev, nev, nev])

        df_mc_samples['mean_is_idle' + str(nev)] = mean_is_idle
        df_mc_samples['mean_time_to_return_min' + str(nev)] = mean_time_to_return_min
        df_mc_samples['mean_idle_time_min' + str(nev)] = mean_idle_time_min
        df_mc_samples['mean_SOC' + str(nev)] = mean_SOC

        mu_n, mu_km, mu_min = np.mean(n_trips), np.mean(tot_km), np.mean(tot_min)
        mu_is_idle, mu_t2return, mu_idle_time, mu_soc = np.mean(mean_is_idle), np.mean(mean_time_to_return_min),\
                                                        np.mean(mean_idle_time_min), np.mean(mean_SOC)

        mean_booking_scores.append([mu_n, mu_km, mu_min, mu_is_idle, mu_t2return, mu_idle_time, mu_soc])

    # Initialize an empty list to store individual data frames
    dfs = []
    for i in fleet_sizes:
        # Create a new DataFrame for each iteration with two columns 'n_booked' and 'sample_no'
        temp_df = pd.DataFrame({
            'n_booked': df_mc_samples['n' + str(i)],
            'km_driven': df_mc_samples['km' + str(i)],
            'min_driven': df_mc_samples['min' + str(i)],
            'step_id': df_mc_samples['step_id' + str(i)],
            'fleet_size': df_mc_samples['fleet_size' + str(i)],
            'episode_no': df_mc_samples['episode_no' + str(i)],
            'mean_is_idle':  df_mc_samples['mean_is_idle' + str(i)],
            'mean_time_to_return_min': df_mc_samples['mean_time_to_return_min' + str(i)],
            'mean_idle_time_min': df_mc_samples['mean_idle_time_min' + str(i)],
            'mean_SOC': df_mc_samples['mean_SOC' + str(i)],
        })

        # Append the temporary DataFrame to the list
        dfs.append(temp_df)

    # Concatenate all the individual DataFrames into a single DataFrame
    return pd.concat(dfs, ignore_index=True), np.array(mean_booking_scores)




def random_fleet_scenario(fleet_conf, stations_conf,
                          name_save='df_fleet_scenario_telaviv.csv',
                          save=False):
    stations_latlon = [[32.04, 34.75], [32.06, 34.765], [32.08, 34.77],
                       [32.12, 34.795], [32.12, 34.83], [32.08, 34.78],
                       [32.06, 34.78], [32.05, 34.8], [32.08, 34.79],
                       [32.10, 34.78], [32.10, 34.79], [32.10, 34.8]]

    stations_latlon = stations_conf['stations_latlon']
    df_stations = pd.DataFrame(stations_latlon, columns=['LAT', 'LON'])

    # Loop through the stations and randomly assign electric vehicles
    fleet_data, EV_ID = [], -1
    n_ev_i = np.array(fleet_conf['sizes'])
    while sum(n_ev_i) > 0:
        car_ids_left = np.where(n_ev_i > 0)[0]
        s_id = np.random.randint(len(df_stations))
        station_row = df_stations.iloc[s_id]
        ev_i = car_ids_left[np.random.randint(len(car_ids_left))]
        n_ev_i[ev_i] -= 1

        model_name = fleet_conf['model_name'][ev_i]
        fuel_type = fleet_conf['fuel_type'][ev_i]
        vehicle_category = fleet_conf['vehicle_category'][ev_i]
        brand_name = fleet_conf['brand_name'][ev_i]
        charge_power = fleet_conf['charge_power'][ev_i]
        battery_capacity = fleet_conf['battery_capacity'][ev_i]
        range_km = fleet_conf['range'][ev_i]
        EV_ID += 1
        # Append the assigned vehicle to the fleet DataFrame
        fleet_data.append({'station_no': s_id, 'vehicle_no': EV_ID, 'ev_tyoe_id': ev_i,
                           'LAT': station_row['LAT'], 'LON': station_row['LON'],
                           'model_name': model_name, 'fuel_type': fuel_type,
                           'vehicle_category': vehicle_category, 'brand_name': brand_name,
                           'charge_power': charge_power, 'SOC': 100,
                           'battery_capacity': battery_capacity, 'range': range_km})

    # Create the fleet DataFrame by converting the list of dictionaries
    df_fleet = pd.DataFrame(fleet_data)

    if save:
        df_fleet.to_csv('data/' + name_save, index=False)  # Save fleet data to a CSV file
        df_stations.to_csv('data/df_charging_stations_telaviv.csv', index=False)  # Save fleet data to a CSV file

    return df_fleet, df_stations

def run_simulation(nev, RelocationGridCityEnv, config_file, n_steps, n_episodes):
    n_trips, tot_km, tot_min, step_id = [], [], [], []
    env = RelocationGridCityEnv(config_file=config_file)  # re-start the environment

    config_file['Fleet']['sizes'] = [nev, nev, nev, nev, nev]
    random_fleet_scenario(config_file['Fleet'], config_file['Stations'],
                          name_save='df_fleet_scenario_telaviv.csv', save=True)


    print(f'{blue_color} EV size 5x{nev}{Style.RESET_ALL}')  # Reset color to default
    for e in range(n_episodes):
        print(f'{orange_color}MC simulation, episode : {e}{Style.RESET_ALL}')  # Reset color to default
        env.reset()
        tic = time.time()
        for i in range(n_steps):
            df_trip_demand_samples = env.sample_trip_demand_scenario()
            df_reservations = env.simulate_reservations(df_trip_demand_samples)
            n_trips.append(len(df_reservations))
            tot_km.append(df_reservations['dis_km'].sum())
            tot_min.append(df_reservations['dur_min'].sum())
            step_id.append(i)
        toc = time.time()
        print(f'{cyan_color} episode time : {round(toc - tic, 2)}{Style.RESET_ALL} seconds')  # Reset color to default

    return nev, n_trips, tot_km, tot_min, step_id

def MonteCarlo_fleet_sizes_parallel(RelocationGridCityEnv,
                                     config_file,
                                     n_steps=5,
                                     n_episodes=10,
                                     fleet_sizes=range(5, 10),
                                     num_processes=4):
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Use multiprocessing to run simulations in parallel
    results = pool.starmap(run_simulation, [(nev, RelocationGridCityEnv, config_file, n_steps, n_episodes) for nev in fleet_sizes])

    # Close the pool to free up resources
    pool.close()
    pool.join()

    # Process the results
    fleet_data = {}
    mean_booking_scores = []
    for nev, n_trips, tot_km, tot_min, step_id in results:
        fleet_data[nev] = {
            'n_trips': n_trips,
            'tot_km': tot_km,
            'tot_min': tot_min,
            'step_id': step_id,
            'fleet_size': sum([nev] * 5)
        }

        mu_n, mu_km, mu_min = np.mean(n_trips), np.mean(tot_km), np.mean(tot_min)
        mean_booking_scores.append([mu_n, mu_km, mu_min])

    return fleet_data, np.array(mean_booking_scores)
