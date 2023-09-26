import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import gymnasium as gym
from gymnasium import spaces
from models.macroscopic_traffic_model import Macroscopic_traffic_model
from utils.functions import data_loader, SOC_end_of_trip, distance_start_end, allocate_stations
from utils.visualizers import plot_close_evs
from sklearn.metrics.pairwise import haversine_distances as h_dist
from utils.monte_carlo import *


class RelocationGridCityEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}  # fixme: update metadata

    def __init__(self, config_file, load_Gtr=False, render_mode=None):
        # Initialize class variables and configuration parameters
        self.conf_env = config_file['Environment']
        self.Gtr = self.load_traffic_model() if load_Gtr else None
        self.load_trips_data()
        # get matrix of average departures [24x7] hours x week_day
        self.mean_matrix_trips_per_day_hour = self._get_mean_daily_hourly_departures()
        self.initialize_stations()
        self.initialize_fleet()
        self.day_week_now = self.conf_env['episode_day_start']  # start day of the week
        self.hour_now = self.conf_env['episode_hour_start']  # start hour is 00:00
        self.steps_counter = 0
        self.convert_coords_to_rads()  # ony when init_ the environment
        ## todo: define actions and observation spaces
        self.action_space = spaces.MultiBinary(len(self.df_fleet)+1, seed=42)
        # observations = [time_stamp_episode, SOCs, ev_locations, distance_ev_2_closest_station, idle_times_evs,...., demand_mobilityforecast?]
        # todo: add a function mapping actions to changes in the EV locatio

    def convert_coords_to_rads(self):
        # Convert latitude and longitude from degrees to radians
        self.df_trips[['LATs', 'LONs', 'LATe', 'LONe']] = np.radians(self.df_trips[['LATs', 'LONs', 'LATe', 'LONe']])
        self.df_fleet[['LAT', 'LON']] = np.radians(self.df_fleet[['LAT', 'LON']])
        self.df_stations[['LAT', 'LON']] = np.radians(self.df_stations[['LAT', 'LON']])

    def _get_mean_daily_hourly_departures(self):
        X = []
        for dw in range(7):
            X.append([len(self.df_trips[(self.df_trips['hour'] == i)
                                        & (self.df_trips['dw'] == dw)]) / 52 for i in range(24)])
        return np.array(X).astype(int)  # [day x hour]

    def load_traffic_model(self):
        return Macroscopic_traffic_model(geographical_area=self.conf_env['query_city'])

    def load_trips_data(self):
        self.df_trips = data_loader(data_dir=self.conf_env['data_dir'], file_name=self.conf_env['df_demand_name'])
        self.df_trips = self.df_trips.drop(columns=['dy', 'dm', 'quarter', 'is_holy'])  # drop some stuff not needed now
        self.df_trips = self.df_trips[self.df_trips.index.year == self.conf_env['year']]  # let's work on 1 year
        self.df_trips['dis_km'] = self.df_trips['dis_km'].astype(float)


    def initialize_fleet(self):
        self.df_fleet = pd.read_csv('data/df_fleet_scenario_telaviv.csv')
        self.df_fleet['ev_is_idle_state'] = True  # all EVs are available when we start the simulation
        self.df_fleet[['range', 'SOC']] = self.df_fleet[['range', 'SOC']].astype(float)
        self.df_fleet['time_to_return_min'] = 0.0
        self.df_fleet['idle_time_min'] = 300.0 # fixme: we are assuming 5 hours idle time when the episode starts

    def initialize_stations(self):
        self.df_stations = pd.read_csv('data/df_charging_stations_telaviv.csv')

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed)  # We need the following line to seed self.np_random
        # todo implement reset method
        self.initialize_fleet()
        self.day_week_now = self.conf_env['episode_day_start']  # start day of the week
        self.hour_now = self.conf_env['episode_hour_start']  # start hour is 00:00
        self.steps_counter = 0

    def step(self, actions):
        # Implement the environment's step function
        """ 1 - apply relocation actions """
        # Update the state based on the action
        self.apply_relocation(actions)

        """ 2 - Simulate a random demand scenario for this step, get reservation data frame  """
        df_mobility_demand_samples = self.sample_trip_demand_scenario()
        df_res_step = self.simulate_reservations(df_mobility_demand_samples)

        """ 3- get new set of observations """
        observations = self._get_obs()

        """ 4- calculate reward score """
        # Define a reward based on the state (e.g., maximize a value)
        reward = self._get_step_reward(df_res_step, actions)

        """ 5- check if episode is completed """
        self.steps_counter += 1
        # Check if the episode is done (e.g., a terminal state is reached)
        if self.steps_counter >= self.conf_env['episode_max_steps']:
            done = True
        else:
            done = False
        info = self._get_info()
        return observations, reward, done, info

    def _get_step_reward(self, df_res, actions):
        # todo calculate reward (relocation_cost (a_tr) + fueling_cost (a_el?) + reward_booking ) reset method
        reward_total_km = df_res['dis_km'].sum() * self.conf_env['reward_booking_km']
        reward_total_minutes = df_res['dis_km'].sum() * self.conf_env['reward_booking_min']
        reward_total_booking_events = len(df_res['dis_km']) * self.conf_env['reward_booking_event']
        reward_transportation = reward_total_km + reward_total_minutes + reward_total_km + reward_total_booking_events
        cost_relocation = self.conf_env['cost_relocation']*sum(actions) #todo implement action cost ----...
        total_step_reward = reward_transportation - cost_relocation
        return total_step_reward

    def apply_relocation(self, actions):

        # todo:
        #  0) first attempt, lets try with only binary actions (0= do nothing, 1 = relocate)
        #  if 1 --> relocate to the closes charging station
        #      assign SOC -> 100% after x_hours hours
        #      assign self.df_fleet['time_to_return_min'] = 300 # assume we need 5 hours to relocate and charge the EV?

        pass

    def time_step(self):
        self.hour_now += 1
        if self.hour_now == 24:
            self.hour_now = 0
            self.day_week_now += 1
        if self.day_week_now == 7:
            self.day_week_now = 0

    def sample_trip_demand_scenario(self):
        trip_samples = []  # sample demand for trips from the df_trips e.g. 1 day
        for hr in range(self.conf_env['step_number_of_hours']):  # Iterate for hours in a step
            self.time_step()  # Advance the time step
            # Filter trips for the current day of the week and hour
            df_trip_hour_dw = self.df_trips[
                (self.df_trips['dw'] == self.day_week_now)
                & (self.df_trips['hour'] == self.hour_now)]
            n_samples = self.mean_matrix_trips_per_day_hour[self.day_week_now, self.hour_now]  # Get the desired number of samples based on a matrix
            trip_samples.append(df_trip_hour_dw.sample(n_samples))  # Sample trips and append to the list
        # Create a DataFrame from the sampled trips
        df_trip_sam = pd.concat(trip_samples, axis=0, ignore_index=True)
        df_trip_sam = df_trip_sam.sort_values(by=['dw', 'hour', 'minute'],
                                              ascending=[True, True, True],
                                              ignore_index=True)
        df_trip_sam['total_minutes'] = df_trip_sam['dw'] * 24 * 60 + df_trip_sam[
            'hour'] * 60 + df_trip_sam['minute']  # Convert day_week and hours to minutes
        # Calculate the time difference in minutes between consecutive rows
        df_trip_sam['time_difference'] = df_trip_sam['total_minutes'].diff()
        df_trip_sam = df_trip_sam.drop('total_minutes', axis=1)  # Drop the 'total_minutes' column
        df_trip_sam['Mode_choice'] = 'Others'
        return df_trip_sam

    def simulate_reservations(self, df_trip_sam):
        """ Start looping trips --- this MUST RUN FAST"""
        for index_trip, trip_i in df_trip_sam.iterrows():
            lat1, lon1, dt = trip_i['LATs'], trip_i['LONs'], trip_i['time_difference']
            """ 2) compute evs-to-user distance """
            distance = distance_start_end(lat1, lon1, self.df_fleet['LAT'].values,
                                                      self.df_fleet['LON'].values)
            # consider for rendering - display
            #plot_close_evs(df_trip_sam, self.df_fleet, lat1, lon1, distance, self.conf_env['acceptance_radius'])
            """ 3) Compute time-to-return and update ev_is_idle_state """
            if index_trip == 0:
                pass
            else:  # recompute time-to-return
                self.df_fleet['time_to_return_min'] = np.maximum(0, self.df_fleet['time_to_return_min'] - dt)
                self.df_fleet['idle_time_min'] = self.df_fleet['idle_time_min'] + dt
                self.df_fleet['ev_is_idle_state'] = self.df_fleet['time_to_return_min'] == 0

            """ 4) Compute mode_choice and ev_choice"""
            mode_choice, ev_choice = self.mode_choice(distance)
            df_trip_sam['Mode_choice'].iat[index_trip] = mode_choice

            if mode_choice == 'CarSharing':
                # todo  3) if car-sharing is accepted, re-assign ev position, state = unavailable until trip is completed
                DRIVE_RANGE = self.df_fleet.at[ev_choice, 'range']
                SOC_s = self.df_fleet.at[ev_choice, 'SOC']
                self.df_fleet.at[ev_choice, 'LAT'] = trip_i['LATe']
                self.df_fleet.at[ev_choice, 'LON'] = trip_i['LONe']
                self.df_fleet.at[ev_choice, 'SOC'] = SOC_end_of_trip(trip_i['dis_km'], SOC_s, DRIVE_RANGE)
                self.df_fleet.at[ev_choice, 'time_to_return_min'] = trip_i['dur_min']  # reset time to return
                self.df_fleet.at[ev_choice, 'idle_time_min'] = 0.0  # reset idle time
                self.df_fleet.at[ev_choice, 'ev_is_idle_state'] = False
        return df_trip_sam[df_trip_sam['Mode_choice'] == 'CarSharing']

    def mode_choice(self, distance_user_to_evs):
        acceptance_radius = self.conf_env['acceptance_radius']  # in meters "as the crow flies / on straight line"
        min_soc_accepted = self.conf_env['min_soc_accepted']  # in %
        is_close = (distance_user_to_evs < acceptance_radius)
        is_charged_and_idle = np.logical_and(self.df_fleet['SOC'].values >= min_soc_accepted, self.df_fleet[
            'ev_is_idle_state'].values)  # assume <= 15% charged --> lack of trust by the user
        ev_is_choosable = np.logical_and(is_close, is_charged_and_idle)
        if np.any(ev_is_choosable):
            mode_choice = 'CarSharing'
            ev_selected = self.df_fleet['SOC'].argmax()  # if within the radius, pick the highest SOC
        else:
            mode_choice = 'Others'
            ev_selected = np.nan
        return mode_choice, ev_selected

    def render(self):
        # todo: Implement rendering logic
        pass

    def _get_info(self):
        info = {}  # todo: Get episode information
        return info

    def _get_obs(self):
        observations = []  # todo: extract information needed for the agent to take actions (SOC, IDLE_TIME, DEMAND_)
        return observations

if __name__ == '__main__':
    config_file = {}

    env_conf = {'query_city': 'Tel Aviv, Israel',
                'df_demand_name': 'df_TelAviv_for_gym_episodes.pkl',
                'ev_fleet_scenario_name': 'df_fleet_scenario_telaviv.pkl',
                'station_scenario_name': 'df_charging_stations_telaviv.csv',
                'data_dir': 'data/autotel',
                'year': 2021,
                'episode_day_start': 0,
                'episode_hour_start': 0,
                'episode_max_steps': 168,  # 168 hours in a week
                'step_number_of_hours': 1,
                'acceptance_radius': 300,  # assume max 300 meters walk to grab the car
                'min_soc_accepted': 20,  # assume lower that 20% is not trusted
                'reward_booking_km': 1,
                'reward_booking_min': 1.7,  # 1.2 or 1.7 depending on the contract in TelAviv
                'reward_booking_event': 5,
                'cost_relocation': 20}

    fleet_conf = {
        'model_name': ['i10', 'ZOE 400', 'Picanto', 'Kona', 'Tesla model s 90'],
        'fuel_type': ['electric', 'electric', 'electric', 'electric', 'electric'],
        'vehicle_category': ['Eco', 'Eco', 'Eco', 'Eco', 'Luxury'],
        'brand_name': ['Eco', 'Renault', 'Kia', 'Hyundai', 'Tesla'],
        'sizes': [10, 10, 10, 10, 5],  # number of ev in the fleet - per model class
        'charge_power': [22, 22, 22, 39.2, 16.5],
        'battery_capacity': [5, 54, 12, 7.2, 90],  # soc-loss-rate [%/km] assumed = 1/range
        'range': [200, 255, 316, 312, 550],
        'revenue_km': [2, 2, 2, 2, 4],  # todo: possibly move elsewhere?
        'revenue_hr': [1, 1, 1, 1, 2],
        # todo: repace with realistic value, do we even wish to differentiate between model revenue types ?
        'revenue_book': [5, 5, 5, 5, 20],
    }

    stations_conf = {'stations_latlon': [[32.04, 34.75], [32.06, 34.765], [32.08, 34.77],
                                          [32.12, 34.795], [32.12, 34.83], [32.08, 34.78],
                                          [32.06, 34.78], [32.05, 34.8], [32.08, 34.79],
                                          [32.10, 34.78], [32.10, 34.79], [32.10, 34.8]] }

    random_fleet_scenario(fleet_conf, stations_conf,
                          name_save='df_fleet_scenario_telaviv.csv',
                          save=True)

    config_file['Environment'] = env_conf
    config_file['Fleet'] = fleet_conf
    config_file['Stations'] = stations_conf


    """ Prepare environemnt"""
    env = RelocationGridCityEnv(config_file=config_file)

    """ initialize stations -- example"""
    centers, node_start, node_end, \
    closest_dist_station = allocate_stations(env.df_trips[['LONs','LATs']].values,
                      end_coords= env.df_trips[['LONe','LATe']].values,
                      n_stations=10,
                      method='voronoi_k_mean',
                      visualize=True)


    observations, reward, done, info = env.step(env.action_space.sample())


    """Montecarlo simulation"""
    df_mc_samples, mean_scores = MonteCarlo_fleet_sizes(RelocationGridCityEnv,
                                                                config_file,
                                                                n_episodes=20,
                                                                n_steps=env_conf['episode_max_steps'],
                                                                fleet_sizes=[5, 20])#   num_processes=6,

    df_mc_samples['km_driven_per_car'] = df_mc_samples['km_driven'] / df_mc_samples['fleet_size']
    df_mc_samples['min_driven_per_car'] = df_mc_samples['min_driven'] / df_mc_samples['fleet_size']
    df_mc_samples['n_booked_per_car'] = df_mc_samples['n_booked'] / df_mc_samples['fleet_size']
    grouped_mean = df_mc_samples.groupby(by='fleet_size').mean()

    df_mc_samples.to_pickle(
        r'C:\Users\roberto.rocchetta.in\OneDrive - SUPSI\Desktop\GAMES project\Codes\gym_relocation\results\mc_simulation_100cars_1h_steps.pkl')


    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    sbn.ecdfplot(df_mc_samples, x='n_booked', hue='fleet_size', ax=ax[0, 0])
    sbn.ecdfplot(df_mc_samples, x='km_driven', hue='fleet_size', ax=ax[0, 1])
    sbn.ecdfplot(df_mc_samples, x='min_driven', hue='fleet_size', ax=ax[0, 2])
    sbn.ecdfplot(df_mc_samples, x='n_booked_per_car', hue='fleet_size', ax=ax[1, 0])
    sbn.ecdfplot(df_mc_samples, x='km_driven_per_car', hue='fleet_size', ax=ax[1, 1])
    sbn.ecdfplot(df_mc_samples, x='min_driven_per_car', hue='fleet_size', ax=ax[1, 2])
    # Adjust layout
    plt.tight_layout()
    # Show or save the plot as desired
    plt.show()


    _, ax = plt.subplots(2, 3, figsize=(14, 8))
    y_label_names = [['mean availability', 'mean time_to_return_min', 'mean idle_time_min'],
                     ['n_booked', 'km_driven', 'min_driven']]
    keys_name = [['mean_is_idle', 'mean_time_to_return_min', 'mean_idle_time_min'],
                 ['n_booked', 'km_driven', 'min_driven']]
    filter_fleet = df_mc_samples['fleet_size'] == 100
    n_ep = 20
    color_plot = ['r','c']
    for j in range(2):
        for k in range(3):
           id_range = lambda i : range(i * 168, (i + 1) * 168)
           [ax[j][k].plot(range(168), df_mc_samples[keys_name[j][k]][filter_fleet][i * 168: (i + 1) * 168], color_plot[j],
                          alpha=0.2) for i in range(n_ep)]
           sample_key_ij = [df_mc_samples[keys_name[j][k]][filter_fleet][i * 168: (i + 1) * 168] for i in range(n_ep)]
           ax[j][k].plot(range(168), np.mean(sample_key_ij, axis=0),
                         'k', alpha=0.9, label='mean trend')
           ax[j][k].set_ylabel(y_label_names[j][k])
           ax[j][k].set_xlabel('hour of week')
           ax[j][k].grid()
    plt.show()


    fig, AXs = plt.subplots(3, 4)
    dw = 0
    for Ax in AXs:
        for ax in Ax:
            dw += 1
            df_day_i = env.df_trips[env.df_trips['dw'] == dw + 1]
            df_day_i = df_day_i[df_day_i['hour'] == 12]
            ax.plot(df_day_i[['LONe', 'LONs']], df_day_i[['LATe', 'LATs']], c='r', alpha=0.1)
            ax.scatter(df_day_i['LONs'], df_day_i['LATs'], 1, c='g', alpha=0.1)
            ax.scatter(df_day_i['LONe'], df_day_i['LATs'], 1, c='k', alpha=0.1)
    plt.show()

    fig, ax = plt.subplots(3,3, figsize=(12, 8))  # Optional: Adjust the figure size

    for i in range(3):
        for j in range(3):
            # Execute your code to sample trip demand and simulate reservations
            df_trip_demand_samples = env.sample_trip_demand_scenario()
            df_reservations = env.simulate_reservations(df_trip_demand_samples)
            # Create scatter plot for LATs and LONs in df_reservations
            ax[i, j].scatter(df_reservations['LONs'].values, df_reservations['LATs'].values, label='Reservations',
                             marker='o')
            # Add LATs and LONs of self.df_fleet to the same plot
            ax[i, j].scatter(env.df_fleet['LON'].values, env.df_fleet['LAT'].values, label='Fleet', marker='x')

            # Customize plot labels, titles, legend, etc. as needed
            ax[i, j].set_xlabel('Longitude')
            ax[i, j].set_ylabel('Latitude')
            ax[i, j].set_title(f'Iteration {i * 5 + j + 1}')
            ax[i, j].legend()
    # Adjust layout
    plt.tight_layout()
    # Show or save the plot as desired
    plt.show()

