import time
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
import argparse
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from os.path import join
import wp3_simulate.sim as sim
from wp3_simulate.sim.mode_choice_model.features import compute_dist_to_station
from wp3_simulate.sim.mode_choice_model.simple_choice_models import *
from wp3_simulate.visualization_utils import *

import sys
path_to_v2g4carsharing = r'C:\Users\roberto.rocchetta.in\OneDrive - SUPSI\Desktop\GAMES project\Codes\GAMES\v2g4carsharing'
sys.path.append(path_to_v2g4carsharing)

def preprocess_presence_dso(dso_label='ewz'):
    load_dir = '../datasets/'
    save_dir = '../datasets/mobility/DSO/'
    save_name = 'mobility_{}.zip'.format(dso_label)
    # load stations metadata
    metadata = pd.read_pickle(load_dir + 'stations_metadata_elcom.zip')
    stations = metadata.index[metadata['DSO_label'] == dso_label]
    dso_metadata = metadata.loc[stations]
    saved_data_dirs = [f.replace('\\', '/') for f in glob(join(save_dir, '*'))]
    if join(save_dir, save_name) in saved_data_dirs:
        station_matrix = pd.read_pickle(join(save_dir, save_name))
    else:
        station_matrix = pd.read_pickle(load_dir + 'mobility/station_matrix.zip')
        station_matrix = station_matrix.T
        station_matrix.index = pd.DatetimeIndex(station_matrix.index)
        ev_to_keep = station_matrix.median().isin(stations)
        station_matrix = station_matrix[ev_to_keep.index[ev_to_keep]]
        station_matrix.to_pickle(join(save_dir, save_name))

    ev_per_station = pd.DataFrame(station_matrix.median().astype(int).value_counts())
    stations_data = ev_per_station.join(dso_metadata[['LAT', 'LON']], how='inner')
    stations_data.rename(columns={0: "n_cars"}, inplace=True)
    stations_data.index.names = ['station_no']
    return station_matrix, stations_data, dso_metadata


def get_kernel_density(df, dso_label: 'str',
                       mode: str = 'parked',
                       hr: int = 0,
                       st: int = 1):
    load_save_dir = 'sim/trained_models/kde_model/{}/'.format(dso_label)
    save_name = 'kde_{}_hr{}_station{}.p'.format(mode, hr, st)
    save_dir_name = join(load_save_dir, save_name)
    saved_data_dirs = [f.replace('\\', '/') for f in glob(join(load_save_dir, '*'))]
    if join(load_save_dir, save_name) in saved_data_dirs:
        kernel_density = pickle.load(open(save_dir_name, 'rb'))
    else:
        os.makedirs(os.path.dirname(load_save_dir))
        kernel_density = KernelDensity(bandwidth=1.0, kernel='gaussian')
        if mode == 'parked':
            kernel_density.fit(df['duration_between_trips_hr'].values[:, None])
        else:
            kernel_density.fit(df['duration_trip_hr'].values[:, None])
        pickle.dump(kernel_density, open(save_dir_name, 'wb'))
    return kernel_density


def get_trips_df_from_station_matrix(dso_label='ewz',
                                     directory='../datasets/mobility/DSO/'):
    save_name = 'mobility_df_trips_{}.zip'.format(dso_label)
    try:
        df_trips = pd.read_pickle(directory + save_name)
    except:
        df_trips = pd.DataFrame()  # prepare trip data set
        station_matrix, stations_data, _ = preprocess_presence_dso(dso_label=dso_label)
        vehicles_id = list(np.unique(station_matrix.T.index))
        trip_time_stamp = station_matrix.index
        dis_array = np.array(station_matrix.T.values == 0)
        trips_id_x_vehicle = [np.where(np.diff(d))[0] for d in dis_array]
        for id, tr in enumerate(trips_id_x_vehicle):
            from_station, to_station = station_matrix[vehicles_id[id]][tr[0::2]], station_matrix[vehicles_id[id]][
                tr[1::2] + 1]
            disconnected_time, connected_time = trip_time_stamp[tr[0::2] + 1], trip_time_stamp[tr[1::2] + 1]

            df = pd.DataFrame(list(zip(disconnected_time, connected_time)),
                              columns=['disconnected_time', 'reconnected_time'])
            trip_duration = df['reconnected_time'] - df['disconnected_time']
            between_duration = df['disconnected_time'].shift(-1) - df['reconnected_time']
            hours_disconnected = [td.total_seconds() / 3600 for td in trip_duration]
            hours_connected = [bt.total_seconds() / 3600 for bt in between_duration]

            df_temp = pd.DataFrame(list(
                zip(from_station, to_station, disconnected_time,
                    connected_time, hours_disconnected, hours_connected)),
                columns=['from_station', 'to_station', 'disconnection_time', 'reconection_time',
                         'duration_trip_hr', 'duration_between_trips_hr'])

            df_temp['vehicle_no'] = vehicles_id[id]
            df_trips = pd.concat([df_trips, df_temp])

        df_trips = df_trips.dropna()
        df_trips.index = range(len(df_trips.index))  # fix trip indices
        df_trips['h_disconnected'] = df_trips['disconnection_time'].dt.hour
        df_trips['h_connected'] = df_trips['reconection_time'].dt.hour
        df_trips['disconnection_time'] = pd.DatetimeIndex(df_trips['disconnection_time'].values)
        df_trips['reconection_time'] = pd.DatetimeIndex(df_trips['reconection_time'].values)
        df_trips[['from_station', 'to_station']] = df_trips[['from_station', 'to_station']].astype(int)
        df_trips.to_pickle(join(directory, save_name))
    return df_trips


class Trips_and_Stations:
    def __init__(self, dso_label='ewz',
                 station_matrix=None,
                 stations_data=None):

        if (station_matrix is None) or (stations_data is None):
            station_matrix, stations_data, _ = preprocess_presence_dso(dso_label=dso_label)
        else:  # load station matrix and station data
            station_matrix = station_matrix
            stations_data = stations_data
        self.smat = station_matrix
        self.station_set = Stations_Set(stations_data=stations_data)
        self.vehicles_id = self.smat.T.index  # the ID of the vehicles
        self.trip_time_stamp = self.smat.index  # index time stamps
        # prepare trip data set
        self.df_trips = pd.DataFrame()
        self.df_trips = self.get_trips_df_from_station_matrix()
        self.df_relocations = self.df_trips[
            self.df_trips['from_station'] != self.df_trips['to_station']]  # filter one-way trips
        self.df_trips = self.df_trips[
            self.df_trips['from_station'] == self.df_trips['to_station']]  # filter returns
        self.df_trips["station_no"] = self.df_trips["from_station"]
        self.df_trips = self.df_trips.drop(["from_station", "to_station"], axis=1)

    def get_trip_index(self):
        dis_array = np.array(self.smat.T.values == 0)
        trips_id_x_vehicle = [np.where(np.diff(d))[0] for d in dis_array]
        return trips_id_x_vehicle

    def get_trips_df_from_station_matrix(self):
        trips_id_x_vehicle = self.get_trip_index()

        if not self.df_trips.empty:
            return self.df_trips
        else:
            for id, tr in enumerate(trips_id_x_vehicle):
                from_station = self.smat[self.vehicles_id[id]][tr[0::2]]
                to_station = self.smat[self.vehicles_id[id]][tr[1::2] + 1]
                # TODO: there could be 1 timestamp step inconsistency (15 minutes max for the mobility data)
                disconnected_time = self.trip_time_stamp[tr[0::2] + 1]
                connected_time = self.trip_time_stamp[tr[1::2] + 1]
                df = pd.DataFrame(list(zip(disconnected_time, connected_time)),
                                  columns=['disconnected_time', 'reconnected_time'])
                trip_duration = df['reconnected_time'] - df['disconnected_time']
                between_duration = df['disconnected_time'].shift(-1) - df['reconnected_time']
                hours_disconnected = [td.total_seconds() / 3600 for td in trip_duration]
                hours_connected = [bt.total_seconds() / 3600 for bt in between_duration]

                df_temp = pd.DataFrame(list(
                    zip(from_station, to_station, disconnected_time, connected_time, hours_disconnected,
                        hours_connected)),
                    columns=['from_station', 'to_station', 'disconnection_time', 'reconection_time',
                             'duration_trip_hr', 'duration_between_trips_hr'])

                df_temp['vehicle_no'] = self.vehicles_id[id]
                self.df_trips = pd.concat([self.df_trips, df_temp])

            self.df_trips = self.df_trips.dropna()
            self.df_trips.index = range(len(self.df_trips.index))  # fix trip indices
            self.df_trips['h_disconnected'] = self.df_trips['disconnection_time'].dt.hour
            self.df_trips['h_connected'] = self.df_trips['reconection_time'].dt.hour
            self.df_trips['disconnection_time'] = pd.DatetimeIndex(self.df_trips['disconnection_time'].values)
            self.df_trips['reconection_time'] = pd.DatetimeIndex(self.df_trips['reconection_time'].values)
            self.df_trips[['from_station', 'to_station']] = self.df_trips[['from_station', 'to_station']].astype(int)
            return self.df_trips

    def get_vehicle_in_station(self):
        vehicles_in_station = self.smat.median().astype(int)
        return vehicles_in_station

    def update_connected_vehicles(self, grid_connected_vehicles):
        self.vehicle_connected = grid_connected_vehicles

    def get_vehicle_list_per_station(self):
        vehicles_is_in_station = self.get_vehicle_in_station()
        Vehicle_per_station = [[] for _ in range(len(self.station_set.df))]
        for id, s in enumerate(self.station_set.df):
            Vehicle_per_station[id].append(vehicles_is_in_station[vehicles_is_in_station == s].index)
        self.station_set.df['vehicle_list'] = Vehicle_per_station
        return Vehicle_per_station, self.station_set.df

    def get_number_and_percentage_of_parked_vehicels(self):
        Number_of_parked_vehicles = pd.DataFrame([])
        Number_of_parked_vehicles.index = self.smat.index  # assign datetime indices
        Percentage_parked_vehicles = pd.DataFrame([])
        Percentage_parked_vehicles.index = self.smat.index  # assign datetime indices

        # TODO: be sure that station_set.df has the 'vehicle_list' in it
        _, station_data = self.get_vehicle_list_per_station()
        for idx, ev_list in enumerate(station_data['vehicle_list']):  # loop stations
            print('add station', idx)
            sta_mat_ve_x_st = self.smat.T.loc[station_data['vehicle_list'].iloc[idx][0]]  # loop stations
            df_new = pd.DataFrame((sta_mat_ve_x_st > 0).sum(), columns=[str(station_data.index[idx])])
            Number_of_parked_vehicles = pd.concat([Number_of_parked_vehicles, df_new], axis=1)

            df_new = pd.DataFrame((sta_mat_ve_x_st > 0).sum() / station_data['n_cars'].iloc[idx],
                                  columns=[str(station_data.index[idx])])
            Percentage_parked_vehicles = pd.concat([Percentage_parked_vehicles, df_new], axis=1)

    def compute_car_densities(self):
        station_matrix = self.smat
        vals = []
        for station_id, n_car_id in zip(self.station_set.stations_idx, self.station_set.n_cars_per_station_nominal):
            is_car_in_station = (station_matrix.values == station_id)
            vals.append(np.sum(is_car_in_station, axis=-1) / n_car_id)

        vehicle_densities_df = pd.DataFrame(vals)
        vehicle_densities_df.index = self.station_set.stations_idx
        vehicle_densities_df.columns = station_matrix.index
        vehicle_densities_df = vehicle_densities_df.T
        vehicle_densities_df.columns.name = 'Base_no'
        return vehicle_densities_df

    def append_cluster_resutls_to_trip_df(self, n_clusters=5, variables_2_cluster=['LAT', 'LON']):
        self.station_set.cluster_elements(n_clusters=n_clusters, names_inp2clust=variables_2_cluster)
        df_trips_with_clusters = pd.merge(self.df_trips, self.station_set.df, on="station_no")
        return df_trips_with_clusters


class Stations_Set:
    def __init__(self, stations_data=None):
        if stations_data is None:
            _, stations_data, _ = preprocess_presence_dso()
            self.df = stations_data
        else:
            self.df = stations_data
        self.stations_idx = self.df.index
        self.total_n_stations = np.shape(self.df)[0]  # number of stations
        self.cluster_id = None
        if 'n_cars' not in self.df:  # if no info provide...number of cars =1 in all rows
            self.df['n_cars'] = 1
        self.n_cars_per_station_nominal = self.df['n_cars']  # number of vehicles (per-station)
        self.n_cars_per_station_available = self.n_cars_per_station_nominal
        self.total_n_cars = sum(self.df['n_cars'])  # number of vehicles (total)

    def cluster_latitude_longitude(self, data=None, names_inp2clust=None, n_clusters: int = 4):
        """ a simple k-mean """
        n_clusters = max(n_clusters, 1)
        if names_inp2clust is None:
            names_inp2clust = ['LAT', 'LON']

        if data is None:
            X_to_cluster = self.df[names_inp2clust].values
            cluster_id = KMeans(n_clusters=n_clusters).fit(X_to_cluster,
                                                           y=None,
                                                           sample_weight=None).predict(X_to_cluster)
            self.cluster_id = cluster_id
            self.df['cluster_id'] = cluster_id
            n_stations, n_vehicles = [], []
            for k in range(n_clusters):
                n_stations.append(np.shape(self.df[self.cluster_id == k])[0])
                n_vehicles.append(self.df[self.cluster_id == k]["n_cars"].values.sum())
            return pd.DataFrame(list(zip(cluster_id, n_stations, n_vehicles)),
                                columns=['cluster_id', 'n_stations', 'n_vehicles'])
        else:
            X_to_cluster = data[names_inp2clust].values
            cluster_id = KMeans(n_clusters=n_clusters).fit(X_to_cluster,
                                                           y=None,
                                                           sample_weight=None).predict(X_to_cluster)
            data['cluster_id'] = cluster_id
            return data


class games_trips_sampler:
    def __init__(self,
                 dso_label='ewz',  # default DSO label
                 sim_start_time="2019-01-01",
                 sim_end_time="2019-02-01"
                 ):

        self.dso_label = dso_label
        self.out_path = os.path.join("", "data/simulated_car_sharing", "rf_sim")
        # path to model, stations, users trips(nina's xgb_model.p, etc..)
        self.model_path = "sim/trained_models/mode_choice/xgb_model.p"
        self.station_data_path = "data/station_scenario/station_scenario_new1000_7500.csv"
        self.acts_path = "data/users/'trips_features_from_Zurich'.csv"
        # load stations_data and station matrix for the selected DSO
        self.station_matrix, self.stations, _ = preprocess_presence_dso(dso_label=dso_label)
        # assign station id vechiles id etc...
        self.station_no = self.stations.index  # e.g. indies [0,1,2,3]
        self.data_datetime_index = self.station_matrix.index
        self.vehicle_no = self.station_matrix.T.index  # e.g, [0,1,2,3]
        self.stations['vehicle_list'] = self.get_vehicle_list_per_station
        # start simulation with all connected vehicles (1 connected, 0 disconnected)
        self.vehicle_connected = [np.ones(np.shape(ve), dtype=bool).tolist()
                                  for ve in self.vehicle_no]
        # the datetime range used by the low-fidelity simulator
        self.sim_dt_idx = pd.date_range(start=sim_start_time, end=sim_end_time, freq="15Min")
        # from the Trips_and_Stations object, compute data_fr with trips/parking times
        self.Trips_df = Trips_and_Stations(dso_label=dso_label,
                                           station_matrix=self.station_matrix,
                                           stations_data=self.stations).df_trips

    @property
    def get_vehicle_in_station(self):
        """this gets the reference station for each vehicle"""
        vehicles_in_station = self.station_matrix.median().astype(int)
        return vehicles_in_station

    @property
    def get_vehicle_list_per_station(self):
        """ it computes a List[List] of vehicles, one list for each station in self.station_no"""
        vehicles_is_in_station = self.get_vehicle_in_station
        Vehicle_per_station = [[] for _ in range(len(self.station_no))]
        for i, s in enumerate(self.station_no):
            Vehicle_per_station[i].append(vehicles_is_in_station[vehicles_is_in_station == s].index)
        return Vehicle_per_station

    def update_connected_vehicles(self, grid_connected_vehicles):
        """this can be used to update/assign a new connectivity structure for the evs"""
        self.vehicle_connected = grid_connected_vehicles

    def sample_next_departure(self, hour: int = 8, station_id: int = None, n_samples: bool = 1):
        """ sample idle times conditional to the hour of the day and station ID"""
        # TODO: fit parametric model
        df = self.Trips_df[self.Trips_df['station_no'] == station_id]  # get trips dataframe x station
        if df.empty:  # if no trips recorded, skip # todo: use average (or a close by/similar) station to sample trips if empty?
            return [1e10]
        df_temp = df[df['h_connected'] == hour]  # filter on h_connected
        # if df_temp.empty:
        hd = 0
        while df_temp.shape[0] < 2:
            df = df[df['h_connected'] <= hour + hd]
            df_temp = df[df['h_connected'] > hour - hd]
            hd += 1
            if hd >= 3:  # use average distirbution
                df_temp = self.Trips_df[self.Trips_df['h_connected'] == hour]

        # kde = self.get_kernel_density(df_temp, mode='parked', hr=hour, st=station_id)
        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(df_temp['duration_between_trips_hr'].values[:, None])
        sam_bt = kde.sample(n_samples).flatten().clip(0.5).tolist()
        return sam_bt

    def sample_next_return(self, hour: int = 20, station_id: int = None, n_samples: bool = 1) -> object:
        """ sample time-before-return time given the hour of the day"""
        # TODO: 
        #  1. use interpolate for the sampling new trips
        #  2. if station_id has no data....use ecdf from similar station ? or groups of hours?

        df = self.Trips_df[self.Trips_df['station_no'] == station_id]  # get trips dataframe x station
        if df.empty:  # if no trips recorded, skip #todo: use another station for sampling trips
            return [1e10]
        df_temp = df[df['h_disconnected'] == hour]  # filter on h_disconnected
        hd = 0
        while df_temp.shape[0] < 2:
            df = df[df['h_disconnected'] <= hour + hd]
            df_temp = df[df['h_disconnected'] > hour - hd]
            hd += 1
            if hd >= 3:  # use average distirbution
                df_temp = self.Trips_df[self.Trips_df['h_disconnected'] == hour]

        kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde.fit(df_temp['duration_trip_hr'].values[:, None])
        sam_td = kde.sample(n_samples).flatten().clip(0.5).tolist()
        return sam_td

    def sample_vehicle_connectivity_matrix(self,
                                           ev_is_connected_t0: object = None,
                                           sim_start_time: object = "2019-01-01",
                                           sim_end_time: object = "2019-02-01",
                                           verbose: int = 0) -> object:
        """ Simple trip sampler:
        it works by sampling from the empirical CDF at a given hour of a day
        Example-1: station dependent duration distributions
        see [self.sample_next_return ,   self.sample_next_departure]
        Example-2: station invariant duration distributions
        car departs at hour H1 --> sample duration ~ F_booking(t|H1)
        car arrives at hour H2 --> sample duration ~ F_idle(t|H2)"""

        ev_in_station = self.get_vehicle_in_station
        ev_list, station_list = ev_in_station.index, ev_in_station.values
        if ev_is_connected_t0 is None:
            ev_is_connected_t0 = self.vehicle_connected
        self.sim_dt_idx = pd.date_range(start=sim_start_time, end=sim_end_time, freq="15Min")
        DURATION_SAMPLER = quantile_duration_sampler()  # prepare a sampler for the durations
        # TODO:  prepare SOC sampler at arrival based on random travelled distance
        # TODO: prepare SOC estimator based on the distance for the next trip

        # preallocate
        sim_ev_connect_mat = np.ones(shape=(len(self.sim_dt_idx), len(self.vehicle_no)), dtype=int)
        ev_id, sim_horizon = 0, len(self.sim_dt_idx)  # simulator time steps

        if verbose > 0:
            print('Simulating trips using statistical time-to-event sampler')
        for ev, st, ev_is_con in zip(ev_list, station_list, ev_is_connected_t0):
            t_0, is_done = 0, False
            #if verbose > 0:
            #    print('simulate trips for vehicle number: ' + str(ev_id) + '/' + str(len(ev_list)))
            while not is_done:
                hr_t0 = self.sim_dt_idx[t_0].hour  # get hour for step id t0
                if ev_is_con == 0:  # vehicle is booked at t0 (sample booking event duration)
                    time2ev = DURATION_SAMPLER.sample_n_durations(n_samples=1, hr=hr_t0, event_type='departure')
                else:  # ev_is_con == 1 vehicle is parked/arrived at t0 (sample idle event duration)
                    time2ev = DURATION_SAMPLER.sample_n_durations(n_samples=1, hr=hr_t0, event_type='arrival')
                t_1 = t_0 + int(time2ev[0] * 4)  # convert to quarters of hours (this is the time index of next event)
                if (t_1 < sim_horizon) & (ev_is_con == 0):
                    hr_t1 = self.sim_dt_idx[t_1].hour
                    time2ev = DURATION_SAMPLER.sample_n_durations(n_samples=1, hr=hr_t1, event_type='arrival')
                elif (t_1 < sim_horizon) & (ev_is_con == 1):
                    hr_t1 = self.sim_dt_idx[t_1].hour
                    time2ev = DURATION_SAMPLER.sample_n_durations(n_samples=1, hr=hr_t1, event_type='departure')
                elif (t_1 >= sim_horizon) & (ev_is_con == 0):
                    sim_ev_connect_mat[t_0:sim_horizon, ev_id] = 0
                    break
                elif (t_1 >= sim_horizon) & (ev_is_con == 1):
                    sim_ev_connect_mat[t_0:sim_horizon, ev_id] = 1
                    break
                t_2 = t_1 + int(time2ev[0] * 4)
                if t_2 >= sim_horizon:  #
                    t_2, is_done = sim_horizon, True
                if ev_is_con == 0:  # if booked, first assign ev_presence 0 = booked then 1 = available/idle, rest t0=t2
                    sim_ev_connect_mat[t_0:t_1, ev_id], sim_ev_connect_mat[t_1:t_2, ev_id], t_0 = 0, 1, t_2
                else:  # if available, first assign ev_presence 0 = booked then 1 = available/idle, rest t0=t2
                    sim_ev_connect_mat[t_0:t_1, ev_id], sim_ev_connect_mat[t_1:t_2, ev_id], t_0 = 1, 0, t_2
            ev_id += 1
        print('Simulated trip history for the last vehicle number: ' + str(len(ev_list)))
        df_sim_ev_connect_mat = pd.DataFrame(sim_ev_connect_mat)
        df_sim_ev_connect_mat.index = self.sim_dt_idx
        df_sim_ev_connect_mat = df_sim_ev_connect_mat.T
        df_sim_ev_connect_mat.index = ev_in_station.index
        return df_sim_ev_connect_mat

    def filter_stations_apply_geom(self, station_scenario=None):
        if station_scenario is None:
            station_scenario = sim.simulate.car_sharing_patterns.load_station_scenario(self.station_data_path)
        station_scenario = self.stations.merge(station_scenario,
                                               left_on='station_no', right_on='station_no',
                                               how='inner')  # use reference station scenario and filter the station DSO
        # station_scenario.drop(columns='vehicle_list')
        station_gdf = gpd.GeoDataFrame(station_scenario, geometry="geom", crs="EPSG:2056")

        column_name = 'vehicle_list'
        if column_name not in station_gdf:
            station_gdf[column_name] = station_gdf['vehicle_list_y']
            station_gdf = station_gdf.drop(columns=['vehicle_list_x', 'vehicle_list_y'])
        return station_gdf

    def ev_to_station_list(self, number_vehicles_per_station=None):
        """ assigns fleet vehicles to stations"""
        ev_list, n_stations = self.vehicle_no.tolist(), len(self.station_no)
        ev_assigned_list = [[] for _ in range(n_stations)]
        for s in range(n_stations):
            if number_vehicles_per_station is None:  # assign uniformly
                n = max(1, round(len(ev_list) / (n_stations - s)))
            else:  # assign a user-defined number
                n = number_vehicles_per_station[s]
            if len(ev_list) > 0:  # is vehicles are yet in the list
                ev_assigned = np.random.choice(ev_list, size=min(n, len(ev_list)), replace=False)
                [ev_list.remove(ev) for ev in ev_assigned]
                ev_assigned_list[s] = ev_assigned.tolist()
            else:
                break
        if len(ev_list) > 0:  # if leftover vehicles append it to the first station
            ev_assigned_list[0] = ev_assigned_list[0] + ev_list
        return ev_assigned_list

    def assign_new_vehicle_distribution(self, n_vehicles_x_st=None):
        if n_vehicles_x_st is None:
            print('assigning a uniform number of vehicle to all the stations')
            n_ev_temp = None
        elif n_vehicles_x_st == 'rand':
            print('assigning multinomial integers to the stations')
            n_ev_temp = np.random.multinomial(
                len(self.vehicle_no), [1 / len(self.station_no)] * len(self.station_no),
                size=1)[0]
        else:
            print('assigning the input number of vehicles to the stations')
            n_ev_temp = np.zeros(len(self.station_no), dtype=int)
            n_ev_temp[:len(n_vehicles_x_st)] = n_vehicles_x_st
        ev_distribution_list = self.ev_to_station_list(number_vehicles_per_station=n_ev_temp)
        station_gdf_dso = self.filter_stations_apply_geom()
        station_gdf_dso['vehicle_list'] = ev_distribution_list  # assign vehicles to stations
        station_gdf_dso['n_cars'] = [len(e) for e in station_gdf_dso['vehicle_list']]  # fix car number
        return station_gdf_dso

    def filter_stations_and_acts_by_closest_station(self, acts_gdf=None):
        if acts_gdf is None:
            acts_gdf = sim.simulate.car_sharing_patterns.load_trips(self.acts_path)
        station_gdf_filtered = self.filter_stations_apply_geom()
        c1 = (acts_gdf['closest_station_destination'].isin(station_gdf_filtered.index))
        c2 = (acts_gdf['closest_station_origin'].isin(station_gdf_filtered.index))
        acts_gdf_filtered = acts_gdf[acts_gdf['person_id'].isin(acts_gdf[c1 | c2]['person_id'].unique())]
        print("filtered trips by persons_id "
              "and trip destinations by the DSO stations, leftover trips:", len(acts_gdf_filtered))
        return station_gdf_filtered, acts_gdf_filtered

    def Ninas_daily_trips_sampler(self, actions_gdf=None,
                                        station_data=None,
                                        simulation_save_name=None,
                                        do_save = False):
        """This function uses nina's simulator and generates synthetic schedules given:
        INPs:
        1. station data (station_data),   2. a choice model (self.model_path is an xgboost now),
        3. actions csv, trips and populations patterns (actions_gdf)
        OUTs:  1. simulated trips,    2. simulated model choice """
        os.makedirs(self.out_path, exist_ok=True)
        # load activities and shared-cars availability
        st = time.time()
        if actions_gdf is None:
            station_scenario, actions_gdf = self.filter_stations_and_acts_by_closest_station()
        if station_data is not None:
            station_scenario = station_data
        if simulation_save_name is None:
            simulation_save_name = 'sim_reservations_{}.csv'.format(self.dso_label)
        acts_save_name = 'acts_gdf' + simulation_save_name
        station_save_name = 'stations_gdf' + simulation_save_name
        # Save station scenario
        assert "geom" in station_scenario.columns, "station scenario must have geometry"
        assert "geom_origin" in actions_gdf.columns, "acts gdf must have geometry to recompute distance to station"
        # load the trained model for user choice of transportation mode:
        with open(self.model_path, "rb") as infile:
            mode_choice_model = pickle.load(infile)
        # compute dist to station for each trip start and end point, sort values
        # acts_gdf = compute_dist_to_station(actions_gdf, station_scenario)
        # acts_gdf.sort_values(["person_id", "activity_index"], inplace=True)
        # get time when decision is made
        #acts_gdf = sim.simulate.car_sharing_patterns.derive_decision_time(acts_gdf)
        # Run: iteratively assign modes
        # acts_gdf_mode = sim.simulate.car_sharing_patterns.assign_mode(acts_gdf, station_scenario, mode_choice_model)
        # Run: iteratively assign modes (if we pre-assigned times and dist -- speed up for a fixed station scenario)
        acts_gdf_mode = sim.simulate.car_sharing_patterns.assign_mode(actions_gdf, station_scenario, mode_choice_model)
        # get shared only and derive the reservations by merging subsequent car sharing trips
        sim_reservations = sim.simulate.car_sharing_patterns.derive_reservations(acts_gdf_mode)
        # Save reservations and trip modes
        if do_save:
            station_scenario.to_csv(os.path.join(self.out_path, station_save_name))
            acts_gdf_mode[["person_id", "activity_index",
                           "mode_decision_time",
                           "mode", "vehicle_no"]].to_csv(os.path.join(self.out_path, acts_save_name), index=False)
            sim_reservations.to_csv(os.path.join(self.out_path, simulation_save_name))
        elapsed_time = time.time() - st
        print('ELAPSED TIME: ', elapsed_time, '   SECONDS')
        return sim_reservations


def sample_from_empirical_dist(quantiles, percentiles):
    # Calculate the slopes for linear interpolation
    slopes = np.diff(quantiles) / np.diff(percentiles)

    def sample_interpolated(percentile):
        # Find the nearest percentiles
        lower_percentile = percentiles[np.searchsorted(percentiles, percentile, side='right') - 1]
        upper_percentile = percentiles[np.searchsorted(percentiles, percentile, side='right')]
        # Find the corresponding quantiles
        lower_quantile = quantiles[np.searchsorted(percentiles, percentile, side='right') - 1]
        upper_quantile = quantiles[np.searchsorted(percentiles, percentile, side='right')]
        # Calculate the interpolated quantile value
        slope = slopes[np.searchsorted(percentiles, percentile, side='right') - 1]
        quantile_value = lower_quantile + slope * (percentile - lower_percentile)
        return quantile_value

    return sample_interpolated


class quantile_duration_sampler:
    def __init__(self):
        with open('../models/mobility/hourly_stat_model/Quantiles_idle.pickle', 'rb') as file:
            self.q_idle_duration_hourly = pickle.load(file)
        with open('../models/mobility/hourly_stat_model/Quantiles_booking.pickle', 'rb') as file:
            self.q_booking_duration_hourly = pickle.load(file)
        self.n_hours, self.n_quantiles = np.shape(self.q_booking_duration_hourly)
        self.percentiles = np.linspace(0, 100, self.n_quantiles)
        self.sample_fun_booking_dur = [
            sample_from_empirical_dist(self.q_booking_duration_hourly[h, :], self.percentiles) for h in
            range(0, self.n_hours)]
        self.sample_fun_idle_dur = [sample_from_empirical_dist(self.q_idle_duration_hourly[h, :], self.percentiles)
                                    for h in range(0, self.n_hours)]

    def sample_n_durations(self, n_samples: int = 10, hr: int = 0, event_type: str = 'departure'):
        """ sample booking durations and idle durations condtionalt to the hour hr"""
        if event_type == 'departure':
            sample_func = self.sample_fun_booking_dur[hr]
        else:
            sample_func = self.sample_fun_idle_dur[hr]
        return [sample_func(np.random.uniform(0, 100)) for _ in range(n_samples)]


def assign_stations_to_simulated_trips(df_sim_with_stations, veh_in_stat):
    """ assigns to a binary trip matrix the corresponding station (for a return/station-based case)"""
    for id, val in zip(veh_in_stat.index, veh_in_stat.values):
        df_sim_with_stations.loc[id] = df_sim_with_stations.loc[id][:] * val
    return df_sim_with_stations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dso", "--dso_code",
                        type=str, default="ewz",
                        help="dso code you want to filter for")
    args = parser.parse_args()
    dso_code = args.dso_code
    # dso_code = 'Energie Wasser Bern ewb'
    sim_start_time = "2019-01-01"
    sim_end_time = "2019-6-30"
    SAMPLER = games_trips_sampler(dso_label=dso_code, sim_start_time=sim_start_time, sim_end_time=sim_end_time)


    """Example 1: Use the statistical sampler to generate daily trips sequentially"""
    sim_start_time , sim_end_time= datetime(2019, 1, 1), datetime(2019, 1, 2)
    df_sim_ev = SAMPLER.sample_vehicle_connectivity_matrix(sim_start_time=sim_start_time,
                                                           sim_end_time=sim_end_time, verbose=1)
    n_days = 5
    for day in range(n_days):
        # Calculate the start and end time for the current day

        current_start_time = sim_start_time + timedelta(days=day)
        current_end_time = sim_start_time + timedelta(days=day + 1)
        print('simulated day:  ' + str(current_start_time) + '    -      ' + str(current_end_time))
        print('grid connected vehicles at time t0:  ' + str(sum(df_sim_ev.iloc[:, -1])) + '  / ' + str(len(df_sim_ev.iloc[:, -1])))
        df_sim_ev = SAMPLER.sample_vehicle_connectivity_matrix(ev_is_connected_t0=df_sim_ev.iloc[:, -1],
                                                                sim_start_time=current_start_time,
                                                                sim_end_time=current_end_time, verbose=1)

    """Example 1: Use the statistical sampler 
                to sample 3-month trip history (default forall EV operational state is True =connected)"""
    df_sim_ev = SAMPLER.sample_vehicle_connectivity_matrix(sim_start_time=datetime(2019, 1, 1),
                                                           sim_end_time=datetime(2019, 3, 1),  verbose=1)
    real_data_trips_and_stations = Trips_and_Stations(dso_label='ewz')
    df_sim_ev = assign_stations_to_simulated_trips(df_sim_ev, real_data_trips_and_stations.get_vehicle_in_station())
    sim_data_trips_and_stations = Trips_and_Stations(station_matrix=df_sim_ev.T,
                                                     stations_data=SAMPLER.stations)

    # show that kernel distirbutions of the data and simulated trips are compatible
    plot_kde_departures_returns(real_data_trips_and_stations.df_trips, cmap="Reds")
    plot_kde_departures_returns(sim_data_trips_and_stations.df_trips, cmap="Blues")

    """------------ RUN Nina's sampler that uses a fixed pool of user trips to simulate mobility patterns ----------"""

    # 1) ESTIMATE MAXIMUM CAR SHARING MOBILITY DEMAND
    # to do this: We assigned a very high number of vehicles to all the stations ( Nina's )
    save_name = 'sim_res_sample_maximum_mobility_demand_{}.csv'.format(dso_code)
    station_max_demand_dso = SAMPLER.assign_new_vehicle_distribution()
    max_ev = 50
    # acts_path = 'data/users/trips_features.csv'
    acts_path = 'data/users/trips_features_from_Zurich.csv'
    acts_gdf = sim.simulate.car_sharing_patterns.load_trips(acts_path)
    ev_distribution_list = [list(range(max_ev * i, max_ev * (i + 1))) for i in range(len(station_max_demand_dso))]
    station_max_demand_dso['vehicle_list'] = ev_distribution_list
    SAMPLER.Ninas_daily_trips_sampler(station_data=station_max_demand_dso,
                                      actions_gdf=acts_gdf,
                                      simulation_save_name=save_name)

    # 2.a) TEST MOBILITY DEMAND SERVED by a random relocation vehicles ( Nina's )
    save_name = 'sim_res_sample_rand_acts_unchanged_v2_{}.csv'.format(dso_code)
    station_rand_demand_dso = SAMPLER.assign_new_vehicle_distribution()
    acts_gdf = sim.simulate.car_sharing_patterns.load_trips(acts_path)
    SAMPLER.Ninas_daily_trips_sampler(station_data=station_rand_demand_dso,
                                      actions_gdf=acts_gdf,
                                      simulation_save_name=save_name)

    # 2.b) TEST SEVERAL RANDOM RELOCATIONS  ( Nina's )
    """    for j in range(0, 160):
            new_ev_dist = np.zeros(len(SAMPLER.station_no), dtype=int)
            n_ev_temp = np.random.multinomial(
                len(SAMPLER.vehicle_no), [1 / len(SAMPLER.station_no)] * len(SAMPLER.station_no),
                size=1)[0]
            save_name_j = 'sim_res_sample_' + str(j) + '_rand_multinomial_{}.csv'.format(dso_code)
            station_rnd_dso = SAMPLER.assign_new_vehicle_distribution(n_vehicles_x_st=n_ev_temp)
            SAMPLER.Ninas_daily_trips_sampler(station_data=station_rnd_dso,
                                                    simulation_save_name=save_name_j)
    """

    # 2.c) TRY WITH A STRUCTURED RELOCATION ( Nina's )
    """    for j in range(0, 20):
        new_ev_dist = np.zeros(len(SAMPLER.station_no), dtype=int)
        ev_n, st_n = 5, 99
        new_ev_dist[j * 5:j * 5 + st_n] = ev_n
        save_name_j = 'sim_res_sample_' + str(j) + '_' + str(ev_n) +'ev_over_'+str(st_n)+'st_{}.csv'.format(dso_code)
        station_data_ev_11_over_45_stations = SAMPLER.assign_new_vehicle_distribution(n_vehicles_x_st=new_ev_dist)
        SAMPLER.Ninas_daily_trips_sampler(station_data=station_data_ev_11_over_45_stations,
                                                simulation_save_name=save_name_j)
    """

    # 2.d) APPLY uniform distribution of vehicles ( Nina's )
    """station_data_ev_uniform = SAMPLER.assign_new_vehicle_distribution()  # uniform relocation by default
    SAMPLER.Ninas_daily_trips_sampler(station_data=station_data_ev_uniform,
                                            simulation_save_name='sim_reservations_uniform_ev_{}.csv'.format(dso_code))
    """
