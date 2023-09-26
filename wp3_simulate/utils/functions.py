import os
import pandas as pd
import numpy as np
from numpy import (sin, cos, arcsin, sqrt)
import holidays
import pickle
from numba import jit
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from scipy.cluster.vq import kmeans, vq


def allocate_stations(start_coords,
                      end_coords,
                      n_stations=10,
                      method='voronoi_k_mean',
                      visualize=False):
    """ devides the area in clusters and voronoi partitions
    it uses - kmean clustering of departures to define 'good' coordinates for the stations
    voronoi-like partitioning
    uses kdtree to assign to end-dooeds a cluster ids"""

    centers, node_start, node_end, closest_dist_station = [], [], [], []
    if method == 'voronoi_k_mean':
        centers = kmeans(start_coords, n_stations)[0]   # K-means clustering
        node_start = vq(start_coords, centers)[0]
        vor = Voronoi(centers)
        voronoi_kdtree = cKDTree(centers)   # closest distance lookup
        closest_dist_station, node_end = voronoi_kdtree.query(end_coords)

    if visualize:
        # Plotting
        fig, ax = plt.subplots()
        # fig, ax = ox.plot_graph(Gtr.G_drive, node_alpha=0.1, bgcolor="#cccccc", node_color='k', edge_color='k', show=False,  edge_alpha=0.1)
        plot_data_points(start_coords, ax=ax)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
        ax.tick_params(axis='both', left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)
        [plt.scatter(start_coords[node_start==cl][:,0], start_coords[node_start==cl][:,1])  for cl in np.unique(node_start)]
        plot_kmeans_clustering(centers, ax=ax, alpha=0.99)
        plt.show()
    return centers, node_start, node_end, closest_dist_station




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

def prepare_df_trip_for_environment(df, save=True):
    df_trip_ep = df.copy()
    df_trip_ep = df_trip_ep[df_trip_ep['distance'] < 30]
    df_trip_ep = df_trip_ep[df_trip_ep['distance'] > 0]
    df_trip_ep = df_trip_ep[df_trip_ep['car Id'] > 0]
    df_trip_ep = df_trip_ep[df_trip_ep['out_duration_trip_min'] > 0]
    df_trip_ep = df_trip_ep[df_trip_ep['out_duration_trip_min'] < 300]

    df_trip_ep['hour'] = df_trip_ep['start_ride_datetime'].dt.hour
    df_trip_ep['quarter'] = df_trip_ep['start_ride_datetime'].dt.quarter
    df_trip_ep['dm'] = df_trip_ep['start_ride_datetime'].dt.days_in_month
    df_trip_ep['minute'] = df_trip_ep['start_ride_datetime'].dt.minute

    keep = ['start_ride_datetime', 'start_day_of_year', 'dm', 'minute', 'start_day_of_week',
            'hour', 'quarter', 'start_reservation_isholiday',
            'startLatitude', 'startLongitude', 'endLatitude',
            'endLongitude', 'distance', 'out_duration_trip_min', 'end_ride_datetime']

    df_trip_ep = df_trip_ep[keep]
    # Create a dictionary to map old column names to new column names
    column_mapping = {
        'start_ride_datetime': 'ts',
        'start_day_of_week': 'dw',
        'start_day_of_year': 'dy',
        'minute': 'minute',
        'start_reservation_isholiday': 'is_holy',
        'startLatitude': 'LATs',
        'startLongitude': 'LONs',
        'endLatitude': 'LATe',
        'endLongitude': 'LONe',
        'distance': 'dis_km',
        'out_duration_trip_min': 'dur_min',
        'end_ride_datetime': 'datetime_end'
    }
    # Rename the columns using the dictionary
    df_trip_ep = df_trip_ep.rename(columns=column_mapping)
    # Sort the DataFrame by the 'ts' column
    df_trip_ep = df_trip_ep.sort_values(by='ts')
    # Set 'ts' as the index
    df_trip_ep = df_trip_ep.set_index('ts')
    df_trip_ep = df_trip_ep[['dy', 'is_holy', 'dm', 'dw', 'hour', 'quarter', 'minute', 'LATs', 'LONs', 'LATe', 'LONe', 'dis_km', 'dur_min',
         'datetime_end']]
    if save:
        df_trip_ep.to_pickle('data/autotel/df_TelAviv_for_gym_episodes.pkl')
    return df_trip_ep


def data_loader(data_dir, file_name):
    """load data"""
    print('Loading data:' + file_name)
    file_extension = os.path.splitext(file_name)[1][1:].lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Get the parent directory
    data_path = os.path.join(parent_dir, data_dir, file_name)
    try:
        if file_extension == 'csv':
            df = pd.read_csv(data_path)
        elif file_extension == 'pkl':
            df = pd.read_pickle(data_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        print('Done -- dataset length = ' + str(len(df)))
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the pickle file: {data_path}")


def filter_df_lat_lon(df, lon_min=34.72, lon_max=34.87, lat_min=32.01, lat_max=32.15):
    # preprocess latitude and longitude data
    print('filtering min(lat,lon) = ' + str((lat_min, lon_min)) + ', and max(lat,lon) = ' + str((lat_max, lon_max)))
    Lon_s, Lat_s, Lon_e, Lat_e = get_lon_lat(df)
    # within the borders of Tel Aviv
    cond1_on_departs = np.logical_and(Lon_s > lon_min, Lon_s <= lon_max)
    cond2_on_departs = np.logical_and(Lat_s > lat_min, Lat_s <= lat_max)
    cond1_on_arrive = np.logical_and(Lon_e > lon_min, Lon_e <= lon_max)
    cond2_on_arrive = np.logical_and(Lat_e > lat_min, Lat_e <= lat_max)
    departure_are_in_zone = np.logical_and(cond1_on_departs, cond2_on_departs)
    arrival_are_in_zone = np.logical_and(cond1_on_arrive, cond2_on_arrive)
    all_filters = np.logical_and(departure_are_in_zone, arrival_are_in_zone)
    df = df[all_filters]
    print('Done -- remaining data length = ' + str(len(df)))
    return df


def get_lon_lat(df):
    Lon_s, Lat_s, = df['startLongitude'], df['startLatitude']
    Lon_e, Lat_e = df['endLongitude'], df['endLatitude']
    return Lon_s, Lat_s, Lon_e, Lat_e



def preprocess_autotel_data(df, max_speed_kmh=130, max_distance_km=70, append_shortest_path_features=False):
    """
    @param df: data frame with mobility data
    @param max_speed_kmh, max_distance_km: removal/filter for maximum velocity and distance
    @param append_shortest_path_features: boolean, if True, append to df the features extracted from a street graph
    @return: data_frame filtered
    """
    print('Pre-processing data')
    # fix datetime format
    date_format = '%d/%m/%Y %H:%M'
    df['end_ride_datetime'] = pd.to_datetime(df['end_ride_datetime'], format=date_format)
    df['start_ride_datetime'] = pd.to_datetime(df['start_ride_datetime'], format=date_format)
    df['start_reservation_datetime'] = pd.to_datetime(df['start_reservation_datetime'], format=date_format)

    # filter to ensure time consistency and remove speed>max_speed_kmh, distance>max_distance_km
    df = df[df['start_ride_datetime'] >= df['start_reservation_datetime']]
    df = df[df['end_ride_datetime'] > df['start_ride_datetime']]
    df['trip_duration'] = df['end_ride_datetime'] - df['start_ride_datetime']
    df['speed_kmh'] = df['distance'] / (df['trip_duration'].dt.seconds / 3600)
    df.drop(columns=['lastIgnitionOffDate', 'firstIgnitionOnDate'], inplace=True)
    df = df[df['speed_kmh'] < max_speed_kmh]
    df = df[df['distance'] < max_distance_km]  # df.drop(df[df['distance'] > max_distance_km].index, inplace=True)

    print('--- filtering speeds <=' + str(max_speed_kmh) + ' [km/h]')
    print('--- filtering distances <=' + str(max_distance_km) + ' [km]')
    df = df.sort_values(by=['start_reservation_datetime'])

    print('add holidays and minutes between reservation and return')
    israel_holidays = holidays.country_holidays('Israel')
    df['out_duration_trip_min'] = df['trip_duration'].values.astype("float64") / (60 * 10 ** 9)
    df['min_between_st_and_res'] = (df['start_ride_datetime'] - df['start_reservation_datetime']).values.astype(
        "float64") / (60 * 10 ** 9)
    df['start_day_of_year'] = df['start_ride_datetime'].dt.dayofyear
    df['start_day_of_week'] = df['start_ride_datetime'].dt.weekday  # [d.weekday() for d in df['start_ride_datetime']]
    df['end_day_of_week'] = df['end_ride_datetime'].dt.weekday  # [d.weekday() for d in df['end_ride_datetime']]
    df['start_reservation_isholiday'] = [d in israel_holidays for d in df['start_reservation_datetime']]
    df['start_ride_isholiday'] = [d in israel_holidays for d in df['start_ride_datetime']]
    df['end_ride_isholiday'] = [d in israel_holidays for d in df['end_ride_datetime']]

    print('--- dropping nan rows')
    df = df.dropna()
    if append_shortest_path_features:
        print('appending shortest path featurs from the street network')
        try:
            with open('TelAviv_trips_dataframe_added_shortest_path_data', 'rb') as fp:
                print('loading Tel Aviv shortest path features')
                df_trip_sp = pickle.load(fp)
                print('- shortest path data loaded')
            cols_to_use = df_trip_sp.columns.difference(df.columns).tolist() + ['startLatitude', 'startLongitude',
                                                                                'endLatitude', 'endLongitude']
            df_merged = df.merge(df_trip_sp[cols_to_use], how='inner',
                                 on=['startLatitude', 'startLongitude', 'endLatitude', 'endLongitude'])
            df_merged = df_merged.dropna()
            # ------------------------------------ filter by dissimilarity between distance travelled ? why?
            # df_merged = df_merged[df_merged['distance'] * 1000 - df_merged['sp_distance_m'] < 10_000]
            # df_merged = df_merged[df_merged['distance'] * 1000 - df_merged['sp_distance_m'] > -5_000]
            df_merged['start_hr'] = df_merged['start_ride_datetime'].dt.hour
            df_merged['start_min'] = df_merged['start_ride_datetime'].dt.minute
            df_merged.drop(columns=['Cluster_start', 'Cluster_end'], inplace=True)
            df_merged.dropna()
            print('Done -- shortest path features attached to df -- processed df length = ' + str(len(df_merged)))
            return df_merged
        except:
            print('Done -- shortest path features not available--- processed df length = ' + str(len(df)))
            return df
    else:
        print('Done -- processed dataset length = ' + str(len(df)))
        return df



AVG_EARTH_RADIUS = 6_371_000

@jit(nopython=True)
def distance_start_end(lat_source, lon_source, lat_targets, lon_targets):
    # Compute the matrix of pairwise differences
    d_lat_mat = lat_source - lat_targets  # lat2.T
    d_lon_mat = lon_source - lon_targets  # lon2.T
    d = sin(d_lat_mat * 0.5) ** 2 + cos(lat_source) * cos(lat_targets) * sin(d_lon_mat * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * arcsin(sqrt(d))
    return h

@jit(nopython=True)
def SOC_end_of_trip(distance_travelled_km, SOC_start, range_percentloss_km):
    """"" Compute SOC at the end of the trip in [%] """
    SOC_end = SOC_start - (1 / range_percentloss_km) * distance_travelled_km *100
    return max(0.0, SOC_end)