import numpy as np
import pandas as pd
import pickle
import os
from pyproj import CRS
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
import osmnx as ox
import seaborn as sns
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

def data_loader(data_dir, file_name):
    """load data"""
    print('Loading data:' + file_name)
    file_extension = os.path.splitext(file_name)[1][1:].lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_dir, file_name)
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


def preprocess_autotel_data(df, max_speed_kmh=130, max_distance_km=70, append_shortest_path_features=False):
    """

    @param df: data frame with mobility data
    @param max_speed_kmh, max_distance_km: removal/filter for maximum velocity and distance
    @param append_shortest_path_features: boolean, if True, append to df the features extracted from a street graph
    @return:
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


def concatenate_departures_and_arrivals(df):
    print('Transform the DataFrame in a longitudinal df:')
    """print('FROM rows : = [(lat,lon)_start, (lat,lon)_end, start_ride_datetime, end_ride_datetime, ...]')
    print('TO rows : = [(LAT,LON), datetime, duration, is_start_event,.. ]')"""
    # prepare  data frame of departure events
    df_start = df[
        ['startLatitude', 'startLongitude', 'start_ride_datetime', 'start_day_of_week', 'start_ride_isholiday',
         'car Id', 'Car Model', 'distance', 'out_duration_trip_min']]
    df_start.insert(0, 'is_start', True)
    df_start = df_start.rename(columns={"startLatitude": "LAT", "startLongitude": "LON",
                                        "start_ride_datetime": "datetime", "start_day_of_week": "week_day",
                                        "start_ride_isholiday": "is_holiday",
                                        "out_duration_trip_min": "duration_min"})

    # prepare return data frame
    df_end = df[['endLatitude', 'endLongitude', 'end_ride_datetime', 'end_day_of_week', 'end_ride_isholiday',
                 'car Id', 'Car Model']]
    df_end.insert(0, 'is_start', False)
    df_end.insert(7, 'distance', 0)
    df_end.insert(8, 'duration_min', np.nan)
    df_end = df_end.rename(columns={"endLatitude": "LAT", "endLongitude": "LON",
                                    "end_ride_datetime": "datetime", "end_day_of_week": "week_day",
                                    "end_ride_isholiday": "is_holiday"})
    df_sequence = pd.concat([df_start, df_end]).sort_values(by='datetime', ignore_index=True)
    df_sequence['minute'] = df_sequence['datetime'].dt.minute
    df_sequence['hour'] = df_sequence['datetime'].dt.hour

    df_sequence = df_sequence.rename(columns={"car Id": "car_id", "Car Model": "car_model"})
    print('Done')
    return df_sequence


def append_inflow_idx(df_sequence):
    in_out_sign = []
    for d in df_sequence['is_start']:
        in_out_sign.append(1) if d else in_out_sign.append(-1)
    df_sequence['in_out_sign'] = in_out_sign
    return df_sequence


def append_event_duration(df_sequence):
    """ EXAMPLE:
    import scipy
    df_sequence, DICTIONARY_idle_times = append_event_duration(df_sequence)
    car_usage_rate = [len(DICTIONARY_idle_times[d]) for d in DICTIONARY_idle_times]
    mean_park_dur =  [np.mean(DICTIONARY_idle_times[d]) for d in DICTIONARY_idle_times]
    plt.scatter(ev_list, car_usage_rate); plt.xlabel('EV index'); plt.ylabel('number of trips'); plt.show()
    def mean_confidence_interval_min_max_mu_std(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        st = np.std(a)
        min, max = np.min(a), np.max(a)
        m, se = np.mean(a), scipy.stats.sem(a) # standard error of the mean
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h, min, max, m-st, m+st
    m_CI =  [mean_confidence_interval_min_max_mu_std(d, confidence=0.95) for d in PARKED_DAYS]
    plt.scatter(ev_list,[a[0] for a in m_CI],c='r', label='mean')
    #plt.scatter(ev_list,[a[1] for a in m_CI],c='b', label='mean - ci')
    #plt.scatter(ev_list,[a[2] for a in m_CI],c='k', label='mean + ci')
    plt.scatter(ev_list,[a[3] for a in m_CI],c='k', label='min')
    plt.scatter(ev_list,[a[4] for a in m_CI],c='k', label='max')
    plt.scatter(ev_list,[a[5] for a in m_CI],c='b', label='mu+std')
    plt.scatter(ev_list,[a[6] for a in m_CI],c='b', label='mu+std');
    plt.xlabel('EV index'); plt.ylabel('mean parked days'); plt.show()
    for d in PARKED_DAYS:
        sns.ecdfplot([k+0.01 for k in d], log_scale= True)
    plt.show()
    """
    print("compute mobility event duration (trips/idle times) in minutes")

    def swap_rows(df, row1, row2):
        df.iloc[row1], df.iloc[row2] = df.iloc[row2].copy(), df.iloc[row1].copy()
        return df

    ev_list = np.unique(df_sequence['car_id'])
    DICTIONARY_EVENT_DURATINS = {}
    for car_id in ev_list:
        df_ev = df_sequence[df_sequence['car_id'] == car_id]
        df_ev = df_ev.sort_values(by='datetime')

        # swap order to have Departure, Arrival, Departure, Arrivals,....
        Sink_return_departures = np.where(np.cumsum(df_ev['in_out_sign']) == 2)[0]
        for row in Sink_return_departures:
            df_ev = swap_rows(df_ev, row, row + 1)

        durations_min = (df_ev['datetime'].diff().dt.seconds[1:] / 60).tolist()
        parked_durations = durations_min[1::2]
        # append to the last event (idle time) a mean duration
        DICTIONARY_EVENT_DURATINS[car_id] = parked_durations
        durations_min.append(np.mean(parked_durations))
        df_sequence.loc[df_sequence['car_id'] == car_id, 'duration_min'] = durations_min
    print("Done -- event durations")
    return df_sequence, DICTIONARY_EVENT_DURATINS



def preprocess_trip_data_frame(df_with_trips,lon_min,lon_max,lat_min,lat_max):
    df = filter_df_lat_lon(df_with_trips, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    df = preprocess_autotel_data(df, max_speed_kmh=130, max_distance_km=70, append_shortest_path_features=False)
    df_sequence = concatenate_departures_and_arrivals(df)
    df_sequence = append_inflow_idx(df_sequence)
    df_sequence, _ = append_event_duration(df_sequence)
    return df, df_sequence


## ------ check the following methods --- ##

def compute_flows_between_zones(df):
    """This function returns a rectangular matrix with from-to zones (rows-cols) """
    print("computing flows between zones")
    # group by the starting and ending zones and count the number of occurrences
    grouped_df = df.groupby(['zone_name_start', 'zone_name_end']).size().reset_index(name='count')
    # create a pivot flows table to reshape the data
    pivot_df_flows = grouped_df.pivot(index='zone_name_start', columns='zone_name_end', values='count')
    # fill NaN values with zeros
    pivot_df_flows.fillna(0, inplace=True)
    pivot_df_flows.rename_axis(columns='zone_name_end', inplace=True)
    pivot_df_flows.rename_axis(index='zone_name_start', inplace=True)
    print("Done --")
    return pivot_df_flows


def kde_event_duration_per_hour(df_sequence, show=True):
    """ explain method """
    hourly_pdf_model = []
    if show:
        fig, ax = plt.subplots(1, 2)
        sns.kdeplot(df_sequence[(df_sequence['duration_min'] < 60) & (df_sequence['is_start'] == True)][:5_000],
                    x='hour',
                    y='duration_min', cut=0, fill=True, alpha=.9, ax=ax[0])
        sns.kdeplot(df_sequence[(df_sequence['duration_min'] < 2_000) & (df_sequence['is_start'] == False)][:5_000],
                    x='hour', y='duration_min', cut=0, fill=True, alpha=.9, ax=ax[1])
        ax[0].set_xlabel('h_disconnected')
        ax[0].set_ylabel('trip duration [min]')
        ax[1].set_xlabel('h_connected')
        ax[1].set_ylabel('idle time [min]')
        plt.show()
    return hourly_pdf_model


def load_street_map(query: str = 'Tel Aviv, Israel'):
    # load Tel Aviv street Map as the default
    # tags = {'admin_level': '5'}
    street_graph = ox.graph_from_place(query, network_type='drive')
    streets_graph = ox.projection.project_graph(street_graph)
    streets = ox.graph_to_gdfs(ox.get_undirected(streets_graph),
                               nodes=False, edges=True,
                               node_geometry=False, fill_edge_geometry=True)
    crs = CRS.from_string('EPSG:4326')
    streets = streets.to_crs(crs=crs)
    return streets


def get_lon_lat(df):
    Lon_s, Lat_s, = df['startLongitude'], df['startLatitude']
    Lon_e, Lat_e = df['endLongitude'], df['endLatitude']
    return Lon_s, Lat_s, Lon_e, Lat_e


def daily_stats(df):
    ntrips = []
    number_of_hourly_trips_x_day = []  # 3 clusters
    for day in range(365):
        temp = df[np.logical_and(df['start_ride_datetime'].dt.dayofyear <= day + 1,
                                 df['start_ride_datetime'].dt.dayofyear > day)]
        ntrips.append(temp.shape[0])
        day_stats = temp.groupby(['hr_ride_start']).size()
        number_of_hourly_trips_x_day.append(day_stats)
        if len(day_stats) > 24:
            plt.plot(day_stats, 'r')
        else:
            plt.plot(day_stats, 'b', alpha=0.1)
    return ntrips, number_of_hourly_trips_x_day


def compute_trip_hours(df):
    return df['trip_duration'].dt.seconds / 3600


def get_departure_arrival_hours(df):
    return df['start_ride_datetime'].dt.hour, df['end_ride_datetime'].dt.hour


def f_cluster(row):
    # 3 clusters A:= Southern part of Tel Aviv including Jaffa and South & East, B:= Central, C := Northern area
    Lat_to_cluster = [32.065, 32.10]
    return 0 if (row < Lat_to_cluster[0]) else 2 if (row >= Lat_to_cluster[1]) else 1


def append_clusters_id(df):
    print(' appedn 3-clusters:'
          'A:= Southern part of Tel Aviv including Jaffa and South & East,'
          'B:= Central area,'
          ' C := Northern area of the city')
    df['Cluster_start'] = df['startLatitude'].apply(f_cluster)
    df['Cluster_end'] = df['endLatitude'].apply(f_cluster)


def compute_flow_matrix(df):
    flow_mat_static = df.groupby(['Cluster_end', 'Cluster_start']).size() / len(df)
    flow_mat_hourly = df.groupby(['Cluster_end', 'Cluster_start', 'hr_ride_start']).size() / len(df)
    return flow_mat_static, flow_mat_hourly, len(df)


# ----- Plotting/visualization methods ---------- #
def plot_TelAviv_arrival_and_departure_nodes(df_filtered, streets):
    _, ax = plt.subplots(1, 4, figsize=(20, 15))
    titles = ['Tel Aviv Streets', 'Tel Aviv departure nodes', 'Tel Aviv arrival nodes', 'Tel Aviv trip edges']
    for a, ti in zip(ax, titles):
        streets.plot(ax=a, linewidth=0.5)
        a.set_title(ti)
    Lon_s_filter, Lat_s_filter, Lon_end_filter, Lat_end_filter = get_lon_lat(df_filtered)
    ax[1].scatter(Lon_s_filter, Lat_s_filter, s=3, marker='o', c='b', edgecolors=None, alpha=0.01)
    ax[2].scatter(Lon_end_filter, Lat_end_filter, s=3, marker='o', c='r', edgecolors=None, alpha=0.01)
    for x0, y0, x1, y1, k in zip(Lon_s_filter, Lon_end_filter, Lat_s_filter, Lat_end_filter, range(20_000)):
        ax[3].plot((y0, x0), (y1, x1), 'k', alpha=0.01)
    plt.rcParams.update({'font.size': 22})
    plt.show()


def plot_kde_24_hr(gdf_streets, df_filtered, n_trips_2plot: int = 500):
    fig, AXIS = plt.subplots(4, 6, figsize=(20, 15))
    start_trip_hr, end_trip_hr = get_departure_arrival_hours(df_filtered)
    plt.rcParams.update({'font.size': 14})
    for hr, ax in enumerate(fig.axes):
        Lon_s, Lat_s, Lon_e, Lat_e = get_lon_lat(
            df_filtered[np.logical_and(np.array(end_trip_hr) == hr, np.array(start_trip_hr) == hr)])
        gdf_streets.plot(ax=ax, linewidth=0.8, color='k')
        sns.kdeplot(x=Lon_e[:n_trips_2plot], y=Lat_e[:n_trips_2plot], color='b', alpha=.5, ax=ax)
        sns.kdeplot(x=Lon_s[:n_trips_2plot], y=Lat_s[:n_trips_2plot], color='r', alpha=.5, ax=ax)
        ax.set_title('arrived at: ' + str(hr) + ' hr')
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.show()


def get_daily_profiles_data(df_sequence):
    """ This function plots the historical number of arrivals and departures/arrivals per hour and day of the week """
    df_sequence['dayofyear'] = df_sequence['datetime'].dt.dayofyear
    df_sequence['hour'] = df_sequence['datetime'].dt.hour
    df_sequence['year'] = df_sequence['datetime'].dt.year

    # vehicle features
    car_ids = df_sequence['car_id'].unique()
    n_cars = len(car_ids)
    car_is_active = np.zeros((3, n_cars))  # 0 if the car is not in the fleet car_id otherwise
    car_is_used = np.zeros((3 * 365, n_cars))  # 0 if the car is not used during a day, car_id otherwise

    mat_time_features = np.zeros((3 * 365, 2))
    mat_departures_day, mat_arrivals_day = [np.zeros((3 * 365, 24)) for _ in range(2)]

    mat_idletime_day, mat_durations_day, mat_distance_day = [np.zeros((3 * 365, 24)) * np.nan for _ in range(3)]
    count_days = 0
    print('Preparing matrix of daily arrivals departures and vehicle presence')
    for id_year, year in enumerate(np.unique(df_sequence['year'])):
        df_year = df_sequence[df_sequence['year'] == year]
        ids_active_vehicles = df_year['car_id'].unique().tolist()
        car_is_active[id_year, [np.where(id == car_ids)[0][0] for id in ids_active_vehicles]] = ids_active_vehicles

        unique_days_year = np.unique(df_year['dayofyear'])
        for day in unique_days_year:
            df_dayofyear = df_year[df_year['dayofyear'] == day]
            grouped_hourly_departures = df_dayofyear[df_dayofyear['is_start'] == True].groupby(by=['hour'])
            grouped_hourly_arrivals = df_dayofyear[df_dayofyear['is_start'] == False].groupby(by=['hour'])

            # get total hourly demand (departures) and total hourly drop-offs (arrivals)
            departures_day = grouped_hourly_departures['in_out_sign'].sum()
            arrivals_day = -grouped_hourly_arrivals['in_out_sign'].sum()
            mat_departures_day[count_days, departures_day.index.values] = departures_day
            mat_arrivals_day[count_days, arrivals_day.index.values] = arrivals_day

            # mean of duration and distance travelled
            mean_distance = grouped_hourly_departures['distance'].mean()
            mean_trip_duration = grouped_hourly_departures['duration_min'].mean()
            mean_idle_time_hr = grouped_hourly_arrivals['duration_min'].mean()

            mat_idletime_day[count_days, mean_distance.index.values] = mean_distance
            mat_durations_day[count_days, mean_trip_duration.index.values] = mean_trip_duration
            mat_distance_day[count_days, mean_idle_time_hr.index.values] = mean_idle_time_hr

            # find out which car was used during the day
            id_active_cars_during_day = df_dayofyear['car_id'].unique()

            car_is_used[count_days, [np.where(id == car_ids)[0][0] for id in
                                     id_active_cars_during_day]] = id_active_cars_during_day

            # it is a new day...get featues
            mat_time_features[count_days, :] = df_dayofyear[['week_day', 'is_holiday']].iloc[0].astype(
                  'float32').values  # week id is holiday

            # is a new day...
            count_days += 1

    print('Done')
    # get rid of extra rows previously allocated
    mat_departures_day = mat_departures_day[:count_days, :]
    mat_arrivals_day = mat_arrivals_day[:count_days, :]
    mat_idletime_day = mat_idletime_day[:count_days, :]
    mat_durations_day = mat_durations_day[:count_days, :]
    mat_distance_day = mat_distance_day[:count_days, :]
    car_is_used = car_is_used[:count_days, :] # n_days x n_cars each day  otherwise
    # mat_time_features = mat_time_features[:count_days, :]
    MU_dep = np.mean(mat_departures_day, axis=0)
    MU_arr = np.mean(mat_arrivals_day, axis=0)

    results = {'mean_daily_departures': MU_dep, 'mean_daily_arrivals': MU_arr, 'car_id_parked_per_day': car_is_used,
               'mat_time_features': mat_time_features,
               'matrix_mean_idle': mat_idletime_day,
               'matrix_mean_durations': mat_durations_day, 'matrix_mean_distances': mat_distance_day,
               'matrix_daily_arrivals': mat_arrivals_day, 'matrix_daily_departures': mat_departures_day}
    return results


def plot_kde_departures_arrivals(streets, df_filtered, hr: int = 18, n_trips_2plot: int = 5_000):
    _, ax = plt.subplots(1, 2, figsize=(20, 15))
    streets.plot(ax=ax[0], linewidth=0.5)
    streets.plot(ax=ax[1], linewidth=0.5)
    start_trip_hr, end_trip_hr = get_departure_arrival_hours(df_filtered)
    if 24 >= hr >= 0:
        df_filtered = df_filtered[np.logical_and(np.array(end_trip_hr) == hr, np.array(start_trip_hr) == hr)]
    else:
        print('invalid t must be integer')
        t = None
    Lon_s, Lat_s, Lon_e, Lat_e = get_lon_lat(df_filtered)
    sns.kdeplot(x=Lon_s[:n_trips_2plot], y=Lat_s[:n_trips_2plot], ax=ax[0], fill=True, cmap='Blues')
    ax[0].set_title('kde departure distribution|' + str(hr) + ' hr')
    sns.kdeplot(x=Lon_e[:n_trips_2plot], y=Lat_e[:n_trips_2plot], ax=ax[1], fill=True, cmap='Reds')
    ax[1].set_title('kde arrival distribution|' + str(hr) + ' hr')
    plt.rcParams.update({'font.size': 22})
    plt.show()
    # plt.savefig('KDE_arrival_depart_TelAviv_startat_' + str(hr) + 'hr.svg')

def create_noisy_tip_data(df,
                          sample_size=10_000,
                          noise_std=0.005,
                          start_date='01/12/2021 00:00',
                          end_date='31/12/2022 00:00'):
    """Create a sample from the DataFrame and add noise to it"""
    mask = (df['start_ride_datetime']>= start_date) & (df['end_ride_datetime'] <= end_date)
    subset = df[mask]
    sample = subset.sample(n=sample_size)  # Select a random sample from the DataFrame
    # Add noise to the sample
    sample['reservation_id'] = [i for i in range(sample_size)]
    sample = sample.set_index('reservation_id')
    for column_name in ['car Id', 'Car Model']:
        sample[column_name] = pd.factorize(sample[column_name])[0]
    sample['startLatitude'] = sample['startLatitude'] + np.random.normal(0, noise_std, size=sample_size)
    sample['startLongitude'] = sample['startLongitude'] + np.random.normal(0, noise_std, size=sample_size)
    sample['endLatitude'] = sample['endLatitude'] + np.random.normal(0, noise_std, size=sample_size)
    sample['endLongitude'] = sample['endLongitude'] + np.random.normal(0, noise_std, size=sample_size)
    return sample


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)
        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


if __name__ == '__main__':
    df = data_loader(data_dir='datasets', file_name='autotel_2021_2022.pkl')
    df = preprocess_autotel_data(df)
    append_clusters_id(df)

    df_filter = filter_df_lat_lon(df)

    trip_duration_hr = df['trip_duration'].dt.seconds / 3600
    Lon_s, Lat_s, Lon_e, Lat_e = get_lon_lat(df_filter)

    streets = load_street_map('Tel Aviv, Israel')
    _, ax = plt.subplots(1, 1, figsize=(20, 15))
    streets.plot(ax=ax, linewidth=2)
    plt.show()

    plot_TelAviv_arrival_and_departure_nodes(df_filter, streets)
    n_trips_2plot = 500
    for hr in range(10):
        plot_kde_departures_arrivals(streets, df_filter, hr=hr, n_trips_2plot=n_trips_2plot)
    plot_kde_24_hr(streets, df_filter, n_trips_2plot=n_trips_2plot)
    flow_mat_static, flow_mat_hourly, ntrips = compute_flow_matrix(df_filter)
    flow_mat_hourly.to_csv('flow_mat_hourly_TelAviv.csv')
    flow_mat_static.to_csv('flow_mat_static_TelAviv.csv')

    df_filter = df_filter.sort_values(by='start_ride_datetime', ascending=True)


    def hour_of_year(dt):
        beginning_of_year = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
        return (dt - beginning_of_year).total_seconds() // 3600


 

