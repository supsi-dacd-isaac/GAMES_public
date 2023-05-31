import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def define_zones(df_sequence, min_max_lat=None, min_max_lon=None, n_disretized_lat_lon=None):
    """ define zones on the map"""
    print('Assign zones to trip data frame')
    if min_max_lat is None:
        min_max_lat = [df_sequence['LAT'].min(), df_sequence['LAT'].max()]
    if min_max_lon is None:
        min_max_lon = [df_sequence['LON'].min(), df_sequence['LON'].max()]
    if n_disretized_lat_lon is None:
        n_dis_lat, n_dis_lon = 3, 3
    elif len(n_disretized_lat_lon) == 1:
        n_dis_lat, n_dis_lon = n_disretized_lat_lon, n_disretized_lat_lon
    elif len(n_disretized_lat_lon) == 2:
        n_dis_lat, n_dis_lon = n_disretized_lat_lon[0], n_disretized_lat_lon[1]
    else:
        n_dis_lat, n_dis_lon = 3, 3
        print('Error, provide (n_disretized_lat_lon)<=2, set to  n_dis_lat, n_dis_lon= 3, 3')
    linspace_lat = np.linspace(min_max_lat[0], min_max_lat[1], n_dis_lat)
    linspace_lon = np.linspace(min_max_lon[0], min_max_lon[1], n_dis_lon)
    df_sequence['row'] = np.searchsorted(linspace_lat, df_sequence['LAT'])
    df_sequence['col'] = np.searchsorted(linspace_lon, df_sequence['LON'])
    df_sequence['zone_lat'] = linspace_lat[df_sequence['row']]
    df_sequence['zone_lon'] = linspace_lon[df_sequence['col']]
    df_sequence['zone_name'] = [str(a) + '-' + str(b) for a, b in zip(df_sequence['row'], df_sequence['col'])]
    print('Done --')
    return df_sequence, linspace_lat, linspace_lon


def parked_prevalence_time_history(df_sequence, id_la=None, jd_lo=None):
    """ ....  """
    cond_lat = df_sequence['row'] > 0 if id_la is None else (df_sequence['row'] == id_la)
    cond_lon = df_sequence['col'] > 0 if jd_lo is None else (df_sequence['col'] == jd_lo)
    is_in_zone = np.logical_and(cond_lat, cond_lon)
    df_in_zone = df_sequence[is_in_zone]
    df_in_zone = df_in_zone.sort_values(by='datetime')
    net_out_trips = []
    if not df_in_zone.empty:  # append +1 if the event is a 'start' and -1 if the event is a return trip
        for d in df_in_zone['is_start']:
            net_out_trips.append(1) if d else net_out_trips.append(-1)
    return df_in_zone['datetime'], np.cumsum(net_out_trips)


def matrix_stats_idle_duration(df_sequence):
    """Define arrays [n_zonesxzones] with mean and standard deviation of the duration events (idle times) in each zone"""
    print('Computing statistical indicator of idle times and trip durations for each zone')
    if 'zone_lat' not in df_sequence.columns or 'zone_lon' not in df_sequence.columns:
        n_zone_default = [5, 5]
        print('Zoning not found in the trip data: assign default zoning: ' + str(n_zone_default))
        df_sequence, _, _ = define_zones(df_sequence, n_disretized_lat_lon=n_zone_default)

    n_lats, n_lons = len(np.unique(df_sequence['zone_lat'])), len(np.unique(df_sequence['zone_lon']))
    mat_mu_idle_min, mat_std_idle_min, mat_sample_size_idle = [np.zeros((n_lons, n_lats)) * np.nan for _ in range(3)]
    mat_mu_trip_min, mat_std_trip_min, mat_sample_size_trip = [np.zeros((n_lons, n_lats)) * np.nan for _ in range(3)]

    grouped_idle = df_sequence[df_sequence['is_start'] == False].groupby(by=['col', 'row'])

    mu_parked = grouped_idle.mean()['duration_min']
    std_parked = grouped_idle.std()['duration_min']
    n_parked_events = grouped_idle.count()['duration_min']

    grouped_trips = df_sequence[df_sequence['is_start']].groupby(by=['col', 'row'])
    mu_trip = grouped_trips.mean()['duration_min']
    std_trip = grouped_trips.std()['duration_min']
    n_trip_event = grouped_trips.count()['duration_min']
    for index in n_parked_events.index:
        mat_mu_idle_min[index] = mu_parked[index]
        mat_std_idle_min[index] = std_parked[index]
        mat_sample_size_idle[index] = n_parked_events[index]

    for index in n_trip_event.index:
        mat_mu_trip_min[index] = mu_trip[index]
        mat_std_trip_min[index] = std_trip[index]
        mat_sample_size_trip[index] = n_trip_event[index]

    Idle_duration_zone_stats, Trips_duration_zone_stats = {}, {}
    Idle_duration_zone_stats['mean'] = np.flip(mat_mu_idle_min, axis=1).T
    Idle_duration_zone_stats['std'] = np.flip(mat_std_idle_min, axis=1).T
    Idle_duration_zone_stats['n_samples'] = np.flip(mat_sample_size_idle, axis=1).T
    Trips_duration_zone_stats['mean'] = np.flip(mat_mu_trip_min, axis=1).T
    Trips_duration_zone_stats['std'] = np.flip(mat_std_trip_min, axis=1).T
    Trips_duration_zone_stats['n_samples'] = np.flip(mat_sample_size_trip, axis=1).T
    print('Done')
    return Idle_duration_zone_stats, Trips_duration_zone_stats


def plot_zone_duration_stats(stat_matrix_1, stat_matrix_2, TelAviv, label1=None, label2=None):
    """ this method visualizes parking time stats on the street map"""
    extent = list(TelAviv.get_minmax_lon_lat())
    fig, ax = plt.subplots(1, 2)
    [TelAviv.street.plot(ax=a, linewidth=0.5, color='k') for a in ax]
    imsh = ax[0].imshow(stat_matrix_1, cmap='Purples', extent=extent)
    # ax[0].set_title('Mean idle time [hrs]')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(imsh, cax=cax)
    if label1 is None:
        cbar.ax.set_ylabel('Mean idle time [minutes]')
    else:
        cbar.ax.set_ylabel(label1)
    imsh = ax[1].imshow(stat_matrix_2, cmap='Greens', extent=extent)
    # ax[1].set_title('STD idle time [hrs]')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(imsh, cax=cax)
    if label1 is None:
        cbar.ax.set_ylabel('Std idle time [minutes]')
    else:
        cbar.ax.set_ylabel(label1)
    fig.tight_layout()
    plt.show()


def plot_density_arrivals_departures_net_out_flows(mat_number_start, mat_number_end, Model_street):
    """plot map with number of events"""
    extent = list(Model_street.get_minmax_lon_lat())
    net_outflow = mat_number_start - mat_number_end
    fig, ax = plt.subplot_mosaic([['upper left', 'right'], ['lower left', 'right']], figsize=(13, 6),
                                 layout="constrained")
    [Model_street.street.plot(ax=ax[a], linewidth=0.5, color='k') for a in ax]
    imsh0 = ax['upper left'].imshow(mat_number_start, cmap='Reds_r', extent=extent)
    ax['upper left'].set_title('Number of departures')
    imsh1 = ax['lower left'].imshow(mat_number_end, cmap='Blues_r', extent=extent)
    ax['lower left'].set_title('Number of arrivals')
    imsh2 = ax['right'].imshow(net_outflow, cmap='seismic',
                               norm=colors.TwoSlopeNorm(vmin=-2000, vcenter=0.0, vmax=2000), extent=extent)
    ax['right'].set_title('Imbalance (depart-arrivals)')
    for a, imsh in zip(['upper left', 'lower left', 'right'], [imsh0, imsh1, imsh2]):
        divider = make_axes_locatable(ax[a])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(imsh, cax=cax)
    plt.show()



def compute_accumulation_traces(df_sequence, linspace_lat, linspace_lon, show=True):
    """ explain what this does"""

    ACCUMULATION = np.zeros((len(linspace_lon), len(linspace_lat)))
    CUMULATIVE_SUM_ZONE, TIME_ZONE, ZONE_NAMEs = [], [], []

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    for id_la, la in enumerate(linspace_lat):
        for jd_lo, lo in enumerate(linspace_lon):
            time, cum_sum = parked_prevalence_time_history(df_sequence, id_la=id_la, jd_lo=jd_lo)
            if len(cum_sum) > 0:
                ACCUMULATION[jd_lo, id_la] = cum_sum[-1]
                if show:
                    rescaled = cum_sum  # /200 # normal scaler
                    if rescaled[-1] > 200:
                        ax[0].plot(time, rescaled, 'r', label=['LAT: {lo:.2f}, LON: {la:.2f}'])
                    elif rescaled[-1] < -200:
                        ax[0].plot(time, rescaled, 'blue', label=['LAT: {lo:.2f}, LON: {la:.2f}'])
                    else:
                        ax[0].plot(time, rescaled, 'k', alpha=0.1, label=[])
            else:
                ACCUMULATION[jd_lo, id_la] = 0
            CUMULATIVE_SUM_ZONE.append(cum_sum)
            TIME_ZONE.append(time)
            ZONE_NAMEs.append(str(id_la) + '-' + str(jd_lo))
    if show:
        ax[0].grid()
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('Accumulation score (net out)')
        sbn.ecdfplot(ACCUMULATION.flatten()[ACCUMULATION.flatten() != 0], ax=ax[1])
        plt.show()
    return ACCUMULATION, CUMULATIVE_SUM_ZONE, TIME_ZONE, ZONE_NAMEs

## check methods -----------
def define_zones_trip_matrix(df, n_disretized_lat_lon=[10, 10]):
    print('Assign zones to trip data frame')

    """df['dayofyear_start'] = df['start_ride_datetime'].dt.dayofyear
    df['dayofweek_start'] = df['start_ride_datetime'].dt.dayofweek
    df['hourofday_start'] = df['start_ride_datetime'].dt.hour
    df['dayofyear_end'] = df['end_ride_datetime'].dt.dayofyear
    df['dayofweek_end'] = df['end_ride_datetime'].dt.dayofweek
    df['hourofday_end'] = df['end_ride_datetime'].dt.hour"""

    # df.rename(columns={"start_day_of_week": "dayofweek", "start_day_of_year": "dayofyear"}, inplace=True)
    df = df.rename(columns={"startLatitude": "LAT", "startLongitude": "LON"})
    df, _, _ = define_zones(df, n_disretized_lat_lon=n_disretized_lat_lon)
    df = df.rename(columns={"row": "row_start", "col": "col_start",
                            "zone_lat": "zone_lat_start", "zone_lon": "zone_lon_start", "zone_name": "zone_name_start",
                            "LAT": "startLatitude", "LON": "startLongitude"})

    df = df.rename(columns={"endLatitude": "LAT", "endLongitude": "LON"})
    df, _, _ = define_zones(df, n_disretized_lat_lon=n_disretized_lat_lon)
    df = df.rename(columns={"row": "row_end", "col": "col_end",
                            "zone_lat": "zone_lat_end", "zone_lon": "zone_lon_end", "zone_name": "zone_name_end",
                            "LAT": "endLatitude",  "LON": "endLongitude"})
    df = df.reset_index(drop=True)
    print('Done --')
    return df
