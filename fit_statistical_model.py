import matplotlib.pyplot as plt

from wp3_simulate.trip_sampler import *
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats
import pickle


def select_best_distribution(data, verbose=0):
    # Fit distributions to the data
    dist_names = ['expon', 'exponweib',
                  'weibull_min', 'lognorm']
    f = Fitter(data, distributions=dist_names)
    f.fit()
    best_dist = f.get_best(method='bic')  # bic, ad_test, aic
    # Retrieve the distribution name and parameters
    best_dist_name = list(best_dist.keys())[0]
    best_dist_params = best_dist[best_dist_name]
    if verbose > 0:
        # f.summary()
        plt.figure(figsize=(12, 12))
        sbn.ecdfplot(data, color='k')
        for dist_n in f.distributions:
            samples = def_distribution_and_sample(dist_n, f.fitted_param[dist_n], n=100)
            sbn.ecdfplot(samples, label=dist_n)
        plt.xscale('log');
        plt.grid();
        plt.legend();
        plt.xlabel('log-duration');
        plt.ylabel('ecdf')
        plt.show()
    return best_dist_name, best_dist_params


def save_empirical_quantile_model(Trips_df):
    # save quantiles matrix for aggregated booking duration distribution and idle time distribution
    p_values = np.linspace(0, 100, 51)
    Quantiles_idle = np.zeros((24, len(p_values)))
    Quantiles_booking = np.zeros((24, len(p_values)))

    for hr in range(24):
        condition_dis = Trips_df['h_disconnected'] == hr
        condition_con = Trips_df['h_connected'] == hr
        duration_trip_hr = Trips_df[condition_dis]['duration_trip_hr'].values
        duration_idle_hr = Trips_df[condition_con]['duration_between_trips_hr'].values
        STATS = get_sample_stats(duration_idle_hr)
        quantile_p = []

        for p in p_values:
            quantile_p.append(STATS['quantile_' + str(p)])
        Quantiles_idle[hr, :] = quantile_p
        STATS = get_sample_stats(duration_trip_hr)
        quantile_p = []
        for p in p_values:
            quantile_p.append(STATS['quantile_' + str(p)])
        Quantiles_booking[hr, :] = quantile_p

    with open('models/mobility/hourly_stat_model/Quantiles_idle.pickle', 'wb') as file:
        pickle.dump(Quantiles_idle, file)
    with open('models/mobility/hourly_stat_model/Quantiles_booking.pickle', 'wb') as file:
        pickle.dump(Quantiles_booking, file)


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

        with open('models/mobility/hourly_stat_model/Quantiles_idle.pickle', 'rb') as file:
            self.q_idle_duration_hourly = pickle.load(file)
        with open('models/mobility/hourly_stat_model/Quantiles_booking.pickle', 'rb') as file:
            self.q_booking_duration_hourly = pickle.load(file)

        self.n_hours, self.n_quantiles = np.shape(self.q_booking_duration_hourly)
        self.percentiles = np.linspace(0, 100, self.n_quantiles)
        self.sample_fun_booking_dur = [
            sample_from_empirical_dist(self.q_booking_duration_hourly[h, :], self.percentiles) for h in
            range(0, self.n_hours)]
        self.sample_fun_idle_dur = [sample_from_empirical_dist(self.q_idle_duration_hourly[h, :], self.percentiles) for
                                    h in range(0, self.n_hours)]

    def sample_n_durations(self, n_samples: int = 1, hr: int = 0, event_type: str = 'departure'):
        """ sample booking durations and idle durations condtionalt to the hour hr"""
        if event_type == 'departure':
            sample_func = self.sample_fun_booking_dur[hr]
        else:
            sample_func = self.sample_fun_idle_dur[hr]
        return [sample_func(np.random.uniform(0, 100)) for _ in range(n_samples)]


def save_best_distribution_hourly(Trips_df, verbose=0):
    hour = range(24)

    list_best_idle_dist_name, list_best_duration_dist_name = [], []
    list_best_idle_dist_param, list_best_duration_dist_param = [], []
    for hr in hour:
        condition_dis = Trips_df['h_disconnected'] == hr
        condition_con = Trips_df['h_connected'] == hr
        duration_trip_hr = Trips_df[condition_dis]['duration_trip_hr'].values
        duration_idle_hr = Trips_df[condition_con]['duration_between_trips_hr'].values

        # fit idle time distribution (conditional to the hr) each hour
        best_dist_name, best_dist_params = select_best_distribution(duration_trip_hr, verbose=verbose)
        list_best_duration_dist_param.append(best_dist_params)
        list_best_duration_dist_name.append(best_dist_name)
        print(f"Best booking duration distribution: {best_dist_name}")
        print(f"Parameters: {best_dist_params}")

        # fit idle time distribution (conditional to the hr) each hour
        best_dist_name, best_dist_params = select_best_distribution(duration_idle_hr)
        list_best_idle_dist_param.append(best_dist_params)
        list_best_idle_dist_name.append(best_dist_name)
        print(f"Best idle duration distribution: {best_dist_name}")
        print(f"Parameters: {best_dist_params}")

    # Save best_dist to a file
    with open('models/mobility/hourly_stat_model/list_best_duration_dist_param.pickle', 'wb') as file:
        pickle.dump(list_best_duration_dist_param, file)
    with open('models/mobility/hourly_stat_model/list_best_duration_dist_name.pickle', 'wb') as file:
        pickle.dump(list_best_duration_dist_name, file)
    with open('models/mobility/hourly_stat_model/list_best_idle_dist_param.pickle', 'wb') as file:
        pickle.dump(list_best_idle_dist_param, file)
    with open('models/mobility/hourly_stat_model/list_best_idle_dist_name.pickle', 'wb') as file:
        pickle.dump(list_best_idle_dist_name, file)


def get_sample_stats(samples):
    stats_values = {}
    stats_values['mean'] = np.mean(samples)
    stats_values['std'] = np.std(samples)
    stats_values['skewness'] = stats.skew(samples)
    stats_values['kurtosis'] = stats.kurtosis(samples)
    for q in np.linspace(0, 100, 51):
        stats_values['quantile_' + str(q)] = np.quantile(samples, q / 100)
    return stats_values


def def_distribution_and_sample(dist_name, parameters, n):
    dist = getattr(stats, dist_name)
    rv = dist(**parameters)
    sample = rv.rvs(size=n)
    return sample


def get_hourly_stats(list_best_dist_name, list_best_dist_param):
    MU, STD = [], []
    Quantiles = np.zeros((24, len(np.linspace(0, 100, 51))))
    i = 0
    for name, param in zip(list_best_dist_name, list_best_dist_param):
        samples = def_distribution_and_sample(name, param, n=5000)
        stats_values = get_sample_stats(samples)
        MU.append(max(0, stats_values['mean']))
        STD.append(stats_values['std'])
        quantile_p = []
        for p in np.linspace(0, 100, 51):
            quantile_p.append(stats_values['quantile_' + str(p)])

        Quantiles[i, :] = quantile_p
        i += 1
    return MU, STD, Quantiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dso", "--dso_code",
                        type=str, default="ewz",
                        help="dso code you want to filter for")
    args = parser.parse_args()
    dso_code = args.dso_code
    # dso_code = 'Energie Wasser Bern ewb'
    sim_start_time = "2019-02-01"
    sim_end_time = "2019-03-01"
    SAMPLER = games_trips_sampler(dso_label=dso_code, sim_start_time=sim_start_time, sim_end_time=sim_end_time)
    Trips_df = SAMPLER.Trips_df

    save_best_distribution_hourly(Trips_df, verbose=1)
    # load
    with open('models/mobility/hourly_stat_model/list_best_duration_dist_param.pickle', 'rb') as file:
        list_best_duration_dist_param = pickle.load(file)
    with open('models/mobility/hourly_stat_model/list_best_duration_dist_name.pickle', 'rb') as file:
        list_best_duration_dist_name = pickle.load(file)
    with open('models/mobility/hourly_stat_model/list_best_idle_dist_param.pickle', 'rb') as file:
        list_best_idle_dist_param = pickle.load(file)
    with open('models/mobility/hourly_stat_model/list_best_idle_dist_name.pickle', 'rb') as file:
        list_best_idle_dist_name = pickle.load(file)

    MU, STD, QUANTILES = get_hourly_stats(list_best_duration_dist_name, list_best_duration_dist_param)

    plt.figure(figsize=(8, 12))
    plt.plot(MU, 'r', label='\mu')
    # plt.plot([i+j for i, j in zip(MU, STD)], ':r', label = '\mu + \sigma')
    # plt.plot([i-j for i, j in zip(MU, STD)], ':r', label = '\mu + \sigma')
    for q in QUANTILES.T:
        plt.plot(q, ':k')
    # Set the tick locations and labels
    plt.xticks(range(24))
    plt.grid()
    plt.legend()
    plt.show()
