"""from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import residuals_plot
from yellowbrick.model_selection import FeatureImportances
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import (RandomForestRegressor)#,  AdaBoostRegressor,  GradientBoostingRegressor,   HistGradientBoostingRegressor)
import numpy as np

class MobilityDemandForecaster:
    def __init__(self, n_workers: int = 4):
        # Create the RandomForestRegressor model
        self.model = RandomForestRegressor(random_state=0, n_jobs=n_workers, verbose=1)
        self.residuals = None  # Store the true residuals
        self.best_parameters = None


    def fit(self, X, y):
        param_grid = dict(n_estimators=[50],# Number of trees
                          max_depth=[50, 150],# Max depth of the tree
                          min_samples_split=[2, 5])  # Minimum number of samples required to split a node
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        # Get the best model and its parameters
        self.model = grid_search.best_estimator_
        self.best_parameters = grid_search.best_params_

        # Evaluate the best model on the test set
        y_pred = self.model.predict(X)# Calculate the true residuals during training
        self.residuals = y - y_pred

    def predict(self, X):
        y_pred = np.maximum(self.model.predict(X), 0)
        y_pred_low = np.maximum(y_pred+np.quantile(self.residuals, 0.05), 0)
        y_pred_up = y_pred+np.quantile(self.residuals, 0.95)
        uncertainty = [y_pred_low, y_pred_up]  # Assuming normal distribution, 95% confidence
        return y_pred, uncertainty




def historical_hourly_demand(df_sequence, show=True):
    """ This function plots the historical number of arrivals and departures/arrivals per hour and day of the week """
    # ------------------------------------ historical hourly demand curves in a day --------------------------- #
    df_sequence['dayofyear'] = df_sequence['datetime'].dt.dayofyear
    df_sequence['hour'] = df_sequence['datetime'].dt.hour
    df_sequence['year'] = df_sequence['datetime'].dt.year
    unique_days = np.unique(df_sequence['dayofyear'])
    fig, ax = plt.subplots(1 + 7, 3, figsize=(10, 14))
    MU_dep, MU_arr, MU_outflow = [], [], []
    STD_dep, STD_arr, STD_outflow = [], [], []
    matrix_all_departures, matrix_all_arrivals, matrix_fri_sat_departures, matrix_fri_sat_arrivals = [np.zeros((1, 24)) * np.nan for _ in range(4)]
    for w_d in np.unique(df_sequence['week_day']):
        mat_departures_day, mat_arrivals_day = [np.zeros((2 * len(unique_days), 24)) * np.nan for _ in range(2)]
        mat_departures_fr_saturday, mat_arrivals_fr_saturday = [np.zeros((1_000, 24)) * np.nan for _ in range(2)]
        count_fridays_saturdays = 0
        for id_year, year in enumerate([2021, 2022]):
            df_year = df_sequence[df_sequence['year'] == year]
            for id, day in enumerate(unique_days):
                df_dayofyear = df_year[df_year['dayofyear'] == day]
                # USE THIS TO check number of departures and arrivals
                departures_day = df_dayofyear[df_dayofyear['is_start'] == True].groupby(by=['hour'])[
                    'in_out_sign'].sum()
                arrivals_day = -df_dayofyear[df_dayofyear['is_start'] == False].groupby(by=['hour'])[
                    'in_out_sign'].sum()
                mat_departures_day[id + 365 * id_year, departures_day.index.values] = departures_day
                mat_arrivals_day[id + 365 * id_year, arrivals_day.index.values] = arrivals_day
                if df_dayofyear['week_day'].iloc[0] == w_d:
                    mat_departures_fr_saturday[count_fridays_saturdays, departures_day.index.values] = departures_day
                    mat_arrivals_fr_saturday[count_fridays_saturdays, arrivals_day.index.values] = arrivals_day
                    count_fridays_saturdays += 1
        mat_departures_fr_saturday = mat_departures_fr_saturday[:count_fridays_saturdays, :]
        mat_arrivals_frid_sat = mat_arrivals_fr_saturday[:count_fridays_saturdays, :]

        matrix_all_departures = np.vstack([matrix_all_departures, mat_departures_day])
        matrix_all_arrivals = np.vstack([matrix_all_arrivals, mat_arrivals_day])
        matrix_fri_sat_departures = np.vstack([matrix_fri_sat_departures, mat_departures_fr_saturday])
        matrix_fri_sat_arrivals  = np.vstack([matrix_fri_sat_arrivals, mat_arrivals_frid_sat])


        if w_d == 0:
            ax[0][0].plot(mat_departures_day.T, color='r', alpha=0.05)
            ax[0][0].plot(np.nanmean(mat_departures_day, axis=0), color='k', alpha=0.75)
            ax[0][0].set_xlabel('hour')
            ax[0][0].set_title('number of departures')
            ax[0][1].plot(mat_arrivals_day.T, color='b', alpha=0.05)
            ax[0][1].set_xlabel('hour')
            ax[0][1].set_title('number of arrivals')
            out_flow = mat_departures_day - mat_arrivals_day
            ax[0][2].plot(out_flow.T, c='g', alpha=0.05)
            ax[0][2].set_xlabel('hour')
            ax[0][2].set_title('net outflow (departures-arrivals)')

            mu_dep_d, std_dep_d = np.nanmean(mat_departures_day, axis=0), np.nanstd(mat_departures_day, axis=0)
            mu_arriv_d, std_arriv_d = np.nanmean(mat_arrivals_day, axis=0), np.nanstd(mat_arrivals_day, axis=0)
            mu_outflow, std_outflow = np.nanmean(out_flow, axis=0), np.nanstd(out_flow, axis=0)


            plot_mean_std_process(mu_dep_d, std_dep_d, ax[0][0])
            plot_mean_std_process(mu_arriv_d, std_arriv_d, ax[0][1])
            plot_mean_std_process(mu_outflow, std_outflow, ax[0][2])

        [ax[w_d + 1][0].plot(d, 'r', alpha=0.05) for d in mat_departures_fr_saturday]
        [ax[w_d + 1][1].plot(a, 'b', alpha=0.05) for a in mat_arrivals_frid_sat]
        [ax[w_d + 1][2].plot(d - a, 'g', alpha=0.05) for d, a in
         zip(mat_departures_fr_saturday, mat_arrivals_frid_sat)]

        mu_d_w, mu_a_w, mu_out_w = np.nanmean(mat_departures_fr_saturday, axis=0), np.nanmean(mat_arrivals_frid_sat,
                                                                                              axis=0), np.nanmean(
            mat_departures_fr_saturday - mat_arrivals_frid_sat, axis=0)
        std_d_w, std_a_w, std_out_w = np.nanstd(mat_departures_fr_saturday, axis=0), np.nanstd(mat_arrivals_frid_sat,
                                                                                               axis=0), np.nanstd(
            mat_departures_fr_saturday - mat_arrivals_frid_sat, axis=0)

        plot_mean_std_process(mu_d_w, std_d_w, ax[w_d + 1][0])
        plot_mean_std_process(mu_a_w, std_a_w, ax[w_d + 1][1])
        plot_mean_std_process(mu_out_w, std_out_w, ax[w_d + 1][2])

        MU_dep.append(mu_d_w)
        MU_arr.append(mu_a_w)
        MU_outflow.append(mu_out_w)

        STD_dep.append(std_d_w)
        STD_arr.append(std_a_w)
        STD_outflow.append(std_out_w)

        ax[w_d + 1][0].set_title('Weekday-' + str(w_d) + ' depa ')
        ax[w_d + 1][1].set_title('Weekday-' + str(w_d) + ' arri ')
        ax[w_d + 1][2].set_title('Weekday-' + str(w_d) + ' net out')
        ax[w_d + 1][0].set_ylim(0, 140)
        ax[w_d + 1][1].set_ylim(0, 140)
        ax[w_d + 1][2].set_ylim(-35, 35)
    fig.tight_layout()
    if show:
        plt.show()

    results = {'mu_departures_daily': MU_dep, 'std_departures_daily': STD_dep, 'mu_arrivals_daily': MU_arr,
               'std_arrivals_daily': STD_arr, 'mu_outmat_arrivals_fr_saturdayflow': MU_outflow, 'std_outflow': STD_outflow,
               'matrix_arrivals': matrix_all_arrivals[1:, :], 'matrix_departures': matrix_all_departures[1:, :],
               'matrix_arrivals_fri_sat': matrix_fri_sat_arrivals[1:, :],
               'matrix_departures_fri_sat': matrix_fri_sat_departures[1:, :]}
    return results

def df_accumulation_score_in_zones(df_sequence, linspace_lat, linspace_lon):
    """ build dataframe with accumulation scores and datetimes"""
    #   compute accumulation (unbalance score)
    ACCUMULATION, CUMULATIVE_SUM_ZONE, TIME_ZONE, ZONE_NAMEs = \
        compute_accumulation_traces(df_sequence, linspace_lat, linspace_lon, show=True)

    n_events_per_zone = [len(p) for p in CUMULATIVE_SUM_ZONE]
    max_sample_size = np.max([len(d) for d in CUMULATIVE_SUM_ZONE])
    df = pd.DataFrame(df_sequence['datetime'], columns=['datetime'])
    new_cols = pd.DataFrame(np.zeros((len(df_sequence['datetime']), len(ZONE_NAMEs))) * np.nan, columns=ZONE_NAMEs)
    df = pd.concat([df, new_cols], axis=1)
    # merge the data using merge_asof
    print('Prepare data_frame of vehicle presence')
    for d, dates, name in zip(CUMULATIVE_SUM_ZONE, TIME_ZONE, ZONE_NAMEs):
        df[name] = np.nan
        if not dates.empty:
            df.loc[dates.index, name] = d.astype(int)
            print('Zone: ' + name + ' - assigned')
    df.loc[0, df.loc[0, :].isna()] = int(0)  # assign zero events to the first row if nan
    df = df.interpolate(method='ffill')  # interpolate
    return df

def ecdf_plot_duration(df):
    """ plot ecdf of trip duration for going to zones i to zones j, for all i,j"""
    Predicted_duration, true_duration, trip_names = [], [], []
    get_colors = lambda n: ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(n)]
    df['start_zone'] = df['row_start'].astype(str) + '-' + df['col_start'].astype(str)
    df['end_zone'] = df['row_end'].astype(str) + '-' + df['col_end'].astype(str)
    y = df['trip_duration'].dt.seconds / 60
    for i in np.unique(df['start_zone']):
        for j in np.unique(df['end_zone']):
            trip_names.append(['from_' + str(i) + '_to_' + str(j)])
            cond_ij = np.logical_and(df['start_zone'] == i, df['end_zone'] == j)
            true_duration.append(y[cond_ij])

    # plot ecdf
    for true_ij, trip_name in zip(true_duration, trip_names):
        x_t = np.sort(true_ij)
        cdf_t = np.arange(len(x_t)) / float(len(x_t))
        col = get_colors(1)[0]
        plt.plot(x_t, cdf_t, c=col, label=[trip_name[0] + '_true'], linewidth=1.0)
    plt.title('duration cdf')
    plt.xlabel('duration [min]')
    plt.ylabel('cdf')
    plt.show()

def plot_mean_std_process(mu, std, axis):
    axis.plot(mu, color='k', alpha=0.95)
    axis.plot(mu + std, ':k', alpha=0.95)
    axis.plot(mu - std, ':k', alpha=0.95)



# ---------- analyze relocations
def is_start_in_end_circle(x1, y1, xc, yc, r=0.0002):
    distance = np.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
    if distance <= r:
        return True
    else:
        return False


def analyze_relocations(df_sequence,
                        show: bool = False,
                        car_id_to_plot: int = 1,
                        skip_analysis: bool = False):
    """ explain method """
    percentage_of_trips_after_relocation = []
    if not skip_analysis:
        car_ids = np.unique(df_sequence['car Id'])
        for car_id in car_ids:
            df_car_id = df_sequence[df_sequence['car Id'] == car_id]
            counter, was_relocated = 0, []
            for row in df_car_id[['LAT', 'LON', 'is_start']].iterrows():
                counter += 1
                lat, lon, is_st = row[1]['LAT'], row[1]['LON'], row[1]['is_start']
                if counter % 3_000 == 0:
                    print('car_id:' + str(car_id) + ', excecuted trips:' + str(counter))
                if is_st:  # if is a start trip event
                    LAT_start, LON_start = lat, lon
                    if counter > 1:
                        is_not_relocated = is_start_in_end_circle(LON_start, LAT_start, LON_end, LAT_end)
                        was_relocated.append(is_not_relocated == False)
                else:  # if is a end trip (return) event
                    LAT_end, LON_end = lat, lon
            percentage_of_trips_after_relocation.append(np.mean(was_relocated))

    if show:
        fig, ax = plt.subplots(figsize=(18, 10))
        TelAviv.street.plot(ax=ax, linewidth=0.5, color='k')
        plt.title('100 trips for the vehicle ID' + str(car_id_to_plot))
        counter = 0
        df_car_id = df_sequence[df_sequence['car Id'] == car_id_to_plot]
        df_car_id = df_car_id[:200]  # keep only 200 trips to visualize the example
        for row in df_car_id[['LAT', 'LON', 'is_start']].iterrows():
            counter += 1
            lat, lon, is_st = row[1]['LAT'], row[1]['LON'], row[1]['is_start']
            if is_st:  # if is a start trip event
                LAT_start, LON_start = lat, lon
                ax.scatter(lon, lat, s=50, c='r')
            else:
                ax.scatter(lon, lat, s=100, c='b')
                ax.plot([LON_start, lon], [LAT_start, lat], 'k', alpha=0.2)
        plt.show()
    return percentage_of_trips_after_relocation


