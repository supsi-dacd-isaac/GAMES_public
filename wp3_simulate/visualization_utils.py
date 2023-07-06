import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sbn
plt.rcParams.update({'font.size': 22})

def plot_relocation_graph(df_relocations, stations_lat_lon, color='green'):
    """ this functio plot a graph with replactions"""
    G = nx.from_pandas_edgelist(df=df_relocations, source='start_station_no', target='end_station_no')
    POS, color_map, edge_cmap = {}, [], []
    for n in G.nodes:
        LATLON = stations_lat_lon[['LAT', 'LON']].loc[[n]]
        POS_n = {n: [LATLON['LAT'].values[0], LATLON['LON'].values[0]]}
        POS = {**POS, **POS_n}
        color_map.append(color)
    for u, v, d in G.edges(data=True):
        d['weight'] = 0 if u == v else 1
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw_networkx(G, pos=POS, edgelist=edges, edge_color=weights, node_color=color_map,
                     node_size=5, with_labels=False,
                     edge_cmap=plt.cm.Blues)
    return G


def plot_clusters(df, cluster_id, names_inp2clust=None):
    """ plot clusters: explain what this does"""
    size_cluster_for_plot = 20
    _, _ = plt.subplots(figsize=(14, 8))
    if names_inp2clust is None:
        names_inp2clust = ['LAT', 'LON']

    Colors = ['green', 'orange', 'brown', 'dodgerblue', 'red', 'black', 'darkorange', 'blue', 'lightgreen']
    if max(cluster_id) + 1 <= len(Colors):
        sbn.scatterplot(data=df, x=names_inp2clust[0], y=names_inp2clust[1],  palette=Colors[:max(cluster_id) + 1],
                        s = df['n_cars'] * size_cluster_for_plot,  hue=cluster_id)
    else:
        sbn.scatterplot(data=df, x=names_inp2clust[0], y=names_inp2clust[1],
                       s= df['n_cars'] * size_cluster_for_plot, hue=cluster_id)
    plt.grid()
    plt.show()


def plot_kde_departures_returns(df_trips,
                                truncate_trip_duration=24,
                                truncate_parked_time=24*3,
                                cmap="Blues"):
    """ kernel density plot: explain what this does"""
    # as a figure of merit, show correlations in the simulated data set

    df_trips = df_trips.iloc[:1_000] # plot pnly a few samples..
    Mask_tr_dur = df_trips['duration_trip_hr'] < truncate_trip_duration
    Mask_between = df_trips['duration_between_trips_hr'] < truncate_parked_time

    fig, AX = plt.subplots(1, 2, figsize=(12, 8))
    sbn.kdeplot(data=df_trips[Mask_tr_dur],
                x = 'h_disconnected', y = 'duration_trip_hr',
                fill=True, cmap=cmap,    cut=0, ax=AX[0])
    AX[0].set_xlabel('Booking hour [hr]')
    AX[0].set_ylabel('Duration (booking) [hr]')
    sbn.kdeplot(data=df_trips[Mask_between],
                x = 'h_connected', y = 'duration_between_trips_hr',
                fill=True, cmap=cmap,  cut=0, ax=AX[1])
    AX[1].set_xlabel('Drop-off hour [hr]')
    AX[1].set_ylabel('Duration (idle) [hr]')
    plt.show()
