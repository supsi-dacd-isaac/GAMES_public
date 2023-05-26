import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from pyproj import CRS


class Macroscopic_traffic_model:
    def __init__(self, geographical_area: str = 'Tel Aviv, Israel',
                 crs_format_str: str = 'EPSG:4326'):
        # Group trip duration and distance by start and end clusters
        self.query = geographical_area
        self.crs_format = CRS.from_string(crs_format_str)
        self.G_drive, self.street = self.get_street_drive_graph()
        # self.trips = self.Trips(Graph=self.G_drive, trips_data=trips_data)
        # trips features
        self.start_nodes = []
        self.end_nodes = []

    def get_nearest_nodes(self, df_trips):
        start_vals = df_trips[['startLongitude', 'startLatitude']].values
        end_vals = df_trips[['endLongitude', 'endLatitude']].values
        self.start_nodes = ox.nearest_nodes(self.G_drive, start_vals[:, 0], start_vals[:, 1])
        self.end_nodes = ox.nearest_nodes(self.G_drive, end_vals[:, 0], end_vals[:, 1])
        return self.start_nodes, self.end_nodes

    def get_street_drive_graph(self):
        G = ox.graph_from_place(self.query, network_type='drive')
        # add edge length, speed_kph, travel_time
        G = ox.speed.add_edge_speeds(G)  # add speed (kmh?) for all edges
        G = ox.speed.add_edge_travel_times(G)  # add travel time (seconds?) for all edges
        G = ox.distance.add_edge_lengths(G)  # add length (meter?) for all edges
        # project, get uni directed grah, transform to geo_df with a default crs format
        streets_graph = ox.projection.project_graph(G)
        streets_graph = ox.get_undirected(streets_graph)
        streets_gdfs = ox.graph_to_gdfs(streets_graph, nodes=False, edges=True,
                                        node_geometry=False, fill_edge_geometry=True)
        street = streets_gdfs.to_crs(crs=self.crs_format)
        return G, street

    def plot_graph_data(self,
                        feature: str = 'speed_kph',
                        data_threshold=50,
                        show: bool = True):
        if type(data_threshold) == str:
            ec = ['y' if d == data_threshold else 'r' for ns, ne, d in self.G_drive.edges(data=feature)]
        else:
            ec = ['y' if d <= data_threshold else 'r' for ns, ne, d in self.G_drive.edges(data=feature)]
        fig, ax = ox.plot_graph(self.G_drive, node_alpha=0.1, edge_color=ec, show=False)
        ax.set_title(feature + ' > ' + str(data_threshold))
        if show:
            plt.show()

    def create_edge_dataframe(self):
        edge_data = []
        for u, v, data in self.G_drive.edges(keys=False, data=True):
            start_longitude = self.G_drive.nodes[u]['x']
            start_latitude = self.G_drive.nodes[u]['y']
            end_longitude = self.G_drive.nodes[v]['x']
            end_latitude = self.G_drive.nodes[v]['y']
            edge_data.append([start_longitude, start_latitude, end_longitude, end_latitude])
        df_edges = pd.DataFrame(edge_data, columns=['startLongitude', 'startLatitude', 'endLongitude', 'endLatitude'])
        return df_edges

    def plot_trip_routes(self, df_trips, show=True, route_alpha=0.2):
        """ plt routes on the street map df_trips is the data set with trips data"""
        street = self.street
        ax = street.plot(alpha=0.5, figsize=(6, 8)) #
        ax.scatter(df_trips['startLongitude'], df_trips['startLatitude'], 100, c='r', alpha=route_alpha)
        ax.scatter(df_trips['endLongitude'], df_trips['endLatitude'], 200, c='b', alpha=route_alpha)
        routes, _ = self.get_shortest_routes_and_features(df_trips)
        if len(df_trips['endLongitude']) == 1:
            ox.plot_graph_route(self.G_drive, routes, orig_dest_size=0, show=show, ax=ax)
        else:
            ox.plot_graph_routes(self.G_drive, [r for r in routes if len(r) > 0], 'r', orig_dest_size=0, route_alpha=route_alpha,
                                 show=show, ax=ax)

    def shortest_path_start_end(self, LO_LA_start, LO_LA_end):
        """ shortest route between starting and ending points   """
        G_drive = self.G_drive
        LO_LA_start, LO_LA_end = list(LO_LA_start), list(LO_LA_end)
        node_start = ox.nearest_nodes(G_drive, LO_LA_start[0], LO_LA_start[1])
        node_end = ox.nearest_nodes(G_drive, LO_LA_end[0], LO_LA_end[1])
        try:
            path = nx.shortest_path(G_drive, node_start, node_end, weight='length')
        except nx.NetworkXNoPath:
            print('no-path')
            path = []
        return path

    def get_minmax_lon_lat(self):
        """get minimum and maximum (lat,lon) of the street map"""
        nodes_latitude = [x[1]['y'] for x in self.G_drive.nodes(data=True)]
        nodes_longitude = [x[1]['x'] for x in self.G_drive.nodes(data=True)]
        lon_min, lon_max = min(nodes_longitude), max(nodes_longitude)
        lat_min, lat_max = min(nodes_latitude), max(nodes_latitude)
        return lon_min, lon_max, lat_min, lat_max

    def get_edge_features(self):
        """get data set of features of all the edges"""
        df_edges = pd.DataFrame([d[2] for d in self.G_drive.edges(data=True)])
        df_edges = df_edges[['length', 'travel_time', 'speed_kph']]
        return df_edges


    def get_shortest_routes_and_features(self, df_trips):
        """get features for the shortest routes"""
        routes = []
        for slo, sla, elo, ela in zip(df_trips['startLongitude'], df_trips['startLatitude'],
                                      df_trips['endLongitude'], df_trips['endLatitude']):
            path = self.shortest_path_start_end([slo, sla], [elo, ela])
            routes.append(path)
        # Create an empty dataframe
        df_route_features = pd.DataFrame(columns=['reservation_id', 'distance', 'shortest_route', 'mean_speed_kph',
                                                  'expected_travel_time', 'shortest_length', 'number_street_segments'])
        i = 0
        for trip, shortest_path in zip(df_trips.iterrows(), routes):
            path_df = pd.DataFrame([d[2] for d in self.G_drive.edges(shortest_path, data=True)])
            reservation_id, distance = [], []
            if 'reservation_id' in trip[1]:
                reservation_id = trip[1]['reservation_id']
            if 'distance' in trip[1]:
                distance = trip[1]['distance']
            shortest_route = shortest_path
            if not shortest_route:
                mean_speed_kph = distance/ (trip[1]['trip_duration'].seconds/3600)
                travel_time = trip[1]['trip_duration'].seconds/3600
                length_km = distance
                number_street_segments = 0
            else:
                mean_speed_kph = path_df['speed_kph'].mean()
                travel_time = path_df['travel_time'].sum()
                length_km = path_df['length'].sum()/1000
                number_street_segments = len(shortest_path)
            # Add the row to the dataframe
            df_route_features.loc[i] = [reservation_id, distance, shortest_route, mean_speed_kph,
                                        travel_time, length_km, number_street_segments]
            i +=1
        return routes, df_route_features



