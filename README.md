# GAMES: Grid Aware Mobility and Energy Sharing: GAMES

 
## Installation 
Use python 3.9\
Needed packages are in requirements.txt\
`pip install -r requirements.txt`

 
## DATA 

## Raw data can be shared among the partners of the project (GAMES) and must be placed in the right folders
`datasets/autotel` , `datasets/windkraft_simonsfeld`, `datasets/mobility` 
 
## Simulated mobility scenarios are available   
* For instance, the dir `./mobility/simulated`  contins:
** files `station_matrix.zip` containins simulated time series of EV presences in the stations  

## What to run
#  
* demo_traffic_model
```python
# Defne macroscopic traffic model &  street network for two geographical regions 
TelAviv = Macroscopic_traffic_model(geographical_area = 'Tel Aviv, Israel')
Zurich = Macroscopic_traffic_model(geographical_area = 'Zurich, CH')

lon_min, lon_max, lat_min, lat_max = TelAviv.get_minmax_lon_lat() # get minimum and maximum latitudes, longitudes from the street graph
TelAviv.plot_graph_data(feature = 'speed_kph', data_threshold = 50) # in red are streets with speed >= 50 kph
#TelAviv.plot_graph_data(feature = 'travel_time', data_threshold = 60) # in red are streets with travel_time >= 60 seconds
#TelAviv.plot_graph_data(feature = 'length', data_threshold = 100) # in red are streets with length >= 20 meters
#TelAviv.plot_graph_data(feature = 'highway', data_threshold = 'residential') # in red are branches labelled as residential streets

df_edges = TelAviv.create_edge_dataframe() # get lat,lon start-end of the edges/streets
path = TelAviv.shortest_path_start_end(LO_LA_start= [34.79, 32.073], LO_LA_end = [34.791, 32.09] ) # get (if exist) the shortest path/route from start to end (latitude,longitude)
df_trips = pd.DataFrame([[1, 34.79, 32.073, 34.789, 32.093, 9],
                         [2, 34.789, 32.138, 34.7923, 32.1, 12]],
                        columns=['reservation_id', 'startLongitude', 'startLatitude',
                                 'endLongitude', 'endLatitude', 'distance']) # example data set with two trips with only (ID,lat,lon,dis)
TelAviv.plot_trip_routes(df_trips, show=True, route_alpha=0.9)
routes, df_route_features = TelAviv.get_shortest_routes_and_features(df_trips) # routes= list of routes, df_route_features= data frame with features of the trip and shortest routes

```
* demo_data_loader_pre_process 
```python
df_autotel = data_loader(data_dir='datasets/autotel', file_name='autotel_2021_2022.pkl') 
df, df_sequence = preprocess_trip_data_frame(df,TelAviv.get_minmax_lon_lat())
results_daily = get_daily_profiles_data(df_sequence)
matrix_day = results_daily['matrix_daily_departures'] # a [n_days x 24] array contining the total number of departures for each day and hour in the data set 
```

* demo_train_total_mobility_demand_forecaster
* demo_space_time_probabilistic_forecaster



