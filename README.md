# GAMES: Grid Aware Mobility and Energy Sharing: GAMES

 
## Installation 
Use python 3.9\
Needed packages are in requirements.txt\
`pip install -r requirements.txt`

 
## DATA 

## Raw data can be shared among the partners of the project (GAMES) and must be placed in the right folders
`datasets/autotel` , `datasets/windkraft_simonsfeld`, `datasets/autotel` 
 
## Simulated mobility scenarios are available in 
`../simulated`
* For instance, the dir `datasets/mobility/simulated` contains simulated trip data for 
* `datasets/data/EWZ/` should contain power time series for EWZ with naming convention `2019_ewz_bruttolastgang.csv`  from 2019 to 2021.
* `datasets/data/mobility/` should contain `station_matrix.zip` containing time series of EV locations for the Mobility dataset 

## What to run
### Mobility 
Run preprocess_utils to get the station_matrix filtered by DSO, default is EWZ
