import numpy as np
import matplotlib.pyplot as plt

def plot_close_evs(df_trip_demand_samples, df_fleet, lat1, lon1, distance, acceptance_radius):
    # Radius in radians
    acceptance_radius_meters = acceptance_radius  # Adjust this value as needed
    # Convert radius from meters to radians
    AVG_EARTH_RADIUS = 6_371_000
    radius_rad = acceptance_radius_meters / AVG_EARTH_RADIUS
    is_close = distance <= acceptance_radius_meters
    # Generate points along the circumference of the circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_lats = lat1 + radius_rad * np.sin(theta)
    circle_lons = lon1 + radius_rad * np.cos(theta)
    # Plot the circle
    plt.figure(figsize=(8, 8))
    plt.plot(circle_lons, circle_lats, label='Acceptance area')
    plt.fill(circle_lons, circle_lats, color='blue', alpha=0.3)
    plt.scatter(df_trip_demand_samples['LONs'], df_trip_demand_samples['LATs'], c='g', alpha=0.2,
                label='Simulated demand points')
    plt.scatter(df_fleet[is_close]['LON'].values, df_fleet[is_close]['LAT'].values, 40, c='b', marker='x',
                label='EV is close')
    plt.scatter(lon1, lat1, 50, alpha=0.99, c='k', label='The customer')
    plt.scatter(df_fleet[is_close == False]['LON'].values, df_fleet[is_close == False]['LAT'].values, 10, c='r',
                label='EV is too far')
    plt.xlabel('Longitude (radians)')
    plt.ylabel('Latitude (radians)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Accpetance area with Radius {acceptance_radius_meters} meter around the user')
    plt.show()