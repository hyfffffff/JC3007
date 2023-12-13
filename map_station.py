"""
    China Meteorological Station Temperature Trends Visualization Script

    Description:
        This script is designed to visualize temperature trends observed at meteorological stations across China. 
        It reads in temperature trend data from CSV files and plots these trends on a map of China. The visualization 
        specifically highlights the temperature changes recorded at individual meteorological stations, with a color-coded 
        representation to illustrate the variation in temperature trends.

    Key Functions:
        1. Data Reading: Loads temperature trend data for meteorological stations from a CSV file.
        2. Map Integration: Uses a shapefile to create a geographic map of China for visualization.
        3. Data Visualization: Plots each meteorological station on the map, using color-coding to represent the 
           observed temperature trends.
        4. Custom Color Mapping: Implements a custom colormap to effectively demonstrate the range of temperature changes.

    Preparation:
        - Ensure that the temperature trend data file, named 'China_city_temperature_trends2.csv', is placed in the 
          same directory as the script.
        - The shapefile for China's map, located in the 'Chinamap' directory, should be accessible to the script.

    Usage:
        Run the script with the following command:
        python map_station.py

"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point

# Reload the CSV file
file_path = 'China_city_temperature_trends3.csv'
city_data = pd.read_csv(file_path)

# Load the shapefile
shapefile_path = 'Chinamap/Chinamap.shp'  # Replace this with the correct path to the shapefile
china_map = gpd.read_file(shapefile_path)

# Creating the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plotting the map
china_map.plot(ax=ax, color='white', edgecolor='black')

# Plotting the cities on the map
city_data['Coordinates'] = list(zip(city_data['LONGTITUE'], city_data['LATITUTE']))
city_data['Coordinates'] = city_data['Coordinates'].apply(lambda x: Point(x))

# Create a GeoDataFrame with cities data
gdf = gpd.GeoDataFrame(city_data, geometry='Coordinates')

# Custom colormap
colors = ['blue', 'yellow', 'orange', 'red']
n_bins = [0, 0.6, 0.8, 1]  # Adjust these values based on the data range and desired color distribution
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(n_bins, colors)))

# Plotting the cities using the scatter method with our colormap
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['Year Coefficient'], cmap=cmap, s=50,
                     vmin=min(gdf['Year Coefficient']), vmax=max(gdf['Year Coefficient']))

# Adding color bar
plt.colorbar(scatter, shrink=0.6, label='Yearly Temperature Increase (°C/year)')

plt.title('Annual Temperature Trends at Meteorological Stations Across China')
plt.savefig('China_stations_temperature_trends.png', bbox_inches='tight', dpi=300)
plt.show()
print(1)
