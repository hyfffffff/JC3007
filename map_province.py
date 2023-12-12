"""
    China Provincial Temperature Trends Visualization Script

    Description:
        This script visualizes temperature trends across Chinese provinces. It merges temperature trend data 
        from a CSV file with a shapefile of provincial boundaries, and then plots these trends on a map. 
        The visualization highlights the temperature changes in each province using a color-coded approach.

    Key Functions:
        1. Data Reading: Loads temperature trend data for Chinese provinces from a CSV file.
        2. Map Integration: Uses a shapefile to create a geographic map of Chinese provinces for visualization.
        3. Data Visualization: Plots the provincial boundaries on the map, color-coded based on the observed 
           temperature trends.
        4. Custom Color Mapping: Implements a custom colormap to effectively demonstrate the range of temperature changes.

    Preparation:
        - Ensure that the temperature trend data file, named 'China_province_temperature_trends3.csv', is in the 
          same directory as the script.
        - The shapefile for China's provincial map, located in the 'Chinamap' directory, should be accessible to the script.

    Usage:
        Run the script with the following command:
        python map_province.py
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors


# File paths
provinces_shapefile_path = 'Chinamap/Chinamap.shp'  # Replace with your provinces shapefile path

# Load the shapefiles
provinces = gpd.read_file(provinces_shapefile_path,  dtype={'adcode': str})

provinces_temperature = pd.read_csv('China_province_temperature_trends3.csv',dtype={'ADCODE': str})

provinces = pd.merge(provinces, provinces_temperature[['ADCODE', 'Year Coefficient']], left_on='adcode', right_on='ADCODE', how='left'    )
provinces['Year Coefficient'] = provinces['Year Coefficient'].fillna(0)

# Plot the shapefiles
# Plotting both shapefiles on the same figure
fig, ax = plt.subplots(figsize=(12, 10))

colors = ['blue', 'yellow', 'orange', 'red']
n_bins = [0, 0.6, 0.8, 1]  # Adjust these values based on the data range and desired color distribution
cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(n_bins, colors)))

vmin = provinces['Year Coefficient'].min()
vmax = provinces['Year Coefficient'].max()

# Plot provinces with colormap
provinces.plot(column='Year Coefficient', ax=ax, edgecolor='black', legend=True, cmap=cmap,
                vmin=vmin, vmax=vmax,
                legend_kwds={'label': "Year Coefficient (°C/Year)", 'shrink': 0.8})

ax.set_title('Annual Temperature Change Trends Across Chinese Provinces', fontsize=15)
plt.savefig('China_provinces_temperature_trends.png', bbox_inches='tight', dpi=300)
plt.show()



