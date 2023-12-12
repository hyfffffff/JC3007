import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def read_csv_files(file_list):
    """
    Reads multiple CSV files into a single pandas DataFrame.
    :param file_list: List of file paths.
    :return: pandas DataFrame containing merged data from all files.
    """
    return pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)

def get_file_list(pattern):
    """
    Returns a list of file paths matching the given pattern.
    :param pattern: File matching pattern (e.g., 'data/3*.csv').
    :return: List of file paths.
    """
    return glob.glob(pattern)

def process_temperature_data(df):
    """
    Processes the temperature data in the DataFrame.
    Converts temperatures from Fahrenheit to Celsius.
    Aggregates data to yearly averages for each city including latitude and longitude.
    :param df: pandas DataFrame with temperature data.
    :return: Processed pandas DataFrame.
    """
    # Convert temperatures from Fahrenheit to Celsius
    for col in ['TMAX', 'TMIN', 'TAVG']:
        if col in df.columns:
            df[col] = (df[col] - 32) * 5/9

    # Convert DATE column to datetime and extract year
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATE'].dt.year

    # Group by city and year, then calculate mean temperatures and get latitude and longitude
    grouped = df.groupby(['NAME', 'YEAR'])
    yearly_averages = grouped[['TMAX', 'TMIN', 'TAVG']].mean()
    yearly_averages['LATITUDE'] = grouped['LATITUDE'].first()
    yearly_averages['LONGITUDE'] = grouped['LONGITUDE'].first()
    yearly_averages = yearly_averages.reset_index()

    return yearly_averages



def plot_temperature_trend(df, location_filter):
    """
    Plots temperature trend for a given location.
    :param df: pandas DataFrame with temperature data.
    :param location_filter: City name or tuple of (latitude_min, latitude_max, longitude_min, longitude_max)
    """
    # Filter data by location
    if isinstance(location_filter, str):
        df_filtered = df[df['NAME'].str.contains(location_filter)]
    else:
        lat_min, lat_max, long_min, long_max = location_filter
        df_filtered = df[(df['LATITUDE'] >= lat_min) & (df['LATITUDE'] <= lat_max) & 
                         (df['LONGITUDE'] >= long_min) & (df['LONGITUDE'] <= long_max)]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df_filtered['YEAR']), df_filtered['TAVG'], label='Average Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Trend for {location_filter}')
    plt.legend()
    plt.show()

file_list = get_file_list('citiesdata/g*.csv')
df = read_csv_files(file_list)
df_processed = process_temperature_data(df)
plot_temperature_trend(df_processed, 'GUANGZHOU, CH') # or plot_temperature_trend(df_processed, (23.0, 24.0, 113.0, 114.0))
