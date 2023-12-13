"""
    China Weather Data Analysis Script

    Description:
        This script processes and analyzes weather data from various locations across China. It cleans the data, 
        applies the Ordinary Least Squares (OLS) algorithm for analysis, and generates four CSV files for in-depth 
        data analysis on temperature trends at different levels - meteorological stations, provinces, regions, and nationally.

    Key Functions:
        1. Data Reading: Imports weather data files specific to China.
        2. Data Cleaning: Ensures the accuracy and consistency of the dataset.
        3. Data Analysis: Utilizes the OLS algorithm for comprehensive analysis.
        4. CSV Generation: Produces four CSV files for subsequent data analysis, detailing temperature trends at 
           meteorological stations, provincial, regional, and national levels.

    Preparation:
        - Ensure that weather data files downloaded from https://www.ncei.noaa.gov/ are placed in the 'Chinadata' 
          subdirectory within the current directory.
        - Verify the availability of all required data files before executing the script.

    Usage:
        Execute the script using the command:
        python ChinaTempTrendsAnalysis.py

"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def get_file_list(pattern):
    """
        Returns a list of file paths matching the given pattern.
        :param pattern: File matching pattern (e.g., 'Chinadata/3*.csv').
        :return: List of file paths.
    """
    return glob.glob(pattern)

def read_and_clean_csv_files(file_list):
    """
        Reads and cleans multiple CSV files into a single pandas DataFrame.
        Only new records that do not exist in the merged DataFrame are added.
        :param file_list: List of file paths.
        :return: pandas DataFrame containing merged and cleaned data from all files.
    """
    def clean_data(df):
        # Convert DATE column to datetime and handle invalid dates
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])

         # Handle missing temperature values based on provided rules
        df['TMAX'] = df.apply(lambda row: (row['TAVG']*2 - row['TMIN']) if pd.isna(row['TMAX']) and not pd.isna(row['TMIN']) else row['TMAX'], axis=1)
        df['TMIN'] = df.apply(lambda row: (row['TAVG']*2 - row['TMAX']) if pd.isna(row['TMIN']) and not pd.isna(row['TMAX']) else row['TMIN'], axis=1)
        df['TMAX'].fillna(df['TAVG'], inplace=True)
        df['TMIN'].fillna(df['TAVG'], inplace=True)

        # Define the maximum and minimum reasonable temperatures
        max_reasonable_temp = 150
        min_reasonable_temp = -70
        for col in ['TMAX', 'TMIN', 'TAVG']:
            if col in df.columns:
                # Mask values outside the reasonable range
                df[col] = df[col].mask((df[col] > max_reasonable_temp) | (df[col] < min_reasonable_temp))
                # Interpolate the missing values
                df[col] = df[col].interpolate(method='linear')

        return df

    # Initialize an empty DataFrame for the merged data
    merged_df = pd.DataFrame()

    # Process each file
    for file in file_list:
        # Read the file into a DataFrame
        df = pd.read_csv(file)

        # Clean the data
        df_cleaned = clean_data(df)

        # Drop duplicates based on 'DATE' and 'STATION' columns
        df_cleaned = df_cleaned[~df_cleaned[['DATE', 'STATION']].duplicated()]

        # Append only new records to the merged DataFrame
        merged_df = pd.concat([merged_df, df_cleaned], ignore_index=True).drop_duplicates(subset=['DATE', 'STATION'])

    return merged_df


def clean_incomplete_station_data(df):
    """
        This function cleans the meteorological station data by removing stations with incomplete records.
        Incompleteness is defined based on two criteria:
            1. A month is missing entirely for a station.
            2. A month has fewer than 25 days of data for a station.

        Parameters:
            df (DataFrame): The DataFrame that contains all the read-in meteorological data, including 
            'DATE' and 'STATION' columns.

        Returns:
            DataFrame: A cleaned DataFrame with only stations that meet the completeness criteria.
    """

    # Ensure the 'DATE' column is converted to datetime type, with coercion for any format errors
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Generate a range of months from January 1980 to December 2022, each as a period
    all_months = pd.period_range('1980-01', '2022-12', freq='M')

    # Function to check each station's data for completeness
    def check_station(station_group):
        # Count the number of records for each month in the station's group
        monthly_records = station_group['DATE'].dt.to_period('M').value_counts()
        # Identify months that are missing from the station's records
        missing_months = set(all_months) - set(monthly_records.index)
        # Identify months with insufficient data (less than 25 days recorded)
        insufficient_days = monthly_records[monthly_records < 25]
        # If there are missing months or insufficient data, return True (indicating the station should be removed)
        return len(missing_months) > 0 or len(insufficient_days) > 0

    # Identify stations to remove based on the check_station criteria
    stations_to_remove = df.groupby('STATION').filter(check_station)['STATION'].unique()

    # Remove all data for the identified stations
    df_cleaned = df[~df['STATION'].isin(stations_to_remove)]

    return df_cleaned


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

    # Group by station, year, and aggregate data
    aggregated_df = df.groupby(['NAME', 'YEAR']).agg({
        'TMAX': 'mean',
        'TMIN': 'mean',
        'TAVG': 'mean',
        'LATITUDE': 'first',
        'LONGITUDE': 'first'
    }).reset_index()

    return aggregated_df


def plot_temperature_trend(df, location_filter):
    """
        Plots temperature trends (maximum, minimum, average) for a given location.
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
    plt.plot(df_filtered['YEAR'], df_filtered['TMAX'], label='Average Maximum Temperature', color='red')
    plt.plot(df_filtered['YEAR'], df_filtered['TMIN'], label='Average Minimum Temperature', color='blue')
    plt.plot(df_filtered['YEAR'], df_filtered['TAVG'], label='Average Temperature', color='green')

    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Trends for {location_filter}')
    plt.legend()
    plt.show()

def calresult(df, unit_name):
    """
        Calculate and analyze the temperature trend for a specified analysis unit (e.g., meteorological station, 
        province, region, or country) based on the unit's name.

        Parameters:
            df (DataFrame): The DataFrame containing comprehensive meteorological data for the specified analysis unit. 
                            Essential columns include 'YEAR', 'TAVG', 'LATITUDE', 'LONGITUDE', and 'ADCODE', among others. 
                            The DataFrame is expected to be aggregated accordingly if unit_name refers to a larger entity 
                            such as a province, in which case df would comprise data from all meteorological stations 
                            within that province.
            unit_name (str): The name of the analysis unit. This could refer to a specific meteorological station, a 
                             province, a region, or the entire country, dictating the scope of data within the DataFrame.
    
        Returns:
            list: A list containing the results of the temperature trend analysis for the unit, including the unit's name, 
            latitude, longitude, administrative code, constant coefficient, yearly temperature change rate (year_coeff), 
            p-value for the trend, and R-squared value from the OLS regression model.
    
        The function performs data aggregation to compute the average annual temperature ('TAVG') and conducts an 
        Ordinary Least Squares (OLS) regression to identify the trend of temperature change over the years. An 
        Augmented Dickey-Fuller (ADF) test checks the stationarity of the temperature data. The OLS regression's 
        slope (year_coeff) indicates the annual rate of temperature change. A p-value under 0.05 signifies a 
        statistically significant trend, permitting rejection of the null hypothesis (H0) that temperature changes 
        are random with no distinct trend.
    """
    # Aggregate the data to get average annual temperature per year for the unit
    average_tavg_per_year = df.groupby('YEAR')['TAVG'].mean()
    # Get the latitude and longitude of the unit from the first entry
    unit_latitue = df['LATITUDE'].iloc[0]
    unit_longtitue = df['LONGITUDE'].iloc[0]
    # Get the administrative code of the unit from the first entry
    unit_adcode = df['ADCODE'].iloc[0]

    # Prepare data for OLS regression to determine the temperature trend
    year_data = average_tavg_per_year.reset_index()['YEAR']
    # OLS regression to determine trend
    X = sm.add_constant(year_data)    # Add a constant term to the predictor for OLS
    Y = average_tavg_per_year.values  # Response variable
    ols_model = sm.OLS(Y, X).fit()    # Fit the OLS model


    # ADF test for stationarity
    adf_test = adfuller(city_data['TAVG'])

    # Retrieve the slope (yearly change rate) and p-value from the OLS model to assess trend significance
    slope = ols_model.params['YEAR']
    p_value = ols_model.pvalues['YEAR']

    print(unit_name, ols_model.summary(), slope, p_value, adf_test)

    const_coeff = ols_model.params['const']
    year_coeff = ols_model.params['YEAR']
    p_value = ols_model.pvalues['YEAR']
    r_squared = ols_model.rsquared

    # Compile the results into a list and return them
    return [unit_name, unit_latitue,  unit_longtitue, unit_adcode, const_coeff, year_coeff, p_value, r_squared]


# Main module for analyzing temperature trends in meteorological data across various geographical units in China.

# Retrieve a list of CSV files containing meteorological data from the 'Chinadata' directory.
file_list = get_file_list('Chinadata/*.csv')

# Read and clean the CSV files and combine them into a single DataFrame.
df = read_and_clean_csv_files(file_list)

# Further clean the DataFrame to remove incomplete data from meteorological stations.
df_complete = clean_incomplete_station_data(df)

# Process the temperature data in the DataFrame.
df_processed = process_temperature_data(df_complete)

# Read and clean the master list of cities for later merging.
city_master = pd.read_csv('citymaster.csv')
city_master = city_master.rename(columns=lambda x: x.strip())  # Clean column names

# Read and clean the master list of provinces for later merging, ensuring ADCODE is read as a string.
province_master = pd.read_csv('provincemaster.csv', dtype={'ADCODE': str})
province_master = province_master.rename(columns=lambda x: x.strip()) # Clean column names

# Standardize city and province names in both datasets for consistency.
city_master['City'] = city_master['City'].str.strip().str.upper()
city_master['Province'] = city_master['Province'].str.strip().str.upper()
province_master['English_Name'] = province_master['English_Name'].str.strip().str.upper()

# Merge the city master data with the province master data on the province name.
city_master = pd.merge(city_master, province_master[['ADCODE','English_Name']], left_on='Province', right_on='English_Name', how='left')

# Clean and standardize the names of cities in the processed DataFrame.
df_processed['NAME'] = df_processed['NAME'].str.replace(', CH', '').str.strip().str.upper()

# Merge the processed meteorological data with the city master data.
df_processed = pd.merge(df_processed, city_master[['City', 'Province', 'Region', 'ADCODE']], left_on='NAME', right_on='City', how='left')


cities = df_processed['NAME'].unique()
provinces = df_processed['Province'].unique()
regions = df_processed['Region'].unique()

# Define the layout for plotting.
cities_per_row = 5
rows_per_screen = 5
total_rows = (len(cities) + cities_per_row - 1) // cities_per_row
total_screens = total_rows // rows_per_screen + 1

# Loop through each screen and plot temperature trends for each city.
for screen in range(total_screens):
     # Setup the subplots for each row and column.
    fig, axs = plt.subplots(rows_per_screen, cities_per_row, figsize=(20, 3 * rows_per_screen))
    axs = axs.flatten()
    
    # Calculate the range of city indices for the current screen.
    start_index = screen * rows_per_screen * cities_per_row
    end_index = start_index + rows_per_screen * cities_per_row
    cities_to_plot = cities[start_index:end_index]

    # Plot the temperature data for each station on the current screen.
    for i, city in enumerate(cities_to_plot):
        city_data = df_processed[df_processed['NAME'] == city]
        ax = axs[i % (rows_per_screen * cities_per_row)]

        ax.tick_params(axis='both', which='both', labelsize=8)
        ax.set_xticks(range(city_data['YEAR'].min(), city_data['YEAR'].max(),5)) # 设置 x 轴刻度
        ax.set_yticks(range(int(city_data['TMIN'].min()), int(city_data['TMAX'].max()) + 1, 5)) # 设置 y 轴刻度

        ax.plot(city_data['YEAR'], city_data['TMAX'], label='TMAX', color='red')
        ax.plot(city_data['YEAR'], city_data['TMIN'], label='TMIN', color='blue')
        ax.plot(city_data['YEAR'], city_data['TAVG'], label='TAVG', color='green')
        ax.set_title(city)
        ax.label_outer()

    # Hide any unused subplots.
    for j in range(i+1, rows_per_screen * cities_per_row):
        axs[j].set_visible(False)

# Adjust layout and display the plots.
plt.tight_layout()
plt.show()
    

# Analysis and results compilation for temperature trends across stations.
results = []
for city in cities:
    # Extract data for the current station.
    city_data = df_processed[df_processed['NAME'] == city]
    # Calculate temperature trend results for the station.
    ret_result = calresult(city_data, city)
    
    # Append the results to the list.
    results.append(ret_result)

# Convert the results into a DataFrame and save to a CSV file for stations.
results_df = pd.DataFrame(results, columns=['City', 'LATITUTE', 'LONGTITUE', 'ADCODE', 'Const Coefficient', 'Year Coefficient', 'P-Value', 'R-Squared'])
results_df.to_csv('China_station_temperature_trends3.csv', index=False)    

# Analysis and results compilation for temperature trends across provinces.
results = []
for province in provinces:
    province_data = df_processed[df_processed['Province'] == province]
    ret_result = calresult(province_data, province)
    results.append(ret_result)

province_df = pd.DataFrame(results, columns=['Province', 'LATITUTE', 'LONGTITUE', 'ADCODE', 'Const Coefficient', 'Year Coefficient', 'P-Value', 'R-Squared'])    
province_df.to_csv('China_province_temperature_trends3.csv', index=False)    

# Analysis and results compilation for temperature trends across regions.
results = []
for region in regions:
    region_data = df_processed[df_processed['Region'] == region]
    ret_result = calresult(region_data, region)
    results.append(ret_result)

region_df = pd.DataFrame(results, columns=['Region', 'LATITUTE', 'LONGTITUE', 'ADCODE', 'Const Coefficient', 'Year Coefficient', 'P-Value', 'R-Squared'])    
region_df.to_csv('China_region_temperature_trends3.csv', index=False)    

# Perform the analysis for the entire dataset to get nationwide trends.
all_data = df_processed
ret_result = calresult(all_data, 'China')
results.append(ret_result)

all_df = pd.DataFrame(results, columns=['Country', 'LATITUTE', 'LONGTITUE', 'ADCODE', 'Const Coefficient', 'Year Coefficient', 'P-Value', 'R-Squared'])    
all_df.to_csv('China_all_temperature_trends3.csv', index=False)

