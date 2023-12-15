# China Weather Data Analysis Project

## Description

This project analyzes and visualizes temperature trends across China at meteorological stations, provinces, and national levels. The analysis is conducted using Python scripts that process weather data, apply statistical methods, and create visual representations of temperature trends.

## Contents

- `ChinaTempTrendsAnalysis.py`: Analyzes weather data and generates CSV files with temperature trends.
- `map_station.py`: Visualizes temperature trends at meteorological stations on a map of China.
- `map_province.py`: Displays temperature trends across provinces using a color-coded map.
- `citymaster.csv`: Maps meteorological stations to geographic information.
- `provincemaster.csv`: This file provides a list of provinces and administrative regions in China, detailing their administrative codes (ADCODE), Chinese names, and English translations. 
- `Chinadata/`: Directory for raw weather data files.
- `Chinamap/`: Contains shapefiles for China's geographical boundaries.
- `environment.yml`: Conda environment configuration file.

## Analysis Process

To perform the full analysis of temperature trends across various geographical units in China and visualize the results, follow these steps:

1. **Data Preparation**:
    - Ensure that the weather data files obtained from [NOAA](https://www.ncei.noaa.gov/) are placed in the `Chinadata` directory.
    - Confirm that the shapefiles are in the `Chinamap` directory.

2. **Conda Environment**:
    - set up Conda enviroment using the `environment.yml` file by running:
  
      ```sh
      conda env create -f environment.yml
      ```

    - Activate the Conda environment by running:
  
      ```sh
      conda activate JC3007
      ```

3. **Run Analysis Script**:
    - Generate the CSV files containing the temperature trends for meteorological stations, provinces, regions, and the entire country by running:
  
      ```sh
      python ChinaTempTrendsAnalysis.py
      ```

      This script processes the weather data and applies the Ordinary Least Squares (OLS) algorithm to identify temperature trends.

4. **Run Visualization Scripts**:
    - Create visual representations of the temperature trend results by executing:
  
      ```sh
      python map_station.py
      ```

      for a map showing meteorological stations, and

      ```sh
      python map_province.py
      ```

      for a map showing provinces.

By following these steps, you will produce CSV files detailing temperature trends and corresponding visualizations mapped geographically for meteorological stations and provinces in China.
