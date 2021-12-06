"""
Script to get and clean data from NOOA. This script is unfortunately prone to connection and timeout errors depending
on the computer and time of day/response speed from NOAA and GLERL. To bypass these errors, the results of this code
(output table from final df in line 147) are stored here on a GCP storage bucket:
'https://storage.googleapis.com/great_lakes/LakeIce_PhysicalProps.csv'

Both functions (lines 23 - 51) query and clean data from .dat files on web servers.

Driver code goes through following steps.
1. Calls get_ice_concentration on a list of urls to .dat files containing ice concentration.
2. Reshapes ice DataFrame to tidy long format and appends new id column to merge with additional datasets.
3. Calls get_surface_temp on a urls to .dat files containing temp. Joins response objects into master temperature df.
4. Merges temperature and ice dataframes. Removes duplicated columns.
5. Creates DataFrame of physical properties of Great Lakes.
6. Exports a csv with the full dataset.

"""

import pandas as pd
import requests
import time

def get_ice_concentration(url):
    """Takes a url to a .dat file as an argument. Returns a df of clean data from the web hosted file."""
    res = requests.get(url, timeout=(10, 30))
    time.sleep(1)
    rows = []
    data = res.text[304:].splitlines() #strips header. creates list of lists containing rows of ice data.
    for i in range(len(data)):
        clean_row = [value for value in data[i].split(' ') if value]  #cleans white spaces so row only contains values.
        rows.append(clean_row)
    df = pd.DataFrame(rows, columns=['Year', 'Day', 'Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr', 'GL Total'],
                      dtype=float)
    return df

def get_surface_temp(url):
    """Takes a url to a .dat file as an argument. Returns a df of clean data from the web hosted file."""
    try:
        res = requests.get(url, timeout=(60, 180))
    except requests.exceptions.ConnectionError as errc:
        print("Connection error", errc, url)
        time.sleep(1)
        res = requests.get(url)
    rows = []
    data = res.text[348:].splitlines() #strips header. creates list of lists containing rows of ice data.
    for i in range(len(data)):
        clean_row = [value for value in data[i].split(' ') if value]  #cleans white spaces so row only contains values.
        rows.append(clean_row)
    df = pd.DataFrame(rows, columns=['Year', 'Day', 'Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr'],
                      dtype=float)
    return df


if __name__ == "__main__":
    """Get from ice.dat files"""
    ice_urls = ['https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2008_2009_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2009_2010_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2010_2011_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2011_2012_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2012_2013_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2013_2014_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2014_2015_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2015_2016_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2016_2017_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2017_2018_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2018_2019_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2019_2020_ice.dat',
                'https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2020_2021_ice.dat']

    ice_con = [get_ice_concentration(url) for url in ice_urls]  # creates a list of dfs with content of each .dat file
    df_ice = pd.concat(ice_con).reset_index(drop=True).astype(
        {'Year': 'int64', 'Day': 'int64'})  # merges all annual files into a single df

    # reshape from wide to tidy long and convert to strings
    ice = pd.melt(df_ice, id_vars=["Year", "Day"],
                  value_vars=['Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr', 'GL Total'],
                  var_name="Lake", value_name="Ice_Con_pct").astype(
        {'Year': 'string', 'Day': 'string', 'Lake': 'string'})

    # create an id containing year, date, and lake
    ice['id'] = ice.Year + '_' + ice.Day + '_' + ice.Lake

    """Get annual surface temperature from NOAA .dat files. Note that we run these individually rather 
    than a loop because the request is prone to connection errors from the ftp server."""

    temp_2008 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2008/glsea-temps2008_1024.dat')
    temp_2009 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2009/glsea-temps2009_1024.dat')
    temp_2010 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2010/glsea-temps2010_1024.dat')
    temp_2011 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2011/glsea-temps2011_1024.dat')
    temp_2012 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2012/glsea-temps2012_1024.dat')
    temp_2013 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2013/glsea-temps2013_1024.dat')
    temp_2014 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2014/glsea-temps2014_1024.dat')
    temp_2015 = get_surface_temp('http://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2015/glsea-temps2015_1024.dat')
    temp_2016 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2016/glsea-temps2016_1024.dat')
    temp_2017 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2017/glsea-temps2017_1024.dat')
    temp_2018 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2018/glsea-temps2018_1024.dat')
    temp_2019 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2019/glsea-temps2019_1024.dat')
    temp_2020 = get_surface_temp('https://coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/2020/glsea-temps2020_1024.dat')

    #Create a list of dfs containing annual surface temperature
    temps = [temp_2008, temp_2009, temp_2010, temp_2011, temp_2012, temp_2013, temp_2014, temp_2015, temp_2016,
             temp_2017, temp_2018, temp_2019, temp_2020]

    #Concatenate all annual temp dfs into a master df
    df_temp = pd.concat(temps).reset_index(drop=True).astype({'Year': 'int64', 'Day': 'int64'})
    # Reshape from wide to tidy long. Convert to lake names to strings.
    temp = pd.melt(df_temp, id_vars=["Year", "Day"], value_vars=['Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr'],
                   var_name="Lake", value_name="Surface_Temp_C").astype(
        {'Year': 'string', 'Day': 'string', 'Lake': 'string'})

    # Create an id containing year, date, and lake to merge with ice.
    temp['id'] = temp.Year + '_' + temp.Day + '_' + temp.Lake

    # Merges ice values to temperature dataset on common id structured 'year_day_lake.
    messy_dataset = pd.merge(temp, ice, on='id', how='outer')

    # Great lakes total values only exist in the ice (right merged) df.
    # This replaces all missing in the columns we keep with values from the right merged data.
    messy_dataset.Year_x.fillna(messy_dataset.Year_y, inplace=True)
    messy_dataset.Day_x.fillna(messy_dataset.Day_y, inplace=True)
    messy_dataset.Lake_x.fillna(messy_dataset.Lake_y, inplace=True)

    clean = messy_dataset.drop(columns=['Year_y', 'Day_y', 'Lake_y'])
    clean = clean.rename(columns={"Year_x": "Year", "Day_x": "Day", "Lake_x": "Lake", "Ice_Con_pct": "Ice_pct"})

    clean.to_csv("Ice_SurfaceTemp_LongTidy.csv")

    # Create df with Lake physical characteristics from this page in the NOAA site:
    # https://coastwatch.glerl.noaa.gov/statistic/physical.html

    lake_properties = {'Elevation_meters': [183, 176, 176, 173, 74],
                       'Length_km': [563, 494, 332, 388, 311],
                       'Breadth_km': [257, 190, 245, 92, 85],
                       'Avg_Depth_meters': [147, 85, 59, 19, 86],
                       'Max_Depth_meters': [406, 282, 229, 64, 244],
                       'Volume_km3': [12100, 4920, 3540, 484, 1640],
                       'Water_Area_km2': [82100, 57800, 59600, 25700, 18960],
                       'Land_Drain_Area_km2': [127700, 118000, 134100, 78000, 64030],
                       'Total_Area_km2': [209800, 175800, 193700, 103700, 82990],
                       'Shore_Length_km': [4385, 2633, 6157, 1402, 1146],
                       'Retention_Time_years': [191, 99, 22, 2.6, 6]}

    df_lake_properties = pd.DataFrame(lake_properties, index=['Sup.',
                                                              'Mich.',
                                                              'Huron',
                                                              'Erie',
                                                              'Ont.'])

    df_lake_properties['Lake'] = ['Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.']

    full_df = pd.merge(df_lake_properties, clean, on='Lake', how='outer')

    full_df.to_csv("LakeIce_PhysicalProps.csv")

