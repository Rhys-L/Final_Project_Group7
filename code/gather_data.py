import pandas as pd
import requests
import numpy as np

def get_ice_concentration(url):
    """Takes in a url to a .dat file as an argument. Returns a df of clean data from the web hosted file."""
    res = requests.get(url, timout=(10, 30))
    rows = []
    data = res.text[304:].splitlines() #strips header. creates list of lists containing rows of ice data.
    for i in range(len(data)):
        clean_row = [value for value in data[i].split(' ') if value]  #cleans spaces so row only contains values.
        rows.append(clean_row)
    df = pd.DataFrame(rows, columns=['Year', 'Day', 'Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr', 'GL Total'],
                      dtype=float)
    return df

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

ice_con = [get_ice_concentration(url) for url in ice_urls] #creates a list of dfs with content of each .dat file
df_ice = pd.concat(ice_con).reset_index(drop=True) #merges all annual files into a single df

#reshape from wide to tidy long
ice = pd.melt(df_ice, id_vars=["Year", "Day"], value_vars=['Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr',
       'GL Total'], var_name = "Lake", value_name = "Ice_Con_pct")

#cast types
ice2 = ice.astype({'Year': 'int64', 'Day':'int64'}).astype({'Year': 'string', 'Day':'string', 'Lake':'string'})

def get_surface_temp(url):
    """Takes in a url to a .dat file as an argument. Returns a df of clean data from the web hosted file."""
    res = requests.get(url)
    rows = []
    data = res.text[347:].splitlines() #strips header. creates list of lists containing rows of ice data.
    for i in range(len(data)):
        clean_row = [value for value in data[i].split(' ') if value]  #cleans spaces so row only contains values.
        rows.append(clean_row)
    df = pd.DataFrame(rows, columns=['Year', 'Day', 'Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr'],
                      dtype=float)
    return df


pd.wide_to_long(df=ice_df, stubnames=['Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr',
       'GL Total'])