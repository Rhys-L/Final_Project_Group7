import pandas as pd
import requests

import numpy as np

res = requests.get('https://coastwatch.glerl.noaa.gov/statistic/ice/dat/g2008_2009_ice.dat')
main_arr = np.array([])
#clears headers and creates a list of strings. Each string is a row in the corresponding .txt with values and spaces.
data = res.text[304:].splitlines()
for i in range(len(data)):
    clean_row = [i for i in data[0].split(' ') if i] #creates a list of clean strings of the values (no spaces) of the first element of data.
    arr = np.array(clean_row, dtype=float)
    main_arr.append(arr)
out = np.asarray(main_arr)

df = pd.DataFrame(out, columns = ['Year', 'Day', 'Sup.', 'Mich.', 'Huron', 'Erie', 'Ont.', 'St.Clr', 'GL Total'])