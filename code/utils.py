import pandas as pd
import numpy as np
import pickle
import os
import datetime


def load_data(data_path):
    """Loading the block location and load data.
    
    :return: gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :return avg_loads: Numpy array with each row containing the average load
    for a day of week and time, where each column is a day of week and hour.
    :return park_data: Dictionary of DataFrames, where each DataFrame contains
    the full load data for a block.
    :return N: Number of samples, i.e. the number of blockfaces.
    :return P: Number of times (day, hour) pairs.
    :return idx_to_day_hour: Dictionary of column in avg_loads to (day, hour) pair.
    :return day_hour_to_idx: Dictionary of (day, hour) pair to column in avg_loads.
    """

    curr_dir = os.getcwd()
    data_path = curr_dir + '/../data/'

    with open(os.path.join(data_path, 'hourlyUtilization100_uncapped.pck'), 'rb') as f:
     load_data = pickle.load(f)
        
    with open(os.path.join(data_path, 'ElementKeytoLatLong.pck'), 'rb') as f:
        locations = pickle.load(f)

    avg_loads = []

    gps_loc = []

    park_data = {}

    elkeys = sorted(load_data.keys())

    N = len(elkeys)

    for key in elkeys:
        
        # Loading the data for a single block.
        block_data = pd.DataFrame(load_data[key].items(), columns=['Datetime', 'Load'])
        
        block_data['Datetime'] = pd.to_datetime(block_data['Datetime'])
        
        block_data.sort_values(by='Datetime', inplace=True)
        block_data.reset_index(inplace=True, drop=True)
        
        block_data['Date'] = block_data['Datetime'].dt.date
        block_data['Hour'] = block_data['Datetime'].dt.hour
        block_data['Day'] = block_data['Datetime'].dt.weekday
        
        # Getting rid of Sunday since there is no paid parking.
        block_data = block_data.loc[block_data['Day'] != 6]
        
        # Getting the dates where the total parking is 0 because of holidays.
        empty_day_dates = []
        
        # No parking on new years eve.
        empty_day_dates.append(datetime.date(2015,1,1))
        
        # No parking on MLK day.
        empty_day_dates.append(datetime.date(2015,1,19))
        
        # No parking on Presidents day.
        empty_day_dates.append(datetime.date(2015,2,16))
        
        # Dropping the days where the total parking is 0.
        block_data = block_data.loc[~block_data['Date'].isin(empty_day_dates)]
        block_data.reset_index(inplace=True, drop=True)
        
        park_data[key] = block_data
        
        # Getting the average load for each hour of the week for the block.
        avg_load = block_data.groupby(['Day', 'Hour'])['Load'].mean().values.reshape((1,-1))
        avg_loads.append(avg_load)
        
        mid_lat = (locations[key][0][0] + locations[key][1][0])/2.
        mid_long = (locations[key][0][1] + locations[key][1][1])/2.
        gps_loc.append([mid_lat, mid_long])
        
    avg_loads = np.vstack((avg_loads))
    gps_loc = np.vstack((gps_loc))

    index = park_data[key].groupby(['Day', 'Hour']).sum().index

    days = index.get_level_values(0).unique().values
    days = np.sort(days)

    hours = index.get_level_values(1).unique().values
    hours = np.sort(hours)

    idx_to_day_hour = {i*len(hours) + j:(days[i], hours[j]) for i in range(len(days)) 
                                                            for j in range(len(hours))}
    day_hour_to_idx = {v:k for k,v in idx_to_day_hour.items()}

    P = len(idx_to_day_hour)

    return gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx
