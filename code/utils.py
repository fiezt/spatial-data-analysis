import pandas as pd
import numpy as np
import pickle
import os
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar


def load_data(data_path):
    """Loading the block location and load data.
    
    :return: gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :return avg_loads: Numpy array with each row containing the average load
    for a day of week and time, where each column is a day of week and hour.
    :return park_data: Dictionary of DataFrames, where each DataFrame contains
    the full load data for a block.
    :param N: Integer number of samples (locations).
    :param P: Integer number of total hours in the week with loads.
    :return idx_to_day_hour: Dictionary of column in avg_loads to (day, hour) pair.
    :return day_hour_to_idx: Dictionary of (day, hour) pair to column in avg_loads.
    """

    with open(os.path.join(data_path, 'hourlyUtilization100_uncapped.pck'), 'rb') as f:
        load_data = pickle.load(f)
        
    with open(os.path.join(data_path, 'ElementKeytoLatLong.pck'), 'rb') as f:
        locations = pickle.load(f)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2000-01-01', end=datetime.datetime.now().date()).to_pydatetime()
    holidays = [hol.date() for hol in holidays]

    avg_loads = []
    gps_loc = []
    park_data = {}

    elkeys = sorted(load_data.keys())
    N = len(elkeys)

    for key in elkeys:
        
        # Loading the data for a single block.
        block_data = pd.DataFrame(load_data[key].items(), columns=['Datetime', 'Load'])

        # Clipping the loads to be no higher than 1.5
        block_data['Load'] = block_data['Load'].clip_upper(1.5)

        block_data['Datetime'] = pd.to_datetime(block_data['Datetime'])
        
        block_data.sort_values(by='Datetime', inplace=True)
        block_data.reset_index(inplace=True, drop=True)
        
        block_data['Date'] = block_data['Datetime'].dt.date
        block_data['Hour'] = block_data['Datetime'].dt.hour
        block_data['Day'] = block_data['Datetime'].dt.weekday
        
        # Getting rid of Sunday since there is no paid parking.
        block_data = block_data.loc[block_data['Day'] != 6]
        
        # Dropping the days where the total parking is 0.
        block_data = block_data.loc[~block_data['Date'].isin(holidays)]
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


def load_daily_data(park_data):
    """Load the data into a multi-index DataFrame sorted by date and block key.
    
    :param park_data: Dictionary of DataFrames, where each DataFrame contains
    the full load data for a block.
    
    :return park_data: Multi-index DataFrame with data sorted by date and block key.
    """

    for key in park_data:
        park_data[key] = park_data[key].set_index('Datetime')

    # Merging the dataframes into multi-index dataframe.
    park_data = pd.concat(park_data.values(), keys=park_data.keys())

    park_data.index.names = ['ID', 'Datetime']

    # Making the first index the date, and the second the element key, sorted by date.
    park_data = park_data.swaplevel(0, 1).sort_index()

    return park_data


def load_daily_data_standalone(data_path):
    """Load the data into a multi-index DataFrame sorted by date and block key.
    
    :param data_path: File path to the directory with the data.
    
    :return park_data: Multi-index DataFrame with data sorted by date and block key.
    :return gps_loc: Numpy array with each row containing the lat, long pair 
    midpoints for a block.
    :param N: Integer number of samples (locations).
    """

    params = load_data(data_path)
    gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx = params

    for key in park_data:
        park_data[key] = park_data[key].set_index('Datetime')

    # Merging the dataframes into multi-index dataframe.
    park_data = pd.concat(park_data.values(), keys=park_data.keys())

    park_data.index.names = ['ID', 'Datetime']

    # Making the first index the date, and the second the element key, sorted by date.
    park_data = park_data.swaplevel(0, 1).sort_index()

    return park_data, gps_loc, N