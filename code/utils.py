import pandas as pd
import numpy as np
import pickle
import os
import glob
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
    holidays = cal.holidays(start='2012-01-01', end=datetime.datetime.now().date()).to_pydatetime()
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


def load_data_new(data_path, load_paths):
    """Process loads 
    
    :param data_path:
    :param load_paths: 
    """
    
    # Load file containing GPS coordinates for blockfaces.
    with open(os.path.join(data_path, 'blockface_locs.p'), 'rb') as f:
        locations = pickle.load(f)
    
    # Load sheet containing blockface info about blockface operating times.
    block_info = pd.read_csv(os.path.join(data_path, 'block_info.csv'))
    keep_columns = ['ElementKey', 'PeakHourStart1', 'PeakHourEnd1', 
                    'PeakHourStart2', 'PeakHourEnd2', 'PeakHourStart3', 
                    'PeakHourEnd3', 'EffectiveStartDate', 'EffectiveEndDate']
    block_info = block_info[keep_columns]
    
    # Converting to datetime format for processing.
    for col in keep_columns:
        if 'Hour' in col:
            block_info.loc[:, col] = pd.to_datetime(block_info[col]).dt.time
        elif 'Date' in col:
            block_info.loc[:, col] = pd.to_datetime(block_info[col])
        else:
            pass
    
    # Loading holiday information for when paid parking is not available.
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2012-01-01', end=datetime.datetime.now().date()).to_pydatetime()
    holidays = [hol.date() for hol in holidays]


    # To contain average loads at each day of week and hour of day for each block.
    avg_loads = []

    # To contain GPS midpoint for each blockface.
    gps_loc = []

    # To hold list of all element keys whose data are processed.
    element_keys = []

    # To be converted to dataframe containing occupancy information.
    park_data = {}
    
    # Convert path to list if only a single path is provided.
    if isinstance(load_paths, list):
        pass
    else:
        load_paths = [load_paths]

    for load_path in load_paths:
        for fi in sorted(glob.glob(load_path + '/*.csv')):
            key = int(fi.split('/')[-1].split('.')[0])

            block_data = pd.read_csv(fi, names=['Datetime', 'Load'])

            # Clipping the loads to be no higher than 1.5
            block_data['Load'] = block_data['Load'].clip_upper(1.5)

            block_data['Datetime'] = pd.to_datetime(block_data['Datetime'])
            block_data.sort_values(by='Datetime', inplace=True)
            block_data.reset_index(inplace=True, drop=True)

            # Dropping days where the supply was 0 for this blockface.
            block_data.dropna(inplace=True)
            block_data.reset_index(inplace=True, drop=True)

            block_data['Date'] = block_data['Datetime'].dt.date
            block_data['Time'] = block_data['Datetime'].dt.time
            block_data['Day'] = block_data['Datetime'].dt.weekday
            block_data['Hour'] = block_data['Datetime'].dt.hour
            block_data['Minute'] = block_data['Datetime'].dt.minute

            # Getting rid of Sunday since there is no paid parking.
            block_data = block_data.loc[block_data['Day'] != 6]

            # Dropping the days where the total parking is 0 because of holidays.
            block_data = block_data.loc[~block_data['Date'].isin(holidays)]
            block_data.reset_index(inplace=True, drop=True)

            # If block contains no data, skip it.
            if len(block_data) == 0:
                continue

            # Get GPS midpoint for blockface and skip if no information for it.
            if key in locations:
                curr_block = locations[key]

                lat1, lat2 = curr_block[1], curr_block[-2]
                lon1, lon2 = curr_block[0], curr_block[-3]

                mid_lat = (lat1 + lat2)/2.
                mid_long = (lon1 + lon2)/2.
                gps_loc.append([mid_lat, mid_long])
            else:
                continue

            # Getting blockface info for the current key about hours of operation.
            curr_block_info = block_info.loc[block_info['ElementKey'] == key]

            # Filling times where paid parking is not allowed for the block with nan.
            for index, row in curr_block_info.iterrows():
                row_null = row.isnull()

                if not row_null['PeakHourStart1'] and not row_null['PeakHourStart2'] and not row_null['PeakHourStart3']:
                    continue

                if not row_null['EffectiveEndDate']:
                    row['EffectiveEndDate'] += datetime.timedelta(hours=23, minutes=59, seconds=59)

                if not row_null['PeakHourStart1']:

                    start1 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourStart1']) for i in xrange(len(block_data))])
                    end1 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourEnd1']) for i in xrange(len(block_data))])

                    if row_null['EffectiveEndDate']:
                        mask1 = ((row['EffectiveStartDate'] <= block_data['Datetime'])
                                 & (start1 <= block_data['Datetime']) 
                                 & (end1 > block_data['Datetime']))
                    else:
                        mask1 = ((row['EffectiveStartDate'] <= block_data['Datetime']) 
                                 & (row['EffectiveEndDate'] >= block_data['Datetime'])
                                 & (start1 <= block_data['Datetime']) 
                                 & (end1 > block_data['Datetime']))

                    block_data.loc[mask1, 'Load'] = np.nan    

                if not row_null['PeakHourStart2']:

                    start2 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourStart2']) for i in xrange(len(block_data))])
                    end2 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourEnd2']) for i in xrange(len(block_data))])

                    if row_null['EffectiveEndDate']:
                        mask2 = ((row['EffectiveStartDate'] <= block_data['Datetime'])
                                & (start2 <= block_data['Datetime']) 
                                & (end2 > block_data['Datetime']))
                    else:
                        mask2 = ((row['EffectiveStartDate'] <= block_data['Datetime']) 
                                 & (row['EffectiveEndDate'] >= block_data['Datetime'])
                                 & (start2 <= block_data['Datetime']) 
                                 & (end2 > block_data['Datetime']))

                    block_data.loc[mask2, 'Load'] = np.nan  

                if not row_null['PeakHourStart3']:

                    start3 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourStart3']) for i in xrange(len(block_data))])
                    end3 = pd.Series([datetime.datetime.combine(block_data.loc[i, 'Date'], row['PeakHourEnd3']) for i in xrange(len(block_data))])

                    if row_null['EffectiveEndDate']:
                        mask3 = ((row['EffectiveStartDate'] <= block_data['Datetime'])
                                 & (start3 <= block_data['Datetime']) 
                                 & (end3 > block_data['Datetime']))
                    else:
                        mask3 = ((row['EffectiveStartDate'] <= block_data['Datetime']) 
                                 & (row['EffectiveEndDate'] >= block_data['Datetime'])
                                 & (start3 <= block_data['Datetime']) 
                                 & (end3 > block_data['Datetime']))

                    block_data.loc[mask3, 'Load'] = np.nan   

            element_keys.append(key)

            park_data[key] = block_data

            # Getting the average load for each hour of the week for the block.
            avg_load = block_data.groupby(['Day', 'Hour'])['Load'].mean().values.reshape((1,-1))
            avg_loads.append(avg_load)
    
    # Each row has load and GPS locations for a block. Ordered as in element_keys.
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
    N = len(element_keys)
    
    for key in park_data:
        park_data[key] = park_data[key].set_index('Datetime')

    # Merging the dataframes into multi-index dataframe.
    park_data = pd.concat(park_data.values(), keys=park_data.keys())

    park_data.index.names = ['ID', 'Datetime']

    # Making the first index the date, and the second the element key, sorted by date.
    park_data = park_data.swaplevel(0, 1).sort_index()

    return element_keys, gps_loc, avg_loads, park_data, N, P, idx_to_day_hour, day_hour_to_idx