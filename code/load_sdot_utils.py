import urllib2
import datetime
import pandas as pd
import numpy as np
import calendar
import os
import string
from collections import defaultdict
import glob


def get_data(month, year, file_path, verbose=False):
    """Load all transaction data for a month to a csv file from SDOT API. 
    
    :param month: Integer month to get the transaction data from.
    :param year: Integer year to get the transaction data from.
    :param file_path: File path to save transaction data to.
    :param verbose: Bool indicating whether to print progress.
    """
    
    parsed_responses = []
    
    month_name = {1: ('Jan','01'), 2: ('Feb','02'), 3: ('Mar','03'), 4: ('Apr','04'), 
                  5: ('May','05'), 6: ('Jun','06'), 7: ('Jul','07'), 8: ('Aug','08'), 
                  9: ('Sep','09'), 10: ('Oct','10'), 11: ('Nov','11'), 12: ('Dec','12')}

    # SDOT API 
    url1 = 'http://web6.seattle.gov/SDOT/wapiParkingStudy/api/ParkingTransaction?from='

    # Part of the API command.
    url2 = '&to='

    start_date = month_name[month][1] + '01' + str(year)
    start_date_static = start_date

    start_date_dt = datetime.datetime.strptime(start_date,'%m%d%Y')

    num_days = calendar.monthrange(year, month)[1]
    final_date = month_name[month][1] + str(num_days) + str(year)
    
    call_count = 0

    while start_date != final_date:
        
        # Incrementing the ending state.
        end_date_dt = start_date_dt + datetime.timedelta(days=1)
        end_date = end_date_dt.strftime('%m%d%Y')

        if verbose:
            print('Retrieving data from %s to %s' % (start_date, end_date))
        
        url = url1 + start_date + url2 + end_date
    
        # Making API call.
        response = urllib2.urlopen(url).read()
        response = response.split('\r\n')
        response = response[:-1]
    
        # Getting rid of the header if not the first iteration.
        if call_count != 0:
            response = response[1:]
        
        for line in response:
            parsed_responses.append(line.split(','))
    
        start_date = end_date
        start_date_dt = end_date_dt
    
        call_count += 1
    
    transactions = pd.DataFrame(data=parsed_responses)

    # Making the first row the header.
    transactions.columns = transactions.iloc[0]
    transactions = transactions[1:]
    
    # Writing transactions to a file.
    transactions.to_csv(os.path.join(file_path, start_date_static + '_' + final_date + '.csv'), index=False)


def get_meter_codes(elkey, paystation_info):
    """Get the meter codes for each paystation associated with an element key.
    
    Note: element keys correspond to blocks which may have multiple meters on them.
    
    :param elkey: Integer block-face element key to get meter codes for.
    :param paystation_info: Dataframe containing paystation information.
    
    :return meter_codes: Numpy array of meter codes associated with a block.
    """
    
    curr_key_meters = paystation_info.loc[paystation_info['ELMNTKEY'] == elkey]
    
    # Getting the meter codes for the current element key.
    meter_codes = curr_key_meters['PANDD_NBR'].values
    
    return meter_codes


def get_supply(elkey, date, block_info):
    """Get the maximum supply (# of spots) for an element key and date.
    
    :param elkey: Integer block-face element key to get the supply for.
    :param date: Datetime object in which to get the supply for.
    :param block_info: Dataframe containing blockface supply data.
    
    :return supply: Numpy array of the supply for the block.
    """
    
    block_supply = block_info.loc[block_info['ElementKey'] == elkey]

    block_supply = block_supply[['ElementKey', 'EffectiveStartDate', 'EffectiveEndDate', 'ParkingSpaces']]
    
    block_supply.loc[:, 'EffectiveStartDate'] = pd.to_datetime(block_supply['EffectiveStartDate'])
    block_supply.loc[:, 'EffectiveEndDate'] = pd.to_datetime(block_supply['EffectiveEndDate'])
    
    block_supply = block_supply.loc[block_supply['EffectiveStartDate'] <= date]
    block_supply = block_supply.loc[(block_supply['EffectiveEndDate'] >= date) | (block_supply['EffectiveEndDate'].isnull())]
    
    supply = block_supply['ParkingSpaces'].values
    
    return supply


def get_element_keys_by_paid_area(paid_area, paystation_info):
    """Get the element keys associated with a paid area.
    
    :param paid_area: String of paid area to get element keys for.
    :param paystation_info: Dataframe containing paystation_info information.
    
    :return elkeys: List of element keys associated with a paid area.
    """
    
    elkeys = paystation_info[paystation_info['PAIDAREA'] == paid_area]['ELMNTKEY'].tolist()
    
    return elkeys


def get_block_load(date, transactions, meter_codes, supply):
    """Get the load for an element key.
    
    :param date: Date in which to get supply for in string format e.g. "2017-01-19"
    :param transactions: Dataframe containing the paid parking transactions. 
    :param meter_codes: List of meter codes for an element key.
    :param supply: Numpy array of the supply for the element key.
    
    return output: Zipped list with two arrays, the first is the normalized load 
    (by supply) and the second is the actual load.
    """
    
    data = []
    
    # Getting the transactions for the date specified as the input.
    time_range = pd.date_range(date + " 00:00:00", date + " 23:59:59", freq='1S')
    mask = ((transactions['TransactionDateTime'] > time_range[0]) & (transactions['TransactionDateTime'] < time_range[-1]))
    date_transactions = transactions[mask]
        
    for code in meter_codes:
        meter_transactions = date_transactions.loc[date_transactions['MeterCode'] == code]
        time_duration = meter_transactions.ix[:, ['PaidDuration', 'TransactionDateTime']].values 
        
        # Convert paid duration column from seconds to minutes.
        time_duration[:, 0] = time_duration[:, 0]/60.0 
        
        # Convert to datetime format from string.
        date_pd = pd.DatetimeIndex([date])[0]
        date_python = date_pd.to_pydatetime()

        for i in range(len(time_duration)):
            
            # Convert TransactionDateTime column to datetime format.
            time_datetime = time_duration[i, 1].to_pydatetime()
            
            # Get the time in seconds from the start of the day the transaction began at.
            time_seconds = float(time_datetime.strftime('%s')) - float(date_python.strftime('%s'))
            time_seconds /= 60.0
            
            # Saving transaction start time and duration in minutes.
            data.append([int(time_seconds), int(time_duration[i, 0])])
    
    start = np.zeros([1, 60*24])
    stop = np.zeros([1, 60*24])
    load = np.zeros([1, 60*24])
    percentage = np.zeros([1, 60*24])

    for i in range(len(data)):
        start[0, data[i][0]] += 1.0
        stop[0, min(data[i][0] + data[i][1], 1439)] += 1.0 

    for j in range(1, 60*24):
        load[0, j] = load[0, j-1] + start[0, j] - stop[0, j] 
    
    if not ((supply == 0).any()):
        percentage[0, :] = 1.0*load[0, :]/supply
        
    output = zip(percentage, load)

    return output


def get_loads(month, year, subarea, paystation_info, block_info, transactions, data_path):
    """Get the loads for a subarea using the paid parking transactions.

    Note that the transaction data will be saved in a subfolder in the path provided
    under a folder named after the subarea. This folder will be created if it does
    not exist.

    :param month: Integer month to get occupancy data for.
    :param year: Integer year to get occupancy for.
    :param subarea: String of zone to get occupancy data for.
    :param paystation_info: Dataframe containing paystation information.
    :param block_info: Dataframe containing blockface supply information.
    :param transactions: Dataframe containing paid parking transaction data.
    :param data_path: Path to save directories containing load files in.
    """
  
    month_name = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    


    subarea_dir = data_path + '/' + subarea.translate(None, string.punctuation).replace(' ', '') + '_Minute'
    if not os.path.exists(subarea_dir):
        os.makedirs(subarea_dir)

    num_days = calendar.monthrange(year, month)[1]
    days = np.arange(1, num_days + 1)

    date_start = str(month) + '/1/2017' 
    date_end = str(month) + '/' + str(num_days) + '/2017'
    dates = pd.date_range(date_start, date_end, freq='1D')
        
    element_keys = get_element_keys_by_paid_area(subarea, paystation_info)
    
    for key in element_keys:

        loads = []

        for date in dates:
            curr_date = str(date.month) + '/' + str(date.day) + '/' + str(date.year)

            supply = get_supply(key, date, block_info)

            if supply:
                meter_codes = get_meter_codes(key, paystation_info)
                block_load = get_block_load(curr_date, transactions, meter_codes, supply)
                loads.append(block_load[0][0])
            else: 
                loads.append(np.nan*np.ones(1440))

        load_frame = pd.DataFrame(loads) 
        load_frame = load_frame.transpose()
        load_frame.columns = days
        load_frame.to_csv(os.path.join(subarea_dir, str(key) + '-2017' 
                                       + month_name[month]  +'-loads.csv'), index=False)


def create_loads(subareas, months, years, file_paths, data_path, verbose=False):
    """

    :param subareas: List of subareas to get loads for.
    :param months: List of integer months to get loads for subareas.
    :param years: List of integer years corresponding to months to get loads for subareas.
    :param file_paths: List of file paths to paid parking transaction month data.
    :param data_path: File path to block_info.csv and paystation_info.csv
    :param verbose: Bool indicating whether to print progress.
    """

    block_info = pd.read_csv(os.path.join(data_path, 'block_info.csv'))
    paystation_info = pd.read_csv(os.path.join(data_path, 'paystation_info.csv'))

    for month, year, file in zip(months, years, file_paths):

        transactions = pd.read_csv(file)
        transactions['TransactionDateTime'] = pd.to_datetime(transactions['TransactionDateTime'])
        
        for subarea in subareas:
            if verbose:
                print('Getting loads for month %d and year %d for subarea %s' % (month, year, subarea))

            get_loads(month, year, subarea, paystation_info, block_info, transactions, data_path)


def aggregate_loads(start_hour, end_hour, minute_interval, file_paths):
    """Aggregate loads to a new interval and write files for each key in sequential fashion.
    
    :param start_hour: Integer 0-23 of hour to start saving the load data at.
    :param end_hour: Integer 1-24 of hour to stop saving the load data at.
    :param minute_interval: Integer number of aggregate loads to, must be divisible by 60.
    :param file_paths: List of directories where minute load data is for a subarea.
    """

    for path in file_paths:
        
        # Creating new directory to write new files to if it does not exist.
        subarea_name = path.split('/')[-1].split('_')[0]
        subdir = path + '/..'
        new_dir = subdir + '/' + subarea_name
        
        if minute_interval == 1:
            dir_addon = '_Minute'
        elif minute_interval == 60:
            dir_addon = '_Hour'
        else:
            dir_addon = '_' + str(minute_interval) + 'Minute' 
            
        new_dir += dir_addon
        
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        park_data = defaultdict(list)

        month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 
                     'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

        for fi in glob.glob(path + '/*.csv'):

            data = pd.read_csv(fi)
            cols = data.columns
            interval_data = []

            # Aggregate from 1 minute intervals to specified interval.
            for i in xrange(0, 1440, minute_interval):
                interval = data.loc[i:i+minute_interval-1]
                interval_avg = interval.values.mean(axis=0)
                interval_data.append(interval_avg)

            interval_data = np.vstack((interval_data))
            interval_data = interval_data[start_hour*(60/minute_interval):end_hour*(60/minute_interval)]
            interval_data = interval_data.T
            interval_data = interval_data.flatten()
                    
            times = [(hour, minute) for hour in xrange(start_hour, end_hour) for minute in xrange(0, 60, minute_interval)]
            
            split = fi.split('/')[-1].split('-')
            key = split[0]
            date = split[1]
            date = date[:4] + '-' + date[4:]
            year, month = int(date.split('-')[0]), date.split('-')[1]
            
            index = [datetime.datetime(year, month_map[month], int(day), hour, minute, 0) for day in cols.tolist() for hour, minute in times]

            new_df = pd.DataFrame(interval_data, index=index, columns=['Load'])
            new_df.index.name = 'Datetime'
            park_data[int(key)].append(new_df.copy())

        for key in park_data:
            park_data[key] = pd.concat(park_data[key], axis=0)
            park_data[key] = park_data[key].sort_index(axis=0)
            park_data[key] = park_data[key].reset_index()
            park_data[key].to_csv(os.path.join(new_dir, str(key)+'.csv'), sep=',', index=False, header=False)