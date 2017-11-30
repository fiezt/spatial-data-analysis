import os
import glob
import datetime
import load_sdot_utils


curr_path = os.getcwd()

data_path = os.path.join(curr_path, '..', 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
raw_transaction_path = os.path.join(data_path, 'RawTransactionData')
if not os.path.exists(raw_transaction_path):
    os.makedirs(raw_transaction_path)

belltown_path = os.path.join(data_path, 'Belltown_Minute')
if not os.path.exists(belltown_path):
    os.makedirs(belltown_path)


# Indicating to pull data from June 2017. Edit this to what you want.
months = [6]
years = [2017]

# Pulling the parking data from the API.
for month, year in zip(months, years):
    load_sdot_utils.get_data(month, year, raw_transaction_path, verbose=True)

# Finding where data is available.
filenames = []
for fi in glob.glob(raw_transaction_path + os.sep + '*.csv'):
    date = fi.split(os.sep)[-1].split('_')[0]
    month_year = (int(date[0:2]), int(date[4:]))

    filenames.append(fi)

filenames = sorted(filenames, key=lambda fi: datetime.datetime.strptime(fi.split(os.sep)[-1].split('_')[0], '%m%d%Y'))

months_years = [fi.split(os.sep)[-1].split('_')[0] for fi in filenames]
months_years = [(int(date[0:2]), int(date[4:])) for date in months_years]
months = [month_year[0] for month_year in months_years]
years = [month_year[1] for month_year in months_years]

# Printing out where there is available data.
print('The following month/years have raw transaction data:')
for month_year in months_years:
    print(month_year)

# Subareas to either get minute loads for. Edit this to what you want.
subareas = ['Belltown']

# Indicating to get occupancies from June 2017. Edit this to what you want.
months_get_loads = [6]
years_get_loads = [2017]
months_years_get_loads = zip(months_get_loads, years_get_loads)

filenames_get_loads = [filenames[months_years.index(val)] for val in months_years if val in months_years_get_loads]

# Creating occupancy data from the transactions and writing to files.
load_sdot_utils.create_loads(subareas, months_get_loads, years_get_loads, 
                             filenames_get_loads, data_path, verbose=True)

# Starting at 8am and ending at 8pm.
start_hour = 8
end_hour = 20

# Hourly Interval
minute_interval = 60

# Paths to the Minute Occupancy Data. These should be the neighborhoods you want to get occupancy for.
file_paths = [belltown_path]

# Indicating aggregate from June 2017. Edit this to what you want.
months_aggregate = [6]
years_aggregate = [2017]
months_years_aggregate = zip(months_aggregate, years_aggregate)

# Aggregating to 1 hour occupancy data and converting to new format for future analysis.
load_sdot_utils.aggregate_loads(start_hour, end_hour, minute_interval, months_years_aggregate, file_paths)

