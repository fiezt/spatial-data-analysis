import os
import glob
import datetime
import load_sdot_utils

curr_path = os.getcwd()
data_path = os.path.join(curr_path, '..', 'data')
raw_transaction_path = os.path.join(data_path, 'RawTransactionData')
belltown_path = os.path.join(data_path, 'Belltown_Minute')
commcore_path = os.path.join(data_path, 'CommercialCore_Minute')
pikepine_path = os.path.join(data_path, 'PikePine_Minute')
firsthill_path = os.path.join(data_path, 'FirstHill_Minute')
dennytriangle_path = os.path.join(data_path, 'DennyTriangle_Minute')

# Indicating to pull data from March-July 2017.
months = [4, 5, 6, 7]
years = [2017, 2017, 2017, 2017]
months_years = [(4, 2017), (5, 2017), (6, 2017), (7, 2017)]

# Pulling the parking data from the API.
for month, year in zip(months, years):
    load_sdot_utils.get_data(month, year, raw_transaction_path, verbose=False)

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

subareas = ['Belltown', 'First Hill', 'Denny Triangle', 'Commercial Core', 'Pike-Pine']

# Creating occupancy data from the transactions and writing to files.
load_sdot_utils.create_loads(subareas, months, years, filenames, data_path, verbose=True)

start_hour = 8
end_hour = 20
minute_interval = 60
file_paths = [belltown_path, firsthill_path, dennytriangle_path, commcore_path, pikepine_path]

# Aggregating to 1 hour occupancy data and converting to new format for future analysis.
load_sdot_utils.aggregate_loads(start_hour, end_hour, minute_interval, months_years, file_paths)