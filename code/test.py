import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import calendar
import glob
import datetime
import load_sdot_utils
import pandas as pd
from collections import defaultdict

curr_path = os.getcwd()
data_path = curr_path + '/../data'
raw_transaction_path = data_path + '/RawTransactionData'
belltown_path = data_path + '/Belltown_Minute'
commcore_path = data_path + '/CommercialCore_Minute'
pikepine_path = data_path + '/PikePine_Minute'
firsthill_path = data_path + '/FirstHill_Minute'
dennytriangle_path = data_path + '/DennyTriangle_Minute'

filenames = []
for fi in glob.glob(raw_transaction_path + '/*.csv'):
    filenames.append(fi)

filenames = sorted(filenames, key=lambda fi: datetime.datetime.strptime(
    fi.split('/')[-1].split('_')[0], '%m%d%Y'))

months_years = [fi.split('/')[-1].split('_')[0] for fi in filenames]
months_years = [(int(date[0:2]), int(date[4:])) for date in months_years]

months_years = months_years[:5]
months = [month_year[0] for month_year in months_years]
years = [month_year[1] for month_year in months_years]
subareas = ['Uptown Triangle', 'Uptown', 'South Lake Union']
filenames = filenames[:5]

load_sdot_utils.create_loads(subareas, months, years, filenames, data_path,
                             verbose=True)
