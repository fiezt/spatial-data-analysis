import sys
import load_sdot_utils

months = [int(sys.argv[1]) + 3]
years = [2016]
subareas = ['Belltown']

data_path = '/Users/tfiez/GitHub/spatial-data-analysis/code/../data'

filenames = ['/Users/tfiez/GitHub/spatial-data-analysis/code/../data/RawTransactionData/03012016_03312016.csv',
             '/Users/tfiez/GitHub/spatial-data-analysis/code/../data/RawTransactionData/04012016_04302016.csv',
             '/Users/tfiez/GitHub/spatial-data-analysis/code/../data/RawTransactionData/05012016_05312016.csv',
             '/Users/tfiez/GitHub/spatial-data-analysis/code/../data/RawTransactionData/06012016_06302016.csv',
             '/Users/tfiez/GitHub/spatial-data-analysis/code/../data/RawTransactionData/07012016_07312016.csv']

filenames = [filenames[int(sys.argv[1])]]

load_sdot_utils.create_loads(subareas, months, years, filenames, data_path, verbose=True)
print 'done'