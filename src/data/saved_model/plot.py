import pandas as pd
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=True, help="the path to the history.csv file")
args = vars(ap.parse_args())
history_df = pd.read_csv(f'{args["file"]}')

def get_names(pandas_df):
    ''' return a list of lists which contain
        the given training a validation counterparts
        of data found in csv.
        returns empty list if no match is found
    '''
    keys = pandas_df.keys().tolist()
    vals = [x for x in keys if 'val_' in x]
    trains = [x for x in keys if f'val_{x}' in vals]
    return [[x,y] for x,y in zip(trains,vals)]

items = get_names(history_df)
for set in items:
    history_df[set].plot()
    plt.show()

history_df['lr'].plot()
plt.show()