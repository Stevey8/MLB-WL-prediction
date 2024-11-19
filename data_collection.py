import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pybaseball
from pybaseball import statcast
pybaseball.cache.enable()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


def collect_pitch_data(start, end, filename = "", save_file = False):
    '''
    
    '''
    #TODO: check date time formats
    df = statcast(start_dt= start ,end_dt= end).iloc[::-1].reset_index(drop=True)

    if save_file:
        df.to_csv(filename, index = False)

    return df




















def main():
    pitches = collect_pitch_data("2023-03-30", "2023-10-01", save_file = False)
    
    return


if __name__ == "__main__":
    main()
