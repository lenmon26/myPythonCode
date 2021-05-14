import pandas as pd
import numpy as np

def solomon_data():
    # Assign the filename: file
    file = 'c102.csv'

    # Read the file into a DataFrame: data
    df = pd.read_csv(file)
    df = np.array(df)
    return df