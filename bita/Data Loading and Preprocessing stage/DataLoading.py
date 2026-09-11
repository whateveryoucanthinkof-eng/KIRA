from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# List of CSV file paths
file_paths = [
    '/content/drive/My Drive/11March_e.csv',
    '/content/drive/My Drive/12March_e.csv',
    '/content/drive/My Drive/13March_e.csv',
    '/content/drive/My Drive/14March_e.csv',
    '/content/drive/My Drive/15March_e.csv',
    '/content/drive/My Drive/16March_e.csv',
    '/content/drive/My Drive/17March_e.csv',
]#

# Initialize an empty list to store DataFrames
dfs = []

# Loop through file paths and read each CSV file
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames into one DataFrame
df = pd.concat(dfs, ignore_index=True)

# Convert 'DetectTime' column to datetime with ISO 8601 format
df['DetectTime'] = pd.to_datetime(df['DetectTime'], format='ISO8601')

# Sort the DataFrame by the 'DetectTime' column
df = df.sort_values('DetectTime')
