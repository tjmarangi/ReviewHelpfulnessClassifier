"""
Just navigate to the repo in terminal and run with:
$ python DataSlicing.py

This script takes a while to run, depending on the amount of RAM available
But only needs to run once to cut the dataset down to a useable size

parse() and getDF() functions are provided by McAuley et al. here:
http://jmcauley.ucsd.edu/data/amazon/

Changeable parameters: helpful_cutoff, file_name

@author: tjm
"""

import pandas as pd
import gzip
import os

print("This can take a while with the bigger files. Maybe grab a coffee.")

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Specify the minimum number of helpful votes for a review to be analyzed
# Recommended to keep at 10
helpful_cutoff = 10

# Specify the directory where data is stored (recommended here)
json_dir = '5core_data'

# Specify the compressed JSON for the category in which you are interested
# Testing for now with Kindle Store data (smallest)
file_name = 'reviews_Kindle_Store_5.json.gz'


final_name = file_name[:-8] + "_sliced.pkl"
full_path = os.path.join(json_dir, file_name)
out_path = os.path.join(json_dir, final_name)

# Read in the compressed JSON as a pandas df (can take a long time)
df = getDF(full_path)

# Split the nested list column 'helpful' into two lists:
# helpful_true, which counts the number of votes for helpful per review
helpful_true = []
for i in range(len(df.helpful)):
    helpful_true.append(df.helpful[i][0])

# helpful_count, which counts the number of total votes per review
helpful_count = []
for i in range(len(df.helpful)):
    helpful_count.append(df.helpful[i][1])

# Append the lists to the df
df['helpful_true'] = helpful_true
df['helpful_count'] = helpful_count

# Remove the old 'helpful' column
del df['helpful']

# Slice the dataset down to only those reviews which have a minimum of
# 10 helpfulness votes
final_df = df[df['helpful_count'] >= helpful_cutoff]

# Export the dataset to pickle format (saved to disk)
final_df.to_pickle(out_path)

# For testing successful re-import of pickle
df_input_test = pd.read_pickle(out_path)




