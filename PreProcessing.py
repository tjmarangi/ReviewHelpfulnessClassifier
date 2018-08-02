"""
Just navigate to the repo in terminal and run with:
$ python PreProcessing.py

Pre-processes the sliced dataframe .pkl from the DataSlicing.py script.
Returns 5 rotated sample subsets split into both train/test and helpful=t/f 
(suitable for five-fold cross-validation) as .txt files with one review 
per line (4 .txt files per subset)

Note that 5-fold cross-validation was not employed in the submitted report
due to prohibitive cost restraints, but the possibility has been engineered

Changeable parameters: max_votes, helpful_cutoff, file_name

1) Drops reviews with number of votes above max_votes (outliers)
2) Calculates and appends to df the helpful_ratio of helpful_true/helpful_count
3) Calculates and appends to df the boolean is_helpful based on helpful_cutoff
4) Reduces sample size to the nearest lesser multiple of 50,000 (readability)
5) Creates 5 unique train/test subsets of the data (for 5-fold cross-val)
6) Writes all reviews to lists based on subset, train/test, and helpful=t/f
7) Writes each list to a .txt file with one-line per review

@author: tjm
"""

import pandas as pd
import numpy as np
import os

# Reviews with more than 1000 votes can be generally be considered outliers
# They make up a very small handful of the total
# But this parameter can be changed or removed
max_votes = 1000

# helpful_cutoff is the parameter for the cutoff in the helpful_ratio
# at or above which a review is considered 'helpful'.
# In the research we tested at both 0.5 and 0.75, others have used 0.6
helpful_cutoff = 0.5

# Specify the directory where data is stored (recommended here)
data_dir = '5core_data'

# Specify the filename of the sliced pickle you wish to pre-process
# Testing for now with Kindle Store data (smallest)
file_name = 'reviews_Kindle_Store_5_sliced.pkl'
full_path = os.path.join(data_dir, file_name)

# Read in the sliced .pkl from the DataSlicing.py script
df = pd.read_pickle(full_path)

# Limit to reviews with no more than 1000 votes (can be considered outliers)
df = df[df['helpful_count'] <= max_votes]

# Calculate the helpful_ratio of helpful_true/helpful_count per review
helpful_ratio = []
for i in range(len(df.helpful_count)):
    helpful_ratio.append(df.helpful_true.iloc[i]/df.helpful_count.iloc[i])

# Append helpful_ratio as a new column to the dataset
df['helpful_ratio'] = helpful_ratio

ratio_array = np.asarray(helpful_ratio)

# Calculate whether each review "is helpful" based on helpful_cutoff (above)
is_helpful = []
for i in range(len(df.helpful_ratio)):
    if df.helpful_ratio.iloc[i] >= helpful_cutoff:
        is_helpful.append(True)
    else:
        is_helpful.append(False)

# Append is_helpful as a new column to the dataset
df['is_helpful'] = is_helpful

# Specify desired sample size as the nearest lesser multiple of 5
# This had to be reduced from nearest lesser multiple of 50000
# (as used with the book review dataset)
# because smaller datasets sometimes have less than 50000
# 5 also works, just the sample size numbers might not be as clean
desired_sample = len(df.reviewerID)-(len(df.reviewerID)%5)

# Randomly shuffle the data (because it is currently ordered by product)
# Then take a sample of desired_sample from it
df_shuffled = df.sample(frac=1, random_state=10).reset_index(drop=True)
df_shuffled = df.sample(n=desired_sample, random_state=10).reset_index(drop=True)

# Subsetting for test/train and 5-fold cross validation
# should work for any sample size cleanly divisible by 5
size = desired_sample
fifth = int(size/5)

# Perform rotating index-based slicing to create 5 unique train/test subsets
df_train_subset_1 = df_shuffled.iloc[0:(4*fifth)]
df_test_subset_1 = df_shuffled.iloc[(4*fifth):size]

df_train_subset_2 = df_shuffled.iloc[fifth:size]
df_test_subset_2 = df_shuffled.iloc[0:fifth]

df_train_subset_3 = pd.concat([df_shuffled.iloc[(2*fifth):size], 
                               df_shuffled.iloc[0:fifth]])
df_test_subset_3 = df_shuffled.iloc[fifth:(2*fifth)]

df_train_subset_4 = pd.concat([df_shuffled.iloc[(3*fifth):size],
                               df_shuffled.iloc[0:(2*fifth)]])
df_test_subset_4 = df_shuffled.iloc[(2*fifth):(3*fifth)]

df_train_subset_5 = pd.concat([df_shuffled.iloc[(4*fifth):size],
                               df_shuffled.iloc[0:(3*fifth)]])
df_test_subset_5 = df_shuffled.iloc[(3*fifth):(4*fifth)]

train_subsets = [df_train_subset_1, df_train_subset_2, df_train_subset_3,
                 df_train_subset_4, df_train_subset_5]

test_subsets = [df_test_subset_1, df_test_subset_2, df_test_subset_3,
                df_test_subset_4, df_test_subset_5]

# Instantiate lists of lists where review texts will be stored
train_1_t = []
train_2_t = []
train_3_t = []
train_4_t = []
train_5_t = []
train_t = [train_1_t, train_2_t, train_3_t, train_4_t, train_5_t]

train_1_f = []
train_2_f = []
train_3_f = []
train_4_f = []
train_5_f = []
train_f = [train_1_f, train_2_f, train_3_f, train_4_f, train_5_f]

test_1_t = []
test_2_t = []
test_3_t = []
test_4_t = []
test_5_t = []
test_t = [test_1_t, test_2_t, test_3_t, test_4_t, test_5_t]

test_1_f = []
test_2_f = []
test_3_f = []
test_4_f = []
test_5_f = []
test_f = [test_1_f, test_2_f, test_3_f, test_4_f, test_5_f]


for i in range(len(test_subsets)):
    for j in range(len(train_subsets[i].reviewerID)):
        if train_subsets[i].is_helpful.iloc[j] == True:
            train_t[i].append(train_subsets[i].reviewText.iloc[j])
        else:
            train_f[i].append(train_subsets[i].reviewText.iloc[j])
    for k in range(len(test_subsets[i].reviewerID)):
        if test_subsets[i].is_helpful.iloc[k] == True:
            test_t[i].append(test_subsets[i].reviewText.iloc[k])
        else:
            test_f[i].append(train_subsets[i].reviewText.iloc[k])


out_dir = 'reviews_stacked'


for i in range(len(train_t)):
    outname = 'train_' + str(i+1) + '_t.txt'
    outpath = os.path.join(out_dir, outname)
    with open(outpath, 'w') as outfile:
        for review in train_t[i]:
            outfile.write(review + '\n')

for i in range(len(train_f)):
    outname = 'train_' + str(i+1) + '_f.txt'
    outpath = os.path.join(out_dir, outname)
    with open(outpath, 'w') as outfile:
        for review in train_f[i]:
            outfile.write(review + '\n')

for i in range(len(test_t)):
    outname = 'test_' + str(i+1) + '_t.txt'
    outpath = os.path.join(out_dir, outname)
    with open(outpath, 'w') as outfile:
        for review in test_t[i]:
            outfile.write(review + '\n')
        
for i in range(len(test_f)):
    outname = 'test_' + str(i+1) + '_f.txt'
    outpath = os.path.join(out_dir, outname)
    with open(outpath, 'w') as outfile:
        for review in test_f[i]:
            outfile.write(review + '\n')

# Can use the below for small-scale functionality testing of the Keras script

# =============================================================================
# small_train_3_t = train_3_t[0:650]
# with open(os.path.join(out_dir, 'small_train_3_t.txt'), 'w') as outfile:
#     for review in small_train_3_t:
#         outfile.write(review + '\n')
#         
# small_train_3_f = train_3_f[0:350]
# with open(os.path.join(out_dir, 'small_train_3_f.txt'), 'w') as outfile:
#     for review in small_train_3_f:
#         outfile.write(review + '\n')
# 
# small_test_3_t = test_3_t[0:130]
# with open(os.path.join(out_dir, 'small_test_3_t.txt'), 'w') as outfile:
#     for review in small_test_3_t:
#         outfile.write(review + '\n')
# 
# small_test_3_f = test_3_f[0:70]
# with open(os.path.join(out_dir, 'small_test_3_f.txt'), 'w') as outfile:
#     for review in small_test_3_f:
#         outfile.write(review + '\n')
# =============================================================================
