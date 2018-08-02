"""
Just navigate to the repo in terminal and run with:
$ python Evaluate.py

Credit to: cmasch at GitHub
https://github.com/cmasch/cnn-text-classification

This script is a combination/adaptation of some scripts he offers here:
https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb
for the purposes of this research

The model file cnn_model.py is entirely his creation, based on
(Kim 2014) as referenced within.

Purpose:
1) Ingests the model file created by the training script
2) Decodes the model on its corresponding test observations
3) Reports the test accuracy and loss of the trained model
    
Make sure your model is in the base of model_history_archive,
or change the path towards the end of this script
    
Changeable parameters:
input_test_t, input_test_f : names of corresponding test observations to use
input_dir : relative directory where files are (may need to add sub-directory)

Parameters: set equal to what was used in training
"""

import os, re, string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# These need to be set to the test files corresponding to the train files
input_test_t = 'test_1_t.txt'
input_test_f = 'test_1_f.txt'

input_dir = 'input_files'



# =============================================================================
# 
# Parameters
# 
# =============================================================================

# SET THESE IDENTICAL TO TRAINING PARAMETERS


MAX_NUM_WORDS  = 15000

MAX_SEQ_LENGTH = 200

RUNS           = 1


# =============================================================================
# 
# Pre-Processing
# 
# =============================================================================

def clean_doc(doc):
    """
    Cleaning a document by several methods:
        - Lowercase
        - Removing whitespaces
        - Removing numbers
        - Removing stopwords
        - Removing punctuations
        - Removing short words
    """
    stop_words = set(stopwords.words('english'))
    
    # Lowercase
    doc = doc.lower()
    # Removing multiple whitespaces
    doc = re.sub(r"\?", " \? ", doc)
    # Remove numbers
    doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # Tokens with less then two characters will be ignored
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def read_files(path):
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename)) as f:
                doc = f.read()
                doc = clean_doc(doc)
                documents.append(doc)
    
    # Read in all lines in a txt file
    if os.path.isfile(path):        
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
    return documents



negative_docs_test = read_files(os.path.join(input_dir, input_test_f))
positive_docs_test = read_files(os.path.join(input_dir, input_test_t))

X_test = negative_docs_test + positive_docs_test
y_test = [0 for _ in range(len(negative_docs_test))] + [1 for _ in range(len(positive_docs_test))]



# =============================================================================
#     
# Test
# 
# =============================================================================




tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_test)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_test = pad_sequences(sequences_test, maxlen=MAX_SEQ_LENGTH, padding='post')

test_loss = []
test_accs = []

for i in range(0,RUNS):
    cnn_ = load_model("model_history_archive/model-%i.h5" % (i+1))
    
    score = cnn_.evaluate(X_test, y_test, verbose=1)
    test_loss.append(score[0])
    test_accs.append(score[1])
    
    print('Running test with model %i: %0.4f loss / %0.4f acc' % (i+1, score[0], score[1]))
    
print('\nAverage loss / accuracy on testset: %0.4f loss / %0.4f acc' % (np.mean(test_loss),
                                                                        np.mean(test_accs)))
print('Standard deviation: (+-%0.4f) loss / (+-%0.4f) acc' % (np.std(test_loss), np.std(test_accs)))