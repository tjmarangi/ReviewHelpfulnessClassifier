"""
Just navigate to the repo in terminal and run with:
$ python TrainModel.py

Credit to: cmasch at GitHub
https://github.com/cmasch/cnn-text-classification

This script is a combination/adaptation of some scripts he offers here:
https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb
for the purposes of this research

The model file cnn_model.py is entirely his creation, based on
(Kim 2014) as referenced within.

Purpose:
1) Ingests the pre-processed data created by PreProcessing.py
2) Labels the reviews either positive or negative (t/f as helpful in this case)
2) Performs some text mining / tokenization with re and nltk
3) Loads pre-trained GloVe embeddings
4) Trains the model with the pre-trained embeddings and review data
5) Periodically saves the model and history to respective files
    
    
Changeable parameters:
input_test_t, input_test_f : names of susbset files to run with
parameters from the parameters section (provided with settings used for research)

Note that this script will not automatically perform cross-validation
This is by design, because a single run takes very long on these settings
To do cross-validation, initiate a new run after changing input_test files
Save the model and history from each run in the archive directory provided
"""


import keras, os, pickle, re, sklearn, string, tensorflow
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split


print('Keras version: \t\t%s' % keras.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)

# Name of input files
input_train_t = 'train_1_t.txt'
input_train_f = 'train_1_f.txt'



# =============================================================================
# 
# Parameters
# 
# =============================================================================

# Parameters have been returned to default for ease of functionality testing
# Results in paper were achieved with following changes
# MAX_NUM_WORDS  = 150000
# FEATURE_MAPS   = [100,100,100]
# BATCH_SIZE     = 400

# EMBEDDING
MAX_NUM_WORDS  = 15000
EMBEDDING_DIM  = 300
MAX_SEQ_LENGTH = 200
USE_GLOVE      = True

# MODEL
FILTER_SIZES   = [3,4,5]
FEATURE_MAPS   = [10,10,10]
DROPOUT_RATE   = 0.5

# LEARNING
BATCH_SIZE     = 200
NB_EPOCHS      = 40
RUNS           = 1
VAL_SIZE       = 0.1



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


# Dataset locations
negative_docs = read_files(os.path.join('input_files', input_train_f))
positive_docs = read_files(os.path.join('input_files', input_train_t))

docs   = negative_docs + positive_docs
labels = [0 for _ in range(len(negative_docs))] + [1 for _ in range(len(positive_docs))]

print('Training samples: %i' % len(docs))






# =============================================================================
# 
# Tokenizer
# 
# =============================================================================



def max_length(lines):
    """
    Calculate the maximum document length
    """
    return max([len(s.split()) for s in lines])

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)

length = max_length(docs)
word_index = tokenizer.word_index

result = [len(x.split()) for x in docs]
print('Text informations:')
print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result),
                                                                                np.min(result),
                                                                                np.mean(result),
                                                                                MAX_SEQ_LENGTH))
print('vacobulary size: %i / limit: %i' % (len(word_index), MAX_NUM_WORDS))

# Padding all sequences to same length of `MAX_SEQ_LENGTH`
data   = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')



# =============================================================================
# 
# # Embeddings
# 
# =============================================================================

def create_glove_embeddings():
    print('Pretrained embeddings GloVe is loading...')

    embeddings_index = {}
    f = open('glove.6B.%id.txt' % EMBEDDING_DIM)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embeddings_index))

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM,
                     input_length=MAX_SEQ_LENGTH,
                     weights=[embedding_matrix],
                     trainable=True
                    )



# =============================================================================
# 
# Training
# 
# =============================================================================


import cnn_model

histories = []

for i in range(RUNS):
    print('Running iteration %i/%i' % (i+1, RUNS))
    
    X_train, X_val, y_train, y_val = train_test_split(data, labels, 
                                                      test_size=VAL_SIZE, 
                                                      random_state=42)
    
    emb_layer = None
    if USE_GLOVE:
        emb_layer = create_glove_embeddings()
    
    model = cnn_model.build_cnn(
        embedding_layer=emb_layer,
        num_words=MAX_NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        filter_sizes=FILTER_SIZES,
        feature_maps=FEATURE_MAPS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout_rate=DROPOUT_RATE
    )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adadelta(clipvalue=3),
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[ModelCheckpoint('model_history_out/model-%i.h5'%(i+1), 
                                   monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='min'),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                     patience=4, min_lr=0.01)
                  ]
    )
    print()
    histories.append(history.history)
    
with open('model_history_out/history.pkl', 'wb') as f:
    pickle.dump(histories, f)
    
print('Be sure to save your model and history somewhere safe!')