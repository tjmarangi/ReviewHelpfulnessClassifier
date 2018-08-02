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
1) Ingests the history.pkl file created by the training script
2) Plots the performance of the model on both training and validation sets
    
Make sure your history.pkl is in the base of model_history_archive,
or change the path towards the end of this script
    
Changeable parameters:
hist : names of history pickle to use
history_archive : location of history pickle (may need to add sub-directory)

Note that this script is NOT using the test set. It is purely for visualizing
the performance over time on the validation set (used as part of training)
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


# =============================================================================
#     
# Evaluation
# 
# =============================================================================

hist = 'history.pkl'

history_archive = 'model_history_archive'

histories = pickle.load(open(os.path.join(history_archive, hist), 'rb'))

def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
    
print('Training: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'loss'),
                                              get_avg(histories, 'acc')))
print('Validation: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'val_loss'),
                                                get_avg(histories, 'val_acc')))

def plot_acc_loss(title, histories, key_acc, key_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Accuracy
    ax1.set_title('Model accuracy (%s)' % title)
    names = []
    for i, model in enumerate(histories):
        ax1.plot(model[key_acc])
        ax1.set_xlabel('epoch')
        names.append('Model %i' % (i+1))
        ax1.set_ylabel('accuracy')
    ax1.legend(names, loc='lower right')
    # Loss
    ax2.set_title('Model loss (%s)' % title)
    for model in histories:
        ax2.plot(model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    plt.show()
    
plot_acc_loss('training', histories, 'acc', 'loss')
plot_acc_loss('validation', histories, 'val_acc', 'val_loss')