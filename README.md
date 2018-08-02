# ReviewHelpfulnessClassifier
This is a fork of [cnn-text-classification](https://github.com/cmasch/cnn-text-classification) by Christopher Masch,  
itself an implementation of the design of Yoon Kim [1].  
It is purpose-built for testing helpfulness prediction of Amazon.com product reviews.  
Credit to cmasch for: cnn_model.py, TrainModel.py (mostly), Evaluate.py (mostly), Test.py (mostly).  

## Requirements
* Python 3.6
* Keras 2.0.8
* TensorFlow 1.1
* Scikit 0.19.1

## Necessary Downloads
5-core dataset(s) from Julian McAuley, UCSD [2] and [3]  
[Datasets](http://jmcauley.ucsd.edu/data/amazon/)<br>
Save them in the 5core_data directory  
(Recommend starting with reviews_Kindle_Store_5.json.gz for functionality testing. Scripts are ready for it by default)  
  
Pre-trained GloVe word vectors [4]  
[GloVe word vectors](http://nlp.stanford.edu/data/glove.6B.zip)<br>
Save it in the base of the repo. 
  
## How to run
Read docstrings of individual scripts for more details on operation.  
  
$ python DataSlicing.py  
(can take some time with the larger datafiles due to paging out of memory)  
  
$ python PreProcessing.py  
(Recommended to move the results from reviews_stacked to input_files to avoid overwrite. Training script looks here by default)  
  
$ python TrainModel.py  
(Recommended to move results from model_history_out to model_history_archive to avoid overwrite. Evaluate and Test scripts look for them here)  
  
$ python Evaluate.py  
(Will print plots which can be saved, plus max results in terminal)  
  
$ python Test.py  
(prints results to terminal)  

## Usage
As per the original repo:  
Feel free to use the [model](https://github.com/cmasch/cnn-text-classification/blob/master/cnn_model.py) and your own dataset. 

## References
[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)<br>
[2] [Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering](https://arxiv.org/pdf/1602.01585.pdf)<br>
[3] [Image-based recommendations on styles and substitutes](https://arxiv.org/pdf/1506.04757.pdf)<br>
[4] [Glove: Global vectors for word representation](http://www.aclweb.org/anthology/D14-1162)<br>
