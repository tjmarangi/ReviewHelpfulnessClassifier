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
5-core dataset(s) from Julian McAuley, UCSD [1] and [2]  
[datasets](http://jmcauley.ucsd.edu/data/amazon/)<br>
Save them in the 5core_data directory  
  
Pre-trained GloVe word vectors [4]
[GloVe word vectors](http://nlp.stanford.edu/data/glove.6B.zip)<br>
Save it in the base of the repo. 

## Usage
As per the original repo:  
Feel free to use the [model](https://github.com/cmasch/cnn-text-classification/blob/master/cnn_model.py) and your own dataset. 

## References
[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)<br>
[2] [Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering](https://arxiv.org/pdf/1602.01585.pdf)<br>
[3] [Image-based recommendations on styles and substitutes](https://arxiv.org/pdf/1506.04757.pdf)<br>
[4] [Glove: Global vectors for word representation](http://www.aclweb.org/anthology/D14-1162)<br>
