# Machine-Learning-with-TF-diamonds
A personal project to practice learning TensorFlow on an independent data set.

## About the project

Recently, I have taken an interest in learning about the machine learning library known as `TensorFlow`. I felt that TensorFlow was a strong starting point for furthering my skills in data science with the Python language. 

##### Motivation

I began my quest of learning TensorFlow through Google's Machine Learning Crash Course and followed the guides on building simple linear regression and binary classification models. From there, I moved on to TensorFlow's documentation and began teaching myself through the guides that they have posted on their website. At some point, there were questions that I had that could not be simply answered by the basic examples that I had covered, and I wanted a chance to test myself on the knowledge that I had been learning. 

One of the main questions that I had been pondering was "how can I take what I have learned, and apply it to a completely new set of data," and more specifically, "if I am able to find an intriguing set of data, how do I know what types of data I can build a model with, and how would I go about framing each of those in a scientific way that my model can understand?" Because of these questions, I set out to learn TensorFlow on one of the tfds datasets that is provided on TensorFlow's website. My goal in this first project with TF was the following:
  - To take a completely new example of data from TensorFlow's provided datasets and create several models that accomplished the following tasks:
    - Establish the types of data in the dataset and possible correlations between the data that might make for an interesting ML model
    - Learn the basics of TF datasets and the subsequent map/filter functions for building data input pipelines
    - Create a basic numerical linear regression model
    - Create a categorical feature linear regression model
    - Create a numerical binary classification model
    - Create a categorical binary classification model
    - Learn to write code to save the models and each model's history
    - Write a script that could load a specific model and the history to plot the metrics on figures


##### The Data Set

For this first project I chose to use the tfds dataset *'diamonds'* that included the following features and labels:
  - Features:
    - carat: Weight of the diamond.
    - cut: Cut quality (ordered worst to best).
    - color: Color of the diamond (ordered best to worst).
    - clarity: Clarity of the diamond (ordered worst to best).
    - x: Length in mm.
    - y: Width in mm.
    - z: Depth in mm.
    - depth: Total depth percentage: 100 * z / mean(x, y)
    - table: Width of the top of the diamond relative to the widest point.
  - Labels:
    - price: The price of the diamond in US dollars

This dataset seemed reasonably simple but with a decent chance of having some small correlations between some features and the price.

##### The Models

Using the *diamonds* dataset, I planned to construct four different models that would help me solidify the knowledge that I had been learning up to this point.
The four models are as follows:
  - First, `diamond_numerical_linreg.py`: A simple *numerical linear regression model* that predicts the *price label* based on the *numerical carat feature*.
  - Second, `diamond_two_by_two_linreg.py`: A 4-model version of the first model consisting of the following:
    - The model `m1` that uses the `adam` optimizer and only the feature_layer and dense output layer.
    - The model `m2` that uses the `adam` optimizer and the feature_layer, a new densely connected *'relu'* layer with 128 nodes, and the dense output layer.
    - The model `m3` that uses the `RMSprop` optimizer and only the feature_layer and dense output layer.
    - The model `m4` that uses the `RMSprop` optimizer and the feature_layer, a new densely connected *'relu'* layer with 128 nodes, and the dense output layer.
    - This script also saves each of the models and history to a directory in Saved_Models/two_by_two_linear_regression
    - as well as `diamond_two_by_two_plot.py`, a script that loads each model and history and plots them all on a 2x2 subplot figure.
  - Third, `diamond_categorical_linreg.py`: A *categorical linear regression model* that predicts the *price label* based on the following features:
    - The *numerical carat feature* from the first model
    - The *categorical color feature* as a *categorical_column_with_vocabulary_list* using integers from [0, 10] as the supplied vocab
      - This feature column was then converted to an *embedding_column*
    - The *categorical clarity feature* as a *categorical_column_with_vocabulary_list* using integers from [0, 10] as the supplied vocab
      - This feature column was also converted to an *embedding_column*
  - Fourth, `diamond_categorical_binary_classification.py`: A *categorical binary classification model* that predicts a binary classification representing whether the *price label* is over a given *price_threshold* based on the following features:
    - The *numerical carat feature* from the first model
    - The *categorical color feature* as a *categorical_column_with_vocabulary_list* using integers from [0, 10] as the supplied vocab
      - This feature column was then converted to an *embedding_column*
    - The *categorical clarity feature* as a *categorical_column_with_vocabulary_list* using integers from [0, 10] as the supplied vocab
      - This feature column was also converted to an *embedding_column*



