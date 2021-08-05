# Placements App
- - - -

link to the web-app: [https://pg-placements.herokuapp.com/](https://pg-placements.herokuapp.com/)

A streamlit app to predict the probability of getting placed after post graduation. 

## Content
- - - -

The web-app is divided into two sections, namely _Plots_ which contains the plots for the existing dataset data and _Check your Chances_ which lets the user to input the new data in the required fields and get a prediction of the chances of them getting placed.

The code for the app is divided into four repositories. 
1. The _data_ repository contains the dataset _placementsdata.csv_ for this project. 
2. The _model_ repository has _train.py_ which has the code for training the model and _test.py_ is there to do a test run of the model. The _pkl_ files have the trained model so that we dont have to train the model everytime we open the web-app.
3. The _app_ repository contains the _main.py_ which has the code for the web-app.
4. The _notebook_ repository contains the jupyter notebook for the project, containing the plots of the dataset, along with logistic regression to predict placement status as well as linear regression model for predicting salary of a placed candidate.

## Installation
- - - -
#### Install the packages.
Using Pip


`pip install requirements.txt`


## Meta
- - - -
Kshitiej Verma - @warmice19 .
    

