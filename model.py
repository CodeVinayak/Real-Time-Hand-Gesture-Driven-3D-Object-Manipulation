import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import Normalizer

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from joblib import dump, load

## Define datset path
datasetPath = "Dataset/hand_dataset_1000_24.csv"

## Read dataset using pandas
dataset = pd.read_csv(datasetPath)

## Split dataset into X and y
## @param X: the landmark coodinates data
## @param y: the corresponding letter
X = dataset.drop('class', axis = 1)
y = dataset['class']

## Create normalizer and normalize coordinate data
normalizer = Normalizer().fit(X)

X = normalizer.transform(X)

''' 
## Code used to test various models accuracy score on data
cross_Validate = KFold(n_splits = 10, random_state=7, shuffle = True)

def test_model(modelName, model):

  print(f'Testing Model {modelName} ...')
  scoring = ['accuracy']
  scores = cross_validate(model, X, y, scoring = scoring, cv=cross_Validate, n_jobs = -1)

  accuracy = np.mean(scores['test_accuracy'])
  print('Mean Accuracy: %.3f\n' % (accuracy))

test_model("GNB", GaussianNB())

test_model("KNN", KNeighborsClassifier())

test_model("LR", LogisticRegression(max_iter = 1000))

test_model("DT", DecisionTreeClassifier())

test_model("SVC", SVC())
'''

## Classifier chosen in KNeighbors
classifier = KNeighborsClassifier()

## Train our classifier using dataset
classifier.fit(X, y)

## Dump classifier to be used in handDetection.py
dump(classifier, "model.joblib")