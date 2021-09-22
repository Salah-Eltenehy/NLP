#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Libraries for cleaning the data
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#nltk.download('stopwords')
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3)

#cleaning the data
ps = PorterStemmer()
corpus = []
for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) or word == 'not']
    review = ' '.join(review)
    corpus.append(review)
   
#Convert Sentences to numbers in sparse array
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

#Splitting the data to Training & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 0, test_size= 0.2)
"""
#More models for classification 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
#Accuracy =  0.775

from sklearn.svm import SVC
classifier = SVC(kernel= 'sigmoid', random_state= 0)
"""
#Best model 
#
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 200, random_state= 0,criterion= 'entropy')
#Training the ML model on training set
classifier.fit(X_train, y_train)

#Prediction
y_pred = classifier.predict(X_test)

#Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0]+cm[1][1]) / (cm.sum())
print('Accuracy = ', accuracy)
