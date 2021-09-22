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
data = pd.read_csv('dataset.csv', encoding='utf-8')
data.rename(columns= {'الله يشافيك و يعافيك من كل داء وهامة أخي سيد الوكيل':'Sentence'}, inplace= True)
#cleaning the data

data['Sentence'][1151] = 'مفقود'
data['Sentence'][1210] = 'مفقود'
data['Sentence'][1268] = 'مفقود'
data['Sentence'][1389] = 'مفقود'


ps = PorterStemmer()
corpus = []
for i in range(0, data.shape[0]):
    review = re.sub('[^ء-ي]', ' ', data['Sentence'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('arabic')) or word == 'ولا']
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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
#More models for classification 
#Accuracy =  0.775

#from sklearn.svm import SVC
#classifier = SVC(kernel= 'rbf', random_state= 0)

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators= 200, random_state= 0,criterion= 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0]+cm[1][1]) / (cm.sum())
print('Accuracy = ', accuracy)





