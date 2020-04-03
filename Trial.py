import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')

X = data.drop('Class', axis=1)
Y = data['Class']

x_train,x_test,y_train,y_test = train_test_split(X,Y, train_size=0.4, random_state=42)

#Decision Tree Classifier
Decision = DecisionTreeClassifier()
Decision.fit(x_train, y_train)
Prediction = Decision.predict(x_test)
score = accuracy_score(y_test, Prediction)
print(score)

#PAC
passive = PassiveAggressiveClassifier(max_iter=50)
passive.fit(x_train,y_train)
passivePrediction = passive.predict(x_test)
PACscore = accuracy_score(y_test, passivePrediction)
print(PACscore)

#KNN Classifier
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNNpredict = KNN.predict(x_test)
Kscore = accuracy_score(y_test, KNNpredict)
print(Kscore)

#Random Forest
forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train,y_train)
forestPrediction = forest.predict(x_test)
score = accuracy_score(y_test, forestPrediction)
print(score)

Labels = ['Normal', 'Fraud']
confusion = confusion_matrix(y_test, forestPrediction)
plt.figure(figsize=(13,13))
sns.heatmap(confusion, xticklabels = Labels, yticklabels=Labels,annot=True,fmt="d")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()
