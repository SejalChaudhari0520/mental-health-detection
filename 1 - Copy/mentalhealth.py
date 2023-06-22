import numpy as np
import pandas as pd

dataset = pd.read_csv('dataset[1].csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
m,n=X.shape

print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range (n):
 X[:,i]=le.fit_transform(X[:,i])

print(X)

y=np.array(y)

y = le.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

import joblib
file_name='mentalHealthFinalised'
joblib.dump(classifier,file_name)

import joblib
loaded_model=joblib.load('mentalHealthFinalised.sav')


y_pred = loaded_model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print(cm)
accuracy_score(y_test, y_pred)

print(loaded_model.predict([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]]))
print(le.classes_)