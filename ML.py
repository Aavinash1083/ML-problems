import numpy as np 
import pandas as pd 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Placement_Data_Full_Class.csv')

data = data.fillna(0)
print(data)

data = data.drop(columns=['sl_no','salary'])
print(data)

x = data.drop(columns=['status'])
print(x)
y = data['status']
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x = x.apply(le.fit_transform)
print(x)
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier as DTC
model = DTC(criterion='entropy')
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,model.predict(X_test)))
a = model.predict(X_test)
print(a)
z= le.inverse_transform(a)
print(z)

from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test, a))
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(y_pred)

z= le.inverse_transform(y_pred)
print(z)
