import numpy as np 
import pandas as pd 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('Placement_Data_Full_Class.csv')

df.head(15)

df = df.fillna(0)
print(df)

x = df.drop(columns=['salary'])
print(x)

y = df['salary']
print(y)

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()  
x= x.apply(label_encoder.fit_transform)
print(x)

y= label_encoder.fit_transform(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1001)

nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#separating numerical and categorical col
numerical_col = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
categorical_col = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

#Creating Pipeline to Missing Data 

#inpute numerical missing data with median
numerical_transformer = make_pipeline(SimpleImputer(strategy='median'),
                                      StandardScaler())

#inpute categorical data with the most frequent value of the feature and make one hot encoding
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                        OneHotEncoder(handle_unknown='ignore'))

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_col),
                                               ('cat', categorical_transformer, categorical_col)])
    

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier())])

#Using GradientBoostingClassifier with GridSearchCV to get better parameters

param_grid = {'model__learning_rate':[0.001, 0.01, 0.1], 
              'model__n_estimators':[100, 150, 200, 300, 350, 400]}

#param_grid = {'model__learning_rate':[0.1], 
#              'model__n_estimators':[150]}

#use recall score
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

grid.fit(x_train, y_train)

grid.best_params_

#CV F1 Score
scores = cross_val_score(grid, x_test, y_test, cv=5, scoring='f1', n_jobs=-1)

print(scores, '\nAverage F1 Score: ',scores.mean())

from sklearn.metrics import classification_report,confusion_matrix
predictions = grid.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

                                               


