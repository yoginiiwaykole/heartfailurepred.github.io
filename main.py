import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle

df = pd.read_csv(r"C:\Users\yogin\Downloads\HEARTFAILUREDATASET_project.csv")
print(df)

print(df.info())

print(df.describe())

imputer = KNNImputer(n_neighbors=3)  #Here we have set neighbors = 3
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[["age","cholesterol","thalach"]]),   #fit_transform - jis data ko transform karna h, usko imputer me fit karte hai
    index=range(df.shape[0]),  #considering columns from 0 to 326(whole data)
    columns = ["age","cholesterol","thalach"]
)

df[["age","cholesterol","thalach"]] = df_imputed[["age","cholesterol","thalach"]]
print(df)

print(df.isna().sum())

label_encoder = LabelEncoder()
df['Smoking'] = label_encoder.fit_transform(df['Smoking'])
df['Depression'] = label_encoder.fit_transform(df['Depression'])
print(df)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Random Forest Classifier
#Setting our x variable to the entire dataset except target
x = df.drop(columns = 'target')
# Making our y variable equal to Taregt
y = df['target']

# The test_size=0.3 inside the function indicates the percentage of the data that should be held over for testing. (70/30)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

# Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems.
model = RandomForestClassifier()

model.fit(X_test,y_test)

model.fit(X_train,y_train)

#training accuracy
y_pred1 = model.predict(X_train)
accuracy_ = accuracy_score(y_train,y_pred1)
print(accuracy_)

#testing accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
print(confusion_matrix(y_test,y_pred))

#Logistic Regression
# instantiate the model (using the default parameters)
#logreg = LogisticRegression(random_state=18)

# fit the model with data
#logreg.fit(X_train, y_train)

#y_pred2 = logreg.predict(X_test)

#from sklearn import metrics

#cnf_matrix = metrics.confusion_matrix(y_test, y_pred2)
#print(cnf_matrix)

#score = logreg.score(X_test, y_test)
#print(score)

#score1 = logreg.score(X_train,y_train)
#print(score1)

#from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred2))

#print(logreg.predict([[40.0, 1 ,1, 0,140, 289.0 ,0,172.0,1.0]]))

pickle.dump(model, open("imaginecup.pkl","wb"))

