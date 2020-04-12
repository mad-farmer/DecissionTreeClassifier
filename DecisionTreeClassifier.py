from sklearn import metrics
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#importing libraries


#%% DATA PREPROCESSING

df=pd.read_csv("iris.csv")
#importing csv file
df=df.iloc[0:100,:]
#first 100 rows(Setosa and Versicolor)
df.species=[1 if each=="Iris-setosa" else 0 
            for each in df.species]
#setosa=1 versicolor=0
y=df.species.values.reshape(-1,1)
#1 and 0 labels
x_=df.drop(["species"],axis=1)
#values
x = (x_-np.min(x_))/(np.max(x_)-np.min(x_))
#normalizing


#%% PREDICTION

x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.20,random_state=42)
#creating train and test values

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
#Fitting DecisionTreeClassifier to train values

y_pred = classifier.predict(x_test)
y_pred=y_pred.reshape(-1,1)
#Predicting y_test values


#%% ACCURACY
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))





