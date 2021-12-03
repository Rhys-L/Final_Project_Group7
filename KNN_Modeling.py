# IMPORT PACKAGES

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

np.random.seed(2021)

# ------------------------------------------

# IMPORT DATA FROM GCP BITBUCKET
url = 'https://storage.googleapis.com/great_lakes/LakeIce_PhysicalProps.csv'

lakedata = pd.read_csv(url)
print(lakedata.head(5))
print(lakedata.columns)
print(lakedata.dtypes)

# For now, filter out NANs

lakedata = lakedata.dropna()

# Rename + reorder columns

lakedata = lakedata.rename(columns={'Unnamed: 0': "index", 'Year': "year", "Day" : 'day', 'Lake': 'lake', 'Surface_Temp_C': 'surface_temp', 'Ice_pct' : "ice_pct"})
print(lakedata.columns)

lakedata = lakedata[['index', 'year', 'day', 'id', 'surface_temp', 'ice_pct', 'lake']]
print(lakedata.head(5))

# ------------------------------------------

# EDA

boxplot1 = lakedata.boxplot(by='lake', column = ['surface_temp'])
plt.show()

boxplot2 = lakedata.boxplot(by='lake', column = ['ice_pct'])
plt.show()

sns.scatterplot(data = lakedata, x = 'surface_temp', y = 'ice_pct', hue = 'lake')
plt.show()

# ------------------------------------------

# CLASSIFICATION ANALYSIS

# Split data into attributes and labels

X = lakedata.values[:, 4:6]
Y = lakedata.values[:, 6]

# Split data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20)

# Normalize variables

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Implement KNN algorithm

classifier = KNeighborsClassifier(n_neighbors = 5) #choose k = 5 to start
classifier.fit(X_train, Y_train)
Y_preds = classifier.predict(X_test)

# Evaluation

print(classification_report(Y_test, Y_preds)) #best precision was St. Claire
print(confusion_matrix(Y_test, Y_preds))

# K value inspection for k = 1 to k = 50

Error = []

for i in range(1, 50):
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model.fit(X_train, Y_train)
    preds = knn_model.predict(X_test)
    Error.append(np.mean(preds != Y_test))

# Visualize results

plt.plot(range(1, 50), Error, color = 'blue', linestyle = 'dotted', marker = 'o',
         markerfacecolor = 'orange', markersize = 5)
plt.title('Error based on K Value Selection')
plt.xlabel('K')
plt.ylabel('Mean Error')
plt.figtext(x = .5, y = .5, s = 'k = 23 is optimal', backgroundcolor = 'blue', color = 'white')
plt.show()

# Build new model with optimal k = 23

classifier = KNeighborsClassifier(n_neighbors = 23)
classifier.fit(X_train, Y_train)
Y_preds12 = classifier.predict(X_test)
print(classification_report(Y_test, Y_preds12)) #best precision was St. Claire again
cf_matrix = confusion_matrix(Y_test, Y_preds12)
print(cf_matrix)
print(accuracy_score(Y_test, Y_preds12) * 100) #39% accuracy

# Heatmap, adapted from Prof. Amir's KNN example

names = lakedata['lake'].unique()
cm_df = pd.DataFrame(cf_matrix, index = names, columns = names)
plt.figure(figsize=(5,5))
hm = sns.heatmap(cm_df, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=cm_df.columns, xticklabels=cm_df.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# ------------------------------------------

# GUI IMPLEMENTATION

from tkinter import *
root = Tk()
root.title('KNN Error Calculator')

e = Entry(root, width=35)
e.pack()

def myClick():
    user_input = eval(e.get())

    GUIclassifier = KNeighborsClassifier(n_neighbors=user_input)
    GUIclassifier.fit(X_train, Y_train)
    GUI_preds = GUIclassifier.predict(X_test)
    accuracy = accuracy_score(Y_test, GUI_preds) * 100
    GUIstring = "The accuracy percentage for your chosen k-value is " + str(round(accuracy, 2)) + "%."
    myLabel = Label(root, text=GUIstring)
    myLabel.pack()

myButton = Button(root, text= "Enter k value", command=myClick)
myButton.pack()

root.mainloop()
