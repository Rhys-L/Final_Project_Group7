

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

#read data
url = 'https://storage.googleapis.com/great_lakes/LakeIce_PhysicalProps.csv'
lake_df = pd.read_csv(url)

#read data
lake_df.head()

#mission values
lake_df.isnull().sum()
count=0
for i in lake_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print(round((count/len(lake_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')
lake_df.dropna(axis=0,inplace=True)

lake_df.describe()
lake_df['iceflag'] = np.where(lake_df['Ice_pct'] > 50 , 1, 0) #53.3% ice percentage https://research.noaa.gov/article/ArtMID/587/ArticleID/2706/NOAA-projects-30-percent-average-Great-Lakes-ice-cover-for-2021-winter
lake_df.iceflag.value_counts()

#Lakes
lake_df.Lake.unique()
lake_df.Lake.value_counts()

#Split data
from sklearn.model_selection import train_test_split
X = lake_df.drop(['id','Year','Day','id','Ice_pct'], axis=1)
Y = lake_df['iceflag']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape # check the shape of X_train and X_test

#Categorical data
X_train = pd.concat([X_train.Elevation_meters,X_train.Length_km,X_train.Breadth_km,X_train.Avg_Depth_meters,
                     X_train.Max_Depth_meters,X_train.Volume_km3,X_train.Water_Area_km2,X_train.Land_Drain_Area_km2,
                     X_train.Total_Area_km2,X_train.Shore_Length_km,X_train.Retention_Time_years,X_train.Surface_Temp_C,
                     pd.get_dummies(X_train.Lake),], axis=1)
X_test = pd.concat([X_test.Elevation_meters,X_test.Length_km,X_test.Breadth_km,X_test.Avg_Depth_meters,
                     X_test.Max_Depth_meters,X_test.Volume_km3,X_test.Water_Area_km2,X_test.Land_Drain_Area_km2,
                     X_test.Total_Area_km2,X_test.Shore_Length_km,X_test.Retention_Time_years,X_test.Surface_Temp_C,
                     pd.get_dummies(X_test.Lake),], axis=1)

#Scaling
X_train.describe()
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()

# train logit model training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
# fit the model
logreg.fit(X_train, Y_train)
THRESHOLD = 0.5
#Predict
Y_pred_test = logreg.predict(X_test)
Y_pred_test
# probability of getting output as 0 - no rain
logreg.predict_proba(X_test)[:,0]
# probability of getting output as 1 - rain
logreg.predict_proba(X_test)[:,1]

#accuracy
from sklearn.metrics import accuracy_score
#	for floating point numbers
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(Y_test, Y_pred_test))) #y_test are the true class labels and y_pred_test are the predicted class labels in the test-set
#Compare the train-set and test-set accuracy
Y_pred_train = logreg.predict(X_train)
Y_pred_train
print('Training set score: {:.4f}'.format(logreg.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(logreg.score(X_test, Y_test))) #no overfitting

# fit the Logsitic Regression model with C=100
# instantiate the model
logreg100 = LogisticRegression(C=100, random_state=0)
# fit the model
logreg100.fit(X_train, Y_train)
print('Training set score: {:.4f}'.format(logreg100.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(logreg100.score(X_test, Y_test)))

# fit the Logsitic Regression model with C=001
# instantiate the model
logreg001 = LogisticRegression(C=0.01, random_state=0)
# fit the model
logreg001.fit(X_train, Y_train)
# print the scores on training and test set
print('Training set score: {:.4f}'.format(logreg001.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(logreg001.score(X_test, Y_test)))

#Compare model accuracy with null accuracy
# check class distribution in test set
Y_test.value_counts()
null_accuracy = (1575/(1575+196))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# Classification
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
Sensitivity = TP / float(TP + FN)
print('Sensitivity : {0:0.4f}'.format(Sensitivity))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

# plot ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_test)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC
from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(Y_test, Y_pred_test)
print('ROC AUC : {:.4f}'.format(ROC_AUC))


# GUI IMPLEMENTATION
from tkinter import *
root = Tk()
root.title('logit Error Calculator')

e = Entry(root, width=35)
e.pack()

def myClick():
    user_input = eval(e.get())

    GUIclassifier = LogisticRegression(C=user_input)
    GUIclassifier.fit(X_train, Y_train)
    GUI_preds = GUIclassifier.predict(X_test)
    accuracy = accuracy_score(Y_test, GUI_preds) * 100
    GUIstring = "The accuracy percentage for your chosen k-value is " + str(round(accuracy, 2)) + "%."
    myLabel = Label(root, text=GUIstring)
    myLabel.pack()

myButton = Button(root, text= "Enter c value", command=myClick)
myButton.pack()

root.mainloop()






