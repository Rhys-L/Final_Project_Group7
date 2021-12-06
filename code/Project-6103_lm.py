

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import hvplot.pandas
import matplotlib.mlab as mlab

#read data
url = 'https://storage.googleapis.com/great_lakes/LakeIce_PhysicalProps.csv'
lake_df = pd.read_csv(url)
lake_df = lake_df.iloc[: , 1:]
#check the data
lake_df.head()
lake_df.columns


#mission values
lake_df.isnull().sum()
count=0
for i in lake_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print(round((count/len(lake_df.index))*100), 'percent of the entire dataset the rows with missing values.')
lake_df.dropna(axis=0,inplace=True)

#EDA
sns.lmplot(x ='Surface_Temp_C', y ='Ice_pct', data = lake_df)
plt.show()
min(lake_df.Ice_pct)

#Categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lake_df['Lake'] = le.fit_transform(lake_df['Lake'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

#Split
from sklearn.model_selection import train_test_split

X = lake_df.drop(['Year','Day','id','Ice_pct'], axis=1)
Y = lake_df['Ice_pct']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Linear regression
from sklearn import metrics
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('#',50*'-')

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

#Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,Y_train)
print(lin_reg.intercept_) #15.433939243587993
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(Y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(Y_train, train_pred)
