#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


housing=pd.read_csv("housing.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


# %matplotlib inline


# In[8]:


# import matplotlib.pyplot as plt


# In[9]:


# housing.hist(bins=50, figsize=(20,15))


# In[10]:


# import numpy as np
# def split(data, splitratio):
#     np.random.seed(42)
#     shuffle=np.random.permutation(len(data))
#     test_size=int(len(data)*splitratio)
#     test=shuffle[:test_size]
#     train=shuffle[test_size:]
#     print(train)
#     return data.iloc[test], data.iloc[train]


# In[11]:


# test,train=split(housing, 0.2)


# In[12]:


# print(len(test), len(train))


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


test,train=train_test_split(housing, test_size=0.2, random_state=42)


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in split.split(housing, housing['CHAS']):
    test_data=housing.loc[test]
    train_data=housing.loc[train]


# In[16]:


test_data['CHAS'].value_counts()


# In[17]:


train_data['CHAS'].value_counts()


# In[18]:


housing=train_data.copy()


# In[19]:


corr=housing.corr()


# In[20]:


corr['MEDV'].sort_values(ascending=False)


# In[21]:


from pandas.plotting import scatter_matrix
# at=['MEDV', 'PTRATIO', 'ZN', 'LSTAT']
# scatter_matrix(housing[at], figsize=(12,8))


# In[22]:


housing=train_data.drop('MEDV', axis=1)
housingla=train_data['MEDV'].copy()


# In[23]:


housing.shape


# In[24]:


housing.describe()


# There are mainly 3 types of objects in skikit learn
# 1. Estimators:- It estimates a parameter based on the data e.g. Imputer. It has fit method and transform method . Fit: fits in dataset and calculates internal parameters
# 2. Transformers:- transfomer takes input and returns output on the base of fit() it also has fit_transform() which fits and then transforms
# 3. Predictors:- LinerRegression is an example of predictors it has fit() and predict() functions in it. It also takes score() function which helps to evaluavate the score and accuracy of the output given by the predictors.

# In[25]:


from sklearn.impute import SimpleImputer
im=SimpleImputer(strategy="median")
im.fit(housing)


# In[26]:


im.statistics_


# In[27]:


x=im.transform(housing)


# In[28]:


housing_tr=pd.DataFrame(x, columns=housing.columns)


# In[29]:


housing_tr.describe()


# for bringing the data at same scale we use scalers and helps us to analsys of the data easier for the ml model they are of two types:
# standardscaler: it is (mean-value)/std this leads to reduce all values into a range from 0 to 1
# Min max scaler: it (value-min)/(max-min) sklearn provides a class called MinMaxScaler this process is called normalization

# In[30]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pip=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])


# In[31]:


housing_tr=pip.fit_transform(housing)


# In[32]:


housing_tr.shape


# In[33]:


housingla.shape


# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_tr, housingla)


# In[35]:


housing.describe()


# In[36]:


housing_tr.shape


# In[37]:


housing.shape


# In[38]:


somedata=housing.iloc[:5]


# In[39]:


samela=housingla.iloc[:5]
samela


# In[40]:


procesda=pip.fit_transform(somedata)


# In[41]:


predict=model.predict(housing_tr)


# In[42]:


from sklearn.metrics import mean_squared_error
mne=mean_squared_error(housingla, predict)
mne


# In[43]:


from sklearn.model_selection import cross_val_score
scoring=cross_val_score(model, housing_tr, housingla, scoring="neg_mean_squared_error", cv=10)
scoring=-scoring
scoring


# In[44]:


from joblib import dump, load
dump(model, 'dragon.joblib')


# In[45]:


test=test_data.drop('MEDV', axis=1)
testla=test_data['MEDV'].copy()
processed_test=pip.transform(test)


# In[46]:


finalprediction=model.predict(processed_test)
finalmse=mean_squared_error(finalprediction, testla)
finalmse


# In[49]:


np.sqrt(finalmse)


# In[89]:


processed_train=np.array(processed_test)
processed_test[3]


# In[48]:


from joblib import dump, load
model=load('dragon.joblib')


# In[90]:


model.predict([[-0.51130008, -0.81649658,  1.10350733,  1.        ,  0.86453303,
       -0.05690275,  1.80268316, -1.02878854, -0.34015067, -0.60673678,
       -1.39707095, -0.42429609,  0.33470326]])


# In[85]:


predict=np.array(predict)


# In[91]:


predict[3]


# In[ ]:




