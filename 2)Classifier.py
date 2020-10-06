#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required libraries
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('dataset/hist.csv')
df.head()


# In[3]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[4]:


# Random forest
rfc = RandomForestClassifier() 
rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)
print('Random Forest accuracy :- ', accuracy_score(preds, y_test) * 100)


# In[5]:


# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)
print('XGBoost accuracy :- ', accuracy_score(preds, y_test) * 100)


# In[6]:


# Decision tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)
print('Decision Tree accuracy :- ', accuracy_score(preds, y_test) * 100)


# In[7]:


# Tuning Random Forest with GridSearchCV
clc = RandomForestClassifier()
parameters = [{ 'max_depth': [20, 30, 40], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 
               'n_estimators': [100, 200, 300] }]
grid_search = GridSearchCV(estimator = clc, param_grid = parameters, 
                scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 2)

grid_search = grid_search.fit(X, y)
best_acc = grid_search.best_score_
best_param = grid_search.best_params_

# Printing best parameters and accuracy
print(best_param)
print(best_acc)


# In[10]:


# Tuning Random Forest with GridSearchCV
clc = RandomForestClassifier()
parameters = [{ 'max_depth': [40, 60, 80], 'min_samples_leaf': [1], 'min_samples_split': [2], 
               'n_estimators': [200, 250] }]
grid_search = GridSearchCV(estimator = clc, param_grid = parameters, 
                scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 2)

grid_search = grid_search.fit(X, y)
best_acc = grid_search.best_score_
best_param = grid_search.best_params_

# Printing best parameters and accuracy
print(best_param)
print(best_acc)


# In[11]:


# Tuning Random Forest with GridSearchCV
clc = RandomForestClassifier()
parameters = [{ 'max_depth': [80, 100, 120], 'min_samples_leaf': [1], 'min_samples_split': [2], 
               'n_estimators': [250] }]
grid_search = GridSearchCV(estimator = clc, param_grid = parameters, 
                scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 2)

grid_search = grid_search.fit(X, y)
best_acc = grid_search.best_score_
best_param = grid_search.best_params_

# Printing best parameters and accuracy
print(best_param)
print(best_acc)


# In[12]:


# Random Forest training on complete data
final_clc = RandomForestClassifier(max_depth=40, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
final_clc.fit(X, y)


# In[13]:


# Save classifier
filename = 'model/RFCmodel.sav'
pickle.dump(final_clc, open(filename, 'wb'))


# In[4]:


# Random Forest training on complete data
final_clc = XGBClassifier()
final_clc.fit(X, y)


# In[5]:


# Save classifier
filename = 'model/XGBmodel.sav'
pickle.dump(final_clc, open(filename, 'wb'))


# In[ ]:




