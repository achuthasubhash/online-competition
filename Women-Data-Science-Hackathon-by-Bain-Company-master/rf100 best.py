# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:50:46 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
train_df=pd.read_csv("E:\\assignment\\ASS 11 women in data science\\train.csv")
train_df.head()
test_df=pd.read_csv("E:\\assignment\\ASS 11 women in data science\\test_QkPvNLx.csv")
big_df=train_df.append(test_df,sort=False)
big_df.head()
big_df['Long_Promotion'].unique()
big_df['Sales'].fillna((big_df['Sales'].mean()),inplace=True)
big_df['User_Traffic'].fillna((big_df['User_Traffic'].median()),inplace=True)
big_df['Competition_Metric'].fillna((big_df['Competition_Metric'].median()),inplace=True)
sxfe=['User_Traffic','Competition_Metric']
data=big_df[sxfe]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc1= StandardScaler()
X_sc = sc_X.fit_transform(data)
cd=pd.get_dummies(big_df['Course_Domain'],drop_first=True)
ct=pd.get_dummies(big_df['Course_Type'],drop_first=True)
df = pd.DataFrame(data=X_sc)
big_df = pd.concat([big_df,cd,ct],axis=1)
big_df.drop(['ID','Day_No','Course_ID','User_Traffic','Competition_Metric','Course_Domain','Course_Type'],axis=1,inplace=True)
f=['Short_Promotion','Public_Holiday','Long_Promotion','Sales','Development','Finance & Accounting','Software Marketing','Degree','Program']
xfea=big_df[f]
df.reset_index(drop=True, inplace=True)
xfea.reset_index(drop=True, inplace=True)

big_dfx = pd.concat([df,xfea],axis=1)
df_train=big_dfx[0:512087]
df_test=big_dfx[512087:]
X=df_train.drop(['Sales'],axis=1)
y=df_train.Sales
y=sc1.fit_transform(pd.DataFrame(y))
Xtest=df_test.drop(['Sales'],axis=1)
# In[24]:


from sklearn.ensemble import RandomForestRegressor
regressor =RandomForestRegressor()
regressor.fit(X, y)

df_test.head()
testX=df_test.drop(['Sales'],axis=1)
y_pred= regressor.predict(Xtest)
# In[ ]:y_pred
y_pred=sc1.inverse_transform(y_pred)

results = np.array(y_pred)

results = pd.Series(results,name="pred")

submission = pd.concat([pd.Series(range(1,36001),name = "ImageId"),results],axis = 1)

submission.to_csv("rf  best.csv",index=False)