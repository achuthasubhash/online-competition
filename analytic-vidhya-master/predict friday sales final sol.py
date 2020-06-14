#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt 


# In[2]:


data=pd.read_csv("E:\\assignment\\ass 3 analytic vidhya\\train.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().values.any()


# In[5]:


features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(data[feature].isnull().mean(), 4),  ' % missing values')


# In[6]:


data.info()


# In[7]:


mean = data['Product_Category_2'].median()
data['Product_Category_2'].fillna(mean, inplace=True)


# In[8]:


data.head()


# In[9]:


mean = data['Product_Category_3'].mean()
data['Product_Category_3'].fillna(mean, inplace=True)


# In[10]:


data.info()


# In[11]:


data.drop(['User_ID','Product_ID'],axis=1,inplace=True)


# In[12]:


data.info()


# In[13]:


data['Stay_In_Current_City_Years'].unique()


# In[14]:


gen=pd.get_dummies(data['Gender'],drop_first=True)


# In[15]:


city=pd.get_dummies(data['City_Category'],drop_first=True)


# In[16]:


int1={'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}


# In[17]:


data['age'] = data.Age.map(int1)


# In[18]:


data.head()


# In[19]:


int2={'0':0,'1':1,'2':2,'3':3,'4+':4}
data['Stay_In_Current_City_Years'] = data.Stay_In_Current_City_Years.map(int2)


# In[20]:


data = pd.concat([data,gen,city],axis=1)
data.drop(['Age','Gender','City_Category'],axis=1,inplace=True)


# In[21]:


data.head()


# In[22]:



fea=['Occupation','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
x=data[fea]


# In[23]:


x.head()


# In[24]:


x.info()


# In[25]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_sc = sc_X.fit_transform(x)


# In[26]:


df = pd.DataFrame(data=X_sc)


# In[27]:


f=['age','M','B','C','Stay_In_Current_City_Years']
xfea=data[f]
datax = pd.concat([df,xfea],axis=1)


# In[28]:


datax.head()


# In[29]:


y=data['Purchase']


# In[35]:


from sklearn.linear_model import Ridge
ridge_regressor=Ridge(alpha=1e-10)
ridge_regressor.fit(datax,y)

X_grid = np.arange(min(datax), max(datax), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(datax, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# In[38]:


tdata=pd.read_csv("E:\\assignment\\ass 3 analytic vidhya\\test.csv")


# In[39]:


tdata.isnull().values.any()


# In[40]:


features_with_na=[features for features in tdata.columns if tdata[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(tdata[feature].isnull().mean(), 4),  ' % missing values')


# In[41]:


features_with_na=[features for features in tdata.columns if tdata[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(tdata[feature].isnull().mean(), 4),  ' % missing values')

mean1 = tdata['Product_Category_2'].median()
tdata['Product_Category_2'].fillna(mean1, inplace=True)


mean = tdata['Product_Category_3'].mean()
tdata['Product_Category_3'].fillna(mean, inplace=True)

tdata.drop(['User_ID','Product_ID'],axis=1,inplace=True)

tdata['Stay_In_Current_City_Years'].unique()


gen=pd.get_dummies(tdata['Gender'],drop_first=True)


city=pd.get_dummies(tdata['City_Category'],drop_first=True)


int1={'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}


tdata['age'] = tdata.Age.map(int1)


tdata.head()


int2={'0':0,'1':1,'2':2,'3':3,'4+':4}
tdata['Stay_In_Current_City_Years'] = tdata.Stay_In_Current_City_Years.map(int2)


tdata = pd.concat([tdata,gen,city],axis=1)
tdata.drop(['Age','Gender','City_Category'],axis=1,inplace=True)

tdata.head()


fea=['Occupation','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
x=tdata[fea]

X_sc = sc_X.fit_transform(x)


df = pd.DataFrame(data=X_sc)

f=['age','M','B','C','Stay_In_Current_City_Years']
xfea=tdata[f]
tdatax = pd.concat([df,xfea],axis=1)


# In[42]:


y_pred = ridge_regressor.predict(tdatax)


# In[43]:


y_pred


# In[46]:


results = np.array(y_pred)

results = pd.Series(results,name="Infect_Prob")

submission = pd.concat([pd.Series(range(1,233600),name = "ImageId"),results],axis = 1)

submission.to_csv("lasso.csv",index=False)


# In[47]:


results.describe()


# In[ ]:




