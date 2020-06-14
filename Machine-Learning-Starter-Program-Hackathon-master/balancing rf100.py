# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:23:56 2020

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt 

data=pd.read_csv("E:\\assignment\\ass 13 ml starter\\train_HK6lq50.csv")
data.head()
count_classes = pd.value_counts(data['is_pass'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
from sklearn.utils import resample
data_majority = data[data.is_pass==1]
data_minority = data[data.is_pass==0]
data_majority .count()
data_minority_upsampled = resample(data_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=50867,    # to match majority class
                                    random_state=123)
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
data_upsampled.is_pass.value_counts()
data_upsampled.head()
data_upsampled.info()
data_upsampled.drop(['id','program_id','trainee_id'],axis=1,inplace=True)
data_upsampled['is_handicapped'].unique()
study_data = pd.DataFrame({'columns':data_upsampled.columns})
study_data['datatypes'] = data_upsampled.dtypes.values
study_data['missing'] =data_upsampled.isnull().sum().values
study_data['unique'] =data_upsampled.nunique().values
print(study_data)
data_upsampled.describe()
data_upsampled['age'].fillna((data_upsampled['age'].mean()),inplace=True)
data_upsampled['trainee_engagement_rating'].fillna((data_upsampled['trainee_engagement_rating'].mean()),inplace=True)
data_upsampled['is_pass'].fillna((data_upsampled['is_pass'].mean()),inplace=True)
study_data = pd.DataFrame({'columns':data_upsampled.columns})
study_data['datatypes'] = data_upsampled.dtypes.values
study_data['missing'] =data_upsampled.isnull().sum().values
study_data['unique'] = data_upsampled.nunique().values
print(study_data)
df_frequency_map = data_upsampled.program_type.value_counts().to_dict()
data_upsampled.program_type =data_upsampled.program_type.map(df_frequency_map)
data_upsampled.head()
tt=pd.get_dummies(data_upsampled['test_type'],drop_first=True)
int2={'intermediate':3, 'easy':4, 'hard':2, 'vary hard':1}
data_upsampled['difficulty_level'] =data_upsampled.difficulty_level.map(int2)
gen=pd.get_dummies(data_upsampled['gender'],drop_first=True)
int3={'Matriculation':3, 'High School Diploma':2, 'Bachelors':4, 'Masters':5,'No Qualification':1}
data_upsampled['education'] = data_upsampled.education.map(int3)
ih=pd.get_dummies(data_upsampled['is_handicapped'],drop_first=True)
data_upsampled.head()
data_upsampled = pd.concat([data_upsampled,tt,gen,ih],axis=1)
data_upsampled.drop(['test_type','gender','is_handicapped'],axis=1,inplace=True)
data_upsampled.head()
fea=['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating']
data=data_upsampled[fea]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(data)
df = pd.DataFrame(data=X_sc)
data_upsampled.drop(['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating'],axis=1,inplace=True)
df = df.loc[df.index.drop_duplicates()]
df.reset_index(drop=True, inplace=True)
data_upsampled.reset_index(drop=True, inplace=True)
datax = pd.concat([df,data_upsampled],axis=1)
datax.tail()
y=datax['is_pass']
y=(pd.DataFrame(y))
datax.drop(['is_pass'],axis=1,inplace=True)

from catboost import CatBoostRegressor
classifier = CatBoostRegressor(iterations=100,learning_rate=0.1)
classifier.fit(datax, y)


tdata=pd.read_csv("E:\\assignment\\ass 13 ml starter\\test_wF0Ps6O.csv")
tdata.drop(['id','program_id','trainee_id'],axis=1,inplace=True)
study_data = pd.DataFrame({'columns':tdata.columns})
study_data['datatypes'] = tdata.dtypes.values
study_data['missing'] =tdata.isnull().sum().values
study_data['unique'] =tdata.nunique().values
print(study_data)
tdata.describe()
tdata['age'].fillna((tdata['age'].mean()),inplace=True)
tdata['trainee_engagement_rating'].fillna((tdata['trainee_engagement_rating'].mean()),inplace=True)
data_upsampled['is_pass'].fillna((data_upsampled['is_pass'].mean()),inplace=True)
study_data = pd.DataFrame({'columns':tdata.columns})
study_data['datatypes'] = tdata.dtypes.values
study_data['missing'] =tdata.isnull().sum().values
study_data['unique'] = tdata.nunique().values
print(study_data)
df_frequency_map = tdata.program_type.value_counts().to_dict()
tdata.program_type =tdata.program_type.map(df_frequency_map)
tdata.head()
tt=pd.get_dummies(tdata['test_type'],drop_first=True)
int2={'intermediate':3, 'easy':4, 'hard':2, 'vary hard':1}
tdata['difficulty_level'] =tdata.difficulty_level.map(int2)
gen=pd.get_dummies(tdata['gender'],drop_first=True)
int3={'Matriculation':3, 'High School Diploma':2, 'Bachelors':4, 'Masters':5,'No Qualification':1}
tdata['education'] = tdata.education.map(int3)
ih=pd.get_dummies(tdata['is_handicapped'],drop_first=True)
tdata.head()
tdata = pd.concat([tdata,tt,gen,ih],axis=1)
tdata.drop(['test_type','gender','is_handicapped'],axis=1,inplace=True)
tdata.head()
fea=['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating']
data=tdata[fea]

X_sc = scaler.fit_transform(data)
df = pd.DataFrame(data=X_sc)
tdata.drop(['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating'],axis=1,inplace=True)
df = df.loc[df.index.drop_duplicates()]
df.reset_index(drop=True, inplace=True)
tdata.reset_index(drop=True, inplace=True)
testx = pd.concat([df,tdata],axis=1)
testx.drop(['is_pass'],axis=1,inplace=True)
y_pred = classifier.predict(testx)

res=[]
y=0.5
for i in y_pred :
     if  i>0.5:
         res.append(1)
     else:
         res.append(0)
results = np.array(res)

results = pd.Series(results,name="pred")

submission = pd.concat([pd.Series(range(1,36001),name = "ImageId"),results],axis = 1)

submission.to_csv("catboost 100 0.1 balance.csv",index=False)