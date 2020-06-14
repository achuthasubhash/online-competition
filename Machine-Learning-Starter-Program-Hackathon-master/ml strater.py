#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt 

traindata=pd.read_csv("E:\\assignment\\ass 13 ml starter\\train_HK6lq50.csv")
testdata=pd.read_csv("E:\\assignment\\ass 13 ml starter\\test_wF0Ps6O.csv")
big_df=traindata.append(testdata,sort=False)
big_df.head()
big_df.drop(['id','program_id','trainee_id'],axis=1,inplace=True)
big_df['is_handicapped'].unique()
study_data = pd.DataFrame({'columns':big_df.columns})
study_data['datatypes'] = big_df.dtypes.values
study_data['missing'] =big_df.isnull().sum().values
study_data['unique'] = big_df.nunique().values
print(study_data)
big_df['age'].fillna((big_df['age'].mean()),inplace=True)
big_df['trainee_engagement_rating'].fillna((big_df['trainee_engagement_rating'].mean()),inplace=True)
big_df['is_pass'].fillna((big_df['is_pass'].mean()),inplace=True)
study_data = pd.DataFrame({'columns':big_df.columns})
study_data['datatypes'] = big_df.dtypes.values
study_data['missing'] =big_df.isnull().sum().values
study_data['unique'] = big_df.nunique().values
print(study_data)
df_frequency_map = big_df.program_type.value_counts().to_dict()
big_df.program_type = big_df.program_type.map(df_frequency_map)
big_df.head()
tt=pd.get_dummies(big_df['test_type'],drop_first=True)
int2={'intermediate':3, 'easy':4, 'hard':2, 'vary hard':1}
big_df['difficulty_level'] = big_df.difficulty_level.map(int2)
gen=pd.get_dummies(big_df['gender'],drop_first=True)
int3={'Matriculation':3, 'High School Diploma':2, 'Bachelors':4, 'Masters':5,'No Qualification':1}
big_df['education'] = big_df.education.map(int3)
ih=pd.get_dummies(big_df['is_handicapped'],drop_first=True)
big_df.head()
big_df = pd.concat([big_df,tt,gen,ih],axis=1)
big_df.drop(['test_type','gender','is_handicapped'],axis=1,inplace=True)
big_df.head()
fea=['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating']
data=big_df[fea]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_sc = sc_X.fit_transform(data)
df = pd.DataFrame(data=X_sc)
big_df.drop(['program_type','program_duration','test_id','difficulty_level','education','city_tier','age','total_programs_enrolled','trainee_engagement_rating'],axis=1,inplace=True)

df = df.loc[df.index.drop_duplicates()]
df.reset_index(drop=True, inplace=True)
big_df.reset_index(drop=True, inplace=True)
datax = pd.concat([df,big_df],axis=1)
datax.head()
train=datax[0:73147]
test=datax[73147:]
y=train['is_pass']
y=(pd.DataFrame(y))
train.drop(['is_pass'],axis=1,inplace=True)
test.drop(['is_pass'],axis=1,inplace=True)


# In[34]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4500, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 12))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# In[35]:

classifier.fit(train, y, batch_size = 64, epochs = 100)
y_pred = classifier.predict(test)


# In[36]:


results = np.concatenate(y_pred)




# In[ ]:
res=[]
y=0.5
for i in results:
     if  i>0.5:
         res.append(1)
     else:
         res.append(0)


results = pd.Series(res,name="pred")

submission = pd.concat([pd.Series(range(1,36001),name = "ImageId"),results],axis = 1)

submission.to_csv("ann 4 100 nn.csv",index=False)         
