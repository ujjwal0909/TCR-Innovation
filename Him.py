#!/usr/bin/env python
# coding: utf-8

# # HR_ATTRITION_DATA
# In this project , we need to predict whether a given employee will leave the organization or not. Our target column is Attrition. We will create a model, perform EDA and predict the target column using ML concepts
# 
# Made by: Om Vispute, Ujjwal Patel

# ### Importing Packages

# In[1]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn import metrics 


# ### Reading CSV File

# In[2]:


data = pd.read_csv(r"/Users/omvispute/Downloads/HR_Employee_Attrition-1.csv")
data.head()


# ### Explainatory Data Analysis (EDA)

# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


numeric_cols = data.select_dtypes(include=['int64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()


# In[6]:


data[numeric_cols].head()


# In[7]:


data.isnull().sum()


# In[8]:


data[numeric_cols].describe()


# In[9]:


data.nunique()


# In[10]:


data.dtypes.sort_index()


# ### Looking for Duplicate Data

# In[11]:


data.duplicated().sum()


# In[12]:


data.describe()


# ### Data Analysis

# In[13]:


attrition_count = pd.DataFrame(data['Attrition'].value_counts())
attrition_count


# In[14]:


f, ax = plt.subplots(figsize=(10,10))
ax = data['Attrition'].value_counts(). plot.pie(explode=[0,0], autopct = '%1.1f%%', shadow=True)
ax.set_title('Attrition Probability')


# In[15]:


fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(
    x='Age', 
    hue='Attrition', 
    data = data, 
    palette="colorblind", 
    ax = ax,  
    edgecolor=sns.color_palette("dark", n_colors = 1),
    )


# In[16]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'EducationField', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Education Field")

ax[1,0] = sns.countplot(x='Attrition', hue= 'Department', data=data,  ax = ax[1,0], palette='Set1' )
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Department")

ax[0,1] = sns.countplot(x='Attrition', hue= 'Education', data=data,  ax = ax[0,1], palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Education")

ax[1,1] = sns.countplot(x='Attrition', hue= 'BusinessTravel', data=data,  ax = ax[1,1], palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Bussiness Travel")


f.tight_layout()


# In[17]:


f, ax = plt.subplots(2,2, figsize=(20,15))

ax[0,0] = sns.countplot(x='Attrition', hue= 'JobRole', data=data, ax = ax[0,0], palette='Set1' )
ax[0,0].set_title("Frequency Distribution of Attrition w.r.t. Job Role")

ax[0,1] = sns.countplot(x='Attrition', hue= 'OverTime', data=data,  ax = ax[0,1],palette='Set1' )
ax[0,1].set_title("Frequency Distribution of Attrition w.r.t. Over Time")

ax[1,1] = sns.countplot(x='Attrition', hue= 'EnvironmentSatisfaction', data=data,  ax = ax[1,1],palette='Set1' )
ax[1,1].set_title("Frequency Distribution of Attrition w.r.t. Environment Satisfaction")

ax[1,0] = sns.countplot(x='Attrition', hue='WorkLifeBalance', data=data, ax = ax[1,0], palette='Set1')
ax[1,0].set_title("Frequency Distribution of Attrition w.r.t. Work Life Balance")

f.tight_layout()


# ### Object Data Types and Their Unique Values

# #### Correlation of the columns

# In[18]:


for column in data.columns:
    if data[column].dtype == object:
        print(str(column) + ' : ' + str(data[column].unique()))
        print(data[column].value_counts())
        print("-"*90)


# In[19]:


#Remove unneeded columns

#Remove the column EmployeeNumber
data = data.drop('EmployeeNumber', axis = 1) # A number assignment 
#Remove the column StandardHours
data = data.drop('StandardHours', axis = 1) #Contains only value 80 
#Remove the column EmployeeCount
data = data.drop('EmployeeCount', axis = 1) #Contains only the value 1 
#Remove the column EmployeeCount
data = data.drop('Over18', axis = 1) #Contains only the value 'Yes'


# In[20]:


data.corr()


# In[21]:


plt.figure(figsize=(15,15))
sns.heatmap(
    data.corr(), 
    annot=True, 
    fmt='.0%',
    linewidths=1,
    cmap='inferno'
)


# In[26]:


data_train, data_test, data_train, data_test = train_test_split(data, data, test_size=0.2,random_state=42 )


# In[27]:


print(data_train.shape, data_test.shape)


# ### converting categorical values to numerical values

# In[28]:


encoder = ce.OrdinalEncoder(cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'])
data_train = encoder.fit_transform(data_train)
data_test = encoder.fit_transform(data_test)


# ### Data Preprocessing

# In[31]:


from sklearn.preprocessing import LabelEncoder
x = data.drop('Attrition', axis=1)
y = data['Attrition']
for column in x.columns:
    if x[column].dtype == np.number:
        continue
    x[column] = LabelEncoder().fit_transform(x[column])


# In[32]:


x['Age_Years'] = x['Age']
x = x.drop('Age', axis=1)
x


# In[33]:


print(x.columns)


# In[34]:


print(y)


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state = 0)
print(X_train.shape, X_test.shape)


# In[37]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
forest.fit(X_train,Y_train)


# In[38]:


score = forest.score(X_train, Y_train)
print('Accuracy Score:', np.abs(score)*100)


# ### Accuracy Score: 99.5%

# In[39]:


predict = forest.predict(X_test)


# In[40]:


predict


# In[41]:


from sklearn.metrics import confusion_matrix, classification_report
M = confusion_matrix(Y_test, predict)
TN = M[0][0]
TP = M[1][1]
FN = M[1][0]
FP = M[0][1]

print(M)
print('Model testing accuracy="{}!"'.format((TP+TN)/(TP+TN+FN+FP)))
print(classification_report(Y_test, predict))

