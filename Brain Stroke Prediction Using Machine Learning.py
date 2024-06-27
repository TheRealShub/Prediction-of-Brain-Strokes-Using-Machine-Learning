#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv("Desktop/prō/healthcare-dataset-stroke-data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


#we have 201 NULL values in bmi column , since we have 5110 instance and bmi is numerical column we will fill NULLs with mean 
df['bmi'].fillna(df['bmi'].mean(), inplace=True)


# In[6]:


#we will drop id column since it is not importnat

df = df.drop("id",axis=1)
df['ever_married'] = df['ever_married'].map( 
                   {'Yes':1 ,'No':0})
df['ever_married'].value_counts()


# In[7]:


df.isna().sum()                                              #  3.9% of rows are having NULLs so we can drop them


# In[8]:


df['gender'].value_counts()


# In[9]:


df = df[df['gender'] != 'Other']

# Now df will only contain rows where the gender is either 'Male' or 'Female'
df['gender'].value_counts()


# In[10]:


df['stroke'].value_counts()               #95,7% of target data is 0 "not stroke"!!! , that is a huge difference we need to balance it


# In[11]:


df.info()


# In[12]:


numerical_df = df[['age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','stroke']]


# In[13]:


numerical_df.shape


# In[14]:


ColumnsForCountPlot = ['gender','hypertension','heart_disease','ever_married','stroke']
for column in ColumnsForCountPlot:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[column])
    plt.title(f'Count Plot of {column}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[15]:


ColumnsForCountPlot = ['gender','hypertension','heart_disease','ever_married']
for column in ColumnsForCountPlot:
    # Calculate the percentage of stroke occurrences for each category in the column
    stroke_percentages = df.groupby(column)['stroke'].mean() * 100

    plt.figure(figsize=(8, 8))
    plt.pie(stroke_percentages, labels=stroke_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Stroke Percentage by {column}')
    plt.show()


# In[16]:


corr = numerical_df.corr()


# In[17]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,mask = np.triu(np.ones_like(corr, dtype=bool)))


# In[18]:


df = pd.get_dummies(df).astype(int)
df.head()


# In[19]:


df.shape


# In[20]:


corr=df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,mask = np.triu(np.ones_like(corr, dtype=bool)))


# In[21]:


df.stroke.value_counts()


# In[22]:


# Calculate the skewness for each column
skewness = df.skew()

# Plot the skewness for each column
plt.figure(figsize=(10, 6))
skewness.plot(kind='bar')
plt.title('Skewness of Columns')
plt.xlabel('Columns')
plt.ylabel('Skewness')
plt.show()


# In[23]:


x = df.drop("stroke",axis=1)
y = df.stroke


# In[24]:


x.shape , y.shape , y.value_counts()


# In[25]:


from imblearn.over_sampling import SMOTE


# In[26]:


smote = SMOTE(sampling_strategy="minority")


# In[27]:


x_smote , y_smote = smote.fit_resample(x,y)                #it was fit_sample  but now it is fit_resample


# In[28]:


y_smote.value_counts()


# In[29]:


x_train , x_test , y_train , y_test = train_test_split(x_smote,y_smote,test_size=0.2,random_state=42,stratify=y_smote)


# In[30]:


scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[31]:


y_train.value_counts() ,y_test.value_counts()


# In[32]:


pip install lazypredict


# In[33]:


import lazypredict
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)

print(models)


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# In[35]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# In[36]:


# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

