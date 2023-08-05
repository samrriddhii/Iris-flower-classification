#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("Iris.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df=df.drop(columns=["Id"])


# In[6]:


df.head()


# In[7]:


df["Species"].replace({"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3},inplace = True)


# In[8]:


df


# In[9]:


x=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values


# In[10]:


x


# In[11]:


y = df.Species.values.reshape(-1,1)


# In[12]:


y


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42) 


# In[15]:


x_train.shape


# In[16]:


y_train.shape


# In[17]:


k=6
knclr=KNeighborsClassifier(k)


# In[18]:


knclr.fit(x_train,y_train)


# In[19]:


y_pred=knclr.predict(x_test)


# In[21]:


metrics.accuracy_score(y_test,y_pred)*100


# In[ ]:





# In[ ]:




