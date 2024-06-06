#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd 


# In[15]:


df = pd.read_csv('500hits.csv', encoding='latin-1')


# In[16]:


df.head()


# In[17]:


df.describe()


# In[18]:


df.info()


# In[19]:


from sklearn.preprocessing import MinMaxScaler


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


from sklearn.metrics import confusion_matrix


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


df = df.drop(columns=['PLAYER','CS'],axis =1)


# In[30]:


df.head()


# In[31]:


X = df.drop('HOF',axis=1)


# In[32]:


y = df['HOF']


# In[33]:


X.shape,y.shape


# In[34]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=43)


# In[36]:


scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[38]:


knn = KNeighborsClassifier(n_neighbors=8)


# In[39]:


knn.fit(X_train,y_train)


# In[40]:


y_pred=knn.predict(X_test)


# In[43]:


accuracy_score(y_test,y_pred)*100


# In[46]:


cm=confusion_matrix(y_test,y_pred)


# In[45]:


from sklearn.metrics import classification_report


# In[48]:


classification_report(y_test,y_pred)


# In[ ]:




