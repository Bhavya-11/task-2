#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# In[2]:


iris=datasets.load_iris()


# In[3]:


x=iris.data[:,:2]
y=iris.target


# In[4]:


plt.scatter(x[:,0],x[:,1],c=y,cmap='gist_rainbow')
plt.xlabel('Speal Length',fontsize=20)
plt.ylabel('Speal Width',fontsize=20)


# In[6]:


km=KMeans(n_clusters=3,n_jobs=4,random_state=20)
km.fit(x)


# In[7]:


centers=km.cluster_centers_
print(centers)


# In[9]:


#this will tell us to which cluster does the data obsevations belongs.
new_labels=km.labels_
#Plot the identifies clusters and compare with answers
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(x[:,0],x[:,1],c=y,cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(x[:, 0], x[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)


# In[ ]:




