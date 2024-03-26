#!/usr/bin/env python
# coding: utf-8

# # PR Assignment -3 Baye's classification model

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from math import pi
from math import e
from sklearn.metrics import accuracy_score


# In[2]:


# importing data from txt file into dataframe
df =pd.read_csv("D:/ISIBangalore/MS-QMScourse/2nd sem/Pattern Recognition/Assignment-3/Vowel Data.txt", sep='\s+', header=None, names=['y','x1','x2','x3'])


# In[3]:


# grouping data classwise
dflist = list(df.groupby('y'))


# In[4]:


# segregating dataframe class wise
df1 = dflist[0][1]
df2 = dflist[1][1]
df3 = dflist[2][1]
df4 = dflist[3][1]
df5 = dflist[4][1]
df6 = dflist[5][1]
# Probabilities of each class
P1 = df1['y'].count()/df['y'].count()
P2 = df2['y'].count()/df['y'].count()
P3 = df3['y'].count()/df['y'].count()
P4 = df4['y'].count()/df['y'].count()
P5 = df5['y'].count()/df['y'].count()
P6 = df6['y'].count()/df['y'].count()


# In[5]:


# segregating input features and output feature for each class
df1_x=df1.drop(columns=['y'])
df1_y=df1.iloc[:,0:1]
df2_x=df2.drop(columns=['y'])
df2_y=df2.iloc[:,0:1]
df3_x=df3.drop(columns=['y'])
df3_y=df3.iloc[:,0:1]
df4_x=df4.drop(columns=['y'])
df4_y=df4.iloc[:,0:1]
df5_x=df5.drop(columns=['y'])
df5_y=df5.iloc[:,0:1]
df6_x=df6.drop(columns=['y'])
df6_y=df6.iloc[:,0:1]


# In[6]:


A1 = [0.1, 0.2, 0.3, 0.4]
acc_table_train = np.empty((4, 10))
acc_table_test  = np.empty((4, 10))
for i in range(4):
    for j in range(10):
        # splitting each data frame into training and test data
        df1_x_train, df1_x_test, df1_y_train, df1_y_test = train_test_split(df1_x, df1_y, test_size=A1[i])
        df2_x_train, df2_x_test, df2_y_train, df2_y_test = train_test_split(df2_x, df2_y, test_size=A1[i])
        df3_x_train, df3_x_test, df3_y_train, df3_y_test = train_test_split(df3_x, df3_y, test_size=A1[i])
        df4_x_train, df4_x_test, df4_y_train, df4_y_test = train_test_split(df4_x, df4_y, test_size=A1[i])
        df5_x_train, df5_x_test, df5_y_train, df5_y_test = train_test_split(df5_x, df5_y, test_size=A1[i])
        df6_x_train, df6_x_test, df6_y_train, df6_y_test = train_test_split(df6_x, df6_y, test_size=A1[i])
        # combining training and test data from each class
        train_x = pd.concat([df1_x_train, df2_x_train, df3_x_train, df4_x_train, df5_x_train, df6_x_train], axis = 0)
        train_y = pd.concat([df1_y_train, df2_y_train, df3_y_train, df4_y_train, df5_y_train, df6_y_train], axis = 0)
        test_x = pd.concat([df1_x_test, df2_x_test, df3_x_test, df4_x_test, df5_x_test, df6_x_test], axis = 0)
        test_y = pd.concat([df1_y_test, df2_y_test, df3_y_test, df4_y_test, df5_y_test, df6_y_test], axis = 0)
        # mean vector from dataframe of each class
        mu1 = np.array(df1_x_train.mean())
        mu2 = np.array(df2_x_train.mean())
        mu3 = np.array(df3_x_train.mean())
        mu4 = np.array(df4_x_train.mean())
        mu5 = np.array(df5_x_train.mean())
        mu6 = np.array(df6_x_train.mean())
        # covariance matrix from dataframe of each class
        cov1= df1_x_train.cov()
        cov2= df2_x_train.cov()
        cov3= df3_x_train.cov()
        cov4= df4_x_train.cov()
        cov5= df5_x_train.cov()
        cov6= df6_x_train.cov()
        # converting train & test input features dataframe to array form
        train_x_arr = train_x.to_numpy()
        test_x_arr = test_x.to_numpy()
        # predicting output class for training data
        y_pred_train = []
        for k in range(train_x['x1'].count()):
            A = train_x_arr[k] 
            p1 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov1), 0.5)))*pow(e,-0.5*(A - mu1)@np.linalg.inv(cov1)@(np.transpose(A - mu1)))
            p2 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov2), 0.5)))*pow(e,-0.5*(A - mu2)@np.linalg.inv(cov2)@(np.transpose(A - mu2)))
            p3 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov3), 0.5)))*pow(e,-0.5*(A - mu3)@np.linalg.inv(cov3)@(np.transpose(A - mu3)))
            p4 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov4), 0.5)))*pow(e,-0.5*(A - mu4)@np.linalg.inv(cov4)@(np.transpose(A - mu4)))
            p5 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov5), 0.5)))*pow(e,-0.5*(A - mu5)@np.linalg.inv(cov5)@(np.transpose(A - mu5)))
            p6 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov6), 0.5)))*pow(e,-0.5*(A - mu6)@np.linalg.inv(cov6)@(np.transpose(A - mu6)))
            Pp_list = [P1*p1, P2*p2, P3*p3, P4*p4, P5*p5, P6*p6]
            y_pred_train.append(Pp_list.index(max(Pp_list)) + 1)
        # converting output into array from list form
        ypred_train = np.array(y_pred_train)
        # checking accuracy score
        acc_table_train[i][j] = accuracy_score(train_y, ypred_train)
        
        # predicting output class for test data
        y_pred_test = []
        for k in range(test_x['x1'].count()):
            A = test_x_arr[k] 
            p1 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov1), 0.5)))*pow(e,-0.5*(A - mu1)@np.linalg.inv(cov1)@(np.transpose(A - mu1)))
            p2 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov2), 0.5)))*pow(e,-0.5*(A - mu2)@np.linalg.inv(cov2)@(np.transpose(A - mu2)))
            p3 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov3), 0.5)))*pow(e,-0.5*(A - mu3)@np.linalg.inv(cov3)@(np.transpose(A - mu3)))
            p4 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov4), 0.5)))*pow(e,-0.5*(A - mu4)@np.linalg.inv(cov4)@(np.transpose(A - mu4)))
            p5 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov5), 0.5)))*pow(e,-0.5*(A - mu5)@np.linalg.inv(cov5)@(np.transpose(A - mu5)))
            p6 = (1/(pow(2*pi, 1.5)*pow(np.linalg.det(cov6), 0.5)))*pow(e,-0.5*(A - mu6)@np.linalg.inv(cov6)@(np.transpose(A - mu6)))
            Pp_list = [P1*p1, P2*p2, P3*p3, P4*p4, P5*p5, P6*p6]
            maxm = max(Pp_list)
            index = Pp_list.index(maxm)
            y_pred_test.append(index + 1)
        # converting output into array from list form
        ypred_test = np.array(y_pred_test)
        # checking accuracy score
        acc_table_test[i][j] = accuracy_score(test_y, ypred_test)

acc_table_train_df = pd.DataFrame(np.transpose(np.round(acc_table_train*100,2)), columns=['90-10', '80-20', '70-30', '60-40'], index=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'])
acc_table_test_df = pd.DataFrame(np.transpose(np.round(acc_table_test*100,2)), columns=['90-10', '80-20', '70-30', '60-40'], index=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'])


# In[7]:


acc_table_train_df.loc['Mean acc train'] = acc_table_train_df.mean()
acc_table_train_df


# In[8]:


acc_table_test_df.loc['Mean acc test'] = acc_table_test_df.mean()
acc_table_test_df


# In[9]:


pd.concat([acc_table_train_df.iloc[[10]], acc_table_test_df.iloc[[10]]], axis = 0).to_csv('D:/ISIBangalore/MS-QMScourse/2nd sem/Pattern Recognition/Assignment-3/accuracy table.csv', sep=',', header=True, index=True)


# In[ ]:




