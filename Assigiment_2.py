#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
univariateData=pd.read_csv('univariateData.csv')


# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
X = univariateData.iloc[:, 0]
Y = univariateData.iloc[:, 1]

plt.scatter(X, Y)
plt.show()


# # Model implementation of Functions

# In[117]:


class GradientDescentLinearRegression:
    
    def fit(self, x1, x2, y, iterations, learning_rate):
        m = 0
        b = 0
        c = 0

        L = learning_rate  # The learning Rate
        epochs = iterations  # The number of iterations to perform gradient descent

        self.n = float(len(x1)) # Number of elements in X

        # Performing Gradient Descent 
        for i in range(epochs): 
            self.y_pred = m*x1 +b*x2 +c  # The current predicted value of Y
            D_m = (-2/self.n) * sum(x1 * (y - self.y_pred))  # Derivative wrt m
            D_b = (-2/self.n) * sum(x2 * (y - self.y_pred))  # Derivative wrt b
            D_c = (-2/self.n) * sum(y - self.y_pred)  # Derivative wrt c
            m = m - L * D_m  # Update m
            b = b- L* D_b    # Update b
            c = c - L * D_c  # Update c

        self.m, self.b, self.c = m, b, c
        print (self.m, self.b, self.c)
    def compute_cost(self,y):
        cost = (1/self.n)*sum([value**2 for value in (y-self.y_pred)])
        print(cost)
    def predict(self, x1, x2):
        self.y_predicted = self.m*x1 + self.b*x2 + self.c
        return self.y_predicted
    def accuarcy(self,y):    
        # sum of square of residuals
        ssr = np.sum((self.y_predicted - y)**2)

        #  total sum of squares
        sst = np.sum((y - np.mean(y))**2)

        # R2 score
        r2_score = 1 - (ssr/sst)
        print(r2_score)


# In[118]:


model = GradientDescentLinearRegression()


# # Univariate Results

# In[119]:


model.fit(X,0,Y,10000,0.01)


# In[120]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.scatter(X, Y, color='black')
plt.plot(X, model.predict(X,0))


# In[121]:


model.accuarcy(Y)


# In[122]:


model.compute_cost(Y)


# # Multivariate Results

# In[123]:


plt.rcParams['figure.figsize'] = (12.0, 9.0)
multivariateData=pd.read_csv('multivariateData.csv')
# Preprocessing Input data
X1 = multivariateData.iloc[:, 0]
X2 = multivariateData.iloc[:,1]
Y1 = multivariateData.iloc[:, 2]

plt.scatter(X1, Y1)
plt.scatter(X2, Y1)
plt.show()


# In[124]:


from sklearn import preprocessing
X1=preprocessing.scale(X1)
X2=preprocessing.scale(X2)
Y1=preprocessing.scale(Y1)
X1,X2,Y1


# In[125]:


model.fit(X1,X2,Y1,1000,0.005)


# In[126]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.scatter(X1, Y1, color='black')
plt.scatter(X2, Y1)
plt.plot([min(X1), max(X1)], [min(model.predict(X1,X2)), max(model.predict(X1,X2))], color='red') # predicted


# In[127]:


model.accuarcy(Y1)


# In[128]:


model.compute_cost(Y1)


# In[ ]:





# In[ ]:




