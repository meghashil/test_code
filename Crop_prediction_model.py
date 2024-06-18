#!/usr/bin/env python
# coding: utf-8

# In[ ]:


<b><h2> CROP YIELD PREDICTION IN INDIA </h2></b>

Predicting yield helps the state to get an estimate of the crop in a
certain year to control the price rates.This model focuses on predicting the crop yield in advance by analyzing
factors like location, season, and crop type  through machine learning techniques on
previously collected datasets.


# # **PRE-PROCESSING**
# 

# In[ ]:


# importing necessary libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# loading the dataset

url = 'https://raw.githubusercontent.com/jdapoorv/Crop-yield-prediction/main/crop_csv_file_1_.csv'
crop_data = pd.read_csv(url)
crop_data


# In[ ]:


crop_data.shape

#rows X columns


# In[ ]:


# dataset columns
crop_data.columns


# In[ ]:


# Statistical summary of data frame.

crop_data.describe()


# In[ ]:


# Checking missing values of the dataset in each column
crop_data.isnull().sum()


# In[ ]:


# Replacing missing values with mean of the production coloumn
crop_data['Production'] = crop_data['Production'].fillna(crop_data['Production'].mean())
crop_data


# In[ ]:


#checking
crop_data.isnull().values.any()


# In[ ]:


# Displaying State Names present in the dataset
print(crop_data.State_Name.unique())
print('Total count of states and Union Territories:', len(crop_data.State_Name.unique()))


# In[ ]:


# Adding a new column Yield which indicates Production per unit Area. 

crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])
crop_data.head(10) 


# In[ ]:


# Dropping unnecessary columns

data = crop_data.drop(['State_Name'], axis = 1)


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data.corr(), annot =True, fmt='.4f')
plt.title('Correlation Matrix')


# In[ ]:


dummy = pd.get_dummies(data)
dummy


# <b><i> Splitting dataset into train and test dataset </i></b>

# In[ ]:


from sklearn.model_selection import train_test_split

x = dummy.drop(["Production","Yield"], axis=1)
y = dummy["Production"]

# Splitting data set - 25% test dataset and 75% 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=42)

print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# In[ ]:


print(x_train)
print(y_train)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# creating the dataset
year = [2012,2013,2014]
values = []
data1 = pd.DataFrame(crop_data)
for x in range(3):
  curyearsum=0
  count=0
  for y in range(1000):
    if(data1.iloc[y]["Crop_Year"] == year[x]):
      count+=1;
      curyearsum += data1.iloc[x]["Humidity"]
  values.append(curyearsum/count)
fig = plt.figure(figsize = (7, 5))
 
# creating the bar plot
x = np.array(["2012","2013","2014"])
y = np.array(values)
plt.xlabel("year")
plt.ylabel("mean value of humidity")
plt.title("mean value of humidity of three years")
plt.bar(x,y,color ='maroon', width = 0.4)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# creating the dataset
year = [2012,2013,2014]
values = []
data1 = pd.DataFrame(crop_data)
for x in range(3):
  curyearsum=0
  count=0
  for y in range(1000):
    if(data1.iloc[y]["Crop_Year"] == year[x]):
      count+=1;
      curyearsum += data1.iloc[x]["Temperature"]
  values.append(curyearsum/count)
fig = plt.figure(figsize = (7, 5))
 
# creating the bar plot
x = np.array(["2012","2013","2014"])
y = np.array(values)
plt.xlabel("year")
plt.ylabel("mean value of Temperature")
plt.title("mean value of Temperature of three years")
plt.bar(x,y,color ='maroon', width = 0.4)
plt.show()


# # **Linear Regression**

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].scatter(crop_data['Temperature'], crop_data['Production'], color='red')
axes[1].scatter(crop_data['Soil_Moisture'], crop_data['Production'], color='blue')
axes[0].set_title("Temperature vs Production")
axes[1].set_title("Soil_Moisture vs Production")
axes[0].set_ylabel("Production")
axes[0].set_xlabel("Temperture")
axes[1].set_ylabel("Production")
axes[1].set_xlabel("Soil_Moisture")
plt.show()


# In[ ]:


# Training the Simple Linear Regression model .

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[ ]:


lr_predict = model.predict(x_test)
lr_predict


# In[ ]:


from sklearn.metrics import mean_squared_error
lr_predict = model.predict(x_test)
scores_regr = mean_squared_error(y_test, lr_predict)
print(scores_regr)
# print(mean_squared_error(y_train, lr_predict))


# In[ ]:


# Predicting the test Results 

lr_predict = model.predict(x_test)
lr_predict


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import r2_score
r = r2_score(y_test,lr_predict)
print("R2 score : ",r)


# In[ ]:


plt.scatter(y_test,lr_predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')


# Clearly, the dataset is not good for linear regression.
# 
# <b> Assumptions of Linear Regression </b>
# <ol>
#     <li> Linearity.</li>
#     <li> Homoscedasticity </li>
#     <li> Multivariate normality </li>
#     <li> Lack of multicollinearity </li>
#     
# 

# R2 score: This is pronounced as R-squared, and this score refers to the coefficient of determination. 
# This tells us how well the unknown samples will be predicted by our model.

# # <b> Decision Tree </b>

# In[ ]:


# Training model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 5)
regressor.fit(x_train,y_train)

# Predicting results
decisiontree_predict = regressor.predict(x_test)
decisiontree_predict


# In[ ]:


regressor.score(x_test,y_test)


# In[ ]:


plt.scatter(y_test,decisiontree_predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Decision Tree ')


# In[ ]:


# Calculating R2 score :

from sklearn.metrics import r2_score
r2 = r2_score(y_test,decisiontree_predict)
print("R2 score : ",r2)


# In[ ]:


# Calculating Adj. R2 score: 

Adjr2_2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_2))


# In[ ]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(decisiontree_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Decision Tree Regression')


# # **Random Forest Algorithm**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 11)
model.fit(x_train,y_train)
rf_predict = model.predict(x_test)
rf_predict


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


# Calculating R2 score

from sklearn.metrics import r2_score
r1 = r2_score(y_test,rf_predict)
print("R2 score : ",r1)


# In[ ]:


# Calculating Adj. R2 score: 

Adjr2_1 = 1 - (1-r1)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_1))


# In[ ]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(rf_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Random Forest Regression')


# In[ ]:


plt.scatter(y_test,rf_predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')


# <b> Comparison between Linear Regression Algorithm and Random Forest Algorithm </b> 

# 
# 
# 1. Linear regression algorithm is not at all accurate for this kind of prediction.
# 2. Random Forest Algorithm has higher accuracy ( between 85 % to 90% ), but it is slow.

# # <b> Support Vector Regression </b> 

# In[ ]:


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# In[ ]:


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
regr = make_pipeline(StandardScaler(), SVR(kernel='rbg'))
regr.fit(x_train,y_train)


# In[ ]:


svr_predict = regr.predict(x_test)
svr_predict


# In[ ]:


print(regr.score(x_test,y_test))


# In[ ]:


from sklearn.metrics import r2_score
r3 = r2_score(y_test,svr_predict)
print(r3)


# In[ ]:


# Calculating Adj. R2 score: 

Adjr2_3 = 1 - (1-r3)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_3))


# In[ ]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(svr_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Support Vector Regression')


# # <b> Cross-validation </b> 

# Random Forest

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = x_train, y=y_train, cv = 10)
accuracies


# In[ ]:


a1 = (accuracies.mean()*100)
b1 = (accuracies.std()*100)


# In[ ]:


# Mean Accuracy and SD of 10 fold results

print("Accuracy : {:.2f}%".format (accuracies.mean()*100))
print("Standard Deviation : {:.2f}%".format(accuracies.std()*100))


# Decision Tree
# 

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y=y_train, cv = 10)


# In[ ]:


a2 = (accuracies.mean()*100)
b2 = (accuracies.std()*100)


# In[ ]:


print("Accuracy : {:.2f}%".format (accuracies.mean()*100))
print("Standard Deviation : {:.2f}%".format(accuracies.std()*100))


# Support Vector Regressor

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y=y_train, cv = 10)


# In[ ]:


a3 = (accuracies.mean()*100)
b3 = (accuracies.std()*100)


# In[ ]:


print("Accuracy : {:.2f}%".format (accuracies.mean()*100))
print("Standard Deviation : {:.2f}%".format(accuracies.std()*100))


# Comparing with Graphs
# 

# In[ ]:


# Mean Accuracy
import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'Support Vector Regression']
Accuracy = [a1, a2, a3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00', '#800000'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Accuracy(in %)')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()


# In[ ]:


# Standard Deviation
import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'Support Vector Regression']
Accuracy = [b1, b2, b3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00', '#800000'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Standard Deviation(in %)')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[ ]:


# Adjusted R2 value
import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'Support Vector Regression']
Accuracy = [Adjr2_1, Adjr2_2, Adjr2_3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00', '#800000'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Adjusted R-Squared Score')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[ ]:


# R2 Score
import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree', 'Support Vector Regression']
Accuracy = [r1, r2, r3]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00', '#800000'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('R-Squared Score')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')

