#!/usr/bin/env python
# coding: utf-8

# ### The solution is divided into the following sections:
# 
# * Data understanding and exploration
# * Data cleaning
# * Data preparation
# * Model building and evaluation

# ## 1. Data understanding and exploration

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

#reading the dataset
data = pd.read_csv(r'C:\Users\ahuja\Desktop\Interview prep\Project ML\Car Price Prediction\CarPrice_Assignment.csv')


# In[2]:


#data summary
print(data.info())


# ### 26 features, 205 entries, no-null values

# In[3]:


data.head()


# ### Understanding the data
# 
# The data dictionary contains the meaning of various attributes; some non-obvious ones are:

# In[4]:


#symboling: -2 (least risky) to +3 most risky
#Most cars are 0,1,2
data['symboling'].astype('category').value_counts()


# In[5]:


#aspiration: An (internal combustion) engine property showing 
#whether the oxygen intake is through standard (atmospheric pressure)
#or through turbocharging (pressurised oxygen intake)

data['aspiration'].astype('category').value_counts()


# In[6]:


#wheelbase: distance between centre of front and rarewheels
sns.distplot(data['wheelbase'],kde = False, color ='red', bins = 30)
plt.show()


# In[7]:


data['wheelbase'].mean()


# In[8]:


#curbweight: weight of car without occupants or baggage
sns.distplot(data['curbweight'],kde = False, color ='blue', bins = 30)
plt.show()


# In[9]:


#compression ration: ration of volume of compression chamber at largest capacity to least capacity
sns.distplot(data['compressionratio'])
plt.show()


# ### Data Exploration
# 
# To perform linear regression, the (numeric) target variable should be linearly related to at least one another numeric variable. Let's see whether that's true in this case.
# 
# We'll first subset the list of all (independent) numeric variables, and then make a correlation heatmap.

# In[10]:


#all numeric (float and int) variables in the dataset
data_numeric = data.select_dtypes(include=['float64','int64'])
data_numeric.head()


# Here, although the variable symboling is numeric (int), we'd rather treat it as categorical since it has only 6 discrete values. Also, we do not want 'car_ID'.

# In[11]:


data_numeric = data_numeric.drop(['symboling','car_ID'], axis=1)
data_numeric.head()


# In[12]:


#correlation matrix
cor = data_numeric.corr()
cor


# In[13]:


#figure size
plt.figure(figsize=(16,8))

#heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# 
# The heatmap shows some useful insights:
# 
# Correlation of price with independent variables:
# 
# Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower (notice how all of these variables represent the size/weight/engine power of the car).
# 
# Price is negatively correlated to citympg and highwaympg (-0.70 approximately). This suggest that cars having high mileage may fall in the 'economy' cars category, and are priced lower (think Maruti Alto/Swift type of cars, which are designed to be affordable by the middle class, who value mileage more than horsepower/size of car etc.)
# 
# Correlation among independent variables:
# 
# Many independent variables are highly correlated (look at the top-left part of matrix): wheelbase, carlength, curbweight, enginesize etc. are all measures of 'size/weight', and are positively correlated.
# Thus, while building the model, we'll have to pay attention to multicollinearity (especially linear models, such as linear and logistic regression, suffer more from multicollinearity).

# ## 2. Data Cleaning
# 
# Let's now conduct some data cleaning steps.
# 
# We've seen that there are no missing values in the dataset. We've also seen that variables are in the correct format, except symboling, which should rather be a categorical variable so that dummy variable are created for the categories.
# 
# Note that it can be used in the model as a numeric variable also.

# In[14]:


#datatype of variables
data.info()


# In[15]:


#converting symboling to categorical
data['symboling'] = data['symboling'].astype('object')
data.info()


# In[16]:


print(data['CarName'][:30]) 
#fetching 30 entries only


# Notice that the carname is what occurs before a space, e.g. alfa-romero, audi, chevrolet, dodge, bmx etc.
# 
# Thus, we need to simply extract the string before a space. There are multiple ways to do that.

# In[17]:


#Extracting carname

#str.split() by space
carnames = data['CarName'].apply(lambda x: x.split(" ")[0])
carnames[:30]


# Let's create a new column to store the compnay name and check whether it looks okay.

# In[18]:


#New column car_company
data['car_company'] = data['CarName'].apply(lambda x: x.split(" ")[0])


# In[19]:


#look at all values 
data['car_company'].astype('category').value_counts()


# Notice that some car-company names are misspelled - vw and vokswagen should be volkswagen, porcshce should be porsche, 
# toyouta should be toyota, Nissan should be nissan, maxda should be mazda etc.
# 
# This is a data quality issue, let's solve it.

# In[20]:


#replacing misspelled car_company names

#volkswagen
data.loc[(data['car_company'] == "vw") | 
         (data['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'

#porsche
data.loc[data['car_company'] == "porcshce", 'car_company'] = 'porsche'

#toyota
data.loc[data['car_company'] == "toyouta", 'car_company'] = 'toyota'

#nissan
data.loc[data['car_company'] == "Nissan", 'car_company'] = 'nissan'

#mazda
data.loc[data['car_company'] == "maxda", 'car_company'] = 'mazda'


# In[21]:


data['car_company'].astype('category').value_counts()


# The car_company variable looks okay now. Let's now drop the car name variable.

# In[22]:


#drop carname variable
data = data.drop('CarName', axis=1)


# In[23]:


data.info()


# ### Data Preparation
# 
# Let's now prepare the data and build the model.

# In[24]:


#split into X and y
X = data.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y = data['price']


# In[25]:


#creating dummy variables for categorical variables

#subset all categorical variables
data_categorical = X.select_dtypes(include=['object'])
data_categorical.head()


# In[26]:


#convert to dummy variables
data_dummies = pd.get_dummies(data_categorical, drop_first=True)
data_dummies.head()


# In[27]:


# drop categorical variables 
X = X.drop(list(data_categorical.columns), axis=1)

# concat dummy variables with X
X = pd.concat([X, data_dummies], axis=1)


# In[28]:


from sklearn.preprocessing import scale

#storing column names in cols, since column names are (annoyingly) lost after 
#scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[30]:


#split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ## 3. Model Building and Evaluation¶

# In[31]:


#building model

#instantiate
lm = LinearRegression()

#fit
lm.fit(X_train, y_train)


# In[32]:


#predict 
y_pred = lm.predict(X_test)

#metrics
from sklearn.metrics import r2_score

print(r2_score(y_true=y_test, y_pred=y_pred))


# Not bad, we are getting approx. 83% r-squared with all the variables. 

# In[ ]:




