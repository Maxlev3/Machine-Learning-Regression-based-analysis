
# timeit

# Student Name : Maxime Levintoff
# Cohort       : Divisadaro 4

# Final model using KNN with standarised data.

################################################################################
# Import Packages
################################################################################


import numpy as np
np.set_printoptions(precision=4)      # To display values upto Four decimal places. 

import pandas as pd
pd.set_option('mode.chained_assignment', None)      # To suppress pandas warnings.


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')    # To apply seaborn whitegrid style to the plots.
plt.rc('figure', figsize=(10, 8))     # Set the default figure size of plots.
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")     # To suppress all the warnings in the notebook.

import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


# ### Data Loading 


original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')

chef = original_df

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

chef.isnull().sum()

# since FAMILY_NAME is the only variable having missing value and is not going to contribute much so we will drop it.
chef = chef.drop(['FAMILY_NAME'], axis=1)
chef = chef.drop(['NAME'], axis=1)
chef = chef.drop(['EMAIL'], axis=1)
chef = chef.drop(['FIRST_NAME'], axis=1)

chef_dummies = pd.get_dummies(chef['CANCELLATIONS_AFTER_NOON'], prefix='CAN', drop_first=True)

# Concatenating the dummy variables into the dataset.

chef = pd.concat([chef,chef_dummies], 1)

chef.drop(['CANCELLATIONS_AFTER_NOON'], 1, inplace=True)


chef_dummies1 = pd.get_dummies(chef['MASTER_CLASSES_ATTENDED'], prefix='MCA', drop_first=True)


chef = pd.concat([chef,chef_dummies1], 1)

chef.drop(['MASTER_CLASSES_ATTENDED'], 1, inplace=True)

chef_dummies2 = pd.get_dummies(chef['MOBILE_LOGINS'], prefix='ML', drop_first=True)

chef = pd.concat([chef,chef_dummies2], 1)

chef.drop(['MOBILE_LOGINS'], 1, inplace=True)

chef_dummies3 = pd.get_dummies(chef['PC_LOGINS'], prefix='PL', drop_first=True)

chef = pd.concat([chef,chef_dummies3], 1)

chef.drop(['PC_LOGINS'], 1, inplace=True)



# Dropping values having REVENUE more than 6500.

chef.drop(chef[chef['REVENUE'] > 6500].index, inplace=True)

# - From boxplot it is obvious that the data do have outliers and we will cap them with threshold for each columns
#     - `TOTAL_PHOTOS_VIEWED`: The values above 1150 will be dropped.
#     - `PREP_VID_TIME`: The values above 400 will be dropped.
#     - `AVG_TIME_PER_SITE_VISIT`: The values above 500 will be dropped.
#     - `TOTAL_MEALS_ORDERED`: The value above 400 will be dropped.


chef.drop(chef[chef['TOTAL_PHOTOS_VIEWED'] > 1150].index, inplace=True)
chef.drop(chef[chef['AVG_PREP_VID_TIME'] > 400].index, inplace=True)
chef.drop(chef[chef['AVG_TIME_PER_SITE_VISIT'] > 500].index, inplace=True)
chef.drop(chef[chef['TOTAL_MEALS_ORDERED'] > 400].index, inplace=True)

# ### Data train/test split

# Splitting the dataset into training and test sets.

chef_train, chef_test = train_test_split(chef, test_size = 0.2, random_state = 2)

# Dropping Interest.Rate from x_train and x_test matrices, and creating y_train and y_test vectors for Interest.Rate values.

x_train = chef_train.drop(['REVENUE'], 1)
y_train = chef_train['REVENUE']
x_test = chef_test.drop(['REVENUE'], 1)
y_test = chef_test['REVENUE']

# Creating a list of the names of x_train columns for future use. 

features = x_train.columns
features

import sklearn.linear_model

# new libraries
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler # standard scaler

# declaring set of x-variables
x_variables = ['REVENUE', 'CROSS_SELL_SUCCESS', 'TOTAL_MEALS_ORDERED',
       'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
       'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER',
       'CANCELLATIONS_BEFORE_NOON', 'TASTES_AND_PREFERENCES', 'WEEKLY_PLAN',
       'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER',
       'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT',
       'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MEDIAN_MEAL_RATING',
       'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED', 'CAN_1', 'CAN_2',
       'CAN_3', 'MCA_1', 'MCA_2', 'MCA_3', 'ML_5', 'ML_6', 'ML_7', 'PL_1',
       'PL_2', 'PL_3']


# applying model in scikit-learn

# Preparing a DataFrame based the the analysis above
chef_data   = chef.loc[ : ,x_variables]


# Preparing the target variable
chef_target = chef.loc[ : , 'REVENUE']

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()

# FITTING the scaler with housing_data
scaler.fit(chef_data)


# TRANSFORMING our data after fit
X_scaled = scaler.transform(chef_data)

# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# Adding the columns back
X_scaled_df.columns = chef_data.columns

# checking the results
X_scaled_df.describe().round(2)

X_scaled_df

# adding labels to the scaled DataFrame
X_scaled_df.columns = chef_data.columns

################################################################################
# Train/Test Split
################################################################################

# ### K-nearest neighbor with standardized data

# this is the exact code we were using before
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            chef_target,
            test_size = 0.25,
            random_state = 222)

# creating lists for training set accuracy and test set accuracy
training_accuracy = []
test_accuracy = []


# building a visualization of 1 to 50 neighbors
neighbors_settings = range(1, 50)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

# finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print(f"""The optimal number of neighbors is {opt_neighbors}""")

# INSTANTIATING a model with the optimal number of neighbors
knn_stand = KNeighborsRegressor(algorithm = 'auto',
                   n_neighbors = opt_neighbors)

# FITTING the model based on the training data
knn_stand.fit(X_train, y_train)

# PREDITCING on new data
knn_stand_pred = knn_stand.predict(X_test)

################################################################################
# Final Model Score (score)
################################################################################


# SCORING the results
print('Training Score:', knn_stand.score(X_train, y_train).round(5))
print('Testing Score:',  knn_stand.score(X_test, y_test).round(5))


# saving scoring data for future use
train_score = knn_stand.score(X_train, y_train).round(5)
test_score  = knn_stand.score(X_test, y_test).round(5)

