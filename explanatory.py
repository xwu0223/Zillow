import numpy as np
import pandas as pd
import sys
import os
import gc
import random
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import src.data_proc as data_proc

pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.options.display.float_format

mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Load in 2016 properties data
prop = data_proc.load_properties_data("data/properties_2016.csv")
print("Number of properties: {}".format(len(prop)))
print("Number of property features: {}".format(len(prop.columns)-1))

# Load in 2016 training data (with logerror labels)
train_2016 = data_proc.load_training_data("data/train_2016_v2.csv")
print("Number of 2016 transaction records: {}".format(len(train_2016)))
print("\n", train_2016.head())

# Rename & retype the feature columns; also unify representations of missing values
data_proc.rename_columns(prop)
data_proc.retype_columns(prop)

# Join the training data with the property table
train_2016 = train_2016.merge(how='left', right=prop, on='parcelid')
train_2016.head(10)

# Look at how complete (i.e. no missing value) each training set feature is
data_proc.print_complete_percentage(train_2016)

# Look at the distribution of the target variable (log-error)
print(train_2016['logerror'].describe())
plt.hist(train_2016.loc[abs(train_2016['logerror']) < 0.6, 'logerror'],bins=40)
plt.show()

# Looks like there are some outliers in the training data (very large logerror)
# abs(logerror) > 0.6 seems abnormal, should probably remove them from the training set
threshold = 0.6
print("{} training examples in total".format(len(train_2016)))
print("{} with abs(logerror) > {}".format((abs(train_2016['logerror']) > threshold).sum(), threshold))

train_2016 = train_2016[abs(train_2016.logerror) <= threshold]

# Let's see if there are any feature value outliers -> looks like things are mostly fine, no need to process
prop.describe().loc[['min', 'max', 'mean']].T

# Analyze time dependency!
datetime = pd.to_datetime(train_2016.transactiondate).dt
train_2016['month'] = datetime.month
train_2016['quarter'] = datetime.quarter
print(train_2016.groupby('month')['month', 'logerror'].median())
print(train_2016.groupby('quarter')['quarter', 'logerror'].median())

# Let's see how a 'neighborhood' is defined
# Maybe we can extract some neighborhood-based aggregate features
df = prop[prop.neighborhood_id == 27080]
plt.scatter(df.latitude, df.longitude, s=1)
plt.show()