import src.data_proc as data_proc

import numpy as np
import pandas as pd
import sys
import os
import gc
import random
pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.options.display.float_format

from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor, Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Save CatBoost models to files
def save_models(models):
    for i, model in enumerate(models):
        model.save_model('checkpoints/catboost_' + str(i))
    print("Saved {} CatBoost models to files.".format(len(models)))

# Load CatBoost models from files
def load_models(paths):
    models = []
    for path in paths:
        model = CatBoostRegressor()
        model.load_model(path)
        models.append(model)
    return models


"""
    Drop id and label columns + Feature selection for CatBoost
"""
def catboost_drop_features(features):
    # id and label (not features)
    unused_feature_list = ['parcelid', 'logerror']

    # too many missing
    missing_list = ['framing_id', 'architecture_style_id', 'story_id', 'perimeter_area', 'basement_sqft', 'storage_sqft']
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = ['fireplace_flag', 'deck_id', 'pool_unk_1', 'construction_id', 'fips', 'county_id']
    unused_feature_list += bad_feature_list

    # hurts performance
    unused_feature_list += ['county_landuse_code_id', 'zoning_description_id']

    return features.drop(unused_feature_list, axis=1, errors='ignore')

# Read DataFrames from hdf5
features_2016 = pd.read_hdf('hdf5/features.h5', 'features_2016')  # All features except for datetime for 2016
features_2017 = pd.read_hdf('hdf5/features.h5', 'features_2017')  # All features except for datetime for 2017
train = pd.read_hdf('hdf5/train.h5', 'train')  # Concatenated 2016 and 2017 training data with labels

catboost_features = catboost_drop_features(train)
print("Number of features for CatBoost: {}".format(len(catboost_features.columns)))
print(catboost_features.head(5))

catboost_label = train.logerror.astype(np.float32)
print(catboost_label.head())

# Transform to Numpy matrices
catboost_X = catboost_features.values
catboost_y = catboost_label.values

# Perform shuffled train/test split
np.random.seed(42)
random.seed(10)
X_train, X_val, y_train, y_val = train_test_split(catboost_X, catboost_y, test_size=0.2)

# Remove outlier examples from X_train and y_train; Keep them in X_val and y_val for proper cross-validation
outlier_threshold = 0.4
mask = (abs(y_train) <= outlier_threshold)
X_train = X_train[mask, :]
y_train = y_train[mask]

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))

# Specify feature names and categorical features for CatBoost
feature_names = [s for s in catboost_features.columns]
categorical_features = ['cooling_id', 'heating_id', 'landuse_type_id', 'year', 'month', 'quarter']

categorical_indices = []
for i, n in enumerate(catboost_features.columns):
    if n in categorical_features:
        categorical_indices.append(i)
print(categorical_indices)

# CatBoost parameters
params = {}
params['loss_function'] = 'MAE'
params['eval_metric'] = 'MAE'
params['nan_mode'] = 'Min'  # Method to handle NaN (set NaN to either Min or Max)
params['random_seed'] = 0

params['iterations'] = 1000  # default 1000, use early stopping during training
params['learning_rate'] = 0.015  # default 0.03

params['border_count'] = 254  # default 254 (alias max_bin, suggested to keep at default for best quality)

params['max_depth'] = 6  # default 6 (must be <= 16, 6 to 10 is recommended)
params['random_strength'] = 1  # default 1 (used during splitting to deal with overfitting, try different values)
params['l2_leaf_reg'] = 5  # default 3 (used for leaf value calculation, try different values)
params['bagging_temperature'] = 1  # default 1 (higher value -> more aggressive bagging, try different values)

# Train CatBoost Regressor with cross-validated early-stopping
val_pool = Pool(X_val, y_val, cat_features=categorical_indices)

np.random.seed(42)
random.seed(36)
model = CatBoostRegressor(**params)
model.fit(X_train, y_train,
          cat_features=categorical_indices,
          use_best_model=True, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

# Evaluate model performance
print("Train score: {}".format(abs(model.predict(X_train) - y_train).mean() * 100))
print("Val score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))

# CatBoost feature importance
feature_importance = [(feature_names[i], value) for i, value in enumerate(model.get_feature_importance())]
feature_importance.sort(key=lambda x: x[1], reverse=True)
for k, v in feature_importance[:10]:
    print("{}: {}".format(k, v))


# Train CatBoost on all given training data (preparing for submission)
outlier_threshold = 0.4
mask = (abs(catboost_y) <= outlier_threshold)
catboost_X = catboost_X[mask, :]
catboost_y = catboost_y[mask]
print("catboost_X: {}".format(catboost_X.shape))
print("catboost_y: {}".format(catboost_y.shape))

params['random_seed'] = 0
params['iterations'] = 1000  # roughly chosen based on public leaderboard score
print(params)
np.random.seed(42)
random.seed(36)
model = CatBoostRegressor(**params)
model.fit(catboost_X, catboost_y, cat_features=categorical_indices, verbose=False)

# Sanity check: score on a small portion of the dataset
print("sanity check score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))

"""
    Helper method that prepares 2016 and 2017 properties features for CatBoost inference
"""


def transform_test_features(features_2016, features_2017):
    test_features_2016 = catboost_drop_features(features_2016)
    test_features_2017 = catboost_drop_features(features_2017)

    test_features_2016['year'] = 0
    test_features_2017['year'] = 1

    # 11 and 12 lead to bad results, probably due to the fact that there aren't many training examples for those two
    test_features_2016['month'] = 10
    test_features_2017['month'] = 10

    test_features_2016['quarter'] = 4
    test_features_2017['quarter'] = 4

    return test_features_2016, test_features_2017


"""
    Helper method that makes predictions on the test set and exports results to csv file
    'models' is a list of models for ensemble prediction (len=1 means using just a single model)
"""


def predict_and_export(models, features_2016, features_2017, file_name):
    # Construct DataFrame for prediction results
    submission_2016 = pd.DataFrame()
    submission_2017 = pd.DataFrame()
    submission_2016['ParcelId'] = features_2016.parcelid
    submission_2017['ParcelId'] = features_2017.parcelid

    test_features_2016, test_features_2017 = transform_test_features(features_2016, features_2017)

    pred_2016, pred_2017 = [], []
    for i, model in enumerate(models):
        print("Start model {} (2016)".format(i))
        pred_2016.append(model.predict(test_features_2016))
        print("Start model {} (2017)".format(i))
        pred_2017.append(model.predict(test_features_2017))

    # Take average across all models
    mean_pred_2016 = np.mean(pred_2016, axis=0)
    mean_pred_2017 = np.mean(pred_2017, axis=0)

    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]
    submission_2016['201611'] = submission_2016['201610']
    submission_2016['201612'] = submission_2016['201610']

    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]
    submission_2017['201711'] = submission_2017['201710']
    submission_2017['201712'] = submission_2017['201710']

    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')

    print("Length of submission DataFrame: {}".format(len(submission)))
    print("Submission header:")
    print(submission.head())
    submission.to_csv(file_name, index=False)
    return submission, pred_2016, pred_2017  # Return the results so that we can analyze or sanity check it


file_name = 'submission/final_catboost_single.csv'
submission, pred_2016, pred_2017 = predict_and_export([model], features_2016, features_2017, file_name)

# Remove outliers (if any) from training data
outlier_threshold = 0.4
mask = (abs(catboost_y) <= outlier_threshold)
catboost_X = catboost_X[mask, :]
catboost_y = catboost_y[mask]
print("catboost_X: {}".format(catboost_X.shape))
print("catboost_y: {}".format(catboost_y.shape))

# Train multiple models
bags = 8
models = []
params['iterations'] = 1000
for i in range(bags):
    print("Start training model {}".format(i))
    params['random_seed'] = i
    np.random.seed(42)
    random.seed(36)
    model = CatBoostRegressor(**params)
    model.fit(catboost_X, catboost_y, cat_features=categorical_indices, verbose=False)
    models.append(model)

# Sanity check (make sure scores on a small portion of the dataset are reasonable)
for i, model in enumerate(models):
    print("model {}: {}".format(i, abs(model.predict(X_val) - y_val).mean() * 100))

# Save the trained models to disk
save_models(models)

# models = load_models(['checkpoints/catboost_' + str(i) for i in range(8)])  # load pretrained models


# Make predictions and export results
file_name = 'submission/final_catboost_ensemble_x8.csv'
submission, pred_2016, pred_2017 = predict_and_export(models, features_2016, features_2017, file_name)