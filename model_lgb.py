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

import lightgbm as lgb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Save LightGBM models to files
def save_models(models):
    for i, model in enumerate(models):
        model.save_model('checkpoints/lgb_' + str(i))
    print("Saved {} LightGBM models to files.".format(len(models)))

# Load LightGBM models from files
def load_models(paths):
    models = []
    for path in paths:
        model = lgb.Booster(model_file=path)
        models.append(model)
    return models

"""
    Drop id and label columns + Feature selection for LightGBM
"""
def lgb_drop_features(features):
    # id and label (not features)
    unused_feature_list = ['parcelid', 'logerror']

    # too many missing (LightGBM is robust against bad/unrelated features, so this step might not be needed)
    missing_list = ['framing_id', 'architecture_style_id', 'story_id', 'perimeter_area', 'basement_sqft', 'storage_sqft']
    unused_feature_list += missing_list

    # not useful
    bad_feature_list = ['fireplace_flag', 'deck_id', 'pool_unk_1', 'construction_id', 'county_id', 'fips']
    unused_feature_list += bad_feature_list

    # really hurts performance
    unused_feature_list += ['county_landuse_code_id', 'zoning_description_id']

    return features.drop(unused_feature_list, axis=1, errors='ignore')

# Read DataFrames from hdf5
features_2016 = pd.read_hdf('hdf5/features.h5', 'features_2016')  # All features except for datetime for 2016
features_2017 = pd.read_hdf('hdf5/features.h5', 'features_2017')  # All features except for datetime for 2017
train = pd.read_hdf('hdf5/train.h5', 'train')  # Concatenated 2016 and 2017 training data with labels


# Training and Tuning
lgb_features = lgb_drop_features(train)
print("Number of features for LightGBM: {}".format(len(lgb_features.columns)))
print(lgb_features.head(5))

# Prepare training and cross-validation data
lgb_label = train.logerror.astype(np.float32)
print(lgb_label.head())

# Transform to Numpy matrices
lgb_X = lgb_features.values
lgb_y = lgb_label.values

# Perform shuffled train/test split
np.random.seed(42)
random.seed(10)
X_train, X_val, y_train, y_val = train_test_split(lgb_X, lgb_y, test_size=0.2)

# Remove outlier examples from X_train and y_train; Keep them in X_val and y_val for proper cross-validation
outlier_threshold = 0.4
mask = (abs(y_train) <= outlier_threshold)
X_train = X_train[mask, :]
y_train = y_train[mask]

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))

# Specify feature names and categorical features for LightGBM
feature_names = [s for s in lgb_features.columns]
categorical_features = ['cooling_id', 'heating_id', 'landuse_type_id', 'year', 'month', 'quarter']

categorical_indices = []
for i, n in enumerate(lgb_features.columns):
    if n in categorical_features:
        categorical_indices.append(i)
print(categorical_indices)

# LightGBM parameters
params = {}

params['objective'] = 'regression'
params['metric'] = 'mae'
params['num_threads'] = 4  # set to number of real CPU cores for best performance

params['boosting_type'] = 'gbdt'
params['num_boost_round'] = 2000
params['learning_rate'] = 0.003  # shrinkage_rate
params['early_stopping_rounds'] = 30  # Early stopping based on validation set performance

# Control tree growing
params['num_leaves'] = 127  # max number of leaves in one tree (default 31)
params['min_data'] = 150  # min_data_in_leaf
params['min_hessian'] = 0.001  # min_sum_hessian_in_leaf (default 1e-3)
params['max_depth'] = -1  # limit the max depth of tree model, defult -1 (no limit)
params['max_bin'] = 255  # max number of bins that feature values are bucketed in (small -> less overfitting, default 255)
params['sub_feature'] = 0.5    # feature_fraction (small values => use very different submodels)

# Row subsampling (speed up training and alleviate overfitting)
params['bagging_fraction'] = 0.7
params['bagging_freq'] = 50  # perform bagging at every k iteration

# Constraints on categorical features
params['min_data_per_group'] = 100  # minimal number of data per categorical group (default 100)
params['cat_smooth'] = 15.0  # reduce effect of noises in categorical features, especially for those with few data (default 10.0)

# Regularization (default 0.0)
params['lambda_l1'] = 0.0
params['lambda_l2'] = 0.0

# Random seeds (keep default values)
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

# Train LightGBM
lgb_train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
lgb_valid_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)

np.random.seed(42)
random.seed(36)
model = lgb.train(params, lgb_train_set, verbose_eval=False,
                valid_sets=[lgb_train_set, lgb_valid_set], valid_names=['train', 'val'],
                categorical_feature=categorical_indices)

# Evaluate on train and validation sets
print("Train score: {}".format(abs(model.predict(X_train) - y_train).mean() * 100))
print("Val score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))

# Plot LightGBM feature importance
lgb.plot_importance(model, height=0.8, figsize=(12.5, 12.5), ignore_zero=False)
plt.title('Feature Importance')
plt.show()


print('Train on all data + Make predictions')

# Train LightGBM on all given training data (preparing for submission)
del params['early_stopping_rounds']
params['num_boost_round'] = 1250  # roughly chosen based on public leaderboard score
print(params)

outlier_threshold = 0.4
mask = (abs(lgb_y) <= outlier_threshold)
lgb_X = lgb_X[mask, :]
lgb_y = lgb_y[mask]

lgb_train_set = lgb.Dataset(lgb_X, label=lgb_y, feature_name=feature_names)
print("lgb_X: {}".format(lgb_X.shape))
print("lgb_y: {}".format(lgb_y.shape))

np.random.seed(42)
random.seed(36)
model = lgb.train(params, lgb_train_set, verbose_eval=True, categorical_feature=categorical_indices)

# Sanity check: make sure the model score is reasonable on a small portion of the data
print("score: {}".format(abs(model.predict(X_val) - y_val).mean() * 100))

"""
    Helper method that prepares 2016 and 2017 properties features for inference
"""


def transform_test_features(features_2016, features_2017):
    test_features_2016 = lgb_drop_features(features_2016)
    test_features_2017 = lgb_drop_features(features_2017)

    test_features_2016['year'] = 0
    test_features_2017['year'] = 1

    # 11 & 12 lead to unstable results, probably due to the fact that there are few training examples for them
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

file_name = 'submission/final_lgb_single.csv'
submission, pred_2016, pred_2017 = predict_and_export([model], features_2016, features_2017, file_name)

print('Ensemble Training and Prediction')
# Remove outliers (if any) from training data
outlier_threshold = 0.4
mask = (abs(lgb_y) <= outlier_threshold)
lgb_X = lgb_X[mask, :]
lgb_y = lgb_y[mask]
lgb_train_set = lgb.Dataset(lgb_X, label=lgb_y, feature_name=feature_names)
print("lgb_X: {}".format(lgb_X.shape))
print("lgb_y: {}".format(lgb_y.shape))

#del params['early_stopping_rounds']
del params['feature_fraction_seed']
del params['bagging_seed']
params['num_boost_round'] = 1250

# Train multiple models
bags = 5
models = []
for i in range(bags):
    print("Start training model {}".format(i))
    params['seed'] = i
    np.random.seed(42)
    random.seed(36)
    model = lgb.train(params, lgb_train_set, verbose_eval=False, categorical_feature=categorical_indices)
    models.append(model)

# Sanity check (make sure scores on a small portion of the dataset are reasonable)
for i, model in enumerate(models):
    print("model {}: {}".format(i, abs(model.predict(X_val) - y_val).mean() * 100))

# Save the trained models to disk
save_models(models)

# models = load_models(['checkpoints/lgb_' + str(i) for i in range(5)])  # load pretrained models

# Make predictions and export results
file_name = 'submission/final_lgb_ensemble_x5.csv'
submission, pred_2016, pred_2017 = predict_and_export(models, features_2016, features_2017, file_name)



