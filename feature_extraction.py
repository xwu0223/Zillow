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

# Load in properties data
prop_2016 = data_proc.load_properties_data("data/properties_2016.csv")
prop_2017 = data_proc.load_properties_data("data/properties_2017.csv")

assert len(prop_2016) == len(prop_2017)
print("Number of properties: {}".format(len(prop_2016)))
print("Number of property features: {}".format(len(prop_2016.columns)-1))


# Rename & retype the feature columns; also unify representations of missing values
def get_landuse_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby('county_landuse_code')['county_landuse_code'].count()
    landuse_codes = list(temp[temp >= 300].index)
    temp = prop_2017.groupby('county_landuse_code')['county_landuse_code'].count()
    landuse_codes += list(temp[temp >= 300].index)
    landuse_codes = list(set(landuse_codes))
    df_landuse_codes = pd.DataFrame({'county_landuse_code': landuse_codes,
                                     'county_landuse_code_id': range(len(landuse_codes))})
    return df_landuse_codes


def get_zoning_desc_code_df(prop_2016, prop_2017):
    temp = prop_2016.groupby('zoning_description')['zoning_description'].count()
    zoning_codes = list(temp[temp >= 5000].index)
    temp = prop_2017.groupby('zoning_description')['zoning_description'].count()
    zoning_codes += list(temp[temp >= 5000].index)
    zoning_codes = list(set(zoning_codes))
    df_zoning_codes = pd.DataFrame({'zoning_description': zoning_codes,
                                    'zoning_description_id': range(len(zoning_codes))})
    return df_zoning_codes


def process_columns(df, df_landuse_codes, df_zoning_codes):
    df = df.merge(how='left', right=df_landuse_codes, on='county_landuse_code')
    df = df.drop(['county_landuse_code'], axis=1)

    df = df.merge(how='left', right=df_zoning_codes, on='zoning_description')
    df = df.drop(['zoning_description'], axis=1)

    df.loc[df.county_id == 3101, 'county_id'] = 0
    df.loc[df.county_id == 1286, 'county_id'] = 1
    df.loc[df.county_id == 2061, 'county_id'] = 2

    df.loc[df.landuse_type_id == 279, 'landuse_type_id'] = 261
    return df


data_proc.rename_columns(prop_2016)
data_proc.rename_columns(prop_2017)

df_landuse_codes = get_landuse_code_df(prop_2016, prop_2017)
df_zoning_codes = get_zoning_desc_code_df(prop_2016, prop_2017)
prop_2016 = process_columns(prop_2016, df_landuse_codes, df_zoning_codes)
prop_2017 = process_columns(prop_2017, df_landuse_codes, df_zoning_codes)

data_proc.retype_columns(prop_2016)
data_proc.retype_columns(prop_2017)

print(prop_2017.head())

# Write current DataFrames to hdf5
prop_2016.to_hdf('hdf5/prop.h5', key='prop_2016', format='table', mode='w')
prop_2017.to_hdf('hdf5/prop.h5', key='prop_2017', format='table', mode='a')

# Read DataFrames from hdf5
prop_2016 = pd.read_hdf('hdf5/prop.h5', 'prop_2016')
prop_2017 = pd.read_hdf('hdf5/prop.h5', 'prop_2017')

# Load in training data (with logerror labels)
train_2016 = data_proc.load_training_data("data/train_2016_v2.csv")
train_2017 = data_proc.load_training_data("data/train_2017.csv")

print("Number of 2016 transaction records: {}".format(len(train_2016)))
print("Number of 2017 transaction records: {}".format(len(train_2017)))
print("\n", train_2016.head())
print("\n", train_2017.head())

# Basic feature engineering + Drop duplicate columns
for prop in [prop_2016, prop_2017]:
    prop['avg_garage_size'] = prop['garage_sqft'] / prop['garage_cnt']

    prop['property_tax_per_sqft'] = prop['tax_property'] / prop['finished_area_sqft_calc']

    # Rotated Coordinates
    prop['location_1'] = prop['latitude'] + prop['longitude']
    prop['location_2'] = prop['latitude'] - prop['longitude']
    prop['location_3'] = prop['latitude'] + 0.5 * prop['longitude']
    prop['location_4'] = prop['latitude'] - 0.5 * prop['longitude']

    # 'finished_area_sqft' and 'total_area' cover only a strict subset of 'finished_area_sqft_calc' in terms of
    # non-missing values. Also, when both fields are not null, the values are always the same.
    # So we can probably drop 'finished_area_sqft' and 'total_area' since they are redundant
    # If there're some patterns in when the values are missing, we can add two isMissing binary features
    prop['missing_finished_area'] = prop['finished_area_sqft'].isnull().astype(np.float32)
    prop['missing_total_area'] = prop['total_area'].isnull().astype(np.float32)
    prop.drop(['finished_area_sqft', 'total_area'], axis=1, inplace=True)

    # Same as above, 'bathroom_cnt' covers everything that 'bathroom_cnt_calc' has
    # So we can safely drop 'bathroom_cnt_calc' and optionally add an isMissing feature
    prop['missing_bathroom_cnt_calc'] = prop['bathroom_cnt_calc'].isnull().astype(np.float32)
    prop.drop(['bathroom_cnt_calc'], axis=1, inplace=True)

    # 'room_cnt' has many zero or missing values
    # On the other hand, 'bathroom_cnt' and 'bedroom_cnt' have few zero or missing values
    # Add an derived room_cnt feature by adding bathroom_cnt and bedroom_cnt
    prop['derived_room_cnt'] = prop['bedroom_cnt'] + prop['bathroom_cnt']

    # Average area in sqft per room
    mask = (prop.room_cnt >= 1)  # avoid dividing by zero
    prop.loc[mask, 'avg_area_per_room'] = prop.loc[mask, 'finished_area_sqft_calc'] / prop.loc[mask, 'room_cnt']

    # Use the derived room_cnt to calculate the avg area again
    mask = (prop.derived_room_cnt >= 1)
    prop.loc[mask, 'derived_avg_area_per_room'] = prop.loc[mask, 'finished_area_sqft_calc'] / prop.loc[
        mask, 'derived_room_cnt']

print(prop_2017.head())


# Compute region-based aggregate features
def add_aggregate_features(df, group_col, agg_cols):
    df[group_col + '-groupcnt'] = df[group_col].map(df[group_col].value_counts())
    new_columns = []  # New feature columns added to the DataFrame

    for col in agg_cols:
        aggregates = df.groupby(group_col, as_index=False)[col].agg([np.mean])
        aggregates.columns = [group_col + '-' + col + '-' + s for s in ['mean']]
        new_columns += list(aggregates.columns)
        df = df.merge(how='left', right=aggregates, on=group_col)

    for col in agg_cols:
        mean = df[group_col + '-' + col + '-mean']
        diff = df[col] - mean

        df[group_col + '-' + col + '-' + 'diff'] = diff
        if col != 'year_built':
            df[group_col + '-' + col + '-' + 'percent'] = diff / mean

    # Set the values of the new features to NaN if the groupcnt is too small (prevent overfitting)
    threshold = 100
    df.loc[df[group_col + '-groupcnt'] < threshold, new_columns] = np.nan

    # Drop the mean features since they turn out to be not useful
    df.drop([group_col + '-' + col + '-mean' for col in agg_cols], axis=1, inplace=True)

    gc.collect()
    return df


group_col = 'region_zip'
agg_cols = ['lot_sqft', 'year_built', 'finished_area_sqft_calc',
            'tax_structure', 'tax_land', 'tax_property', 'property_tax_per_sqft']
prop_2016 = add_aggregate_features(prop_2016, group_col, agg_cols)
prop_2017 = add_aggregate_features(prop_2017, group_col, agg_cols)

print(prop_2017.head(10))

# Write feature DataFrames to hdf5
prop_2016.to_hdf('hdf5/features.h5', key='features_2016', format='table', mode='w')
prop_2017.to_hdf('hdf5/features.h5', key='features_2017', format='table', mode='a')

# Join the training data with the property table
train_2016 = train_2016.merge(how='left', right=prop_2016, on='parcelid')
train_2017 = train_2017.merge(how='left', right=prop_2017, on='parcelid')
train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)

# Combine the 2016 and 2017 training sets
train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)
print("\nCombined training set size: {}".format(len(train)))

# Add datetime features to training data
data_proc.add_simple_datetime_features(train)

print(train.head(10))

# Write training DataFrame to hdf5
train.to_hdf('hdf5/train.h5', key='train', format='table', mode='w')