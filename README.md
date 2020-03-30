# Zillow
This repo tackles the first round of Zillow’s Home Value Prediction Competition, which challenges competitors to predict the log error between Zestimate and the actual sale price of houses. And the submissions are evaluated based on Mean Absolute Error between the predicted log error and the actual log error. The competition was hosted from May 2017 to October 2017 on Kaggle, and the final private leaderboard was revealed after the evaluation period ended in January 2018.  The score of the stacked model in this repo would have ranked in the top-100 and qualified for the private second round, which involves building a home valuation algorithm from ground up.

### Please note, this is an replica of Junjie Dong (junjied@stanford.edu)'s work.
### All large size files such as raw data, submission data, hdf5, checkpoints are available at https://drive.google.com/open?id=1nQbj8m8MQmMPcSHSraRaCECrmhZSg6RV
### All running results are in the "output" folder

#### From the feature importance plot , the most important features in house prices is year_built, lot_sqft, region_zip-year_built_diff, and bed_avg_area_per_room. the least important features are poll_total_size, bathroom_cnt_calc, and bathroom_small_cnt.
#### In this project, catboost, Light GBM, catboost_x8(with 8 Bootstraps training set),combination of catboost_x8 and Light GBM models are used for training and testing the log error of home values.
#### CatBoost is an algorithm for gradient boosting on decision trees. Therefore it is excellent in prediction and recommendation models, in the meanwhile, catboost requires the least amount of RAM and time(you can get the best result after the first run).
#### Light GBM is also a high performance algorithm for gradient boosting on decision trees, as well as handling large scale of data with distributed framework. In this model, the outlier is set to be 0.4, this is because the standard deviation of the log error is 0.161, and we usually obtain data within 3*SD, so 0.4 is an ideal threshold number. We also have to drop some useless features such as 'framing_id', 'architecture_style_id', 'story_id', 'perimeter_area', 'basement_sqft', 'storage_sqft','fireplace_flag', 'deck_id', 'pool_unk_1', 'construction_id', 'county_id', 'fips' for the training processing
#### Therefore with tuned model 0.7 of Catboost and 0.3 of Light GBM gives the highest performance


| __Model__  | __Public Board Score__ | __Private Board Score__ |
|-------------|------------|------------|
| Catboost_x8|      0.06434       | 0.07523 |
| Catboost   | 0.06435            |  0.07523 |
| Light GBM_x5   | 0.06440           |  0.07523 |
| Light GBM | 0.06437 |0.07520 
| Stack   | 0.06428           |  0.07523|

