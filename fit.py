import joblib
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

def create_folds(data):
    # create folds for training data
    # create kfold column and fill -1
    data['kfold'] = -1

    # randomize data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule
    # Take floor or round it
    #num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    # cut to sort data into bins
    data.loc[:, "bins"] = pd.cut(
        data["last_sold_usd"], bins=20, labels=False
    )

    # initiate kfold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new k-fold column
    # use bins for target since regression
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column
    data = data.drop("bins", axis=1)
    # return dataframe with folds
    return data


model = RandomForestRegressor(n_jobs=-1, n_estimators=60,
        max_samples=2500, max_features=0.5,
        min_samples_leaf=8, oob_score=True)

def run(fold, df, model):
    # get training data with folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data with folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # fit model on training data
    model.fit(x_train, df_train.last_sold_usd_log.values)
    
    # get predictions from validation data
    valid_preds = model.predict(x_valid)

    # get scores
    mse = metrics.mean_squared_error(df_valid.last_sold_usd_log.values, valid_preds)
    mae = metrics.mean_absolute_error(df_valid.last_sold_usd_log.values, valid_preds)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(df_valid.last_sold_usd_log.values, valid_preds)
    # print score
    print(f"Fold = {fold}, RMSE = {rmse}, MSE = {mse}, r2 = {r2}, MAE = {mae}")

def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

if __name__ == "__main__":
    # read data
    df = pd.read_csv('data/data.csv', index_col=0)

    features = [f for f in df.columns if f not in ('id','token_id','image_original_url','image_url','permalink','address', 
                                                    'last_sold_token','last_sold_eth','last_sold_usd','listing_eth_price',
                                                    'payment_token.usd_price', 'stats.floor_price',
                                                    'last_sold_usd_log','stats.num_owners',
                                                    'Background','Body', 'Face', 'Head', 'Skin','preds_usd', 'last_sold_date',
                                                'mean_sale_eth_day','max_sale_eth_day','Trait_count_avg_price','last_sold_eth_log')]

    # Split data into train and test set
    train_df, test_df = model_selection.train_test_split(df, train_size = 0.8)

    df_pengu = create_folds(train_df)

    # fit training data
    for fold_ in range(5):
        run(fold_, df_pengu, model)
    
    # Get test scores
    preds_usd = np.stack([t.predict(test_df[features]) for t in model.estimators_])
    test_df['preds_usd'] = np.expm1(preds_usd[-1])
    print(r_mse(preds_usd.mean(0), test_df['last_sold_usd_log'].values))

    # fit full model
    model.fit(df[features], df.last_sold_usd_log.values)
    joblib.dump(model, "models/rf_ppg_model.joblib")
