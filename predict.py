import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics


if __name__ == "__main__":
    pengus = pd.read_csv('data.csv', index_col=0)

    features = [f for f in pengus.columns if f not in ('id','token_id','image_url', 'permalink','address', 
                                                    'last_sold_token','last_sold_eth','last_sold_usd','listing_eth_price',
                                                    'payment_token.usd_price', 'stats.floor_price',
                                                    'last_sold_usd_log','stats.num_owners',
                                                    'Background','Body', 'Face', 'Head', 'Skin','preds_usd', 'last_sold_date',
                                                'mean_sale_eth_day','max_sale_eth_day','Trait_count_avg_price')]
        
    loaded_rf = joblib.load("rf_ppg_model.joblib")
    preds_usd = np.stack([t.predict(pengus[features]) for t in loaded_rf.estimators_])
    pengus['preds_usd'] = np.expm1(preds_usd[-1])
    pengus['preds_eth'] = pengus['preds_usd'] / pengus['payment_token.usd_price'].median()
    pengus = pengus[['last_sold_usd','last_sold_eth','preds_usd','preds_eth']]
    plt.scatter(x='last_sold_usd', y='preds_usd', s=10, data=pengus)
    plt.show()
    pengus.to_csv('preds.csv')
