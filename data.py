# python3 -m venv nft_pricing
# source nft_pricing/bin/activate
# python3 -m pip install -r requirements.txt

import joblib
import pandas as pd
import numpy as np
import json
import requests
import math
from tqdm.auto import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.style as style
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

token_ids = [x for x in range(0,8888)]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

id_groups = list(chunks(token_ids, 50))

def get_penguins():
    penguin_list = []
    for id in tqdm(id_groups):
        url = "https://api.opensea.io/api/v1/assets"

        querystring = {"token_ids": id, "asset_contract_address":"0xbd3531da5cf5857e7cfaa92426877b022e612cf8","order_direction":"asc","offset":"0","limit":"50"}

        headers = {"X-API-KEY": "41d5b498b5214d489dfe018df71ad8b5"}

        response = requests.request("GET", url, headers=headers , params=querystring).json()

        penguin_list.append(response)
    return penguin_list

def get_stats():
    url = "https://api.opensea.io/api/v1/collection/pudgypenguins"
    querystring = {"offset":0, "limit": 1}
    stat_response = requests.request("GET", url, params=querystring).json()
    stats = pd.json_normalize(stat_response['collection'])[['stats.floor_price']]
    #['stats.seven_day_volume','stats.seven_day_change','stats.seven_day_sales','stats.seven_day_average_price','stats.num_owners']
    return stats

def process_data(nft_asset_list, stats):
    penguins = [x['assets'] for x in nft_asset_list]
    df_penguins = pd.DataFrame(list(chain(*penguins)))

    df_penguins['sell_orders'] = df_penguins['sell_orders'].apply(lambda x: {} if pd.isna(x) else x)
    so = pd.json_normalize(list(chain(*df_penguins['sell_orders'])))
    so['listing_token_price'] = (so['current_price'].astype(float)/10**so['payment_token_contract.decimals'].astype(float))
    so['listing_usd_price'] = (so['listing_token_price'].astype(float) * so['payment_token_contract.usd_price'].astype(float))    
    so['listing_eth_price'] = (so['listing_usd_price'].astype(float) / so['payment_token_contract.usd_price'].median().astype(float))
    so = so[['metadata.asset.id','listing_usd_price', 'listing_eth_price']]

    df_penguins = df_penguins.merge(so, how='left', left_on= 'token_id', right_on='metadata.asset.id')

    cols = ['id','token_id','num_sales','image_original_url', 'permalink', 'listing_usd_price', 'listing_eth_price'] #'listing_usd_price', 'listing_eth_price'
    desc = df_penguins[cols]

    owner = pd.json_normalize(df_penguins['owner'])['address']

    traits = pd.json_normalize(list(chain(*df_penguins['traits']))).pivot(columns='trait_type', values=['value', 'trait_count']).apply(lambda x: pd.Series(x.dropna().values))

    df_penguins['last_sale'] = df_penguins['last_sale'].apply(lambda x: {} if pd.isna(x) else x)
    last_sale_cols = ['created_date','last_sold_token','last_sold_usd','last_sold_eth','payment_token.usd_price']
    last_sale = pd.json_normalize(df_penguins['last_sale'])
    last_sale['last_sold_token'] = last_sale['total_price'].astype(float)/10**last_sale['payment_token.decimals']
    last_sale['last_sold_usd'] = last_sale['payment_token.usd_price'].astype(float) * last_sale['last_sold_token'].astype(float)
    last_sale['last_sold_eth'] = (last_sale['last_sold_usd'].astype(float) / last_sale['payment_token.usd_price'].median().astype(float))
    last_sale.loc[:, ['last_sold_eth']] = last_sale['last_sold_eth'].astype(float).fillna(0.03)
    last_sale['last_sold_usd'] = last_sale['payment_token.usd_price'].median().astype(float) * last_sale['last_sold_eth'].astype(float)

    last_sale = last_sale[last_sale_cols]
    last_sale.columns = ['last_sold_date','last_sold_token','last_sold_usd', 'last_sold_eth', 'payment_token.usd_price']
    last_sale['last_sold_date'] = pd.to_datetime(last_sale['last_sold_date'])
    last_sale['last_sold_date'].fillna(value=pd.to_datetime('2021/07/22'), inplace=True) 
    last_sale['last_sold_date'] = last_sale['last_sold_date'].dt.normalize()
    last_sale['mean_sale_eth_day'] = last_sale.groupby('last_sold_date')['last_sold_eth'].transform('mean')
    last_sale['max_sale_eth_day'] = last_sale.groupby('last_sold_date')['last_sold_eth'].transform('max')
    last_sale['mean_sale_usd_day'] = last_sale['mean_sale_eth_day'] * last_sale['payment_token.usd_price'].median().astype(float)
    last_sale['max_sale_usd_day'] = last_sale['max_sale_eth_day'] * last_sale['payment_token.usd_price'].median().astype(float)

    penguins_combine = pd.concat([desc, owner, traits, last_sale], axis=1)
    new_name = ['id','token_id','num_sales','image_original_url','permalink', 'listing_usd_price', 'listing_eth_price',
                'address','Background','Body','Face','Head','Skin','Background_count','Body_count',
                'Face_count','Head_count','Skin_count','last_sold_date','last_sold_token','last_sold_usd','last_sold_eth','payment_token.usd_price',
                'mean_sale_eth_day','max_sale_eth_day','mean_sale_usd_day','max_sale_usd_day']
    penguins_combine.columns = new_name
    penguins_combine.loc[penguins_combine['Head'] == 'None', ['Head_count']] = '0'
    for col in ('Background','Body','Face','Head','Skin'):
        penguins_combine[col+'1'] = penguins_combine[col].apply(lambda x: 1 if x != 'None' else 0)
    penguins_combine['Trait_count'] = penguins_combine['Background1'] + penguins_combine['Body1'] + penguins_combine['Face1'] + penguins_combine['Head1'] + penguins_combine['Skin1']
    penguins_combine.drop(['Background1','Body1','Face1','Head1','Skin1'], axis=1, inplace=True)

    penguins_combine['rarity_score'] = penguins_combine['Background_count'].astype(int) + penguins_combine['Body_count'].astype(int) + penguins_combine['Face_count'].astype(int) + penguins_combine['Head_count'].astype(int) + penguins_combine['Skin_count'].astype(int)
    penguins_combine['number_owned'] = penguins_combine.groupby('address')['address'].transform('count')
    trait_count = ['Background_count','Body_count','Face_count','Head_count','Skin_count']
    for col in trait_count:
        penguins_combine[col] = penguins_combine[col].astype(float)/8888

    pengu_all = pd.concat([penguins_combine,stats],1).ffill()

    pengu_all['stats.floor_price_usd'] = pengu_all['stats.floor_price'].astype(float) * pengu_all['payment_token.usd_price'].astype(float).median()
    pengu_all['mint_multiple'] = pengu_all['last_sold_eth'].astype(float) / 0.03
    pengu_all['floor_multiple'] = pengu_all['last_sold_eth'].astype(float) / pengu_all['stats.floor_price_usd'].astype(float).median()

    pengu_all['last_sold_usd_log'] = pengu_all['last_sold_usd'].apply(lambda x: np.log1p(x))
    pengu_all['last_sold_eth_log'] = pengu_all['last_sold_eth'].apply(lambda x: np.log1p(x))
    #pengu_all['listing_usd_log'] = pengu_all['listing_usd_price'].apply(lambda x: np.log1p(x))
    pengu_all['num_sold_day'] = pengu_all.groupby('last_sold_date')['num_sales'].transform('count')

    num_cols = ['num_sales', 'Background_count', 'Body_count','Face_count', 'Head_count', 
            'Skin_count','rarity_score','number_owned', 'mint_multiple', 'floor_multiple']

    for col in num_cols:
        # do not fill numerical columns
        pengu_all.loc[:, col] = pengu_all[col].astype(float).fillna(0)
    
    pengu_all['Trait_count_avg_price'] = pengu_all.groupby('Trait_count')['last_sold_eth'].transform('mean')
    pengu_all['Trait_count_avg_usd_price'] = pengu_all['Trait_count_avg_price'] * pengu_all['payment_token.usd_price'].astype(float).median()

    pengu_all.loc[pengu_all['listing_eth_price'] > 150, ['listing_eth_price']] = '200'
    pengu_all['listing_usd_price'] = pengu_all['listing_eth_price'].astype(float) * pengu_all['payment_token.usd_price'].astype(float).median()
    pengu_all['listing_usd_price'].fillna(pengu_all['listing_usd_price'].median(), inplace=True)
    pengu_all['listing_multiple'] = pengu_all['listing_usd_price'].astype(float) / pengu_all['stats.floor_price_usd'].astype(float)

    pengu_all.loc[:, "rarity_bins"] = pd.cut(
        pengu_all["rarity_score"], bins=5, labels=False)

    return pengu_all

if __name__ == "__main__":
    stats = get_stats()
    penguin_list = get_penguins()
    pengus = process_data(penguin_list, stats)
    pengus.to_csv('data/data.csv')