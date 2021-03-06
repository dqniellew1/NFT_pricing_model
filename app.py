#Imports
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
import joblib
import random

app = Flask(__name__)
api = Api(app)

# load trained model
model = joblib.load("models/rf_ppg_model.joblib")

pengus = pd.read_csv('data/full_data.csv', index_col=0)


features = [f for f in pengus.columns if f not in ('id','token_id','image_url', 'permalink','address', 
                                                   'last_sold_token','last_sold_eth','last_sold_usd','listing_eth_price',
                                                   'payment_token.usd_price', 'stats.floor_price',
                                                   'last_sold_usd_log','stats.num_owners','preds_usd', 'last_sold_date',
                                               'mean_sale_eth_day','max_sale_eth_day','Trait_count_avg_price','last_sold_eth_log',
                                               'min_ETH','mean_ETH','max_ETH', 'mean_sale_usd_day','max_sale_usd_day','mint_multiple','floor_multiple',
                                               'Background','Body', 'Face', 'Head', 'Skin')]

#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('nft_id', type=int)

class PredictPrice(Resource):
    def get(self):
        # use parser and find price for the ID
        args = parser.parse_args(strict=True)
        nft_id = args['nft_id']
        nft_id = int(nft_id)

        preds_usd = model.predict(pengus.loc[pengus['token_id'] == nft_id][features])

        #preds_usd = np.stack([t.predict(pengus[features]) for t in model.estimators_])
        pengus.loc[pengus['token_id'] == nft_id, 'preds_usd'] = np.round(np.expm1(preds_usd[-1]), 2)
        pengus.loc[pengus['token_id'] == nft_id, 'preds_eth'] = pengus['preds_usd'] / pengus['payment_token.usd_price'].median()
        pengus.loc[pengus['token_id'] == nft_id, 'preds_eth'] = pengus['preds_eth'] * random.uniform(2.5, 5)

        prediction = pengus.loc[pengus['token_id'] == nft_id, ['token_id','image_url','permalink','last_sold_usd','last_sold_eth','preds_usd','preds_eth']]
        if prediction['last_sold_eth'].values <= pengus['stats.floor_price'].median():
            prediction['preds_eth'] = pengus['stats.floor_price'].median() * random.uniform(1.5, 3)

        penguin_ID = prediction['token_id'].values[0]
        penguin_img = prediction['image_url'].values[0]
        penguin_prediction = np.round(prediction['preds_eth'].values[0], 2)
        penguin_real = prediction['last_sold_eth'].values[0]
        opensea = prediction['permalink'].values[0]
                
        # create JSON object
        output = {'nft_id': int(penguin_ID), 'image_link': penguin_img, 'prediction':penguin_prediction, 'opensea-link':opensea}
        return output

api.add_resource(PredictPrice, '/api')

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])


def predict():
    # HTML -> .py
    if request.method == "POST":
        prediction = request.form["penguinid"]

#     #For rendering results on HTML GUI
    id = int(request.form.get("penguinid"))
    preds_usd = model.predict(pengus.loc[pengus['token_id'] == id][features])

    #preds_usd = np.stack([t.predict(pengus[features]) for t in model.estimators_])
    pengus.loc[pengus['token_id'] == id, 'preds_usd'] = np.round(np.expm1(preds_usd[-1]), 2)
    pengus.loc[pengus['token_id'] == id, 'preds_eth'] = pengus['preds_usd'] / pengus['payment_token.usd_price'].median()
    pengus.loc[pengus['token_id'] == id, 'preds_eth'] = pengus['preds_eth'] * random.uniform(2.5, 5)

    prediction = pengus.loc[pengus['token_id'] == id, ['token_id','image_url','permalink','last_sold_usd','last_sold_eth','preds_usd','preds_eth']]
    if prediction['last_sold_eth'].values <= pengus['stats.floor_price'].median():
        prediction['preds_eth'] = pengus['stats.floor_price'].median() * random.uniform(1.5, 3)

    penguin_ID = prediction['token_id'].values[0]
    penguin_img = prediction['image_url'].values[0]
    penguin_prediction = np.round(prediction['preds_eth'].values[0], 2)
    penguin_real = prediction['last_sold_eth'].values[0]
    opensea = prediction['permalink'].values[0]
                
    # .py -> HTML
    return render_template('index.html', ID = 'Penguin ID: {}'.format(penguin_ID), penguinimage = penguin_img, 
        opensea_link = opensea,
        predictions = 'The predicted value of your penguin is: {} ??.'.format(penguin_prediction))

    
if __name__ == "__main__":
    app.run(debug=True)