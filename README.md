# NFT pricing model for Pudgy Penguins NFT

## About
NFT pricing model.

## Getting Started
1. Install [Python](https://www.python.org/downloads/) and start a virtual environment
2. Install required packages
```
pip install -r requirement.txt
```
3. Run `data.py` to get data for model - calls from opensea api, process data and saves into `data.csv`
```
python3 data.py
```
4. Run `predict.py` to get predictions - saves into `preds.csv`
```
python3 predict.py 
```
5. Alternatively run notebooks in colab 
 - Get `data.csv` from repository to load into colab notebooks
```Penguins_data_&_model.ipynb & Penguins_analysis.ipynb```

## File Structure
* NFT pricing model
  * app.py: Flask API/web application
  * data.py: function to get and process the data
  * fit.py: initiates a new model, trains the model, and saves as joblib model
  * predict.py: takes in data and returns predictions
  * requirements.txt: list of packages that the app will import
  * lib
      * data: directory that contains the data and prediction files
      * models: directory that contains the pickled model files

## Testing the API
1. Run the Flask API locally for testing. Go to directory with `app.py`.

```bash
python app.py
```
2. In a new terminal window, use HTTPie to make a GET request at the URL of the API.

```bash
http http://127.0.0.1:5000/api nft_id==1 
```

3. Example of successful output.

```bash
HTTP/1.0 200 OK
Content-Length: 203
Content-Type: application/json
Date: Thu, 19 Aug 2021 05:48:52 GMT
Server: Werkzeug/1.0.1 Python/3.7.7
{
    "image_link": "https://api.pudgypenguins.io/penguin/image/1",
    "nft_id": 1,
    "opensea-link": "https://opensea.io/assets/0xbd3531da5cf5857e7cfaa92426877b022e612cf8/1",
    "prediction": 3.4
}
```


