# NFT_pricing_model

## About
NFT pricing model.

## Getting Started
1. Install [Python](https://www.python.org/downloads/)

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

5. Alternatively run notebook in colab
