import pandas as pd
import gzip
import pprint as pp
import re

#functions parse and getDF from http://jmcauley.ucsd.edu/data/amazon/
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def format_columns(df):
  df = df.drop(['reviewerName', 'salesRank', 'categories', 'title', 'brand', 'unixReviewTime', 'description', 'imUrl','related', 'reviewTime'], axis=1)
  df = df.astype({"helpful": str})
  df = df.reindex(columns=['reviewerID', 'asin', 'helpful', 'helpful_count', 'total_count', 'percent_helpful', 'reviewText', 'overall', 'summary', 'price'], fill_value=0)
  return df

def format_helpfulness(df):
  df['helpful_count'] = df.apply (lambda row: get_helpful_count(row), axis=1)
  df['total_count'] = df.apply (lambda row: get_total_count(row), axis=1)
  df = df.drop('helpful', axis=1)
  df = df.astype({"helpful_count": int, "total_count": int})
  df = df[df.total_count >= 10]
  df['percent_helpful'] = df.apply (lambda row: get_helpful_percent(row), axis=1)
  return df
  
def get_helpful_count (row):
  result = re.search('\[\d+', row['helpful']).group()
  return re.sub('[^0-9]','', result)

def get_total_count (row):
  result = re.search('\d+\]', row['helpful']).group()
  return re.sub('[^0-9]','', result)

def get_helpful_percent (row):
    return float(row['helpful_count']) / row['total_count']


def create(products_path, reviews_path):
  products_df = getDF(products_path)
  reviews_df = getDF(reviews_path)
  reviews_df = reviews_df[reviews_df.helpful != '[0, 0]']
  combo_df = pd.merge(reviews_df, products_df, on='asin', how='inner')
  combo_df = format_columns(combo_df)
  combo_df = format_helpfulness(combo_df)
  combo_df.to_json('clean.json')
  return combo_df


