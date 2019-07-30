import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt

from subprocess import check_output
from datetime import time

def trainer(data, desc, file):
    file.write(desc + "\n")
    X = data[:, 1:]
    y = data[:, 0] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = MLPRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    file.write("Score: " + str(score) + "\n")

    predictions = model.predict(X_test)

    errors = list()

    for i in (range(0, len(predictions) - 1)):
        errors.append(predictions[i] - y_test[i])

    errors_squared = [x ** 2 for x in errors]
    average = sum(errors_squared)/len(errors_squared)
    RMSE = average**0.5
    file.write("RMSE: " + str(RMSE) + "\n")

    plt.scatter(y_test, predictions, alpha=0.1)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.savefig(desc)
    plt.clf()
    
def run(df):
    df = df.dropna()
    df_logprice = df.drop(["asin", "price", "reviewerID", "helpful_count", "total_count", "reviewText", "overall", "summary"], axis=1)
    df_price = df.drop(["asin", "reviewerID", "helpful_count", "total_count", "reviewText", "overall", "summary", "price_log"], axis=1)
    df_none = df.drop(["asin", "price", "reviewerID", "helpful_count", "total_count", "reviewText", "overall", "summary", "price_log"], axis=1)
    
    f = open("results.txt","w+")
    
    trainer(df_logprice.values, "withlogprice", f)
    trainer(df_price.values, "withprice", f)
    trainer(df_none.values, "withoutall", f)

    f.close()

