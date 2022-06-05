import pandas as pd
import argparse
import datetime 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow import keras
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
np.random.seed(5)
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

n_steps_out = 24

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def output(path, data):
    
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def Model():
    nb_days = 96
    n_features = 1
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,input_shape = (nb_days, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=False))
    model.add(Dense(n_steps_out))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

"""預測未來一天的產量和消耗量"""
def predict_future(cons_df,gene_df):
    consum = cons_df.iloc[-96:]['consumption']
    consum = np.array(consum)
    consum = consum.reshape((1, 96, 1))

    model_consum = Model()
    model_consum.load_weights('./consum.h5')
    predict_consum = model_consum.predict(consum)
    for index in range(len(predict_consum[0])):
        if predict_consum[0][index] < 0:
            predict_consum[0][index] = 0

    gene = gene_df.iloc[-96:]['generation']
    gene = np.array(gene)
    gene = gene.reshape((1, 96, 1))

    model_gene = Model()
    model_gene.load_weights('./gene.h5')
    predict_gene = model_gene.predict(gene)
    for index in range(len(predict_gene[0])):
        if predict_gene[0][index] < 0:
            predict_gene[0][index] = 0

    """ 計算產出和消耗差值"""
    diff = []
    for i in range(len(predict_consum[0])):
        sub = predict_gene[0][i] - predict_consum[0][i]
        diff.append(round(sub,2))

    return diff


if __name__ == "__main__":
    args = config()


    consum_path = args.consumption
    gener_path = args.generation
    bid_path = args.bidresult

    """ Load data   """
    cons_df = pd.read_csv(consum_path)
    gene_df = pd.read_csv(gener_path)

    last_date = gene_df['time'].values.tolist()[-1]
    standard_last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')

    dif = predict_future(cons_df,gene_df)
    data = []
    
    for i in range(1,25):
        hours_add = datetime.timedelta(hours=i)
        future_date = standard_last_date + hours_add

        if dif[i-1] < 0:
            data.append([future_date,"buy",2,1])
            # data.append([future_date,"buy",2,-dif[i-1]])
        elif dif[i-1] >= 0 :
            # data.append([future_date,"sell",3,dif[i-1]])
            data.append([future_date,"sell",2,1])
            data.append([future_date,"sell",2.6,1])
     
    
    output(args.output, data)
