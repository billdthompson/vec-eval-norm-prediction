# coding: utf-8
# Copyright (c) 2018 - Present Bill Thompson (biltho@mpi.nl) 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler as Scaler

import click
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

N = 10000 # model max vocabulary size
D = 300 # model vector dimension

# lightweight wrapper for interfacing with .vec files 
class Vectors:
    def __init__(self, modelpath=''):
        with open(modelpath, 'r') as f:
            # skip header
            next(f)
            
            self.vectors = np.zeros((N, D))
            self.word = np.empty(N, dtype = object)
            for i, line in enumerate(f):
        
                if i >= N: break

                rowentries = line.rstrip('\n').split(' ')
                self.word[i] = rowentries[0]
                self.vectors[i] = rowentries[1:D + 1]

            self.vectors = self.vectors[:i]
            self.word = self.word[:i]



@click.command()
@click.option('--norms', '-n', default='norms-vadc-aoa-en.csv')
@click.option('--vecfile', '-v', default='wiki.en.vec')
def run(norms, vecfile):
    
    ### Data Preprocessing --->
    logging.info("Reading {0} and {1}.".format(norms, vecfile))
    data = pd.read_csv(norms).drop_duplicates(subset = 'word').copy()
    data['word'] = data.word.str.lower()

    # obtain semantics
    model = Vectors(vecfile)
    
    # merge norms and semantics
    modeldata = pd.DataFrame(dict(word = model.word))
    covered = modeldata.word.isin(data.word)
    modeldata = modeldata.merge(data, on = 'word', how = 'left')
    
    # silo training material
    training_vectors = model.vectors[covered]

    results = {}
    logging.info("Computing predictions.")
    for norm in ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']:
    
        # format norms
        data[norm] -= data[norm].mean()

        # mask nas
        keep = modeldata[covered][norm].notnull()

        # learn the regression
        X, y = training_vectors[keep], modeldata[covered][norm][keep].values
        lr = LR()
        lr.fit(X, y)

        results[norm] = lr.score(X, y)

    results = pd.DataFrame(results, index = [0])
    results['vectors'] = vecfile.split('/')[-1]
    logging.info('=================================================================================')
    logging.info("=================================================================================:\n{}".format(results[['valence', 'arousal', 'dominance', 'aoa', 'concreteness']].head()))
    logging.info('=================================================================================')
    results.to_csv('{0}-normscores.csv'.format(vecfile.split('/')[-1].strip('.vec')))


if __name__ == '__main__':
    run()