import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import csv

from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split as tts

import click
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

D = 300 # skipgram vector dimension
VECTORDIMENSIONS = ['d_{0}'.format(d) for d in range(D)]

@click.command()
@click.option('--norms', '-n', default='norms-vadc-aoa-en.csv')
@click.option('--vecfile', '-v', default='wiki.en.vec')
def run(norms, vecfile):

	# norms
	data = pd.read_csv(norms).drop_duplicates(subset = 'word').copy()
	data['word'] = data.word.str.lower()

	# vectors
	vecs = pd.read_csv(vecfile, sep = ' ', quoting = csv.QUOTE_NONE, skiprows = 1, header = None, names = ['word'] + VECTORDIMENSIONS + ['ignore']).drop_duplicates(subset = ['word']).merge(data, on = 'word', how = 'inner')

	# train // test split
	train, test = tts(vecs, test_size = .25)

	norms, predictions, trainsizes, testsizes  = [], [], [], [] 
	
	logging.info("Computing predictions")
	for norm in ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']:
	
		# format norms
		mu = train[norm].mean()

		# mask nas
		keep = train[norm].notnull()

		logging.info("{0}: trained on #{1} (of #{2})".format(norm, keep.sum(), keep.sum() + test[norm].notnull().sum()))

		# learn the regression
		X, y = train[keep][VECTORDIMENSIONS].values, train[keep][norm].values - mu
		lr = LR()
		lr.fit(X, y)

		norms.append(norm)
		predictions.append(pearsonr(lr.predict(test[test[norm].notnull()][VECTORDIMENSIONS].values), test[test[norm].notnull()][norm].values)[0])
		trainsizes.append(keep.sum())
		testsizes.append(test[norm].notnull().sum())


	results = pd.DataFrame(dict(norm = norms, correlation = predictions, trained_on = trainsizes, tested_on = testsizes))
	results['vectors'] = vecfile.split('/')[-1]
	logging.info("\n{}".format(results.sort_values('correlation')))
	results.to_csv('{0}-normscores.csv'.format(vecfile.split('/')[-1].strip('.vec')))


if __name__ == '__main__':
	run()