import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import Levenshtein
import multiprocessing as mp
from multiprocessing import Pool, Manager, Process
import itertools
import pandas as pd
import numpy as np
from time import time
import logging

class MetricsCalculator(object):

    def __init__(self, metrics, workers, dataset, columns, pass_through_columns, logging_level):
        self.metrics = metrics
        self.workers = workers
        self.dataset = dataset
        self.columns = columns
        self.pass_through_columns = pass_through_columns
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger('requests').setLevel(logging.CRITICAL)
        logger = logging.getLogger(__name__)

    def calculate(self):
        if self.validate():
            result = []
            for column_tuple in [e for e in self.columns]:
                calculated_values = self.calculateColumn(self.dataset[column_tuple[0]], self.dataset[column_tuple[1]])
                df = pd.DataFrame.from_dict(calculated_values)                
                result.append(df.rename(columns={e: column_tuple[2] + '_' + e for e in df.columns}))
            return self.attachPassThroughColumns(pd.concat(result, axis=1))

    def attachPassThroughColumns(self, dataset):
        for column_tuple in self.pass_through_columns:
            dataset[column_tuple[0]] = self.dataset[column_tuple[0]]
            dataset[column_tuple[1]] = self.dataset[column_tuple[1]]
        return dataset
                
    def calculateColumn(self, seriesA, seriesB):
        result_dict = {}
        for metric in self.metrics:
            ts = time()
            result_dict[metric] = self.calculateColumnMetric(seriesA, seriesB, metric)
            logging.info('Column: %s, %s metric took: %s seconds', seriesA.name, metric, time() - ts)
        return result_dict

    def validate(self):
        return True

    def calculateColumnMetric(self, seriesA, seriesB, metric):
        index = [e for e in range(0, seriesA.size)]
        with mp.Manager() as manager:
            result = manager.dict()
            with manager.Pool(self.workers) as p:
                p.starmap(self.calculateItemMetricAndSave, [(result, e[2], e[0], e[1], metric) for e in 
                      [(item[0], item[1], item[2]) for item in zip(seriesA, seriesB, index)]])
            return pd.Series(result)
        

    def calculateItemMetricAndSave(self, output, index, strA, strB, metric):
        logging.info('Index: %s, metric: %s, A: %s, B: %s', index, strA, strB, metric)
        output[index] = self.calculateMetric(strA, strB, metric) 

    def calculateMetric(self, strA, strB, metric): 
        if metric == 'ratio':
            return fuzz.ratio(strA, strB)
        if metric == 'partial_ratio':
            return fuzz.partial_ratio(strA, strB)
        if metric == 'token_sort_ratio':
            return fuzz.token_sort_ratio(strA, strB)
        if metric == 'token_set_ratio':
            return fuzz.token_set_ratio(strA, strB)
        if metric == 'distance':
            return Levenshtein.distance(strA, strB)
        if metric == 'l_ratio':
            return Levenshtein.ratio(strA, strB)
        if metric == 'jaro':
            return Levenshtein.jaro(strA, strB)
        if metric == 'jaro_winkler':
            return Levenshtein.jaro_winkler(strA, strB)
        if metric == 'setratio':
            return Levenshtein.setratio(strA, strB)
        if metric == 'seqratio':
            return Levenshtein.seqratio(strA, strB)
        if metric == 'longestnumericseq':
            return longestNumericSubstringMetric(strA, strB)
        return None

def longestNumericSubstringMetric(value1, value2):    
    '''Metric determines the length of longest numeric sequence in the feature
    '''
    return abs(longestNumericSequence(value2) - longestNumericSequence(value1))

def longestNumericSequence(value):    
    '''Function determines the boundaries of the longest numeric sequence
    '''
    return len(re.sub("[^\d.]+", '', value))
    
