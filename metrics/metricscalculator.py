import numpy as np
import re
import time
from queue import Queue
import pickle
import pandas as pd
from .custommetrics import LongestNumericSequence, LongestNumericSubstringMetric
from .workerpool import WorkerPool
from .metricsconfig import MetricsConfig


class MetricsCalculator(object):

    def __init__(self, columns, pass_through_columns, metrics_config):
        self.columns = columns
        self.metrics_config = metrics_config
        self.pass_through_columns = pass_through_columns

    def applyMetrics(self, df_1, df_2, n_jobs = 1):
        df_source_1 = df_1.copy()
        df_target_1 = df_2.copy()
        df_source_1['Key'] = 0
        df_target_1['Key'] = 0
        df_merged = df_source_1.merge(df_target_1, left_on = ["Key"], right_on = ["Key"]).reset_index()
        df_metrics = self.metricCalculator(df_merged, columns = self.columns, size = n_jobs).reset_index()
        df_metrics = df_metrics.merge(df_merged[[e + '_x' for e in self.pass_through_columns] 
                                + [e + '_y' for e in self.pass_through_columns]], how = 'inner', left_index = True, right_index = True)
        df_metrics = df_metrics.drop(df_metrics.columns[df_metrics.columns.str.contains('Index',case = False)],axis = 1)
        return df_merged, df_metrics

    def metricCalculator(self, df, columns, size):
        '''Calculate metric for the given source dataframe and concat resulting metric set to it
        '''
        column_results = []
        for column in [e for e in columns if e not in self.pass_through_columns]:
            column_results.append(self.producer(df = df, size = size, column = column))
        return pd.concat(column_results, axis = 1)        

    def producer(self, df, size, column):
        queue = Queue()
        workerPool = WorkerPool(queue = queue, input_df = df, column = column, metrics_config = self.metrics_config, size = size)
        workerPool.SplitData()
        worker_threads = workerPool.BuildWorkerPool()
        start_time = time.time()

        # Add the poison pillv
        for worker in worker_threads:
            queue.put('quit')
        for worker in worker_threads:
            worker.join()

        print('Column "{}" done. Time taken: {}'.format(column, time.time() - start_time))
        workerPoolResult = workerPool.GetCombinedResult().copy()
        del workerPool
        return workerPoolResult 

