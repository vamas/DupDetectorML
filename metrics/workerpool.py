from .consumer import Consumer

import os
import numpy as np
import pandas as pd

class WorkerPool:

    def __init__(self, queue, input_df, column, metrics_config, size): 
        self.queue = queue
        self.input_df = input_df[[column+'_x',column+'_y']]
        self.column = column
        self.metrics_config = metrics_config
        self.size = size
        self.results_out = [{} for x in range(0, size)]
        self.task_dict = [{} for x in range(0, size)]
    def __enter__(self):
            return self
        
    def GetCombinedResult(self):
        return self.CombineData()
    
    def __exit__(self, exc_type, exc_value, traceback):
        os.unlink(self.task_dict)
        
    def SplitData(self):
        index_task = dict([(x, x % self.size) for x in self.input_df.index.values])        
        for task in range(0, self.size):
            self.task_dict[task] = [k for k, v in index_task.items() if v == task] 
        column_src = self.column + '_x'
        column_dest = self.column + '_y'
        worker_df = {}
        for task in range(0, self.size):
            df = self.input_df.iloc[self.task_dict[task]][[column_src, column_dest]]
            df = df.rename(index=str, columns={column_src: 'Source', column_dest: 'Target'})
            worker_df[task] = df
        self.worker_df = worker_df
        return worker_df   
    
    def BuildWorkerPool(self):
        workers = []    
        for task in range(0, self.size):
            worker = Consumer(task, self.queue, self.worker_df[task], self.metrics_config, self.results_out)    
            worker.start() 
            workers.append(worker)
        self.workers = workers
        return workers
    
    def CombineData(self):
        final_result_df = pd.concat(self.results_out)
        final_result_df.index = final_result_df.index.astype(np.int64)
        final_result_df = final_result_df.sort_index()
        final_result_df = final_result_df.drop(['Source', 'Target'], axis=1)
        new_names = self.metrics_config.getMetricNames(self.column)
        final_result_df = final_result_df.rename(index=str, columns=new_names)
        return final_result_df