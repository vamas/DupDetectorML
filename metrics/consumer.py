import threading
import time
import numpy as np
import Levenshtein
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from .custommetrics import LongestNumericSequence, LongestNumericSubstringMetric

class Consumer(threading.Thread): 

    def __init__(self, task, queue, df, metrics_config, results): 
        threading.Thread.__init__(self)
        self._queue = queue
        self.df = df.copy()
        self.metrics_config = metrics_config
        self.task = task
        self.results = results
        
    def run(self):
        start_time = time.time()      
        try:
            #print('Thread {} is started at {}'.format(self.task, start_time))
            if 'Ratio' in self.metrics_config.getMetrics():        
                self.df['Ratio'] = self.df.apply(lambda row: fuzz.ratio(row['Source'], row['Target']), axis=1)
            if 'PartialRatio' in self.metrics_config.getMetrics():        
                self.df['PartialRatio'] = self.df.apply(lambda row: fuzz.partial_ratio(row['Source'], row['Target']), axis=1)  
            if 'TokenSortRatio' in self.metrics_config.getMetrics():        
                self.df['TokenSortRatio'] = self.df.apply(lambda row: fuzz.token_sort_ratio(row['Source'], row['Target']), axis=1) 
            if 'TokenSetRatio' in self.metrics_config.getMetrics():        
                self.df['TokenSetRatio'] = self.df.apply(lambda row: fuzz.token_set_ratio(row['Source'], row['Target']), axis=1)   
            if 'distance' in self.metrics_config.getMetrics():            
                self.df['distance'] = self.df.apply(lambda row: Levenshtein.distance(row['Source'], row['Target']), axis=1) 
                self.df['distance'] = self.df['distance'].astype(np.float16)
            if 'ratio' in self.metrics_config.getMetrics():        
                self.df['ratio'] = self.df.apply(lambda row: Levenshtein.ratio(row['Source'], row['Target']), axis=1) 
                self.df['ratio'] = self.df['ratio'].astype(np.float32)
            if 'jaro' in self.metrics_config.getMetrics():        
                self.df['jaro'] = self.df.apply(lambda row: Levenshtein.jaro(row['Source'], row['Target']), axis=1) 
                self.df['jaro'] = self.df['jaro'].astype(np.float32)
            if 'jaro_winkler' in self.metrics_config.getMetrics():        
                self.df['jaro_winkler'] = self.df.apply(lambda row: Levenshtein.jaro_winkler(row['Source'], row['Target']), axis=1)    
                self.df['jaro_winkler'] = self.df['jaro_winkler'].astype(np.float32)
            if 'setratio' in self.metrics_config.getMetrics():        
                self.df['setratio'] = self.df.apply(lambda row: Levenshtein.setratio(row['Source'], row['Target']), axis=1)    
                self.df['setratio'] = self.df['setratio'].astype(np.float32) 
            if 'seqratio' in self.metrics_config.getMetrics():        
                self.df['seqratio'] = self.df.apply(lambda row: Levenshtein.seqratio(row['Source'], row['Target']), axis=1)    
                self.df['seqratio'] = self.df['seqratio'].astype(np.float32) 
            if 'longestnumericseq' in self.metrics_config.getMetrics():        
                self.df['longestnumericseq'] = self.df.apply(lambda row: LongestNumericSubstringMetric(row['Source'], row['Target']), axis=1)
                self.df['longestnumericseq'] = self.df['longestnumericseq'].astype(np.float16) 
            self.results[self.task] = self.df    
        except Exception:
            print('Error calculating metrics')
        end_time = time.time()   
        