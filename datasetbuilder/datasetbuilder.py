import pandas as pd
from sklearn import preprocessing
import re
from texttransformation import TransformDataset
from metrics import MetricsCalculator

def doLabel(a, b):
    if a == b:
        return 1
    return 0
    
def duplicatePattern(text, value):
    if len(text) > 0:
        if value in text:
            return text.replace(value, value+value+value)    
    return text 

def duplicateNumericSequence(text, factor = 2):
    '''Increase weight of numeric sequences to reduce overfitting e.g. PO BOX 34 -> PO BOX 343434
    '''
    if len(text) > 0:
        seq = re.sub("[^\d]+", '', text)
        new_seq = ''
        if len(seq) > 0:
            for i in range(1, factor + 1):
                new_seq = new_seq + seq
            return text.replace(seq, new_seq)    
    return text    

def increaseWeightOfShortWords(df, columns):  
    '''Some short words are important for the calculation
    but due to it's length might have low impact and must be expanded
    '''
    vocabulary = ['']
    for column in columns:
        for word in vocabulary:
            #print("{}-{}".format(column, word))
            df[column] = df.apply(lambda row: duplicatePattern(row[column], word), axis=1) 
    return df

def increaseWeightOfLongestNumericSequence(df, columns):
    for column in columns:
        df[column] = df.apply(lambda row: duplicateNumericSequence(row[column]), axis=1) 
    return df

class DatasetBuilder(object):

    def __init__(self, datasource_columns, datasource_index, one_hot_encoding_columns, 
        text_metrics, pass_through_columns, alteration_rules, high_importance_columns, workers, logging_level):
        self.datasource_columns = datasource_columns
        self.datasource_index = datasource_index
        self.one_hot_encoding_columns = one_hot_encoding_columns
        self.text_metrics = text_metrics
        self.pass_through_columns = pass_through_columns
        self.alteration_rules = alteration_rules        
        self.high_importance_columns = high_importance_columns
        self.workers = workers
        self.logging_level = logging_level

    def getCompleteDataset(self):
        return self.completeData

    def generateTrainingDataset(self, dataset):
        enrichedData = self.enrichData(dataset)
        alteredData = self.alterateData(enrichedData)
        self.completeData, meteredData = self.calculateMetrics(dataset, alteredData)
        labeledData = self.addLabel(meteredData)
        oneHotEncodedData = self.addOneHotEncodings(labeledData)
        self.trainingDataset = oneHotEncodedData
        return self.trainingDataset

    def generatePredictionDataset(self, dataset):
        enrichedData = self.enrichData(dataset)
        self.completeData, meteredData = self.calculateMetrics(enrichedData, enrichedData)
        oneHotEncodedData = self.addOneHotEncodings(meteredData)
        self.predictingDataset = oneHotEncodedData
        return self.predictingDataset

    def enrichData(self, dataset):
        dataset = increaseWeightOfShortWords(dataset, self.high_importance_columns)
        dataset = increaseWeightOfLongestNumericSequence(dataset, self.high_importance_columns)
        return dataset

    def alterateData(self, dataset):
        transformer = TransformDataset(self.alteration_rules)
        return pd.DataFrame(transformer.execute(dataset.values.tolist()), columns=self.datasource_columns)

    def calculateMetrics(self, dataset, altered_dataset):
        merged_dataset = self.datasetsProduct(dataset, altered_dataset)
        columns_tuples = [(col + '_x', col + '_y', col) for col in 
            list(set([e.replace('_x','').replace('_y','') for e in self.datasource_columns 
            if e not in self.pass_through_columns]))]
        pass_through_columns_tuples = [(col + '_x', col + '_y', col) for col in 
            list(set([e.replace('_x','').replace('_y','') for e in self.pass_through_columns]))]
        calculator = MetricsCalculator(
                               metrics = self.text_metrics,
                               workers = self.workers,                               
                               dataset = merged_dataset,
                               columns = columns_tuples,
                               pass_through_columns = pass_through_columns_tuples,
                               logging_level = self.logging_level)
        calculated = calculator.calculate()
        return merged_dataset, calculated

    def datasetsProduct(self, dataset1, dataset2):
        df_source_1 = dataset1.copy()
        df_target_1 = dataset2.copy()
        df_source_1['Key'] = 0
        df_target_1['Key'] = 0
        return df_source_1.merge(df_target_1, left_on = ["Key"], right_on = ["Key"]).reset_index()

    def addLabel(self, dataset):
        dataset['Label'] = dataset.apply(lambda row: 
            doLabel(row[self.datasource_index + '_x'], row[self.datasource_index + '_y']),axis=1)
        return dataset

    def addOneHotEncodings(self, dataset):
        for ehec in self.one_hot_encoding_columns.keys():
            le = preprocessing.LabelEncoder()
            le.fit(self.one_hot_encoding_columns[ehec])
            dataset[ehec + '_x'] = le.transform(dataset[ehec + '_x'])
            dataset[ehec + '_y'] = le.transform(dataset[ehec + '_y'])

        del dataset[self.datasource_index + '_x']
        del dataset[self.datasource_index + '_y'] 
        return dataset