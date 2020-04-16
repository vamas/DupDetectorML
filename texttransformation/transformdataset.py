from texttransformation import RowTextTransform

class TransformDataset(object):

    def __init__(self, transform_rules):
        self.transform_rules = transform_rules

    def execute(self, dataset):
        '''Performs text transformations for all rows
        dataset: list of rows
        '''
        result = []
        if dataset != None:
            rowTransformer = RowTextTransform(self.transform_rules)
            for row in dataset:
                result.append(rowTransformer.execute(row))
        return result
            


