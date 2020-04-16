from texttransformation import RowTextTransform

class TransformDataset(object):

    def __init__(self, transform_rules):
        self.transform_rules = transform_rules


    def executeRuleset(self, dataset, ruleset):
        '''Execute single ruleset
        dataset: list of rows
        '''
        result = []
        if dataset != None:
            rowTransformer = RowTextTransform(ruleset)
            for row in dataset:
                result.append(rowTransformer.execute(row))
        return result
            
    def execute(self, dataset):
        '''Performs text transformations for all rows applying all rulesets
        dataset: list of rows
        '''
        result = []
        for ruleset_index in range(0, len(self.transform_rules)):
            result = result + self.executeRuleset(dataset, self.transform_rules[ruleset_index])
        return result


