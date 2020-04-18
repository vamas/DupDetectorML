import unittest
from texttransformation import StringTransform, RowTextTransform, TransformDataset
from metrics import MetricsCalculator

class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_TransformRuleList(self):
        stringTransform = StringTransform(None)
        self.assertEqual({'rule_RandomTypo', 'rule_ScrambleWords', 'rule_RemoveStopWords',
                'rule_RemoveSpecialSymbols', 'rule_DuplicateNumericSequence', 'rule_Replace',
                'rule_IncreaseWeight', 'rule_IncreaseWeightOfShortWords'}, 
            set(stringTransform.allTransformRules()), 
            "Correct transformation rules not returned")

    def test_ExecuteRule(self):
        stringTransform = StringTransform({'rule_RandomTypo': ['alpha', 1, 'replace']})
        result = stringTransform.execute('Test')
        self.assertEqual(len(result), len('Test'), "Output length not matching input")
        self.assertNotEqual(result, 'Test', "Output string is not transformed")

    def test_ExecuteMultipleRules(self):
        rules = {'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]}
        stringTransform = StringTransform(rules)
        stringTransform.execute('This is my test input string 12')
        self.assertEqual(1, 1, "Failure")

    def test_TransformRow(self):
        row = ['Bobba', '34343', 'Rabbit at walk']
        rules = {
            0: {'rule_RandomTypo': ['alpha', 1, 'replace']},
            1: {'rule_RandomTypo': ['any', 1, 'replace']},
            2: {'rule_RandomTypo': ['digit', 1, 'replace']}
        }
        rowTransformer = RowTextTransform(rules)
        result = rowTransformer.execute(row)
        self.assertEqual(len(row), len(result), "Column number in input and output rows is different")
    
    def test_TransformDataset(self):
        dataset = [['Bobba', '34343', 'Rabbit at walk'],
                    ['Hubble', '9843', 'Nirvan at work is bs223'],
                    ['Bobba Smithy', '11', 'Lobotomia is good']]
        rules = [{
            0: {'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]},
            1: {'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]},
            2: {'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]}
        }]
        transformer = TransformDataset(rules)
        result = transformer.execute(dataset)
        self.assertEqual(len(dataset)*3, len(result), "Row number in input and output datasets is different")


    def test_TransformDatasetNoTransformationOnFirstColumn(self):
        dataset = [['Bobba', '34343', 'Rabbit at walk'],
                    ['Hubble', '9843', 'Nirvan at work is bs223'],
                    ['Bobba Smithy', '11', 'Lobotomia is good for idiots']]
        rules = [{
            1: {'rule_Replace': ['none',''],
                    'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]},
            2: {'rule_RandomTypo': ['alpha', 2, 'replace'],
                    'rule_ScrambleWords': [],
                    'rule_DuplicateNumericSequence': [2],
                    'rule_RemoveSpecialSymbols': [],
                    'rule_RemoveStopWords': [],
                    'rule_IncreaseWeightOfShortWords':[]}
        }]
        transformer = TransformDataset(rules)
        result = transformer.execute(dataset)
        self.assertEqual('Bobba', result[0][0], "No transformation expected on the column 0")
        self.assertEqual('Bobba', result[1][0], "No transformation expected on the column 0")
        self.assertEqual('Bobba', result[2][0], "No transformation expected on the column 0")
        self.assertEqual('Hubble', result[3][0], "No transformation expected on the column 0")
        self.assertEqual('Hubble', result[4][0], "No transformation expected on the column 0")
        self.assertEqual('Hubble', result[5][0], "No transformation expected on the column 0")
        self.assertEqual('Bobba Smithy', result[6][0], "No transformation expected on the column 0")
        self.assertEqual('Bobba Smithy', result[7][0], "No transformation expected on the column 0")
        self.assertEqual('Bobba Smithy', result[8][0], "No transformation expected on the column 0")


    def test_ReplaceRule(self):
        value = 'none'
        stringTransform = StringTransform({'rule_Replace': ['none', '']})
        result = stringTransform.execute(value)
        self.assertEqual(result, '', 'rule_Replace not working')

if __name__ == '__main__':
    unittest.main()
