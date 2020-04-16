from texttransformation import StringTransform

class RowTextTransform(object):

    def __init__(self, transform_rules):
        self.transform_rules = transform_rules

    def execute(self, row):
        if row != None:
            result = []
            column_index = 0
            for value in row:
                if column_index in self.transform_rules.keys():
                    column_rule = self.transform_rules[column_index]
                    transformer = StringTransform(column_rule)
                    result.append(transformer.execute(value))
                else:
                    result.append(value)
                column_index = column_index + 1
            return result
        return None
