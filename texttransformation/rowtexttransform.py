from texttransformation import StringTransform

class RowTextTransform(object):

    def __init__(self, transform_rules):
        self.transform_rules = transform_rules

    def execute(self, row):
        if row != None:
            result = []            
            column_index = 0
            for value in row:
                row_copy = row.copy()
                if column_index in self.transform_rules.keys():
                    column_rule = self.transform_rules[column_index]
                    transformer = StringTransform(column_rule)
                    row_copy[column_index] = transformer.execute(value)
                result.append(row_copy)
                column_index = column_index + 1
            return result
        return None
