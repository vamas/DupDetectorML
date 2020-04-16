import re
    
def LongestNumericSequence(value):    
    '''Function determines the boundaries of the longest numeric sequence
    '''
    return len(re.sub("[^\d.]+", '', value))

def LongestNumericSubstringMetric(value1, value2):    
    '''Metric determines the length of longest numeric sequence in the feature
    '''
    return abs(LongestNumericSequence(value2) - LongestNumericSequence(value1))
