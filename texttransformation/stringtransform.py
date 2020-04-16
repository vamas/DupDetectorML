import re
import random
from nltk.corpus import stopwords

def replace_str_index(text, index=0, replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def duplicatePattern(text, value):
    if len(text) > 0:
        if value in text:
            return text.replace(value, value+value+value)    
    return text   

class StringTransform(object):

    def __init__(self, transform_rules):
        self.transform_rules = transform_rules
        self.vocabulary = ['']
        self.value = ''

    def validate(self):
        return True

    def execute(self, value):
        self.value = value
        for rule in self.transform_rules.keys():
            args = self.transform_rules[rule]
            self.value = getattr(self, rule)(*args)
        return self.value
    
    def allTransformRules(self):
        rules = [func for func in dir(StringTransform) if callable(getattr(StringTransform, func)) and func.startswith('rule_')]
        return rules

    # def rule_RandomTypo(self, source_set_id, transform_count, method):
    def rule_RandomTypo(self, *param):
        ''' 
        Transformation function that alters input string by adding artificial typos    
        param[0]: string, type of dataset will be used to inject typos             
            'alpha': only letters
            'digits': only numbers
            'any': both        
        param[1]: int, number of transformations to be added 
        param[2]: string, typo injection method
            'add': add symbol
            'replace': replace symbol
            'delete': delete symbol (typo_source_set is ignored)
        '''
        source_set_id = param[0]
        transform_count = param[1]
        method = param[2]
        if self.value == '':
            return self.value
        if source_set_id == 'alpha':
            typo_source = 'abcdefghijklmnopqrstuvwxyz'
        elif source_set_id == 'digits':
            typo_source = '0123456789'
        else:
            typo_source = 'abcdefghijklmnopqrstuvwxyz0123456789'
            
        for i in range(1, transform_count + 1):                
            position = random.randint(0, len(self.value) - 1)        
            if method == 'delete':
                self.value = self.value[:position] + self.value[position + 1:]
            else:            
                symbol = random.choice(typo_source)
                if method == 'replace':
                    self.value = replace_str_index(self.value, index = position, replacement = symbol)
                if method == 'add':
                    self.value = self.value[:position] + symbol + self.value[position + 1:]
        return self.value

    def rule_ScrambleWords(self):
        '''Scramble words in the sentence that represents input attribute value        
        '''
        words = self.value.split()
        random.shuffle(words)
        self.value = ' '.join(words)
        return self.value

    def rule_DuplicateNumericSequence(self, *param):
        '''Increase weight of numeric sequences to reduce overfitting e.g. PO BOX 34 -> PO BOX 343434
        param[0]: int, number of repetitions performed on sequence
        '''
        factor = param[0]
        if len(self.value) > 0:
            seq = re.sub("[^\d]+", '', self.value)
            new_seq = ''
            if len(seq) > 0:
                for i in range(1, factor + 1):
                    new_seq = new_seq + seq
                return self.value.replace(seq, new_seq)    
        return self.value

    def rule_RemoveSpecialSymbols(self):
        '''Remove special symbols from input value
        '''
        self.value = self.value.replace(".", "")
        self.value = self.value.replace(",", "")
        self.value = self.value.replace("-", "")
        self.value = self.value.replace("/", "")
        self.value = re.sub('\[.*?\]', '', self.value)
        self.value = re.sub('<.*?>', '', self.value)
        self.value = re.sub('\(.*?\)', '', self.value)
        return self.value

    def rule_RemoveStopWords(self):
        '''Remove stop words from input value
        '''
        stop = set(stopwords.words('english'))
        stop.add('inc')
        stop.add('ltd')
        stop.add('limited')
        stop.add('la')
        words = [i for i in self.value.split() if i not in stop]
        return ''.join(word for word in words)

    def rule_IncreaseWeight(self, *param):
        '''Increase weight of the feature by duplicating it's value
        e.g. orgname may have higher weigh than other features
        so duplicting its value makes it to stand out
        param[0]: string substring which will be duplicated
        '''
        text = param[0]
        self.value = duplicatePattern(self.value, text)

    def rule_IncreaseWeightOfShortWords(self):
        '''Some short words are important for the calculation
        but due to it's length might have low impact and must be expanded
        '''        
        for word in self.vocabulary:
            self.value = self.value = duplicatePattern(self.value, word)
        return self.value

    def rule_Replace(self, *param):
        '''Replace substring with new value
        param[0]: string - value idicating that entire string has to be replaced with new value
        param[1]: string - substring to replace found substring with
        '''        
        if self.value == param[0]:
            self.value = param[1]
        return self.value
