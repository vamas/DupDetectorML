# DupDetectorML

Master data is the most critical component of any organization. Like any other component, master data has its own lifecycle. Information is acquired, verified, updated, archived. Not following the lifecycle introduces bad master data and increases the organization's operation cost (see [The True Cost of Duplicate Data](https://www.freshbusinessthinking.com/the-true-cost-of-duplicate-data/)). Duplicated data is one of the most common problems. 
To counteract the problem, organizations need to carry out data deduplication. This process requires a combination of domain knowledge and knowledge of data algorithms. DupDetectorML is a Python application that helps to identify master data duplicates with minimal domain knowledge using supervised machine learning algorithms.


As with any supervised machine learning algorithms, properly labeled data sample is critical to achieving good result. DupDetectorML uses a synthetic learning dataset that is a sample of a master data with artificially introduced errors and inconsistencies which can be observed in real-life scenarios. For example, two entries that have the name "Qlpha Industries, Toronto" and "Alpha Industries, Toronto" are likely the duplicates. Creating a rule that creates learning dataset entry simulating a typo - "Alpha" and "Qlpha" and labeling these two entries as a duplicate will help to detect all such cases.
Machine learning algorithms don't understand strings, so we use various text metrics to represent the difference between attributes. 

## Documentation

Folder structure
```
root
|___dupdetectorML
    |___datasetbuilder          -   dataset builder module, handles alterations, transformations, labeling etc.
    |___metrics                 -   text metric calculator class
    |___texttransformation      -   text transformation logic
    |___trainer                 -   trainer abstraction module
    |___predictor               -   predictor abstraction module
    |___helpers                 -   helper functions
    |   Training.ipynb          -   jupyter notebook example of training application
    |   Predicting.ipynb        -   jupyter notebook example of predicting application
```

Detection workflow:
```
-   Configure global parameters
Training
-   Import source dataset for training
-   Generate training data
-   Train
Predicting
-   Import source dataset for predicting
-   Generate predicting data
-   Predict
```

## Install

Copy content of the dupdetectorML to your project location. Install third-party libraries:
```bash 
pip install requirements.txt
```

## Usage

### Import trainer and predictor to you python host application:
```python
import from dupdetectorML import DatasetBuilder, Trainer, Predictor, SaveModel, RestoreModel
```

### Configure global parameters  

Define a dataset columns
```python
DATASOURCE_COLUMNS = ['Id', 'Name', 'Phone', 'Fax', 'OtherPhone', \
                       'Email', 'WebsiteURL', 'MailingAddressFreeform', \
                       'MailingAddressCity', 'MailingAddressPostalCode', 'MailingAddressState']
```

Define an index field for the dataset:
```python
DATASOURCE_INDEX = 'Id'
```

Define one-hot-encoding values enums (see [One-Hot Encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)):  
```python
ONE_HOT_ENCONDING_COLUMNS = {'MailingAddressState': ['nunavut ',
                                                        'saskatchewan',
                                                        'ontario  ',
                                                        'alberta',
                                                        'british columbia ',
                                                        'prince edward island ',
                                                        'yukon territory',
                                                        'newfoundland',
                                                        'northwest territories',
                                                        'new brunswick',
                                                        'manitoba',
                                                        'nova scotia ',
                                                        'quebec']}
```

Define columns that will be ignored during text metric calculation (this should also include all one-hot-encoding columns):
```python
PASS_THROUGH_COLUMNS = ['Id','MailingAddressState'] 
```

Define what text metrics to be calulated for each nonpass through column (adding more metrics affects performance of both training and predicting, adding less metrics may affect detection accuracy):
```python
TEXT_METRICS = ['ratio','partial_ratio','token_sort_ratio','token_set_ratio','distance',
                'l_ratio','jaro','jaro_winkler','setratio','seqratio','longestnumericseq']                      
```

If you want you can define the critical attributes. Weight of these column will be higher during detection:                     
```python
HIGH_IMPORTANCE_COLUMNS = ['Name']
```
Define rules for creating the synthetic learning data for each column in DATASOURCE_COLUMNS you want alterations to be simulated. That is where some domain knowledge is still required:
```python
ALTERATION_RULES = {
    1: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    2: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    3: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    4: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    5: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    7: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },
    8: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    },  
    9: {
        'rule_Replace': ['none',''],
        'rule_RandomTypo': ['alpha', 2, 'replace'],
        'rule_ScrambleWords': [],
        'rule_DuplicateNumericSequence': [2],
        'rule_RemoveSpecialSymbols': [],
        'rule_RemoveStopWords': [],
        'rule_IncreaseWeightOfShortWords':[]
    }           
}
```

### Training
Import dataset and perform initial transformations:
```python
df_source = pd.read_csv('train_input.csv', header=0, names=DATASOURCE_COLUMNS)
df_source.Id = df_source.Id.astype(str).str.lower()
df_source.Name = df_source.Name.astype(str).str.lower()
df_source.Phone = df_source.Phone.astype(str).str.lower()
df_source.Fax = df_source.Fax.astype(str).str.lower()
df_source.OtherPhone = df_source.OtherPhone.astype(str).str.lower()
df_source.Email = df_source.Email.astype(str).str.lower()
df_source.WebsiteURL = df_source.WebsiteURL.astype(str).str.lower()
df_source.MailingAddressFreeform = df_source.MailingAddressFreeform.astype(str).str.lower()
df_source.MailingAddressCity = df_source.MailingAddressCity.astype(str).str.lower()
df_source.MailingAddressPostalCode = df_source.MailingAddressPostalCode.astype(str).str.lower()
df_source.MailingAddressState = df_source.MailingAddressState.astype(str).str.lower()
```

Instantiate an instance of DatasetBuilder class providing configuration parameters. Two extra parameters are workers and logging which define number of parallel processes when calculating metrics (roughly number of cores of your CPU) and logging level (see [Python logging](https://docs.python.org/2/library/logging.html)):
```python
builder = DatasetBuilder(DATASOURCE_COLUMNS, DATASOURCE_INDEX, 
                         ONE_HOT_ENCONDING_COLUMNS, TEXT_METRICS, 
                         PASS_THROUGH_COLUMNS, COLUMN_ALTERATIION_RULES,
                         HIGH_IMPORTANCE_COLUMNS, workers=8, logging=logging.DEBUG)
```

Create syntethic dataset for learning:
```python
training_df = builder.generateTrainingDataset(df_source)
```

Instantiate a trainer class, perform training and save the model in the file:
```python
trainer = Trainer(training_df, 0.2)
SaveModel(trainer.execute(), "TrainedModel.sav")
```

### Prediction
Import two datasets with the same structure as the training dataset above, one is defining what entries to find duplicates for, another - where to search for duplicates. Instantiate an instance of DatasetBuiler class (see above) and create predicting dataset:
```python
predicting_df = builder.generatePredictionDataset(df_what_to_search, df_where_to_search)
```

Restore a machine learning model from file and run a prediction against predicting dataset:
```python
model = RestoreModel("TrainedModel.sav")
predictor = Predictor(model)
predictor.execute(predicting_df, builder.getCompleteDataset())
```

The result is available via result of getPrediction() method of Predictor class combined with dataset returned by getCompleteDataset() method of DatasetBuilder:
```python
matches = builder.getCompleteDataset()[
    [e + '_x' for e in DATASOURCE_COLUMNS] + [e + '_y' for e in DATASOURCE_COLUMNS]].iloc[predictor.getPrediction().index]
```

## License
DupDetectorML is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Feel free to do whatever you want with it.

    
