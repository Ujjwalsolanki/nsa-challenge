#nsa
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class Make_Predictions:

    def __init__(self):
        self.artifacts_folder_path:Path = Path('artifacts/')
        self.training_data_path:Path = Path('training_data/')
        self.numerical_features = ["funded_amnt", "int_rate", "annual_inc", "dti", "fico_range_high", "revol_bal", 
                                  "revol_util", "bc_util", "total_bal_ex_mort", "inq_last_12m", "num_accts_ever_120_pd"]
        self.categorical_features = ["term", "home_ownership", "delinq_2yrs", "mort_acc"]

    
    def initiate_predictions(self):
        ##we will start prediction pipeline
        current_loans = pd.read_csv(os.path.join(self.training_data_path,"current_loans.csv"))

        ##for this project only, remove in version 2. I am making prediction only for sample of 100 rows
        sample_current_loans = current_loans.sample(100)

        df = pd.concat([sample_current_loans[self.numerical_features + self.categorical_features]], axis=1, ignore_index=False)

        ## data preprocessing
        #int_rate we have to remove % sign so that we can 
        df["int_rate"] = df["int_rate"].apply(lambda x: x.replace('%', ''))
        df['int_rate'] = df['int_rate'].astype('float64')

        #revol_util
        df['revol_util'] = df['revol_util'].apply(lambda x: 0 if x == 'nan' else x)
        df['revol_util'] = df['revol_util'].apply(lambda x: str(x).replace('%', ''))
        df['revol_util'] = df['revol_util'].astype('float64')

        ##here categorical data will start

        #term
        df['term'] = df['term'].apply(lambda x: 0 if x=='36 months' else 1)
        df['term'].astype('Int32')

        #home_ownership ['RENT', 'MORTGAGE', 'OWN', 'ANY'] => [0,1,2,3]
        df['home_ownership'] = df['home_ownership'].replace(['RENT', 'MORTGAGE', 'OWN', 'ANY','NONE'], [0,1,2,3,3])

        # inq_last_12m Numeric feature
        df.replace(np.nan, 0, inplace=True)
        df['inq_last_12m'] = df['inq_last_12m'].astype('int64')

        # Open the pickle file in binary read mode
        with open(os.path.join(self.artifacts_folder_path,'preprocessing_object.pickle'), 'rb') as file:
            # Load the data from the pickle file
            preprocessor = pickle.load(file)

        prediction_features = preprocessor.transform(df)

        # Open the pickle file in binary read mode
        with open(os.path.join(self.artifacts_folder_path,'best_model.pickle'), 'rb') as file:
            # Load the data from the pickle file
            best_model = pickle.load(file)

        predicted_probabilities = best_model.predict_proba(prediction_features)[:,1] # this is for the class 1

        sample_current_loans['investments'] = 0
        sample_current_loans['predictions'] = predicted_probabilities
        
        ##here your SQL code goes, which will update predictions of the all rows
        ##Since we have to large db, we can utilize batch processing using PySpark DataFrames 
        sample_current_loans.to_csv(os.path.join(self.artifacts_folder_path,'sample_loans_with_predictions.csv'),index=False)