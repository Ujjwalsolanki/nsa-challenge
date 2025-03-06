#nsa
from pathlib import Path
import pandas as pd
from logger import logging
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

class DataPreprocessing():
    def __init__(self):
        ##this all values come from YAML file or config file in version 2
        self.artifacts_folder_path:Path = Path('artifacts/')
        self.training_data_path:Path = Path('training_data/')
        self.numerical_features = ["funded_amnt", "int_rate", "annual_inc", "dti", "fico_range_high", "revol_bal", 
                                  "revol_util", "bc_util", "total_bal_ex_mort", "inq_last_12m", "num_accts_ever_120_pd"]
        self.categorical_features = ["term", "home_ownership", "delinq_2yrs", "mort_acc"]

        os.makedirs(self.artifacts_folder_path, exist_ok=True)
        os.makedirs(self.training_data_path, exist_ok=True)

    def initiate_data_preprocessing(self):
        
        try:
            logging.info("data preparation started")
            data = pd.read_csv(os.path.join(self.training_data_path, 'data.csv'))
            logging.info(f"Shape of all the data: {data.shape}")

            current_loans = data[data["loan_status"] == "Current"]
            if current_loans.shape[0] > 0:
                current_loans.to_csv(os.path.join(self.training_data_path,'current_loans.csv'), index=False)
            
            df = data[data["loan_status"] != "Current"]
            logging.info(f'These are the columns of the files: {df.columns}')

            #here are defining funciton for good loan and bad loan
            # 1 for good loan and 0 for bad loan

            df["loan_status"] = df["loan_status"].apply(lambda x: 1 if x=="Fully Paid" else 0)
            
            #first we will decide which columns are required and which columns are not
            #this can be used for column mapping thru evedently for future data drifts
            target = "loan_status"
            prediction = "prediction"

            #removed unwanted columns 
            df = pd.concat([df[self.numerical_features + self.categorical_features], df[["loan_status"]]], axis=1, ignore_index=False)

            #int_rate we have to remove % sign so that we can 
            df["int_rate"] = df["int_rate"].apply(lambda x: x.replace('%', ''))
            df['int_rate'] = df['int_rate'].astype('float64')
            logging.info('int_rate column transformed')

            #annual_inc columns has zeros negative numbers and outliers
            df= df[df['annual_inc'] != 0]

            ##now we are dealing with outliers
            # Define bounds for outliers, as 1.5 * IQR is not acceptable based on data, we have to take 99 percentile values 
            upper_bound = df['annual_inc'].quantile(0.99)
            df = df[(df['annual_inc'] < upper_bound)]
            logging.info(f'value counts for customer above $250000 : {df[(df['annual_inc'] > upper_bound)].value_counts().sum()}')
            ##since value counts are not that bigger and may impact on our data because of high income, we will remove those outliers

            ##dti column
            #removing outlies below zeros
            df= df[df['dti'] > 0]
            # Define bounds for outliers
            upper_bound = df['dti'].quantile(0.99)
            df = df[(df['dti'] < upper_bound)]

            ##fico_range_high It is good at the moment but in version 2.0 implement box-cox transformation

            #revol_bal
            Q1 = df['revol_bal'].quantile(0.25)
            Q3 = df['revol_bal'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            # Define bounds for outliers
            df = df[(df['dti'] < upper_bound)]

            #revol_util
            df['revol_util'] = df['revol_util'].apply(lambda x: 0 if x == 'nan' else x)
            df['revol_util'] = df['revol_util'].apply(lambda x: str(x).replace('%', ''))
            df['revol_util'] = df['revol_util'].astype('float64')

            #bc_util
            upper_bound = df['bc_util'].quantile(0.99)
            df = df[(df['bc_util'] < upper_bound)]
                       
            #total_bal_ex_mort
            Q1 = df['total_bal_ex_mort'].quantile(0.25)
            Q3 = df['total_bal_ex_mort'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df['total_bal_ex_mort'] < upper_bound)]

            ##here categorical data will start

            #term
            df['term'] = df['term'].apply(lambda x: 0 if x=='36 months' else 1)
            df['term'].astype('Int32')

            #home_ownership ['RENT', 'MORTGAGE', 'OWN', 'ANY'] => [0,1,2,3]
            df['home_ownership'] = df['home_ownership'].replace(['RENT', 'MORTGAGE', 'OWN', 'ANY'], [0,1,2,3])
            
            # delinq_2yrs
            Q1 = df['delinq_2yrs'].quantile(0.25)
            Q3 = df['delinq_2yrs'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = df['delinq_2yrs'].quantile(0.999) #again we have to take .999 percentile to accomodate reasonable values
            df = df[(df['delinq_2yrs'] < upper_bound)]

            # inq_last_12m Numeric feature
            df.replace(np.nan, 0, inplace=True)
            df['inq_last_12m'] = df['inq_last_12m'].astype('int64')

            # mort_acc
            Q1 = df['mort_acc'].quantile(0.25)
            Q3 = df['mort_acc'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['mort_acc'] < upper_bound)]

            #num_accts_ever_120_pd
            Q1 = df['num_accts_ever_120_pd'].quantile(0.25)
            Q3 = df['num_accts_ever_120_pd'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = df['num_accts_ever_120_pd'].quantile(0.99)
            df = df[(df['num_accts_ever_120_pd'] < upper_bound)]


            self.initiate_data_transformation(df)

        except Exception as e:
            logging.exception(e)
            raise(e)
        

    def get_data_transformer_object(self):
        num_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )
        
        cat_pipeline = Pipeline(
            steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, self.numerical_features),
                ("cat_pipeline", cat_pipeline, self.categorical_features)
            ]
        )

        return preprocessor
    
    def initiate_data_transformation(self, df):
        #before we start transformation, we will create train, test split
        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        logging.info(f'X_train.shape, X_test.shape : {X_train.shape, X_test.shape}')

        # since target variable is imbalance
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=0,
            k_neighbors=5,
        )

        X_train_resampled, target_resampled = smote.fit_resample(X_train, y_train)

        logging.info(f'X_train_resampled.shape, target_resampled.shape : {X_train_resampled.shape, target_resampled.shape}')

        preprocessing_object=self.get_data_transformer_object()
        input_feature_train_arr=preprocessing_object.fit_transform(X_train_resampled)
        input_feature_test_arr=preprocessing_object.transform(X_test)

        ##save train and test csv and also save preprocessingobject for prediction pipeline
        ##below code also you can save in utils folders tosave files

        train_arr = np.c_[input_feature_train_arr, np.array(target_resampled)]
        test_arr = np.c_[input_feature_test_arr, np.array(y_test)]
        
        np.savetxt(os.path.join(self.training_data_path,'train.csv'), train_arr, delimiter=',')
        np.savetxt(os.path.join(self.training_data_path,'test.csv'), test_arr, delimiter=',')

        logging.info("Train and test csv files are save")

        ##Below funsiton will save preprocessor to pickle file
        
        with open(os.path.join(self.artifacts_folder_path,"preprocessing_object.pickle"), "wb") as file:
            pickle.dump(preprocessing_object, file)
        logging.info("save preprocessor to pickle file")

        