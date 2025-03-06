#nsa
import os
from pathlib import Path
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from logger import logging
import pickle

class ModelTrainer():
    def __init__(self):
        ##this all values come from YAML file or config file in version 2
        self.artifacts_folder_path:Path = Path('artifacts/')
        self.training_data_path:Path = Path('training_data/')
        
        os.makedirs(self.artifacts_folder_path, exist_ok=True)

    def initiate_training(self):
        
        logging.info("Model training started")
        train_array = np.loadtxt(os.path.join(self.training_data_path,'train.csv'), delimiter=',')
        test_array = np.loadtxt(os.path.join(self.training_data_path,'test.csv'), delimiter=',')

        X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=5)

        # Train a new model using the best parameters
        best_model = RandomForestClassifier(random_state=42, **study.best_params)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        logging.info(f"test_accuracy: {test_acc}")
        logging.info(f"F1_Score: {f1score}")

        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, 
            y_pred, 
            average='binary'
        )

        logging.info(f"test_accuracy:   {test_acc}")
        logging.info(f"test_precision:  {test_precision}")
        logging.info(f"test_recall:     {test_recall}")
        logging.info(f"test_f1_score:   {test_f1}")


        with open(os.path.join(self.artifacts_folder_path,"best_model.pickle"), "wb") as file:
            pickle.dump(best_model, file)
        logging.info("best model to pickle file")

        
    #trainig started
    def objective(self, trial, X_train, y_train):
        # Number of trees in random forest
        n_estimators = trial.suggest_int(name="n_estimators", low=100, high=500, step=100)

        # Number of features to consider at every split
        max_features = trial.suggest_categorical(name="max_features", choices=['log2', 'sqrt']) 

        # Maximum number of levels in tree
        max_depth = trial.suggest_int(name="max_depth", low=10, high=110, step=20)

        # Minimum number of samples required to split a node
        min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=10, step=2)

        # Minimum number of samples required at each leaf node
        min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
        
        params = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        model = RandomForestClassifier(random_state=42, **params)
        
        cv_score = cross_val_score(model, X_train, y_train, n_jobs=4, cv=5)
        mean_cv_accuracy = cv_score.mean()
        return mean_cv_accuracy

    

