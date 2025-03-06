#nsa
from logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.make_predictions import Make_Predictions
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    
    STAGE_NAME = "Data Ingestion stage - Creating Master csv"

    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        object = DataIngestion()
        object.create_master_data_file()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
    
    STAGE_NAME = "Data Preprocessing"

    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        object = DataPreprocessing()
        object.initiate_data_preprocessing()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e

    STAGE_NAME = "Model Training"

    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainer()
        obj.initiate_training()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
    
    STAGE_NAME = "Make Predictions"

    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = Make_Predictions()
        obj.initiate_predictions()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e