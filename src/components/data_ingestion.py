import os
import pandas as pd
from pathlib import Path
from logger import logging

class DataIngestion():
    def __init__(self):
        self.data_path:Path = Path('data')
        self.archive_folder_path:Path = Path('archive')
        self.training_data_path:Path = Path('training_data')
        os.makedirs(self.archive_folder_path, exist_ok=True)
        os.makedirs(self.training_data_path, exist_ok=True)

    

    def create_master_data_file(self):
        try:
            logging.info("data ingestion started")
            files = [f for f in os.listdir(self.data_path)]

            if files:
                df_list = []
                for file in files:
                    file_path = os.path.join(self.data_path , file)
                    
                    df = pd.read_csv(file_path, compression='gzip')
                    
                    df_list.append(df)
                    ##attention: I have commented below line because due to training phase
                    # shutil.move(file_path, self.archive_folder_path)
                    logging.info(f"File '{str(file_path)}' moved to '{str(self.archive_folder_path)}'")

                big_df = pd.concat(df_list)

                big_df.to_csv(os.path.join(self.training_data_path, "data.csv"), index=False)
            
            else:
                logging.exception("No files found")
                raise(Exception)
        except Exception as e:
            logging.exception(e)
            raise(e)
        


