from functions import ModelPipeline
from data_loader import DataProcessor
import pandas as pd
import time

INCLUDE_HOUR = False

def main():
    """
        Performs model inference using new dataset

        Option 1: query the DB to get inference data
        - Will not work for you unless you have credentials
          or else load the data somewhere else

        Option 2: Manually create inference DF

        Option 3: load inference data from csv file
    """
    # Option 1:
    # dproc = DataProcessor(
    #     credentials_path='niologic-assessment-33f145533e28.json',
    #     project_id='niologic-assessment'
    # )
    # df_infer = dproc.query_db('inference')
    
    # Option 2:
    # df_infer = pd.DataFrame(
    #     {'year': [], 
    #     'month': [],
    #     'day': [],
    #     'hour': [],
    #     'Longitude': [],
    #     'Latitude': []
    #     }
    # )

    # Option 3:
    df_infer = pd.read_csv('data/input/inference_data.csv')

    print('Inference started')
    inference_pipeline = ModelPipeline(df_infer, include_hour=INCLUDE_HOUR)
    infer = inference_pipeline.inference(scoring=False)
    timestr = time.strftime('%Y%m%d-%H%M%S')
    pd.DataFrame(infer).to_csv('data/output/inference_{}.csv'.format(timestr), 
    index=None)
    print('Inference completed\n')


if __name__ == '__main__':
    main()