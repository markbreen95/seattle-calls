from model_pipeline import ModelPipeline
from data_processor import DataProcessor, Mode
import pandas as pd
import time
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_hour', action='store_true')
    parser.add_argument(
        '--credentials_path', type=str, help='Path to google cloud credentials json'
    )
    parser.add_argument(
        '--project_id', type=str, help='project id where data is stored in BigQuery'
    )
    arguments = parser.parse_args()
    return arguments

def main():
    """
        Performs model inference using new dataset

        Option 1: query the DB to get inference data
        - Will not work for you unless you have credentials
          or else load the data somewhere else

        Option 2: load inference data from csv file
    """
    args = parse_arguments()

    if args.credentials_path:
        dproc = DataProcessor(
            credentials_path=args.credentials_path,
            project_id=args.project_id
        )
        df_infer = dproc.query_db(Mode.INFERENCE)
    else:
        df_infer = pd.read_csv('data/input/inference_data.csv')

    print('Inference started')
    inference_pipeline = ModelPipeline(df_infer, include_hour=args.include_hour)
    infer = inference_pipeline.inference(scoring=False)
    timestr = time.strftime('%Y%m%d-%H%M%S')
    pd.DataFrame(infer).to_csv('data/output/inference_{}.csv'.format(timestr), 
    index=None)
    print('Inference completed\n')


if __name__ == '__main__':
    main()