from model_pipeline import ModelPipeline
from data_processor import DataProcessor, Mode
import pandas as pd
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
    """Performs model training and evaluation of test and validation set"""
    args = parse_arguments()

    if args.credentials_path:
        dproc = DataProcessor(
            credentials_path=args.credentials_path,
            project_id=args.project_id
        )

        df_train = dproc.query_db(Mode.TRAIN)
        df_test = dproc.query_db(Mode.TEST)
        df_val = dproc.query_db(Mode.VALIDATION)
    else:
        df_train = pd.read_csv('data/input/df_train.csv')
        df_test = pd.read_csv('data/input/df_test.csv')
        df_val = pd.read_csv('data/input/df_val.csv')

    print('Training started')
    training_pipeline = ModelPipeline(df_train, include_hour=args.include_hour)
    training_pipeline.fit()
    print('Training completed\n')

    print('Testing started')
    inference_pipeline = ModelPipeline(df_test, include_hour=args.include_hour)
    inference_pipeline.inference()
    print('Testing completed\n')

    print('Validation started')
    validation_pipeline = ModelPipeline(df_val, include_hour=args.include_hour)
    validation_pipeline.inference()
    print('Validation completed')


if __name__ == '__main__':
    main()