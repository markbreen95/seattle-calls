from functions import ModelPipeline
from data_loader import DataProcessor
import pandas as pd

INCLUDE_HOUR = False

def main():
    dproc = DataProcessor(
        credentials_path='niologic-assessment-33f145533e28.json',
        project_id='niologic-assessment'
    )

    df_train = dproc.query_db('train')
    df_test = dproc.query_db('test')
    df_val = dproc.query_db('validation')

    print('Training started')
    training_pipeline = ModelPipeline(df_train, include_hour=INCLUDE_HOUR)
    training_pipeline.fit()
    print('Training completed\n')

    print('Testing started')
    inference_pipeline = ModelPipeline(df_test, include_hour=INCLUDE_HOUR)
    inference_pipeline.inference()
    print('Testing completed\n')

    print('Validation started')
    validation_pipeline = ModelPipeline(df_val, include_hour=INCLUDE_HOUR)
    validation_pipeline.inference()
    print('Validation completed')


if __name__ == '__main__':
    main()