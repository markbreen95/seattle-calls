from functions import ModelPipeline
from data_loader import DataProcessor
import pandas as pd


def main():
    dproc = DataProcessor(
        credentials_path='niologic-assessment-33f145533e28.json',
        project_id='niologic-assessment'
    )

    df_train = dproc.query_db('train')
    df_test = dproc.query_db('test')

    training_pipeline = ModelPipeline(df_train, include_hour=False)

    training_pipeline.fit()

    inference_pipeline = ModelPipeline(df_test, include_hour=False)

    inference_pipeline.inference()

    return True

if __name__ == '__main__':
    main()