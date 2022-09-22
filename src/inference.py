from functions import ModelPipeline
from data_loader import DataProcessor
import pandas as pd

INCLUDE_HOUR = False

def main():
    dproc = DataProcessor(
        credentials_path='niologic-assessment-33f145533e28.json',
        project_id='niologic-assessment'
    )

    df_infer = dproc.query_db('inference')

    print('Inference started')
    training_pipeline = ModelPipeline(df_infer, include_hour=INCLUDE_HOUR)
    infer = training_pipeline.inference(scoring=False)
    print(len(infer))
    print('Inference completed\n')


if __name__ == '__main__':
    main()