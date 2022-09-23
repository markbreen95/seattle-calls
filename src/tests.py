from data_processor import DataProcessor, Mode
from return_cluster import ReturnCluster
from model_pipeline import ModelPipeline
import pandas as pd
import numpy as np
import random

class TestDataProcessor:
    def test_db_query(self):
        dproc = DataProcessor(
            credentials_path='',
            project_id=''
        )
        df = dproc.query_db(Mode.TEST)
        assert df.shape[0] == 106748, 'Incorrect number of rows in dataframe'


class TestModellingPipeline:
    def test_inference(self):
        df = pd.DataFrame(
            {'year': [2020, 2021],
            'month': [1, 2],
            'day': [10, 11],
            'Longitude': [-122.375, -122.293490],
            'Latitude': [47.549467, 47.658445]}
        )
        inference_pipeline = ModelPipeline(df, include_hour=False)
        infer = inference_pipeline.inference(scoring=False)

        assert len(infer) == df.shape[0], 'Inference return not correct'

    def test_clustering(self):
        X = np.random.rand(100, 1)
        clust = ReturnCluster()
        clust.fit(X)
        ret = clust.transform(X)
        assert len(np.unique(ret)) == 5, 'Incorrect number of clusters returned'