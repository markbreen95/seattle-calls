from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import joblib
import time
from return_cluster import ReturnCluster
import numpy as np

CLUSTER_PIPELINE_PKL_PATH = 'pickled_objects/cluster_pipeline_{}.pkl'
REG_PIPELINE_PKL_PATH = 'pickled_objects/regressor_{}.pkl'


class ModelPipeline():
    """
    A class to allow clusters to be returned in the sklearn pipeline.

    ...

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe that will be used in the model pipeline
    include_hour : bool, optional
        Whether or not to include the hour column in model fitting

    """

    def __init__(self, df, include_hour=True):
        """
        Constructs all the necessary attributes for the model pipeline object.

        Parameters
        ----------
            df : pd.DataFrame
                The dataframe that will be used in the model pipeline
            include_hour : bool, optional
                Whether or not to include the hour column in model fitting
        """
        self.df = df
        self.include_hour = include_hour

    def fit(self):
        """
        Fits the regression pipeline to supplied dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        start = time.time()
        df = self.df

        cluster_pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('cluster', ReturnCluster(n_clusters=6))
        ])

        clusters = cluster_pipeline.fit_transform(df[['Longitude', 'Latitude']])

        joblib.dump(cluster_pipeline, CLUSTER_PIPELINE_PKL_PATH.format(self.include_hour))

        self.clusters = clusters

        X, y  = self.prepare_data(scoring=True)

        reg = HistGradientBoostingRegressor()
        reg.fit(X, y)

        joblib.dump(reg, REG_PIPELINE_PKL_PATH.format(self.include_hour))
    
        print('Training R2: {:.2f}'.format(reg.score(X, y)))
        print('Time taken: {:.2f}s'.format(time.time()-start))

    def inference(self, scoring=True) -> np.array:
        """
        Performs inference on a dataframe.

        Parameters
        ----------
        scoring : bool, optional
            Whether or not to return scoring metrics (false for new data)

        Returns
        -------
        np.array
            The predicted number of calls vector for the input data supplied
        """
        start = time.time()
        df = self.df

        cluster_pipeline = joblib.load(CLUSTER_PIPELINE_PKL_PATH.format(self.include_hour))

        clusters = cluster_pipeline.transform(df[['Longitude', 'Latitude']])

        self.clusters = clusters

        X, y = self.prepare_data(scoring)

        reg = joblib.load(REG_PIPELINE_PKL_PATH.format(self.include_hour))
        
        if scoring:
            print('Inference R2: {:.2f}'.format(reg.score(X, y)))
        print('Time taken: {:.2f}s'.format(time.time()-start))

        return reg.predict(X)


    def prepare_data(self, scoring: bool):
        """
        Prepares data before pipeline fitting or inference.

        Parameters
        ----------
        scoring : bool
            Whether or not to return scoring metrics (false for new data)

        Returns
        -------
        X : np.array
            The transformed feature matrix
        y : np.array
            The target vector
        """
        df = self.df
        df.loc[:, 'Cluster'] = self.clusters

        df.drop(['Longitude'], axis=1, inplace=True)
        df.rename({'Latitude': 'Incident_Number'}, axis=1, inplace=True)

        cols_with_hour = ['Cluster', 'year', 'month', 'day', 'hour']
        cols_without_hour = ['Cluster', 'year', 'month', 'day']

        cols = cols_with_hour if self.include_hour else cols_without_hour

        df_agg = df.groupby(cols).count().reset_index()
        X = df_agg[cols]
        y = df_agg[['Incident_Number']].values.reshape((-1,))

        """
        if self.include_hour:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day', 'hour']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day', 'hour']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))
        else:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))
        """

        if scoring:
            return X, y
        else:
            return X, None


