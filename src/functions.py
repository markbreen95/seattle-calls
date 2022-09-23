from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import joblib
import time


class ReturnCluster(BaseEstimator, TransformerMixin):
    """
    A class to allow clusters to be returned in the sklearn pipeline.

    ...

    Attributes
    ----------
    clusters : int
        number of clusters to use for the KMeans algorithm

    """

    def __init__(self, clusters=5): 
        """
        Constructs all the necessary attributes for the cluster return object.

        Parameters
        ----------
            clusters : int
                number of clusters to use for the KMeans algorithm
        """
        self.clusters = clusters
           
    def fit(self, X):
        """
        Fits the KMeans clustering algorithm to supplied array.

        Parameters
        ----------
        X : np.array
            The numpy array containing latitude and longitude coordinates

        Returns
        -------
        None
        """
        self.X=X
        self.model = KMeans(n_clusters = self.clusters, n_init=10, max_iter=300, random_state=42)
        self.model.fit(self.X)
        return self
       
    def transform(self, X):
        """
        Transforms a numpy array of coordinates to return k means clusters.

        Parameters
        ----------
        X : np.array
            The numpy array containing latitude and longitude coordinates

        Returns
        -------
        np.array
            The numpy array containing clusters for input coordinates
        """
        self.X=X
        X_ = X.copy() 
        return self.model.predict(X_)


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
            ('cluster', ReturnCluster(clusters=6))
        ])

        clusters = cluster_pipeline.fit_transform(df[['Longitude', 'Latitude']])

        joblib.dump(cluster_pipeline, 'pickled_objects/cluster_pipeline.pkl')

        self.clusters = clusters

        X, y  = self.prepare_data(scoring=True)

        reg = HistGradientBoostingRegressor()
        reg.fit(X, y)

        joblib.dump(reg, 'pickled_objects/regressor.pkl')
    
        print('Training R2: {:.2f}'.format(reg.score(X, y)))
        print('Time taken: {:.2f}s'.format(time.time()-start))

    def inference(self, scoring=True):
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

        cluster_pipeline = joblib.load('pickled_objects/cluster_pipeline.pkl')

        clusters = cluster_pipeline.transform(df[['Longitude', 'Latitude']])

        self.clusters = clusters

        X, y = self.prepare_data(scoring)

        reg = joblib.load('pickled_objects/regressor.pkl')
        
        if scoring:
            print('Inference R2: {:.2f}'.format(reg.score(X, y)))
        print('Time taken: {:.2f}s'.format(time.time()-start))

        return reg.predict(X)


    def prepare_data(self, scoring):
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

        if self.include_hour:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day', 'hour']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day', 'hour']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))
        else:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))

        if scoring:
            return X, y
        else:
            return X, None


